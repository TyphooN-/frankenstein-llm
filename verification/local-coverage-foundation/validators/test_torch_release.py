"""Runtime workspace release is cleanup, never an exemption from unload checks."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gatelib


def fake_torch(failing_sync=False):
    events = []
    state = {'allocated': 32 << 20, 'reserved': 32 << 20}

    def sync(index):
        events.append(('sync', index))
        if failing_sync and index == 0:
            raise RuntimeError('sync failed')

    def clear():
        events.append('workspaces')
        state['allocated'] = 0

    def empty():
        events.append('cache')
        state['reserved'] = state['allocated']

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 3,
                             synchronize=sync, empty_cache=empty),
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear))
    return torch, events, state


def test_workspace_is_released_before_allocator_cache():
    torch, events, state = fake_torch()
    assert gatelib.release_torch_memory(torch) == []
    assert events == [('sync', 0), ('sync', 1), ('sync', 2), 'workspaces', 'cache']
    assert state == {'allocated': 0, 'reserved': 0}


def test_cleanup_failure_is_recorded_without_skipping_other_devices():
    torch, events, state = fake_torch(failing_sync=True)
    errors = gatelib.release_torch_memory(torch)
    assert len(errors) == 1 and 'sync failed' in errors[0]
    assert ('sync', 2) in events and events[-1] == 'cache'


def test_absent_private_api_does_not_hide_retained_allocation():
    torch, events, state = fake_torch()
    torch._C = SimpleNamespace()
    assert gatelib.release_torch_memory(torch) == []
    verdict = gatelib.unload_verdict({'card0': 0}, {'card0': state['reserved']},
        allocator={'cuda:0': {'allocated_bytes': state['allocated'], 'reserved_bytes': state['reserved']}})
    assert verdict['pass'] is False
