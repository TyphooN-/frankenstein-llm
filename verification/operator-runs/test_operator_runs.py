import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('operator_runs', ROOT / 'scripts/model-runs.py')
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_config_defaults_and_cli_override(tmp_path):
    p = tmp_path / 'settings.json'
    p.write_text('{"schema": 1, "benchmark": {"threads": 7, "repetitions": 2}, "qualify": {"timeout": 120}}')
    a = m.configured_args(['benchmark', '--model', '/tmp/a.gguf', '--config', str(p), '--threads', '3'])
    assert a.threads == 3 and a.repetitions == 2 and not a.execute


def test_config_cannot_authorize_execution(tmp_path):
    p = tmp_path / 'settings.json'
    p.write_text('{"schema": 1, "benchmark": {"execute": true}}')
    with pytest.raises(ValueError):
        m.configured_args(['benchmark', '--model', '/tmp/a.gguf', '--config', str(p)])


def test_real_child_report_and_timeout(tmp_path):
    a = m.parser().parse_args(['qualify', '--timeout', '1'])
    ok = tmp_path / 'ok'; ok.mkdir()
    assert m.execute(a, [sys.executable, '-c', 'print("fixture only")'], ok) == 0
    import json
    assert json.loads((ok / 'report.json').read_text())['status'] == 'passed'
    slow = tmp_path / 'slow'; slow.mkdir()
    assert m.execute(a, [sys.executable, '-c', 'import time; time.sleep(30)'], slow) == 124
    assert json.loads((slow / 'report.json').read_text())['status'] == 'interrupted'


def test_default_topology_remains_three_gpus():
    a = m.configured_args(['benchmark', '--model', '/tmp/a.gguf'])
    assert a.devices == 'ROCm0/ROCm1/ROCm2'
    assert a.gpu_layers == 999
    assert len(a.tensor_split.split('/')) == 3


def test_benchmark_split_matches_the_router_default_proportions():
    """Two files configure a split; only their ratio has to agree.

    ``config/model-runs.json`` and the ``[*]`` section of ``llama-models.ini``
    are edited by hand at different times, and a benchmark that measures a
    different placement than the router serves measures the wrong thing. Only
    the ratio is compared, because ``--tensor-split`` takes proportions:
    ``6/6/6`` and ``1,1,1`` are the same placement written two ways, and an
    assertion on the literal would fail on an edit that changed nothing.
    """
    import configparser
    from math import gcd
    from functools import reduce

    def ratio(text):
        values = [int(part) for part in text.replace('/', ',').split(',')]
        divisor = reduce(gcd, values, 0)
        return [value // divisor for value in values] if divisor else values

    ini = configparser.ConfigParser(interpolation=None)
    with (ROOT / 'llama-models.ini').open() as handle:
        ini.read_file(handle)
    a = m.configured_args(['benchmark', '--model', '/tmp/a.gguf'])
    assert ratio(a.tensor_split) == ratio(ini['*']['tensor-split'])
    assert ratio(m.parser().parse_args(
        ['benchmark', '--model', '/tmp/a.gguf']).tensor_split) == ratio(
            ini['*']['tensor-split'])


def test_gpu_selection_is_explicit():
    a = m.parser().parse_args(['benchmark', '--model', '/tmp/a.gguf', '--devices', 'ROCm0/ROCm1', '--tensor-split', '1/2'])
    cmd = m.command(a)
    assert cmd[cmd.index('-dev') + 1] == 'ROCm0/ROCm1'
    assert cmd[cmd.index('-ts') + 1] == '1/2'


def test_defaults_are_plan_only():
    a = m.parser().parse_args(['qualify'])
    assert not a.execute
    assert m.command(a)[-1].endswith('run_qualification.py')


def test_benchmark_requires_model():
    with pytest.raises(SystemExit):
        m.parser().parse_args(['benchmark'])


def test_benchmark_command_is_argv_not_shell(tmp_path):
    model = tmp_path / 'model;touch nope.gguf'
    model.write_bytes(b'GGUF')
    a = m.parser().parse_args(['benchmark', '--model', str(model)])
    cmd = m.command(a)
    assert cmd[cmd.index('-m') + 1] == str(model)
    assert '-o' in cmd and 'json' in cmd
    assert '--confirm-kernel' not in cmd


def test_invalid_numeric_arguments_fail():
    for value in ('0', '-1', '999999'):
        with pytest.raises(SystemExit):
            m.parser().parse_args(['benchmark', '--model', '/tmp/m.gguf', '--repetitions', value])


def test_unknown_mode_fails():
    with pytest.raises(SystemExit):
        m.parser().parse_args(['benchmak'])


def test_process_scan_does_not_read_cmdline(tmp_path):
    p = tmp_path / '10'; p.mkdir()
    (p / 'stat').write_text('10 (makepkg) S 1 1 1')
    (p / 'cmdline').mkdir()
    assert m.process_blockers(tmp_path) == ['10:makepkg']


def test_zombies_are_not_workloads(tmp_path):
    p = tmp_path / '10'; p.mkdir()
    (p / 'stat').write_text('10 (makepkg) Z 1 1 1')
    assert m.process_blockers(tmp_path) == []


def test_storage_requires_explicit_healthy_message():
    assert m.storage_healthy(0, 'all pools are healthy\n')
    assert not m.storage_healthy(0, 'pool ONLINE\nerrors: 1 data errors')
    assert not m.storage_healthy(1, 'permission denied')


def test_btrfs_preflight_uses_the_actual_filesystem_not_zpool(monkeypatch, tmp_path):
    import json
    import subprocess

    calls = []
    counters = ''.join(f'[/dev/mapper/root].{key} 0\n' for key in (
        'write_io_errs', 'read_io_errs', 'flush_io_errs',
        'corruption_errs', 'generation_errs'))

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'findmnt':
            return subprocess.CompletedProcess(argv, 0, json.dumps({'filesystems': [{
                'target': '/home', 'source': '/dev/mapper/root[/@home]',
                'fstype': 'btrfs', 'options': 'rw,relatime',
            }]}), '')
        if argv == ['btrfs', 'device', 'stats', '--check', '/home']:
            return subprocess.CompletedProcess(argv, 0, counters, '')
        return subprocess.CompletedProcess(argv, 1, '', 'not applicable to Btrfs')

    read_text = Path.read_text

    def fixture_read_text(path, *args, **kwargs):
        if path == Path('/proc/meminfo'):
            return 'MemAvailable: 67108864 kB\n'
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m.subprocess, 'run', run)
    monkeypatch.setattr(m, 'process_blockers', lambda **kwargs: [])
    monkeypatch.setattr(Path, 'read_text', fixture_read_text)
    m.preflight(m.parser().parse_args(['qualify']))
    assert any(argv[0] == 'findmnt' for argv in calls)
    assert any(argv[:4] == ['btrfs', 'device', 'stats', '--check'] for argv in calls)
    assert not any(argv[0] == 'zpool' for argv in calls)


def test_benchmark_execute_requires_kernel_ack(capsys):
    assert m.main(['benchmark', '--model', '/tmp/m.gguf', '--execute']) == 2
    assert 'confirm-kernel' in capsys.readouterr().err


def test_qualification_plan_does_not_probe_or_execute(monkeypatch, capsys):
    def forbidden(*a, **k):
        raise AssertionError('plan performed side effects')
    monkeypatch.setattr(m.subprocess, 'run', forbidden)
    monkeypatch.setattr(m.subprocess, 'Popen', forbidden)
    assert m.main(['qualify']) == 0
    assert 'plan_only' in capsys.readouterr().out


def btrfs_stats(device='/dev/mapper/root', **values):
    return ''.join(f'[{device}].{key} {values.get(key, 0)}\n' for key in (
        'write_io_errs', 'read_io_errs', 'flush_io_errs',
        'corruption_errs', 'generation_errs'))


def mount_observation(**changes):
    row = {'target': '/home', 'source': '/dev/mapper/root[/@home]',
           'fstype': 'btrfs', 'options': 'rw,relatime'}
    row.update(changes)
    return {'filesystems': [row]}


@pytest.mark.parametrize('rc,text,healthy', [
    (0, btrfs_stats(), True),
    (0, btrfs_stats() + btrfs_stats('/dev/mapper/second'), True),
    (64, btrfs_stats(), False),
    (0, btrfs_stats(corruption_errs=1), False),
    (0, btrfs_stats(read_io_errs=1), False),
    (0, '', False),
    (0, 'all pools are healthy\n', False),
    (0, btrfs_stats().splitlines()[0], False),
    (0, btrfs_stats() + btrfs_stats().splitlines()[0], False),
    (0, btrfs_stats() + 'unexpected output\n', False),
    (0, btrfs_stats() + '[/dev/mapper/second].write_io_errs 0\n', False),
    (0, btrfs_stats().replace('generation_errs 0', 'generation_errs -1'), False),
])
def test_btrfs_requires_complete_zero_counters_for_every_device(rc, text, healthy):
    assert m.btrfs_healthy(rc, text) is healthy


@pytest.mark.parametrize('rc,payload', [
    (1, mount_observation()),
    (0, 'not-json'),
    (0, {}),
    (0, {'filesystems': []}),
    (0, {'filesystems': [None]}),
    (0, {'filesystems': [mount_observation()['filesystems'][0]] * 2}),
    (0, mount_observation(source=None)),
    (0, mount_observation(target='relative')),
    (0, mount_observation(options='ro,rw,relatime')),
    (0, mount_observation(options='relatime')),
    (0, mount_observation(fstype='unknown')),
])
def test_unusable_mount_observations_refuse_before_health_commands(monkeypatch, tmp_path, rc, payload):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == 'findmnt'
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return subprocess.CompletedProcess(argv, rc, body, '')

    monkeypatch.setattr(m.subprocess, 'run', run)
    with pytest.raises(RuntimeError):
        m.storage_preflight([tmp_path])
    assert len(calls) == 1


def test_storage_resolves_each_input_and_output_and_preserves_zfs_checks(monkeypatch, tmp_path):
    output = tmp_path / 'results'
    model = tmp_path / 'weights.gguf'
    paths = [output, model, tmp_path / 'other-weights.gguf']
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'findmnt':
            observation = (mount_observation() if argv[3] == str(output) else
                           mount_observation(target='/models', source='pool/models', fstype='zfs'))
            return subprocess.CompletedProcess(argv, 0, json.dumps(observation), '')
        if argv[0] == 'btrfs':
            return subprocess.CompletedProcess(argv, 0, btrfs_stats(), '')
        if argv == ['zpool', 'status', '-x']:
            return subprocess.CompletedProcess(argv, 0, 'all pools are healthy\n', '')
        raise AssertionError(argv)

    monkeypatch.setattr(m.subprocess, 'run', run)
    m.storage_preflight(paths)
    assert [argv[3] for argv in calls if argv[0] == 'findmnt'] == list(map(str, paths))
    assert sum(argv[0] == 'btrfs' for argv in calls) == 1
    assert sum(argv[0] == 'zpool' for argv in calls) == 1


@pytest.mark.parametrize('failure', ['healthy', 'btrfs-error', 'btrfs-missing', 'findmnt-timeout', 'zfs-error'])
def test_storage_failure_never_starts_a_child_or_creates_a_run(
        monkeypatch, runnable_benchmark, capsys, failure):
    args, attempts, root = runnable_benchmark

    def run(argv, **kwargs):
        if argv[0] == 'findmnt':
            if failure == 'findmnt-timeout':
                raise subprocess.TimeoutExpired(argv, 15)
            payload = mount_observation(fstype='zfs' if failure == 'zfs-error' else 'btrfs')
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload), '')
        if failure == 'btrfs-missing':
            raise FileNotFoundError('missing btrfs fixture')
        if failure == 'healthy':
            return subprocess.CompletedProcess(argv, 0, btrfs_stats(), '')
        return subprocess.CompletedProcess(argv, 64, btrfs_stats(corruption_errs=1), '')

    monkeypatch.setattr(m.subprocess, 'run', run)
    code = m.main(args)
    if failure == 'healthy':
        # The shared setup must reach Popen; its observer deliberately creates
        # no actual child and makes execute return a recorded failure.
        assert code == 1
        assert len(attempts) == 1
        assert len(list((root / 'logs/model-runs').iterdir())) == 1
    else:
        assert code == 75
        assert 'Preflight refused' in capsys.readouterr().err
        assert attempts == []
        assert not (root / 'logs').exists()


def test_healthy_storage_does_not_ignore_an_existing_inference_owner(monkeypatch, tmp_path):
    def run(argv, **kwargs):
        text = json.dumps(mount_observation()) if argv[0] == 'findmnt' else btrfs_stats()
        return subprocess.CompletedProcess(argv, 0, text, '')

    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m.subprocess, 'run', run)
    monkeypatch.setattr(m, 'process_blockers', lambda **kwargs: ['123:llama-server'])
    with pytest.raises(RuntimeError, match='active competing workload'):
        m.preflight(m.parser().parse_args(['benchmark', '--model', str(tmp_path / 'not-loaded.gguf')]))


def test_healthy_readonly_model_storage_remains_usable(monkeypatch, tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        text = (json.dumps(mount_observation(options='ro,relatime'))
                if argv[0] == 'findmnt' else btrfs_stats())
        return subprocess.CompletedProcess(argv, 0, text, '')

    monkeypatch.setattr(m.subprocess, 'run', run)
    m.storage_preflight([tmp_path / 'readonly-model.gguf'])
    assert [argv[0] for argv in calls] == ['findmnt', 'btrfs']


@pytest.fixture
def runnable_benchmark(monkeypatch, tmp_path):
    """Satisfy every non-storage guard; observe dispatch without a real child."""
    import struct

    root = tmp_path / 'repository'
    root.mkdir()
    config = root / 'settings.json'
    config.write_text('{"schema": 1}')
    model = root / 'fixture.gguf'
    # Complete minimal GGUF v3 container, not real model weights.
    model.write_bytes(struct.pack('<4sIQQ', b'GGUF', 3, 0, 0))
    binary = root / 'llama-bench-fixture'
    binary.write_text('#!/bin/sh\nexit 99\n')
    binary.chmod(0o700)
    qualification = root / 'qualification'
    qualification.mkdir()
    attempts = []
    read_text = Path.read_text

    def fixture_read_text(path, *args, **kwargs):
        if path == Path('/proc/meminfo'):
            return 'MemAvailable: 67108864 kB\n'
        return read_text(path, *args, **kwargs)

    def popen(*args, **kwargs):
        attempts.append(args[0])
        raise OSError('offline dispatch observer; no child was created')

    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(m, 'QUALIFICATION', qualification)
    monkeypatch.setattr(m, 'BIN', binary)
    monkeypatch.setattr(m, 'process_blockers', lambda **kwargs: [])
    monkeypatch.setattr(Path, 'read_text', fixture_read_text)
    monkeypatch.setattr(m.subprocess, 'Popen', popen)
    args = ['benchmark', '--model', str(model), '--config', str(config),
            '--confirm-kernel', '--execute']
    return args, attempts, root


def test_benchmark_reuses_one_resolved_model_path(monkeypatch, runnable_benchmark):
    args, attempts, root = runnable_benchmark
    model = root / 'fixture.gguf'
    resolutions, storage_paths, header_paths = [], [], []
    original_resolve, original_open = Path.resolve, Path.open

    def observe_resolve(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        if path == model:
            resolutions.append(resolved)
        return resolved

    def observe_header(path, *args, **kwargs):
        if path == model:
            header_paths.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', observe_resolve)
    monkeypatch.setattr(Path, 'open', observe_header)
    monkeypatch.setattr(m, 'storage_preflight', lambda paths: storage_paths.append(tuple(paths)))

    # The existing fixture observes the would-be dispatch without a child.
    assert m.main(args) == 1
    assert len(attempts) == 1
    command = attempts[0]
    assert command[command.index('-m') + 1] == str(model)
    assert len(storage_paths) == 1 and storage_paths[0][1] == model
    assert header_paths == [model]
    assert resolutions == [model], 'benchmark must resolve its model path exactly once'


def test_actual_symlinked_output_storage_refuses_before_dispatch(
        monkeypatch, tmp_path, runnable_benchmark):
    args, attempts, root = runnable_benchmark
    external = tmp_path / 'separate-results'
    external.mkdir()
    (root / 'logs').symlink_to(external, target_is_directory=True)
    output = external / 'model-runs'
    state = {'healthy': True}
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'findmnt':
            path = Path(argv[3])
            containing = external if path.is_relative_to(external) else root
            payload = mount_observation(target=str(containing),
                                        source='/dev/output' if containing == external else '/dev/root')
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload), '')
        assert argv[:4] == ['btrfs', 'device', 'stats', '--check']
        unhealthy = argv[-1] == str(external) and not state['healthy']
        return subprocess.CompletedProcess(
            argv, 64 if unhealthy else 0,
            btrfs_stats(corruption_errs=1) if unhealthy else btrfs_stats(), '')

    monkeypatch.setattr(m.subprocess, 'run', run)
    # Positive control proves the identical model/binary/process/RAM setup reaches
    # the real production dispatch boundary, where our observer creates no child.
    assert m.main(args) == 1
    assert len(attempts) == 1
    before = set(output.iterdir())
    assert len(before) == 1
    state['healthy'] = False
    code = m.main(args)
    assert code == 75, (code, len(attempts))
    assert len(attempts) == 1
    assert set(output.iterdir()) == before
    assert any(argv[0] == 'findmnt' and Path(argv[3]).is_relative_to(external)
               for argv in calls)


@pytest.mark.parametrize('mode', ['benchmark', 'qualify'])
@pytest.mark.parametrize('failure', ['corrupt-btrfs', 'unsupported'])
@pytest.mark.parametrize('layout', [
    'logs-mount', 'output-mount', 'logs-symlink', 'output-symlink',
    'dangling-output-symlink',
])
def test_actual_output_mount_refuses_without_creating_run_directories(
        monkeypatch, tmp_path, runnable_benchmark, mode, failure, layout):
    args, attempts, root = runnable_benchmark
    logs = root / 'logs'
    if layout == 'logs-mount':
        logs.mkdir()
        mount = logs
    elif layout == 'output-mount':
        mount = logs / 'model-runs'
        mount.mkdir(parents=True)
    else:
        mount = tmp_path / 'output-device'
        mount.mkdir()
        if layout == 'logs-symlink':
            logs.symlink_to(mount, target_is_directory=True)
        else:
            logs.mkdir()
            target = mount / 'not-created' if layout == 'dangling-output-symlink' else mount
            (logs / 'model-runs').symlink_to(target, target_is_directory=True)
    output = (logs / 'model-runs').resolve()
    before = set(output.iterdir()) if output.exists() else None
    if mode == 'qualify':
        (root / 'models').mkdir()
        args = ['qualify', '--config', str(root / 'settings.json'), '--execute']
    state = {'healthy': False}
    probes = []

    def run(argv, **kwargs):
        if argv[0] == 'findmnt':
            path = Path(argv[3])
            probes.append(path)
            is_output = path.is_relative_to(mount)
            fstype = 'ext4' if is_output and not state['healthy'] and failure == 'unsupported' else 'btrfs'
            payload = mount_observation(
                target=str(mount if is_output else root),
                source='/dev/output' if is_output else '/dev/root', fstype=fstype)
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload), '')
        assert argv[:4] == ['btrfs', 'device', 'stats', '--check']
        unhealthy = argv[-1] == str(mount) and not state['healthy']
        return subprocess.CompletedProcess(
            argv, 64 if unhealthy else 0,
            btrfs_stats(corruption_errs=1) if unhealthy else btrfs_stats(), '')

    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m.main(args) == 75
    assert attempts == []
    assert probes[-1] == mount  # Existing output, or its nearest existing ancestor.
    if before is None:
        assert not output.exists()
    else:
        assert set(output.iterdir()) == before
    # Change only output storage health; all other inputs/guards are identical.
    state['healthy'] = True
    assert m.main(args) == 1
    assert len(attempts) == 1
    assert output.is_dir()
