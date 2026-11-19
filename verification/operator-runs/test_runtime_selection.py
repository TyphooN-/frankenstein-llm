"""Benchmark plumbing proof with private synthetic files and mocked effects."""
import json
import os
from pathlib import Path

import pytest

from test_operator_runs import (btrfs_stats, m, mount_observation,
                                runnable_benchmark)


def parse_benchmark(model, *extra):
    try:
        return m.parser().parse_args(['benchmark', '--model', str(model), *extra])
    except SystemExit as error:
        pytest.fail(f'explicit benchmark projection is not supported: parser exit {error.code}')


def test_private_binary_selected_by_command_and_preflight(monkeypatch, runnable_benchmark):
    args, attempts, root = runnable_benchmark
    selected = root / 'private bench;literal'
    selected.write_text('synthetic executable identity only\n')
    selected.chmod(0o700)
    monkeypatch.setattr(m, 'BIN', root / 'missing-default')
    a = parse_benchmark(root / 'fixture.gguf', '--bench-binary', str(selected))
    command = m.command(a)
    assert command[0] == str(selected)
    assert command.count(str(selected)) == 1
    monkeypatch.setattr(m, 'storage_preflight', lambda paths: None)
    access = os.access
    checked = []

    def executable(path, mode):
        checked.append((path, mode))
        return access(path, mode)

    monkeypatch.setattr(m.os, 'access', executable)
    m.preflight(a)
    assert checked == [(selected, os.X_OK)]
    assert attempts == []


def test_cpu_none_uses_no_rocm_split(tmp_path):
    a = parse_benchmark(tmp_path / 'fixture.gguf', '--devices', 'none', '--gpu-layers', '0')
    try:
        command = m.command(a)
    except ValueError as error:
        pytest.fail(f'explicit native CPU device selection is rejected: {error}')
    assert command[command.index('-dev') + 1] == 'none'
    assert command[command.index('-ngl') + 1] == '0'
    assert '-ts' not in command
    assert not any('ROCm' in value for value in command)


def test_safe_cpu_smoke_projection(tmp_path):
    # The added controls are opt-in; use the same native spellings as the
    # source-bound parser, not arbitrary extra argv or a new tuning config.
    a = parse_benchmark(tmp_path / 'fixture.gguf', '--load-mode', 'mmap',
                        '--lazy-mode', 'on', '--no-repack', '--no-warmup', '--offline',
                        '--prompt-tokens', '16', '--generation-tokens', '8',
                        '--repetitions', '1', '--threads', '2', '--timeout', '300')
    command = m.command(a)
    assert command[-8:] == ['-lm', 'mmap', '-lzm', 'on', '--repack', '0',
                            '--no-warmup', '--offline']
    assert command[command.index('-p') + 1] == '16'
    assert command[command.index('-n') + 1] == '8'
    assert command[command.index('-r') + 1] == '1'
    assert command[command.index('-t') + 1] == '2'
    assert a.timeout == 300 and not a.execute


@pytest.mark.parametrize('devices,layers,split,reason', [
    ('none', '1', '1', 'CPU-only'),
    ('none/ROCm0', '0', '1/1', 'devices'),
    ('CPU', '0', '1', 'devices'),
    ('ROCm0,ROCm1', '1', '1/1', 'devices'),
    ('ROCm0/ROCm1', '1', '1', 'match devices'),
    ('ROCm0', '1', '-1', 'nonnegative'),
    ('ROCm0', '1', '0', 'positive share'),
    ('none', '0', 'broken', 'nonnegative'),
])
def test_cpu_and_gpu_selection_validation_stays_strict(tmp_path, devices, layers, split, reason):
    a = parse_benchmark(tmp_path / 'fixture.gguf', '--devices', devices,
                        '--gpu-layers', layers, '--tensor-split', split)
    with pytest.raises(ValueError, match=reason):
        m.command(a)


@pytest.mark.parametrize('flag,value', [('--load-mode', 'mlock'), ('--load-mode', 'none'),
                                       ('--lazy-mode', 'off'), ('--lazy-mode', 'invalid')])
def test_projection_is_only_the_requested_mmap_lazy_surface(tmp_path, flag, value):
    with pytest.raises(SystemExit) as refused:
        m.parser().parse_args(['benchmark', '--model', str(tmp_path / 'fixture.gguf'), flag, value])
    assert refused.value.code == 2


@pytest.mark.parametrize('field,value', [
    ('bench_binary', '/private/llama-bench'), ('load_mode', 'mmap'), ('lazy_mode', 'on'),
    ('no_repack', True), ('no_warmup', True), ('offline', True),
])
def test_new_controls_are_cli_only_not_persistent_config(tmp_path, field, value):
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps({'schema': 1, 'benchmark': {field: value}}))
    with pytest.raises(ValueError, match='unknown or unsafe benchmark config keys'):
        m.configured_args(['benchmark', '--model', str(tmp_path / 'fixture.gguf'), '--config', str(config)])


def test_defaults_do_not_project_new_flags(tmp_path):
    a = parse_benchmark(tmp_path / 'fixture.gguf', '--threads', '2')
    assert m.command(a) == [str(m.BIN), '-m', str(tmp_path / 'fixture.gguf'),
                            '-dev', 'ROCm0/ROCm1/ROCm2', '-ts', '1/1/1',
                            '-t', '2', '-ngl', '999', '-p', '512', '-n', '128',
                            '-r', '3', '-o', 'json']


@pytest.mark.parametrize('flag', ['--bench-binary', '--load-mode', '--lazy-mode',
                                  '--no-repack', '--no-warmup', '--offline'])
def test_qualify_mode_does_not_accept_benchmark_controls(flag):
    values = {'--bench-binary': '/private/llama-bench', '--load-mode': 'mmap', '--lazy-mode': 'on'}
    with pytest.raises(SystemExit) as refused:
        m.parser().parse_args(['qualify', flag, *([values[flag]] if flag in values else [])])
    assert refused.value.code == 2


def healthy_storage(monkeypatch):
    def observe(argv, **kwargs):
        text = json.dumps(mount_observation()) if argv[0] == 'findmnt' else btrfs_stats()
        return m.subprocess.CompletedProcess(argv, 0, text, '')
    monkeypatch.setattr(m.subprocess, 'run', observe)


@pytest.mark.parametrize('kind', ['valid', 'missing', 'not-executable', 'directory'])
def test_selected_binary_preflight_and_mock_dispatch(monkeypatch, runnable_benchmark, kind, capsys):
    args, attempts, root = runnable_benchmark
    selected = root / 'selected-bench'
    if kind == 'directory':
        selected.mkdir()
    elif kind != 'missing':
        selected.write_text('not executed\n')
        selected.chmod(0o700 if kind == 'valid' else 0o600)
    healthy_storage(monkeypatch)
    args += ['--bench-binary', str(selected), '--devices', 'none', '--gpu-layers', '0',
             '--load-mode', 'mmap', '--lazy-mode', 'on', '--no-repack', '--no-warmup', '--offline']
    code = m.main(args)
    if kind == 'valid':
        assert code == 1  # Exact Popen observer raises before child creation.
        assert len(attempts) == 1 and attempts[0][0] == str(selected)
    else:
        assert code == 75
        assert 'llama-bench is not built' in capsys.readouterr().err
        assert attempts == [] and not (root / 'logs').exists()


@pytest.mark.parametrize('blocker', ['140890:llama-server', '156118:llama-server',
                                    '56113:make', '56170:clang'])
def test_private_runtime_does_not_bypass_competing_workloads(monkeypatch, runnable_benchmark, blocker, capsys):
    args, attempts, root = runnable_benchmark
    healthy_storage(monkeypatch)
    monkeypatch.setattr(m, 'process_blockers', lambda **kwargs: [blocker])
    args += ['--bench-binary', str(m.BIN), '--devices', 'none', '--gpu-layers', '0']
    assert m.main(args) == 75
    assert blocker in capsys.readouterr().err
    assert attempts == [] and not (root / 'logs').exists()


def test_private_runtime_does_not_bypass_ram_floor(monkeypatch, runnable_benchmark, capsys):
    args, attempts, root = runnable_benchmark
    healthy_storage(monkeypatch)
    read_text = Path.read_text

    def low_memory(path, *a, **kw):
        return 'MemAvailable: 1024 kB\n' if path == Path('/proc/meminfo') else read_text(path, *a, **kw)

    monkeypatch.setattr(Path, 'read_text', low_memory)
    assert m.main(args + ['--bench-binary', str(m.BIN)]) == 75
    assert 'less than 16 GiB' in capsys.readouterr().err
    assert attempts == [] and not (root / 'logs').exists()


def test_private_runtime_keeps_nonblocking_lease_refusal(monkeypatch, runnable_benchmark, capsys):
    args, attempts, root = runnable_benchmark
    healthy_storage(monkeypatch)
    operations = []

    def busy(fd, operation):
        operations.append(operation)
        raise BlockingIOError('synthetic busy lease; no OS flock')

    monkeypatch.setattr(m.fcntl, 'flock', busy)
    assert m.main(args + ['--bench-binary', str(m.BIN)]) == 75
    assert operations == [m.fcntl.LOCK_EX | m.fcntl.LOCK_NB]
    assert 'lock busy' in capsys.readouterr().err
    assert attempts == [] and not (root / 'logs').exists()


def test_custom_runtime_plan_does_not_call_preflight_or_dispatch(monkeypatch, runnable_benchmark, capsys):
    args, attempts, root = runnable_benchmark
    def forbidden(*args, **kwargs):
        pytest.fail('plan reached an execution guard or dispatch')
    monkeypatch.setattr(m, 'preflight', forbidden)
    monkeypatch.setattr(m.fcntl, 'flock', forbidden)
    args = args[:-2] + ['--bench-binary', str(m.BIN), '--devices', 'none', '--gpu-layers', '0']
    assert m.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['plan_only'] is True
    assert result['command'][0] == str(m.BIN)
    assert result['command'][result['command'].index('-dev') + 1] == 'none'
    assert attempts == [] and not (root / 'logs').exists()


@pytest.mark.parametrize('backend,devices,layers,split', [
    ('cpu', 'none', '0', []),
    ('hip', 'ROCm0/ROCm1/ROCm2', '1', ['-ts', '6/6/6']),
])
def test_exact_small_cpu_and_partial_hip_plans(
        monkeypatch, runnable_benchmark, capsys, backend, devices, layers, split):
    args, attempts, root = runnable_benchmark
    selected = root / ('build-' + backend) / 'bin/llama-bench'

    def forbidden(*args, **kwargs):
        pytest.fail('small preview reached preflight or lease acquisition')

    monkeypatch.setattr(m, 'preflight', forbidden)
    monkeypatch.setattr(m.fcntl, 'flock', forbidden)
    argv = args[:-2] + [
        '--bench-binary', str(selected), '--devices', devices, '--gpu-layers', layers,
        '--tensor-split', '6/6/6', '--prompt-tokens', '16', '--generation-tokens', '8',
        '--repetitions', '1', '--threads', '2', '--timeout', '300',
        '--load-mode', 'mmap', '--lazy-mode', 'on', '--no-repack', '--no-warmup', '--offline',
    ]
    assert m.main(argv) == 0
    assert json.loads(capsys.readouterr().out) == {
        'plan_only': True, 'timeout': 300,
        'command': [str(selected), '-m', str(root / 'fixture.gguf'), '-dev', devices,
                    *split, '-t', '2', '-ngl', layers, '-p', '16', '-n', '8', '-r', '1',
                    '-o', 'json', '-lm', 'mmap', '-lzm', 'on', '--repack', '0',
                    '--no-warmup', '--offline'],
    }
    assert attempts == [] and not (root / 'logs').exists()
