import importlib.util
from pathlib import Path
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
