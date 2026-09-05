import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('serve_model', ROOT / 'scripts/serve-model.py')
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_listing_shows_full_name_and_use_case(capsys):
    assert m.main(['--list']) == 0
    out = capsys.readouterr().out
    assert 'Qwen3.8-27B-Heretic-Abliterated-Uncensored Q6_K (General-purpose Hermes agent and tool use)' in out
    assert 'Qwen3-Coder-Next Q4_K_M (Coding and repository-agent work)' in out
    assert '.gguf' not in out
    assert 'projector:' not in out
    assert 'heretic  (' not in out


def test_listing_details_retains_identifiers(capsys):
    assert m.main(['--list', '--details']) == 0
    out = capsys.readouterr().out
    assert 'alias: heretic' in out
    assert 'RVN-Q6_K-multilingual-mtp.gguf' in out


def test_catalog_covers_every_preset():
    import json
    catalog = json.loads((ROOT / 'config/model-catalog.json').read_text())
    assert set(catalog) == set(m.presets(ROOT / 'llama-models.ini'))
    assert all(set(row) == {'name', 'purpose'} and all(row.values()) for row in catalog.values())


def test_all_presets_have_wrappers():
    for alias in m.presets(ROOT / 'llama-models.ini'):
        assert (ROOT / f'scripts/serve-{alias}.sh').is_file()


def test_shared_and_specific_settings_merge():
    values = m.presets(ROOT / 'llama-models.ini')['qwen3-coder-next']
    assert values['ctx-size'] == '65536'
    assert values['tensor-split'] == '13,26,6'
    cmd = m.command('qwen3-coder-next', values, {'host': '127.0.0.1', 'port': 8080})
    assert '--no-mmap' in cmd and '--jinja' in cmd
    assert cmd[cmd.index('--alias') + 1] == 'qwen3-coder-next'


def test_unknown_preset_key_fails_closed():
    with pytest.raises(ValueError):
        m.command('x', {'model': '/tmp/m.gguf', 'shell': 'evil'}, {'host': '127.0.0.1', 'port': 8080})


def test_non_loopback_is_rejected():
    with pytest.raises(ValueError):
        m.command('x', {'model': '/tmp/m.gguf'}, {'host': '0.0.0.0', 'port': 8080})


def test_arguments_remain_literal():
    cmd = m.command('x', {'model': '/tmp/a;touch bad.gguf'}, {'host': '127.0.0.1', 'port': 8080})
    assert cmd[cmd.index('--model') + 1] == '/tmp/a;touch bad.gguf'


def test_plan_does_not_launch(monkeypatch,capsys):
    def forbidden(*a, **kw):
        raise AssertionError('unexpected exec')
    monkeypatch.setattr(m.os, 'execv', forbidden)
    assert m.main(['heretic']) == 0
    assert 'plan_only' in capsys.readouterr().out


def test_unknown_alias_is_rejected():
    assert m.main(['unknown-alias']) == 2
