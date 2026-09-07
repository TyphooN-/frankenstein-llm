import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
# Running the script puts scripts/ on sys.path; loading it out of band has to
# arrange the same thing, or its shared display module is not importable.
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('serve_model', ROOT / 'scripts/serve-model.py')
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

import model_catalog


def test_listing_leads_with_the_weight_filename_not_the_alias(capsys):
    assert m.main(['--list']) == 0
    out = capsys.readouterr().out
    # The engineering identity is the primary label. "heretic" is a handle that
    # names no file on this disk, and the publisher's release name names a
    # different string again, so neither may lead the line.
    assert 'RVN-Q6_K-multilingual-mtp.gguf (General-purpose Hermes agent and tool use)' in out
    assert 'Qwen3.8-27B-Heretic-Abliterated-Uncensored' not in out
    assert 'Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf (Coding and repository-agent work)' in out
    assert 'Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf' in out
    assert '[compatibility alias: heretic]' in out
    for line in out.splitlines():
        assert not line.startswith('heretic'), line


def test_every_listed_preset_leads_with_identity_and_tags_its_alias(capsys):
    assert m.main(['--list']) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    registry = model_catalog.presets(ROOT / 'llama-models.ini')
    catalog = model_catalog.load_catalog(ROOT / 'config/model-catalog.json')
    assert len(lines) == len(registry)
    for line, (alias, preset) in zip(lines, sorted(registry.items())):
        first_artifact = model_catalog.artifacts(preset)[0]
        assert line.startswith(first_artifact), (alias, line)
        for artifact in model_catalog.artifacts(preset):
            assert artifact in line, (alias, artifact)
        assert f"({catalog[alias]['purpose']})" in line, alias
        assert line.rstrip().endswith(f"[compatibility alias: {alias}]"), alias


def test_listing_details_add_source_and_paths_without_replacing_identity(capsys):
    assert m.main(['--list', '--details']) == 0
    out = capsys.readouterr().out
    assert 'source: Qwen3.8-27B-Heretic-Abliterated-Uncensored Q6_K' in out
    assert 'model: /home/typhoon/git/frankenstein-llm/models/RVN-Q6_K-multilingual-mtp.gguf' in out
    assert 'projector: /home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf' in out
    assert 'RVN-Q6_K-multilingual-mtp.gguf (General-purpose Hermes agent and tool use)' in out


def test_catalog_covers_every_preset():
    catalog = model_catalog.load_catalog(ROOT / 'config/model-catalog.json')
    assert set(catalog) == set(model_catalog.presets(ROOT / 'llama-models.ini'))


def test_listing_fails_closed_on_an_uncatalogued_preset(tmp_path, capsys):
    catalog = tmp_path / 'catalog.json'
    catalog.write_text('{"ridge": {"source": "s", "purpose": "p"}}')
    assert m.main(['--list', '--catalog', str(catalog)]) == 2
    assert 'missing a source and purpose' in capsys.readouterr().err


def test_listing_fails_closed_on_a_blank_catalog_field(tmp_path, capsys):
    catalog = tmp_path / 'catalog.json'
    catalog.write_text('{"ridge": {"source": "s", "purpose": "  "}}')
    assert m.main(['--list', '--catalog', str(catalog)]) == 2
    assert 'invalid catalog entry: ridge' in capsys.readouterr().err


def test_the_source_label_is_not_the_artifact_identity():
    """The distinction this module exists for, stated against the real files."""
    registry = model_catalog.presets(ROOT / 'llama-models.ini')
    catalog = model_catalog.load_catalog(ROOT / 'config/model-catalog.json')
    assert model_catalog.identity(registry['heretic']) == 'RVN-Q6_K-multilingual-mtp.gguf'
    assert catalog['heretic']['source'] == 'Qwen3.8-27B-Heretic-Abliterated-Uncensored Q6_K'
    assert model_catalog.artifacts(registry['obliterated-vision']) == [
        'Qwen3.8-27B-OBLITERATED-Q6_K.gguf', 'Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf']
    # Same weights, different projector: the identity has to tell them apart.
    assert (model_catalog.identity(registry['obliterated'])
            != model_catalog.identity(registry['obliterated-vision']))


def test_every_preset_artifact_is_the_path_the_ini_serves():
    registry = model_catalog.presets(ROOT / 'llama-models.ini')
    for alias, preset in registry.items():
        cmd = m.command(alias, dict(preset), {'host': '127.0.0.1', 'port': 8080})
        served = [Path(cmd[cmd.index(f'--{key}') + 1]).name
                  for key in ('model', 'mmproj') if f'--{key}' in cmd]
        assert served == model_catalog.artifacts(preset), alias


def test_local_registry_is_empty_rather_than_raising_on_unreadable_config(monkeypatch, tmp_path):
    monkeypatch.setattr(model_catalog, 'PRESETS', tmp_path / 'absent.ini')
    monkeypatch.setattr(model_catalog, 'CATALOG', tmp_path / 'absent.json')
    assert model_catalog.local_registry() == ({}, {})
    assert model_catalog.describe_alias('heretic', {}, {}) == 'heretic (not a configured preset)'


def test_the_alias_is_labelled_as_compatibility_metadata_not_as_a_name():
    registry = model_catalog.presets(ROOT / 'llama-models.ini')
    catalog = model_catalog.load_catalog(ROOT / 'config/model-catalog.json')
    line = model_catalog.labelled('heretic', registry, catalog)
    assert line.startswith('RVN-Q6_K-multilingual-mtp.gguf')
    assert line.endswith('[compatibility alias: heretic]')
    # An id with no preset has no identity to lead with; that is the only case
    # where the bare handle is all there is to print.
    assert model_catalog.labelled('zeta', registry, catalog) == 'zeta (not a configured preset)'


def test_all_presets_have_wrappers():
    for alias in m.presets(ROOT / 'llama-models.ini'):
        assert (ROOT / f'scripts/serve-{alias}.sh').is_file()


def test_shared_and_specific_settings_merge():
    presets = m.presets(ROOT / 'llama-models.ini')
    values = presets['qwen3-coder-next']
    assert values['ctx-size'] == '65536'
    # A preset's own placement has to win over the shared [*] default, and this
    # is the preset that needs all three cards. Asserting that it overrides,
    # rather than asserting the numbers of the day, keeps the merge covered
    # while placement stays a thing the evaluator is allowed to change.
    assert values['tensor-split'] != presets['heretic']['tensor-split']
    assert len(values['tensor-split'].split(',')) == 3
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
