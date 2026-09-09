"""Offline rebrand migration and vocabulary regressions."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('migration', ROOT / 'scripts/migrate_qualification_state.py')
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def fixture(root):
    old = root / 'verification' / migration.LEGACY
    (old / 'qualification-cache').mkdir(parents=True)
    receipt = b'{"status":"passed","key":"unchanged","updated_at":"original"}\n'
    (old / 'qualification-cache' / 'receipt.json').write_bytes(receipt)
    (old / 'mission-state.json').write_text(json.dumps({
        'steps': {'asr': {'status': 'passed', 'exit_code': 0}},
        'log': str(old / 'mission.log')}))
    return old, receipt


def test_migration_preserves_receipts_and_is_idempotent(tmp_path, monkeypatch):
    old, receipt = fixture(tmp_path)
    monkeypatch.setattr(migration, 'assert_no_supervisor', lambda root: None)
    assert len(migration.migrate(tmp_path)) == 2
    new = tmp_path / 'verification' / migration.CURRENT
    assert not new.exists()
    migration.migrate(tmp_path, True)
    assert (new / 'qualification-cache/receipt.json').read_bytes() == receipt
    state = json.loads((new / 'qualification-state.json').read_text())
    assert state['steps']['asr'] == {'status': 'passed', 'exit_code': 0}
    assert state['log'] == str(new / 'qualification.log')
    assert (old / 'mission-state.json').is_file()
    assert len(migration.migrate(tmp_path, True)) == 2


def test_divergent_state_refuses_without_overwriting(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(migration, 'assert_no_supervisor', lambda root: None)
    new = tmp_path / 'verification' / migration.CURRENT
    new.mkdir()
    target = new / 'qualification-state.json'
    target.write_text('do not overwrite')
    with pytest.raises(ValueError, match='divergent'):
        migration.migrate(tmp_path, True)
    assert target.read_text() == 'do not overwrite'
    assert not (new / 'qualification-cache').exists()


def test_active_supervisor_refuses(tmp_path, monkeypatch):
    fixture(tmp_path)
    def active(root):
        raise RuntimeError('a supervisor is alive')
    monkeypatch.setattr(migration, 'assert_no_supervisor', active)
    with pytest.raises(RuntimeError, match='alive'):
        migration.migrate(tmp_path, True)
    assert not (tmp_path / 'verification' / migration.CURRENT).exists()


def test_symlink_source_refuses(tmp_path):
    old, _ = fixture(tmp_path)
    (old / 'other.log').symlink_to(old / 'mission-state.json')
    with pytest.raises(ValueError, match='regular'):
        migration.migrate(tmp_path)


def test_operating_vocabulary_has_no_old_brand():
    exceptions = {'scripts/migrate_qualification_state.py',
                  'verification/qualification-supervisor/test_qualification_rebrand.py',
                  'docs/reference/QUALIFICATION-MIGRATION.md', '.gitignore'}
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    pattern = re.compile(r'(?<![a-zA-Z])mission(?:s)?(?![a-zA-Z])|(?<![a-zA-Z])Mission[A-Z]')
    for name in names:
        if not name or name in exceptions or not (ROOT / name).is_file():
            continue
        try:
            text = (ROOT / name).read_text()
        except UnicodeError:
            continue
        assert not pattern.search(name), name
        assert not pattern.search(text), name
