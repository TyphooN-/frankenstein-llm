import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('proof_layout', Path(__file__).resolve().parents[2] / 'scripts/proof_layout.py')
assert SPEC is not None and SPEC.loader is not None
layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layout)


def test_migration_and_future_writes(tmp_path):
    old = tmp_path / layout.DIRECTORIES[1]
    old.mkdir(parents=True)
    (old / 'report.json').write_bytes(b'{"pass":false}')
    with pytest.raises(ValueError, match='--migrate'):
        layout.prepare(tmp_path)
    layout.prepare(tmp_path, migrate=True)
    target = tmp_path / 'proofs' / layout.DIRECTORIES[1]
    assert old.is_symlink()
    assert (target / 'report.json').read_bytes() == b'{"pass":false}'
    (old / 'next.json').write_bytes(b'{"pass":true}')
    assert (target / 'next.json').read_bytes() == b'{"pass":true}'
    assert layout.prepare(tmp_path, migrate=True) == 0


def test_collision_refuses_before_moving_anything(tmp_path):
    relative = layout.DIRECTORIES[-1]
    old, new = tmp_path / relative, tmp_path / 'proofs' / relative
    old.mkdir(parents=True); new.mkdir(parents=True)
    (old / 'report.json').write_text('old')
    (new / 'report.json').write_text('new')
    with pytest.raises(ValueError, match='collision'):
        layout.prepare(tmp_path, migrate=True)
    assert not (tmp_path / layout.DIRECTORIES[0]).exists()
    assert (old / 'report.json').read_text() == 'old'
    assert (new / 'report.json').read_text() == 'new'


def test_external_target_refused(tmp_path):
    root=tmp_path/'repo'; root.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (root/'proofs').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        layout.prepare(root)
    assert list(outside.iterdir()) == []
