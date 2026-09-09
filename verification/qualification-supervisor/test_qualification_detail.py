"""Status detail reads never dispatch model work or promote saved verdicts."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('qualification_detail_tests', ROOT / 'scripts/qualification_detail.py')
assert SPEC is not None and SPEC.loader is not None
detail = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detail)


def artifact(tmp_path, **changes):
    path = tmp_path / 'verification/router-functional/evidence/router-functional.json'
    path.parent.mkdir(parents=True)
    data = {'started_at': '2026-09-09T06:00:00+00:00', 'boot_id': 'boot', 'models': [
        {'model': 'good', 'pass': True, 'qualification_reused': True,
         'checks': {'coherence': {'pass': True}}},
        {'model': 'bad', 'pass': False, 'required_checks': ['tool_call', 'coherence'],
         'checks': {'tool_call': {'pass': False, 'content': 'DO NOT PRINT THIS'}}}]}
    data.update(changes)
    path.write_text(json.dumps(data))
    source = path.parent.parent / 'gate_router_models.py'
    source.write_text("CHAT_MODELS = ['good', 'bad', 'pending']\nVISION_MODELS = []\nraise RuntimeError('never import gates')\n")
    return path


def test_partial_models_and_failed_checks(tmp_path):
    artifact(tmp_path)
    result = detail.detail(tmp_path, 'router-models', {'started_at': '2026-09-09T05:59:59+00:00'}, 'boot')
    assert result['models_recorded'] == 2
    assert result['configured_models'] == 3
    assert result['outcomes'] == {'passed': 1, 'failed': 1}
    assert result['checks_passed'] == result['checks_failed'] == 1
    assert result['models'][1]['failed_checks'] == ['tool_call']
    assert result['models'][1]['missing_checks'] == ['coherence']
    assert result['evidence_scope'] == 'current-attempt'
    assert 'DO NOT PRINT' not in json.dumps(result)


def test_stale_artifact_does_not_claim_current_attempt(tmp_path):
    artifact(tmp_path)
    result = detail.detail(tmp_path, 'router-models', {'started_at': '2026-09-09T07:00:00+00:00'}, 'boot')
    assert result['evidence_scope'] == 'previous-attempt'
    result = detail.detail(tmp_path, 'router-models', {'started_at': '2026-09-09T05:59:59+00:00'}, 'new-boot')
    assert result['evidence_scope'] == 'previous-boot'


def test_missing_time_never_guesses_from_mtime(tmp_path):
    artifact(tmp_path, started_at=None)
    result = detail.detail(tmp_path, 'router-models', {'started_at': '2026-09-09T05:59:59+00:00'})
    assert result['evidence_scope'] == 'unverified-attempt'


def test_duplicate_models_refused(tmp_path):
    artifact(tmp_path, models=[{'model': 'same', 'pass': True}, {'model': 'same', 'pass': True}])
    assert not detail.detail(tmp_path, 'router-models', {})['available']


@pytest.mark.parametrize('data', ['{', '[]', 'x' * (detail.LIMIT + 1)])
def test_bad_or_large_artifact_is_unavailable(tmp_path, data):
    path = artifact(tmp_path)
    path.write_text(data)
    assert not detail.detail(tmp_path, 'router-models', {})['available']


def test_external_symlink_not_read(tmp_path):
    path = artifact(tmp_path)
    outside = tmp_path / 'outside.json'
    outside.write_text(path.read_text())
    path.unlink()
    path.symlink_to(outside)
    assert not detail.detail(tmp_path, 'router-models', {})['available']


def test_fifo_does_not_block(tmp_path):
    path = artifact(tmp_path)
    path.unlink()
    os.mkfifo(path)
    assert not detail.detail(tmp_path, 'router-models', {})['available']


def test_refusal_not_reclassified_as_model_failure():
    row = detail.model_row({'model': 'm', 'pass': False, 'outcome': 'inconclusive'}, 'fallback')
    assert row['outcome'] == 'inconclusive'


def test_residency_is_only_bounded_get(monkeypatch):
    calls = []
    class Connection:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, method, path):
            calls.append((method, path))
        def getresponse(self):
            return self
        status = 200
        def read(self, limit):
            assert limit <= 256 * 1024 + 1
            return b'{"data":[{"id":"heretic","status":{"value":"loaded"}}]}'
        def close(self):
            pass
    monkeypatch.setattr(detail.http.client, 'HTTPConnection', Connection)
    result = detail.router_residency()
    assert calls == [('GET', '/models')]
    assert result['models'] == [{'model': 'heretic', 'state': 'loaded'}]
    assert 'not proof' in result['note']
