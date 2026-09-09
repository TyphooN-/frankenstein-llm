"""Offline qualification observer regression tests."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('qualification_status', Path(__file__).resolve().parents[2] / 'scripts/qualification_status.py')
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_counts_ignore_stale_remaining(tmp_path):
    path = tmp_path / 'state.json'
    state = {'status': 'running', 'input_fingerprint': 'new', 'remaining': ['done'],
             'steps': {'done': {'status': 'passed', 'input_fingerprint': 'new'},
                       'old': {'status': 'passed', 'input_fingerprint': 'old'},
                       'failed': {'status': 'failed', 'exit_code': 1, 'input_fingerprint': 'new'},
                       'active': {'status': 'running', 'input_fingerprint': 'new'}}}
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    report = MODULE.snapshot(path, now=100)
    assert report['passed'] == 1
    assert report['total'] == 4
    assert report['terminal_attempts'] == 2
    assert report['counts']['stale-pass'] == 1
    assert '1/4 gates passed' in MODULE.render(report)
    assert path.read_bytes() == before


@pytest.mark.parametrize('raw', ['{', '[]', '{"steps": []}', '{"steps":{"x":null}}'])
def test_invalid_state(tmp_path, raw):
    path = tmp_path / 'state.json'
    path.write_text(raw)
    with pytest.raises(ValueError):
        MODULE.snapshot(path)


def test_elapsed_and_unknown_times(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'steps': {'x': {'status': 'failed',
        'started_at': '2026-01-01T00:00:00+0000', 'finished_at': '2026-01-01T00:02:00+0000'}}}))
    assert MODULE.snapshot(path)['steps'][0]['elapsed_seconds'] == 120
    assert MODULE.timestamp('bad') is None
    assert MODULE.timestamp('2026-01-01T00:00:00') is None


def test_old_failure_is_not_a_finished_current_attempt(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'input_fingerprint': 'new', 'steps': {
        'old-failure': {'status': 'failed', 'input_fingerprint': 'old'}}}))
    report = MODULE.snapshot(path)
    assert report['terminal_attempts'] == 0
    assert report['counts'] == {'stale-failed': 1}


def test_redirected_runner_log_is_observed(tmp_path):
    state_dir = tmp_path / 'qualification-supervisor'
    state_dir.mkdir()
    log = tmp_path / 'tts-local/evidence/tts-runner.log'
    log.parent.mkdir(parents=True)
    log.write_text('runner progress')
    path = state_dir / 'state.json'
    path.write_text(json.dumps({'steps': {'tts-asr-roundtrip': {'status': 'running'}}}))
    row = MODULE.snapshot(path, now=log.stat().st_mtime + 10)['steps'][0]
    assert row['log_age_seconds'] == 10
    assert row['log_path'] == str(log)


def state_with(tmp_path, **overrides):
    path = tmp_path / 'state.json'
    document = {'status': 'running', 'current_step': 'grounding', 'boot_id': 'boot-a',
                'input_fingerprint': 'f',
                'steps': {'grounding': {'status': 'running', 'input_fingerprint': 'f'}}}
    document.update(overrides)
    path.write_text(json.dumps(document))
    return path


def only(path, **kwargs):
    return MODULE.snapshot(path, now=100, **kwargs)['steps'][0]['status']


def test_the_running_current_step_of_this_boot_is_reported_running(tmp_path):
    assert only(state_with(tmp_path), boot='boot-a') == 'running'


def test_a_step_left_running_by_a_previous_boot_is_interrupted(tmp_path):
    # The host crashed under the gate, so no signal handler marked it stopped.
    assert only(state_with(tmp_path), boot='boot-b') == 'interrupted'


def test_only_the_current_step_can_still_be_running(tmp_path):
    # The supervisor is serialized: it runs one step at a time.
    path = state_with(tmp_path, current_step='reranker')
    assert only(path, boot='boot-a') == 'interrupted'


def test_a_supervisor_that_is_not_running_a_step_has_none_running(tmp_path):
    path = state_with(tmp_path, status='waiting-safe-host')
    assert only(path, boot='boot-a') == 'interrupted'


def test_an_unreadable_host_boot_identity_does_not_invent_a_crossing(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, 'BOOT_ID', tmp_path / 'absent')
    report = MODULE.snapshot(state_with(tmp_path), now=100)
    assert report['boot_crossed'] is False
    assert report['steps'][0]['status'] == 'running'


def test_a_state_without_a_boot_identity_does_not_invent_a_crossing(tmp_path):
    report = MODULE.snapshot(state_with(tmp_path, boot_id=None), now=100, boot='boot-a')
    assert report['boot_crossed'] is False
    assert report['steps'][0]['status'] == 'running'


def test_interruption_composes_with_a_stale_fingerprint(tmp_path):
    path = state_with(tmp_path, input_fingerprint='new',
                      steps={'grounding': {'status': 'running', 'input_fingerprint': 'old'}})
    report = MODULE.snapshot(path, now=100, boot='boot-b')
    assert report['counts'] == {'stale-interrupted': 1}
    assert report['terminal_attempts'] == 0
    assert 'nothing it left running is live' in MODULE.render(report)


def test_current_boot_reads_the_kernel_identity(monkeypatch, tmp_path):
    identity = tmp_path / 'boot_id'
    identity.write_text('boot-c\n')
    monkeypatch.setattr(MODULE, 'BOOT_ID', identity)
    assert MODULE.current_boot() == 'boot-c'
    monkeypatch.setattr(MODULE, 'BOOT_ID', tmp_path / 'absent')
    assert MODULE.current_boot() is None


def test_recorded_cause_ledger_is_reported_and_bounded(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({
        'status': 'functional-foundation-incomplete', 'input_fingerprint': 'f',
        'error': 'x' * 900,
        'failed_steps': [{'name': 'asr', 'exit_code': 1}] * (MODULE.LEDGER_LIMIT + 5),
        'blocked_steps': [{'name': 'tts', 'exit_code': 75}, 'not-a-record'],
        'remaining': ['asr', 'tts', 42],
        'steps': {'asr': {'status': 'failed', 'exit_code': 1, 'input_fingerprint': 'f',
                          'error': 'qualification inputs changed during execution'}}}))
    report = MODULE.snapshot(path, now=100)
    assert len(report['error']) == MODULE.TEXT_LIMIT
    assert len(report['failed_steps']) == MODULE.LEDGER_LIMIT
    assert report['blocked_steps'] == [{'name': 'tts', 'exit_code': 75}]
    assert report['remaining'] == ['asr', 'tts']
    assert report['steps'][0]['error'] == 'qualification inputs changed during execution'
    rendered = MODULE.render(report)
    assert 'Blocked (nothing ran): tts(exit 75)' in rendered
    assert 'inputs changed during execution' in rendered
    assert 'Remaining: asr; tts' in rendered


def test_a_missing_or_malformed_ledger_is_empty_not_invented(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text(json.dumps({'input_fingerprint': 'f', 'failed_steps': 'asr',
                                'remaining': {'asr': 1}, 'error': 7,
                                'steps': {'asr': {'status': 'passed', 'input_fingerprint': 'f'}}}))
    report = MODULE.snapshot(path, now=100)
    assert report['error'] is None
    assert report['failed_steps'] == [] and report['remaining'] == []
    assert 'Recorded cause' not in MODULE.render(report)

