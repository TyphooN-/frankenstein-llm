"""Passive telemetry proof: real loopback HTTP, no GPU or model loads."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import qualification_performance as performance


@pytest.fixture
def output(tmp_path, monkeypatch):
    path = tmp_path / 'performance.jsonl'
    monkeypatch.setenv('QUALIFICATION_PERFORMANCE_PATH', str(path))
    monkeypatch.setattr(performance, '_written', 0)
    monkeypatch.setattr(performance, 'synchronize_devices', lambda: None)
    return path


def test_native_and_end_to_end_rates_are_distinct():
    row = performance.observation({'timings': {'predicted_per_second': 50, 'prompt_per_second': 100},
                                   'usage': {'completion_tokens': 20}}, 2)
    assert row['decode_tokens_per_second'] == 50
    assert row['prompt_tokens_per_second'] == 100
    assert row['end_to_end_tokens_per_second'] == 10
    assert row['benchmark'] is False


def test_missing_invalid_and_nontext_rates():
    assert performance.number(10 ** 1000) is None
    row = performance.observation({'timings': {'predicted_per_second': float('nan')},
                                   'usage': {'completion_tokens': True}}, 0)
    assert not row['token_rate_available']
    assert 'decode_tokens_per_second' not in row
    audio = performance.observation(elapsed=2, audio_seconds=8)
    assert audio['real_time_factor'] == 0.25
    assert not audio['token_rate_available']
    assert performance.observation(elapsed=2, items=8)['items_per_second'] == 4


def test_failed_request_retains_observation_without_payload(output):
    with pytest.raises(RuntimeError):
        with performance.measure('failure'):
            raise RuntimeError('private text never persisted')
    raw = output.read_text()
    assert json.loads(raw)['request_failed']
    assert 'private text' not in raw


def test_process_sample_cap_and_bad_sink(output, monkeypatch):
    monkeypatch.setattr(performance, 'LIMIT', 2)
    for _ in range(5):
        performance.emit(performance.observation(elapsed=1))
    assert len(output.read_text().splitlines()) == 2
    monkeypatch.setattr(performance, '_written', 0)
    monkeypatch.setenv('QUALIFICATION_PERFORMANCE_PATH', str(output.parent))
    performance.emit(performance.observation(elapsed=1))  # Must not fail a gate.


def test_generate_preserves_arguments_and_excludes_input_tokens(output):
    prompt = SimpleNamespace(shape=(1, 100))
    result = SimpleNamespace(shape=(1, 120))
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return result
    assert performance.call(generate, operation='generate', input_ids=prompt,
                            do_sample=False, max_new_tokens=20) is result
    assert calls == [{'input_ids': prompt, 'do_sample': False, 'max_new_tokens': 20}]
    row = json.loads(output.read_text())
    assert row['completion_tokens'] == 20
    assert row['end_to_end_tokens_per_second'] > 0


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_and_repository_http_callers_keep_native_timings(output, monkeypatch):
    payload = {'choices': [{'message': {'content': 'PONG'}}],
               'timings': {'predicted_per_second': 42},
               'usage': {'completion_tokens': 7}}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *_):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}'
        router = load('performance_router', 'verification/router-functional/gate_router_models.py')
        monkeypatch.setattr(router, 'BASE', base)
        assert router.http_json('/v1/chat/completions', {'model': 'fixture'}) == payload
        repo = load('performance_repo', 'verification/repository-agent/gate_repo_agent.py')
        monkeypatch.setattr(repo, 'ROUTER', base)
        assert repo.router_call([], 'fixture') == payload
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == 2
        assert all(row['decode_tokens_per_second'] == 42 for row in rows)
        assert all(row['completion_tokens'] == 7 for row in rows)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_media_existing_workflow_records_elapsed_and_output_count(output, monkeypatch):
    media = load('performance_media', 'verification/generative-media/functional_gate.py')
    result = {'outputs': {'1': {'images': [{'filename': 'fixture.png'}]}}}
    monkeypatch.setattr(media, '_submit', lambda *args: result)
    assert media.submit({}) is result
    row = json.loads(output.read_text())
    assert row['items'] == 1
    assert row['wall_seconds'] > 0
    assert not row['token_rate_available']


def test_supervisor_supplies_unique_sidecars_and_preserves_failed_attempts(tmp_path, monkeypatch):
    supervisor = load('performance_supervisor', 'verification/qualification-supervisor/run_qualification.py')
    monkeypatch.setattr(supervisor, 'HERE', tmp_path)
    monkeypatch.setattr(supervisor, 'wait_for_quiet', lambda state: None)
    monkeypatch.setattr(supervisor, 'atomic_json', lambda state: None)
    monkeypatch.setattr(supervisor, 'log', lambda message: None)
    state = {'steps': {}, 'input_fingerprint': 'fixture'}
    program = (f'import sys; sys.path.insert(0, {str(ROOT / "scripts")!r}); '
               'import qualification_performance as p; '
               'p.emit(p.observation(elapsed=1, items=2)); sys.exit(1)')
    command = [sys.executable, '-c', program]
    assert supervisor.run_step('fixture', command, state) == 1
    first = Path(state['steps']['fixture']['performance_log'])
    assert json.loads(first.read_text())['items_per_second'] == 2
    assert state['steps']['fixture']['status'] == 'failed'
    assert supervisor.run_step('fixture', command, state) == 1
    assert Path(state['steps']['fixture']['performance_log']) != first
    assert first.exists()
