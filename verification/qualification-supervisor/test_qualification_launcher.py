"""Exercise the public shell entry point without services or inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'scripts/qualification-status.sh'


def test_launcher_reports_qualification_from_another_directory(tmp_path):
    state = tmp_path / 'saved state.json'
    state.write_text(json.dumps({'status': 'incomplete', 'current_step': None,
                                 'steps': {'asr': {'status': 'passed', 'exit_code': 0}}}))
    result = subprocess.run([str(LAUNCHER), '--state', str(state)], cwd=tmp_path,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith('LOCAL AI QUALIFICATION: incomplete | current: -\n')


def test_launcher_forwards_json_and_space_containing_path(tmp_path):
    state = tmp_path / 'saved state.json'
    state.write_text(json.dumps({'status': 'incomplete', 'current_step': None,
                                 'steps': {'asr': {'status': 'passed', 'exit_code': 0}}}))
    result = subprocess.run([str(LAUNCHER), '--state', str(state), '--json'],
                            cwd=tmp_path, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'incomplete'
    assert report['total'] == 1


def test_launcher_preserves_reader_failure(tmp_path):
    result = subprocess.run([str(LAUNCHER), '--invalid-option'], cwd=tmp_path,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 2


def test_launcher_requires_current_reader_without_fallback(tmp_path):
    launcher = tmp_path / LAUNCHER.name
    shutil.copy2(LAUNCHER, launcher)
    result = subprocess.run([str(launcher)], cwd=tmp_path,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert 'qualification_status.py' in result.stderr
    assert not result.stdout


@pytest.fixture
def fake_commands(tmp_path):
    calls = tmp_path / 'calls'
    for name in ('systemctl', 'python3'):
        command = tmp_path / name
        command.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$COMMAND_CALLS"\n'
                           'exit "${COMMAND_EXIT:-0}"\n')
        command.chmod(0o755)
    return calls, dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
                       COMMAND_CALLS=str(calls))


@pytest.mark.parametrize('args', [[], ['--start']])
def test_qualification_start_queues_service(args, fake_commands, tmp_path):
    calls, env = fake_commands
    result = subprocess.run([str(ROOT / 'scripts/qualify-models.sh'), *args],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == [
        '--user', 'start', '--no-block', 'local-ai-qualification.service']
    assert 'queued' in result.stdout
    assert 'not yet verified' in result.stdout


def test_qualification_start_propagates_service_error(fake_commands, tmp_path):
    calls, env = fake_commands
    env['COMMAND_EXIT'] = '9'
    result = subprocess.run([str(ROOT / 'scripts/qualify-models.sh')], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 9
    assert calls.exists()
    assert 'queued' not in result.stdout


def test_qualification_plan_does_not_start_service(fake_commands, tmp_path):
    calls, env = fake_commands
    result = subprocess.run([str(ROOT / 'scripts/qualify-models.sh'), '--plan'],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == [
        str(ROOT / 'verification/qualification-supervisor/run_qualification.py'), '--plan']


def test_qualification_help_does_not_start_service(fake_commands, tmp_path):
    calls, env = fake_commands
    result = subprocess.run([str(ROOT / 'scripts/qualify-models.sh'), '--help'],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not calls.exists()
    assert '--plan' in result.stdout
