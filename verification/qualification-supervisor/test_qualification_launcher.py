"""Exercise the public shell entry point without services or inference."""
import json
from pathlib import Path
import shutil
import subprocess

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
