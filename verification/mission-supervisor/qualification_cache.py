"""Durable, bounded last-verdict cache; no inference or host admission side effects.

Receipts describe qualifications, not current host health. Metadata identities for
large local weights assume trusted local storage; this is not checksum validation.
"""
from __future__ import annotations

import configparser
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'verification/mission-supervisor/qualification-cache'
MAX_RECORD = 2 << 20
# Only model inputs used by these fixed workflows. Adding another queue candidate
# does not alter an existing workflow's identity.
REPOSITORIES = {
    'embeddings': {'Qwen/Qwen3-Embedding-8B-GGUF'},
    'reranker': {'Qwen/Qwen3-Reranker-8B'},
    'fim': {'ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF'},
    'asr': {'Qwen/Qwen3-ASR-1.7B-hf'},
    'tts-asr-roundtrip': {'Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'Qwen/Qwen3-ASR-1.7B-hf'},
    'computer-use-grounding': {'ByteDance-Seed/UI-TARS-1.5-7B'},
    'generative-media-functional': {'Comfy-Org/ace_step_1.5_ComfyUI_files',
        'Comfy-Org/z_image_turbo', 'Comfy-Org/Qwen-Image-Edit_ComfyUI',
        'Comfy-Org/HunyuanVideo_1.5_repackaged', 'Comfy-Org/Qwen-Image_ComfyUI',
        'lightx2v/Qwen-Image-Edit-2511-Lightning'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_identity(path):
    path = Path(path)
    st = path.stat()
    if not path.is_file():
        raise ValueError(f'not a regular input file: {path}')
    if st.st_size <= MAX_RECORD:
        return [str(path), hashlib.sha256(path.read_bytes()).hexdigest()]
    return [str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def runtime_identity():
    result: dict = {'kernel': Path('/proc/version').read_text(),
              'cmdline': Path('/proc/cmdline').read_text()}
    cards = sorted(p for p in Path('/sys/class/drm').glob('card*')
                   if re.fullmatch(r'card\d+', p.name))
    result['devices'] = [file_identity(p / 'device/device') for p in cards]
    # Persisted tuning files may be merely staged. Effective controls below,
    # not un-applied /etc edits, define the running hardware configuration.
    result['gpu_controls'] = [file_identity(p) for card in cards for pattern in
        ('device/pp_od_clk_voltage', 'device/hwmon/hwmon*/power1_cap')
        for p in sorted(card.glob(pattern))]
    result['libraries'] = [file_identity(p) for p in
                          (Path('/opt/rocm/lib/libamdhip64.so'), Path('/opt/rocm/lib/librocblas.so'))
                          if p.exists()]
    binary = ROOT / 'upstream/llama.cpp/build/bin/llama-server'
    if binary.exists():
        result['runtime'] = file_identity(binary)
    lock = ROOT / 'upstream/llama-cpp.lock.json'
    if lock.exists():
        result['lock'] = file_identity(lock)
    return result


def source_identity(paths):
    files = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.update(p for p in path.rglob('*') if p.is_file()
                         and p.suffix in {'.py', '.sh', '.jinja', '.json', '.yaml', '.png', '.txt'}
                         and not any(x in p.parts for x in ('evidence', '__pycache__'))
                         and (not p.name.startswith('test_') or 'fixture' in p.parts))
        else:
            files.add(path)
    return [file_identity(p) for p in sorted(files)]


def preset_identity(model):
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / 'llama-models.ini')
    if model not in config:
        raise ValueError(f'unknown model: {model}')
    values = dict(config['*']) if '*' in config else {}
    values.update(config[model])
    artifacts = []
    for name in ('model', 'mmproj', 'chat-template-file'):
        if name not in values:
            continue
        path = Path(values[name])
        match = re.search(r'-\d{5}-of-(\d{5})\.gguf$', path.name)
        paths = [path]
        if match:
            paths = [path.with_name(re.sub(r'-\d{5}-of-', f'-{i:05d}-of-', path.name))
                     for i in range(1, int(match[1]) + 1)]
        artifacts.extend(file_identity(p) for p in paths)
    return {'preset': values, 'artifacts': artifacts}


def model_key(gate, model, sources, extra=None):
    return digest({'schema': 1, 'gate': gate, 'model': model,
                   'preset': preset_identity(model), 'sources': source_identity(sources),
                   'runtime': runtime_identity(), 'extra': extra,
                   'cache_contract': file_identity(__file__)})


def step_key(name, command, *, details=False):
    if name in ('router-models', 'repository-agent'):
        # These aggregates must dispatch: their individual models own receipts.
        return None
    paths = [Path(x) for x in command if isinstance(x, str) and x.endswith(('.py', '.sh'))]
    if not paths:
        return None  # service actions / unknown workflows never claim cached qualification
    scopes = []
    for p in paths:
        scopes.append(p if p.parent.name == 'validators' else p.parent)
    if name == 'wemm-embeddings':
        base = ROOT / 'verification/candidate-qualification'
        scopes = [base / 'run_wemm.sh', base / 'gate_wemm.py', base / 'wemm_remote_code_review.py']
    if any(p.parent.name == 'validators' for p in paths):
        scopes.append(ROOT / 'verification/local-coverage-foundation/validators/gatelib.py')
    for service in (ROOT / f'services/sidecar-{name}.env',
                    ROOT / 'services/systemd/llama-sidecar@.service'):
        if service.exists():
            scopes.append(service)
            for value in re.findall(r'(/[^\s\"\']+)', service.read_text()):
                if Path(value).is_file():
                    scopes.append(Path(value))
    artifacts = []
    repos = REPOSITORIES.get(name, set())
    for queue in sorted((ROOT / 'verification/local-coverage-foundation').glob('download-queue*.json')):
        for artifact in json.loads(queue.read_text()).get('artifacts', []):
            if artifact.get('repository') in repos:
                artifacts.append({'repository': artifact['repository'], 'revision': artifact.get('revision'),
                                  'files': [file_identity(f['destination']) for f in artifact['files']]})
    if name == 'wemm-embeddings':
        model_dir = ROOT / 'models/embedding/WeMM-Embedding-2B'
        files = sorted(p for p in model_dir.rglob('*') if p.is_file()
                       and '__pycache__' not in p.parts and '.cache' not in p.parts)
        if not files:
            return None
        artifacts.append({'repository': 'local-WeMM', 'files': [file_identity(p) for p in files]})
    if not repos and name != 'wemm-embeddings':
        return None
    found = {a['repository'] for a in artifacts}
    if not repos <= found:
        return None
    venv_name = {'asr': 'asr', 'tts-asr-roundtrip': 'tts',
                 'wemm-embeddings': 'candidates', 'computer-use-grounding': 'computer-use',
                 'generative-media-functional': 'comfy'}.get(name)
    if name == 'generative-media-functional':
        scopes.append(ROOT / 'tools/ComfyUI')
    environment = []
    if venv_name:
        venv = ROOT / 'venvs' / venv_name
        environment = [file_identity(p) for p in sorted(venv.glob('lib/python*/site-packages/*.dist-info/RECORD'))]
        if not environment:
            return None
    inputs = {'schema': 1, 'gate': name, 'command': command, 'environment': environment,
                   'source': source_identity(scopes), 'artifacts': artifacts,
                   'runtime': runtime_identity(), 'cache_contract': file_identity(__file__)}
    return inputs if details else digest(inputs)


def adopt_legacy_step(name, command, state, store):
    """Import only a complete same-kernel pass whose inputs predate its start.

    This is a local provenance migration, not fresh GPU proof. Large files use
    trusted filesystem metadata, like the existing supervisor. Missing timing,
    changed inputs and ambiguous per-model legacy records require a real test.
    """
    previous = state.get('steps', {}).get(name, {})
    if (store.read(name, name) is not None or previous.get('status') != 'passed'
            or previous.get('exit_code') != 0 or previous.get('command') != command
            or previous.get('input_fingerprint') != state.get('input_fingerprint')
            or state.get('kernel_build_signature') != Path('/proc/version').read_text().strip()):
        return False
    try:
        start = datetime.fromisoformat(previous['started_at']).timestamp()
        end = datetime.fromisoformat(previous['finished_at']).timestamp()
        if end < start:
            return False
        inputs = step_key(name, command, details=True)
        if not isinstance(inputs, dict):
            return False
        # The newly introduced receipt implementation did not participate in the
        # old execution; all actual gate, artifact and environment inputs did.
        def unchanged(value):
            if isinstance(value, dict):
                return all(unchanged(v) for k, v in value.items() if k != 'cache_contract')
            if isinstance(value, list):
                if value and isinstance(value[0], str) and value[0].startswith('/'):
                    path = Path(value[0])
                    if path.is_file() and not str(path).startswith(('/proc/', '/sys/')):
                        st = path.stat()
                        if max(st.st_mtime, st.st_ctime) > start:
                            return False
                return all(unchanged(v) for v in value)
            return True
        if not unchanged(inputs):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    store.publish(name, name, digest(inputs), {'pass': True, 'step': previous,
                  'legacy_migration': True, 'original_finished_at': previous['finished_at']})
    return True


class Store:
    def __init__(self, root=CACHE):
        self.root = Path(root)

    def path(self, gate, model):
        return self.root / (digest([gate, model]) + '.json')

    def read(self, gate, model):
        try:
            with self.path(gate, model).open('rb') as f:
                raw = f.read(MAX_RECORD + 1)
            if len(raw) > MAX_RECORD:
                return None
            record = json.loads(raw)
            if (not isinstance(record, dict) or record.get('schema') != 1
                    or record.get('gate') != gate or record.get('model') != model):
                return None
            return record
        except (OSError, ValueError):
            return None

    def reuse(self, gate, model, key, force=False):
        record = self.read(gate, model)
        if (force or not key or not record or record.get('key') != key
                or record.get('status') != 'passed'):
            return None
        result = record.get('result')
        if not isinstance(result, dict) or result.get('pass') is not True:
            return None
        answer: dict = dict(result)
        answer.update(qualification_reused=True, qualified_at=record['updated_at'])
        return answer

    def publish(self, gate, model, key, result=None):
        old = self.read(gate, model) or {}
        passed = isinstance(result, dict) and result.get('pass') is True
        record = {'schema': 1, 'gate': gate, 'model': model, 'key': key,
                  'status': 'running' if result is None else ('passed' if passed else 'failed'),
                  'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'result': result,
                  'last_pass': old.get('result') if old.get('status') == 'passed' else old.get('last_pass')}
        raw = json.dumps(record, sort_keys=True).encode()
        if len(raw) > MAX_RECORD:
            raise ValueError('qualification receipt exceeds bound')
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.root, prefix='.receipt-')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            os.replace(temp, self.path(gate, model))
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temp).unlink(missing_ok=True)


def run_cached(gate, model, key, run, force=False, store=None):
    store = store or Store()
    store.root.mkdir(parents=True, exist_ok=True)
    with store.path(gate, model).with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_cached(gate, model, key, run, force, store)


def _run_cached(gate, model, key, run, force, store):
    cached = store.reuse(gate, model, key, force)
    if cached is not None:
        return cached
    # Revoke eligibility before running, so a crash or failed forced retest cannot
    # silently expose an old pass. last_pass remains historical, never reusable.
    store.publish(gate, model, key)
    result = run()
    store.publish(gate, model, key, result)
    return dict(result, qualification_reused=False)
