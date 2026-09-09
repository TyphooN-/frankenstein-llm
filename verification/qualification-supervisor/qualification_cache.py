"""Durable, bounded last-verdict cache; no inference or host admission side effects.

Receipts describe qualifications, not current host health. Metadata identities for
large local weights assume trusted local storage; this is not checksum validation.

A receipt records one of four outcomes, and only the first is reusable:

``passed``        the gate ran and the model met the contract;
``failed``        the gate ran and the model did not -- this revokes an older pass;
``inconclusive``  the gate ran but produced no verdict, because its inputs moved
                  under it or its execution boundary broke. Fail-closed like a
                  failure, but it is not a statement about the model, and it must
                  never resurrect the pass its own retest invalidated;
``blocked``       nothing ran. Admission was refused because the host was busy
                  with work that would confound the measurement. A refusal spends
                  nothing: the receipt it found is the receipt it leaves.
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
CACHE = ROOT / 'verification/qualification-supervisor/qualification-cache'
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


PASSED, FAILED, INCONCLUSIVE, BLOCKED = 'passed', 'failed', 'inconclusive', 'blocked'
# Exit codes a gate uses to tell the supervisor which of the two non-verdicts it
# reached. 75 is already this repository's idiom for "refused, host not idle":
# every serialized runner exits 75 for it, so the supervisor now reads the code
# those runners were already producing instead of filing it as a gate failure.
EXIT_ADMISSION_REFUSED = 75
EXIT_INCONCLUSIVE = 76
# A receipt written before per-component provenance existed carries only the
# composite key. That still decides reuse exactly as before -- a match is a
# match -- but a miss cannot be attributed to a dependency, and saying so is the
# whole point: an unexplained miss must not be read as an excusable one.
LEGACY_PROVENANCE = 'unavailable: receipt predates per-component provenance'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def key_components(inputs):
    """One digest per top-level key of a cache-key document.

    The key itself stays a single digest over the whole document, so this
    changes no reuse decision. It only makes a miss diagnosable: without it,
    "the key moved" is the entire diagnosis, and finding out which dependency
    moved means recomputing the inputs by hand. Bounded by construction --
    there are a handful of components and each contributes one digest of the
    identity records that were already gathered, never a fresh read of a weight
    file and never a value copied out of them.
    """
    if not isinstance(inputs, dict):
        return {}
    return {name: digest(value) for name, value in sorted(inputs.items())}


def outcome_of(result):
    """Classify a gate result: passed, blocked, inconclusive, or failed.

    Fail-closed on anything unrecognised. ``blocked`` and ``inconclusive`` are
    only ever reached when a result says so explicitly.
    """
    if isinstance(result, dict):
        stated = result.get('outcome')
        if stated in (BLOCKED, INCONCLUSIVE, FAILED):
            return stated
        if result.get('pass') is True:
            return PASSED
    return FAILED


def mark_inconclusive(result, reason):
    """Record that a run finished without producing a verdict about the model."""
    result['pass'] = False
    result['outcome'] = INCONCLUSIVE
    result.setdefault('problems', []).append(reason)
    return result


def unreadable_identity(gate, model, error):
    """Refuse a qualification whose own inputs could not be read.

    A weight file that is mid-download, quarantined or replaced by a ``.partial``
    makes the identity unreadable. Nothing can be measured against inputs that
    cannot be enumerated, so this is a refusal on the same terms as a busy host:
    it spends no receipt and states nothing about the model. Left unhandled it
    escaped as ``FileNotFoundError``, which exited the gate 1 -- the code every
    runner reads as "this model failed".
    """
    return dict(mark_blocked(
        {'gate': gate, 'model': model, 'checks': {}, 'problems': []},
        [{'reason': 'qualification-inputs-unreadable', 'gate': gate, 'model': model,
          'error': f'{type(error).__name__}: {error}'[:300]}],
        'refusing to qualify: this preset\'s inputs could not be read'),
        qualification_reused=False)


def mark_blocked(result, blockers, reason):
    """Record that admission was refused, so nothing was measured."""
    result['pass'] = False
    result['outcome'] = BLOCKED
    result['admission_refused'] = True
    result['blocked_by'] = blockers
    result.setdefault('problems', []).append(reason)
    return result


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


def preset_values(model):
    """A preset's effective settings, without touching the weights it names.

    Separate from :func:`preset_identity` because a caller that only needs to
    know what a preset *is* -- whether it declares a projector, say -- must not
    fail because a weight file is mid-download. Reading the configuration and
    reading the artifacts are different questions with different failure modes.
    """
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / 'llama-models.ini')
    if model not in config:
        raise ValueError(f'unknown model: {model}')
    values = dict(config['*']) if '*' in config else {}
    values.update(config[model])
    return values


def preset_identity(model):
    values = preset_values(model)
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


def model_key(gate, model, sources, extra=None, *, details=False):
    inputs = {'schema': 1, 'gate': gate, 'model': model,
              'preset': preset_identity(model), 'sources': source_identity(sources),
              'runtime': runtime_identity(), 'extra': extra,
              'cache_contract': file_identity(__file__)}
    return inputs if details else digest(inputs)


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
                  'legacy_migration': True, 'original_finished_at': previous['finished_at']},
                  components=key_components(inputs))
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
        if (not isinstance(result, dict) or outcome_of(result) != PASSED
                or not isinstance(record.get('updated_at'), str)):
            return None
        answer: dict = dict(result)
        answer.update(qualification_reused=True, qualified_at=record['updated_at'])
        return answer

    def explain(self, gate, model, key, components=None, force=False):
        """Why this gate will or will not reuse its receipt, in read-only terms.

        Never writes and never runs anything, so ``--plan`` and the status
        reader can both call it. ``changed_components`` is ``None`` -- not an
        empty list -- when the stored receipt cannot attribute the miss, so
        "nothing changed" and "we cannot say what changed" stay distinguishable.
        """
        record = self.read(gate, model)
        answer = {'reason': 'reusable', 'receipt_status': record.get('status') if record else None,
                  'updated_at': record.get('updated_at') if record else None,
                  'changed_components': [], 'component_provenance': 'recorded'}
        stored = record.get('components') if isinstance(record, dict) else None
        if not isinstance(stored, dict):
            answer['component_provenance'] = LEGACY_PROVENANCE
        if not key:
            answer.update(reason='no-identity', changed_components=None,
                          component_provenance='not applicable: this workflow claims no cached identity')
            return answer
        if not record:
            answer.update(reason='no-receipt', changed_components=None,
                          component_provenance='not applicable: no receipt exists')
            return answer
        if force:
            answer.update(reason='forced-retest', changed_components=None)
            return answer
        if record.get('status') != 'passed':
            answer.update(reason=record.get('status') or 'unknown', changed_components=None)
            return answer
        if record.get('key') != key:
            answer['reason'] = 'inputs-changed'
            answer['changed_components'] = (
                sorted(name for name in set(stored) | set(components or {})
                       if stored.get(name) != (components or {}).get(name))
                if isinstance(stored, dict) and isinstance(components, dict) else None)
            return answer
        if self.reuse(gate, model, key) is None:
            answer.update(reason='invalid-receipt', changed_components=None)
        return answer

    def publish(self, gate, model, key, result=None, components=None):
        old = self.read(gate, model) or {}
        record = {'schema': 1, 'gate': gate, 'model': model, 'key': key,
                  'status': 'running' if result is None else outcome_of(result),
                  'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'result': result,
                  'components': components if isinstance(components, dict) else old.get('components'),
                  'last_pass': old.get('result') if old.get('status') == 'passed' else old.get('last_pass')}
        self._write(gate, model, record)

    def restore(self, gate, model, record):
        """Put a receipt back exactly as it was, or remove one that was not there.

        Used when a run turns out never to have begun. Rewriting the old record
        rather than editing the new one keeps ``last_pass`` and ``updated_at``
        honest: no attempt happened, so nothing about the receipt should move.
        """
        if record is None:
            self.path(gate, model).unlink(missing_ok=True)
            self._fsync_directory()
            return
        self._write(gate, model, record)

    def _fsync_directory(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _write(self, gate, model, record):
        raw = json.dumps(record, sort_keys=True).encode()
        if len(raw) > MAX_RECORD:
            raise ValueError('qualification receipt exceeds bound')
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.root, prefix='.receipt-')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            os.replace(temp, self.path(gate, model))
            self._fsync_directory()
        finally:
            Path(temp).unlink(missing_ok=True)


def run_cached(gate, model, key, run, force=False, store=None, admit=None,
               components=None):
    """Reuse, refuse, or run one qualification -- in that order.

    ``admit`` is the host-admission check: a callable returning the workloads
    that would confound this measurement, or an empty result when the host is
    usable. It is consulted only when a test would actually run, and always
    *before* the receipt is touched, because refusing to start is not a verdict.
    """
    store = store or Store()
    store.root.mkdir(parents=True, exist_ok=True)
    with store.path(gate, model).with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another qualification already owns this receipt. Losing the race
            # means nothing ran here, which is an admission outcome and not a
            # verdict: propagating the OSError instead exited the gate 1 and
            # published a model failure out of a lock conflict.
            holder = [{'reason': 'qualification-receipt-locked',
                       'gate': gate, 'model': model, 'pid': os.getpid()}]
            return dict(mark_blocked({'gate': gate, 'model': model}, holder,
                                     'another qualification holds this receipt; nothing was run'),
                        qualification_reused=False)
        return _run_cached(gate, model, key, run, force, store, admit, components)


def _run_cached(gate, model, key, run, force, store, admit, components):
    cached = store.reuse(gate, model, key, force)
    if cached is not None:
        # Reuse dispatches no model, so a busy host cannot confound it and
        # admission is not consulted: the receipt was earned on a quiet one.
        return cached
    blockers = admit() if admit is not None else None
    if blockers:
        return dict(mark_blocked({'gate': gate, 'model': model}, blockers,
                                 f'refusing confounded qualification; active workloads: {blockers}'),
                    qualification_reused=False)
    # Revoke eligibility before running, so a crash or failed forced retest cannot
    # silently expose an old pass. last_pass remains historical, never reusable.
    # Keep the receipt this attempt found: a run that turns out never to have
    # begun has to be able to put it back untouched.
    previous = store.read(gate, model)
    store.publish(gate, model, key, components=components)
    result = run()
    if outcome_of(result) == BLOCKED:
        # Admission was refused after the outer check, by the workload's own
        # guard, before it dispatched anything. Nothing was measured, so nothing
        # is spent -- including a pass this attempt would otherwise have revoked.
        store.restore(gate, model, previous)
    else:
        store.publish(gate, model, key, result, components=components)
    return dict(result, qualification_reused=False)
