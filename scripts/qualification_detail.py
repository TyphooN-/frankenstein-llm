"""Bounded, read-only detail from existing gate artifacts; never a verdict source."""
import ast
from collections import Counter
from datetime import datetime
import http.client
import json
import os
from itertools import islice
from pathlib import Path
import stat

LIMIT = 2 * 1024 * 1024
MAX_ROWS = 128
ARTIFACTS = {
    'router-models': 'router-functional/evidence/router-functional.json',
    'computer-use-grounding': 'computer-use-grounding/evidence/computer-use-grounding.json',
    'generative-media-functional': 'generative-media/evidence/media-functional.json',
    'tts-asr-roundtrip': 'tts-local/evidence/gate-tts.json',
    'asr': 'local-coverage-foundation/evidence/gate-asr.json',
    'embeddings': 'local-coverage-foundation/evidence/gate-embeddings.json',
    'reranker': 'local-coverage-foundation/evidence/gate-reranker.json',
    'fim': 'local-coverage-foundation/evidence/gate-fim.json',
    'wemm-embeddings': 'candidate-qualification/evidence/wemm-functional.json',
    'candidate-policy': 'candidate-qualification/evidence/candidate-policy.json',
}
SECTIONS = ('load', 'unload', 'grounding', 'action_selection', 'coordinate_bounds',
            'distractor', 'injection', 'malformed_inputs', 'lifecycle', 'gpu_execution',
            'generated', 'gpu_use', 'intelligibility', 'long_input', 'memory_safety')


def label(value):
    if not isinstance(value, (str, int)):
        return 'unknown'
    return ''.join(c if c.isprintable() else ' ' for c in str(value)[:120])


def stamp(value):
    try:
        dt = datetime.fromisoformat(value)
        return dt.timestamp() if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def read_json(path, boundary):
    resolved = path.resolve()
    resolved.relative_to(boundary.resolve())
    fd = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError('not a regular artifact')
        raw = handle.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError('artifact exceeds read limit')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('artifact is not a mapping')
    return value


def checks(item):
    found = {}
    values = item.get('checks', {})
    if isinstance(values, dict):
        for key, value in list(values.items())[:MAX_ROWS]:
            verdict = value.get('pass') if isinstance(value, dict) else value
            if type(verdict) is bool:
                found[label(key)] = verdict
    for key in SECTIONS:
        value = item.get(key)
        if isinstance(value, dict) and type(value.get('pass')) is bool:
            found[key] = value['pass']
    workflows = item.get('workflows', {})
    if isinstance(workflows, dict):
        for key, value in list(workflows.items())[:MAX_ROWS]:
            if isinstance(value, dict) and type(value.get('nontrivial')) is bool:
                found['workflow:' + label(key) + ':nontrivial'] = value['nontrivial']
    cases = item.get('cases', [])
    if isinstance(cases, list):
        for index, value in enumerate(cases[:MAX_ROWS]):
            if isinstance(value, dict) and type(value.get('matched')) is bool:
                found['case:' + label(value.get('name', index))] = value['matched']
    return found


def model_row(item, fallback):
    outcomes = checks(item)
    required = item.get('required_checks')
    missing = ([label(k) for k in required if isinstance(k, str) and k not in outcomes]
               if isinstance(required, list) and len(required) <= MAX_ROWS else [])
    passed = item.get('pass')
    outcome = item.get('outcome') or item.get('status')
    if outcome not in ('passed', 'failed', 'inconclusive', 'blocked', 'interrupted'):
        outcome = 'passed' if passed is True else 'failed' if passed is False else 'unknown'
    return {'model': label(item.get('model') or fallback), 'outcome': outcome,
            'reused': item.get('qualification_reused') is True,
            'checks_passed': sum(outcomes.values()),
            'checks_failed': sum(not v for v in outcomes.values()),
            'checks_recorded': len(outcomes), 'failed_checks': [k for k, v in outcomes.items() if not v],
            'missing_checks': missing, 'problem_count': len(item.get('problems', []))
            if isinstance(item.get('problems'), list) else 0}


def router_defaults(root):
    """Read literal inventory without importing or executing a gate."""
    path = root / 'verification/router-functional/gate_router_models.py'
    with path.open('rb') as handle:
        raw = handle.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError('router source exceeds read limit')
    values = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('CHAT_MODELS', 'VISION_MODELS'):
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, (list, tuple)) or len(value) > MAX_ROWS:
                        raise ValueError('invalid model inventory')
                    if not all(isinstance(item, str) for item in value):
                        raise ValueError('invalid model identifier')
                    values[target.id] = value
    return list(dict.fromkeys(values.get('CHAT_MODELS', []) + values.get('VISION_MODELS', [])))


def detail(root, name, step, boot=None):
    boundary = root / 'verification'
    paths = [boundary / ARTIFACTS[name]] if name in ARTIFACTS else []
    if name == 'repository-agent':
        paths = list(islice((boundary / 'repository-agent/evidence').glob('gate-repo-agent-*.json'), MAX_ROWS + 1))
        if len(paths) > MAX_ROWS:
            return {'available': False, 'reason': 'Repository model inventory exceeds limit'}
        paths.sort()
    if not paths:
        return {'available': False, 'reason': 'No detailed artifact adapter'}
    rows, scopes, sources = [], [], []
    for path in paths:
        try:
            data = read_json(path, boundary)
        except (OSError, ValueError, TypeError, RecursionError):
            continue
        observed = stamp(data.get('started_at') or data.get('recorded_at') or data.get('finished_at'))
        start, finish = stamp(step.get('started_at')), stamp(step.get('finished_at'))
        scope = 'unverified-attempt'
        if observed is not None and start is not None:
            scope = ('current-attempt' if start <= observed and (finish is None or observed <= finish + 5)
                     else 'previous-attempt')
        if boot and data.get('boot_id') and data['boot_id'] != boot:
            scope = 'previous-boot'
        scopes.append(scope)
        sources.append(str(path))
        models = data.get('models') if name == 'router-models' else [data]
        if not isinstance(models, list) or len(models) > MAX_ROWS:
            return {'available': False, 'reason': 'Invalid or oversized model results'}
        seen = set()
        for item in models:
            if not isinstance(item, dict):
                continue
            row = model_row(item, name)
            if row['model'] in seen:
                return {'available': False, 'reason': 'Duplicate model results'}
            seen.add(row['model'])
            row['evidence_scope'] = scope
            rows.append(row)
    if not rows:
        return {'available': False, 'reason': 'No readable model/check results yet'}
    counts = dict(Counter(row['outcome'] for row in rows))
    planned, plan_basis = None, 'not published'
    if name == 'router-models':
        try:
            planned = router_defaults(root)
            plan_basis = 'configured defaults; runtime selection is not published'
        except (OSError, ValueError, SyntaxError, TypeError):
            pass
    return {'available': True, 'evidence_scope': scopes[0] if len(set(scopes)) == 1 else 'mixed-attempts',
            'sources': sources, 'models_recorded': len(rows), 'outcomes': counts,
            'checks_passed': sum(r['checks_passed'] for r in rows),
            'checks_failed': sum(r['checks_failed'] for r in rows),
            'checks_recorded': sum(r['checks_recorded'] for r in rows),
            'configured_models': len(planned) if planned else None, 'plan_basis': plan_basis,
            'models': rows,
            'note': 'Checks are recorded evidence, not total planned work or a replacement gate verdict. '
                    'Router results publish after each model; the current check is not emitted.'}


def router_residency():
    """One local GET; no model-loading endpoint, redirects, or request payload."""
    connection = http.client.HTTPConnection('127.0.0.1', 8080, timeout=1)
    try:
        connection.request('GET', '/models')
        response = connection.getresponse()
        if response.status != 200:
            return {'available': False}
        raw = response.read(256 * 1024 + 1)
        if len(raw) > 256 * 1024:
            return {'available': False}
        rows = json.loads(raw).get('data')
        if not isinstance(rows, list) or len(rows) > MAX_ROWS:
            return {'available': False}
        resident = [{'model': label(r.get('id')), 'state': label(r['status']['value'])}
                    for r in rows if isinstance(r, dict) and isinstance(r.get('status'), dict)
                    and r['status'].get('value') in ('loaded', 'loading')]
        return {'available': True, 'models': resident,
                'note': 'Observed router residency; not proof this gate owns or is testing the model.'}
    except (OSError, ValueError, http.client.HTTPException):
        return {'available': False}
    finally:
        connection.close()
