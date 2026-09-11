#!/usr/bin/env python3
"""Read-only qualification snapshot. Never starts, stops, or loads a model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualification_detail import detail, router_residency

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / 'verification/qualification-supervisor/evidence/qualification-state.json'
BOOT_ID = Path('/proc/sys/kernel/random/boot_id')
RUNNER_LOGS = {
    'tts-asr-roundtrip': 'tts-local/evidence/tts-runner.log',
    'repository-agent': 'repository-agent/evidence/repo-agent-runner.log',
    'generative-media-functional': 'generative-media/evidence/media-functional-runner.log',
}


def current_boot():
    """The running kernel's boot identity, or None when it cannot be read."""
    try:
        return BOOT_ID.read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


# The recorded cause ledger is read back bounded: it is a state file the reader
# does not control, and a status command must not become a way to print an
# arbitrary amount of arbitrary text.
LEDGER_LIMIT = 32
TEXT_LIMIT = 500


def text(value):
    """One recorded string, truncated, or None. Never a summary of one."""
    return value[:TEXT_LIMIT] if isinstance(value, str) else None


def named_steps(value):
    """A recorded {name, exit_code} ledger, entries kept exactly as written."""
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[:LEDGER_LIMIT]:
        if isinstance(item, dict):
            code = item.get('exit_code')
            rows.append({'name': text(item.get('name')),
                         'exit_code': code if isinstance(code, int) and not isinstance(code, bool) else None})
    return rows


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
        return result.timestamp() if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def snapshot(path, now=None, boot=None, live=False):
    now = time.time() if now is None else now
    boot = current_boot() if boot is None else boot
    with path.open('rb') as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('qualification state exceeds 4 MiB')
    state = json.loads(raw)
    if not isinstance(state, dict) or not isinstance(state.get('steps'), dict):
        raise ValueError('qualification state has no valid steps mapping')
    # The supervisor records its boot id at startup, runs exactly one step at a
    # time, and sets current_step and the top-level running status before it
    # starts the child. A step left marked running that fails any of those three
    # is a record from an invocation that died without reconciling -- a host
    # crash or a hard kill, where no signal handler ran to mark it interrupted.
    recorded_boot = state.get('boot_id')
    crossed_boot = bool(boot and recorded_boot and boot != recorded_boot)
    rows = []
    for name, step in sorted(state['steps'].items()):
        if not isinstance(step, dict):
            raise ValueError(f'invalid step: {name}')
        status = step.get('status', 'unknown')
        if not isinstance(status, str):
            status = 'unknown'
        if status == 'running' and (crossed_boot
                                    or state.get('status') != 'running'
                                    or name != state.get('current_step')):
            status = 'interrupted'
        if (not state.get('input_fingerprint') or
                step.get('input_fingerprint') != state['input_fingerprint']):
            status = 'stale-pass' if status == 'passed' else 'stale-' + status
        start, finish = timestamp(step.get('started_at')), timestamp(step.get('finished_at'))
        elapsed = None if start is None else max(0, (finish if finish is not None else now) - start)
        log = path.parent / f'{name}.log'
        if name in RUNNER_LOGS:
            log = path.parent.parent / RUNNER_LOGS[name]
        # State is data, never a command or an arbitrary log-path authority.
        safe_name = Path(name).name == name and name not in ('.', '..')
        log_age = None
        if safe_name:
            try:
                log_age = max(0, now - log.stat().st_mtime)
            except OSError:
                pass
        rows.append(dict(name=name, status=status, elapsed_seconds=elapsed,
                         exit_code=step.get('exit_code'), log_age_seconds=log_age,
                         log_path=str(log) if safe_name else None,
                         performance_log=step.get('performance_log'),
                         # Why this attempt ended the way it did, when the
                         # supervisor recorded a reason. An exit code alone does
                         # not distinguish a model verdict from a refusal.
                         error=text(step.get('error')),
                         blocked_by=text(step.get('blocked_by'))))
    for row in rows:
        row['detail'] = detail(path.parent.parent.parent, row['name'], state['steps'][row['name']], boot)
        if row['status'].startswith('stale-') and row['detail'].get('available'):
            row['detail']['evidence_scope'] = 'previous-inputs'
            for model in row['detail']['models']:
                model['evidence_scope'] = 'previous-inputs'
    counts = dict(Counter(row['status'] for row in rows))
    updated = timestamp(state.get('updated_at'))
    liveness = ('State-file observation only; running does not prove a live process'
                if not crossed_boot else
                f'State records boot {recorded_boot}; host is now on {boot}, '
                'so nothing it left running is live')
    return dict(status=state.get('status', 'unknown'), current_step=state.get('current_step'),
                router_observation=router_residency() if live else {'available': False},
                total=len(rows), passed=counts.get('passed', 0), counts=counts,
                terminal_attempts=sum(counts.get(k, 0) for k in ('passed', 'failed', 'inconclusive')),
                state_age_seconds=None if updated is None else max(0, now - updated),
                steps=rows, eta='unknown: host waits and failed gates prevent a reliable completion estimate',
                scope='Recorded functional gates, not model count or optimization completeness',
                boot_id=recorded_boot, host_boot_id=boot, boot_crossed=crossed_boot,
                liveness=liveness,
                # The supervisor's own terminal ledger, reported rather than
                # recomputed: it clears these at the start of every invocation,
                # so what is here belongs to the run that wrote this state.
                error=text(state.get('error')),
                failed_steps=named_steps(state.get('failed_steps')),
                blocked_steps=named_steps(state.get('blocked_steps')),
                remaining=[text(item) for item in state.get('remaining', [])[:LEDGER_LIMIT]
                           if isinstance(item, str)] if isinstance(state.get('remaining'), list) else [])


def render(data, details=False):
    lines = [f"LOCAL AI QUALIFICATION: {data['status']} | current: {data['current_step'] or '-'}",
             f"Qualification: {data['passed']}/{data['total']} gates passed; "
             f"{data['terminal_attempts']}/{data['total']} attempts finished",
             'States: ' + ', '.join(f'{key}={value}' for key, value in sorted(data['counts'].items())),
             'ETA: ' + data['eta'], data['scope'], data['liveness'], '',
             f"{'GATE':30} {'STATUS':18} {'ELAPSED':>10} {'EXIT':>5} {'LOG AGE':>10}"]
    for row in data['steps']:
        elapsed = '-' if row['elapsed_seconds'] is None else f"{row['elapsed_seconds'] / 60:.1f}m"
        age = '-' if row['log_age_seconds'] is None else f"{row['log_age_seconds'] / 60:.1f}m"
        lines.append(f"{row['name']:30} {row['status']:18} {elapsed:>10} {str(row['exit_code']):>5} {age:>10}")
        info = row.get('detail', {})
        if info.get('available'):
            outcomes = ', '.join(f'{k}={v}' for k, v in sorted(info['outcomes'].items()))
            denominator = str(info['configured_models']) if info['configured_models'] else '?'
            unit = 'models' if row['name'] in ('router-models', 'repository-agent') else 'result records'
            lines.append(f"  [{info['evidence_scope']}] {unit}: {info['models_recorded']}/{denominator}; "
                         f"{outcomes}; recorded checks: {info['checks_passed']} passed, {info['checks_failed']} failed")
            if info['configured_models']:
                lines.append('  Denominator: ' + info['plan_basis'])
            for model in info['models']:
                failed = model['failed_checks']
                if details or failed or model.get('not_claimed_checks') or model['outcome'] not in ('passed', 'unknown'):
                    lines.append(f"    {model['model']}: {model['outcome']}"
                                 + (' (reused)' if model['reused'] else '')
                                 + f"; checks {model['checks_passed']}/{model['checks_recorded']} passed"
                                 + (f"; FAILED: {', '.join(failed)}" if failed else '')
                                 + (f"; missing: {', '.join(model['missing_checks'])}" if model['missing_checks'] else '')
                                 + (f"; NOT CLAIMED: {', '.join(model['not_claimed_checks'])}" if model.get('not_claimed_checks') else '')
                                 + (f"; other problems: {model['problem_count']}" if model['problem_count'] else ''))
                    if model.get('missing_workflows'):
                        lines.append('      Workflows not completed: ' + ', '.join(model['missing_workflows']))
                    for hint in model.get('problem_hints', []):
                        lines.append('      Cause: ' + hint)
        elif details:
            lines.append('  Detail unavailable: ' + info.get('reason', 'not recorded'))
    observation = data.get('router_observation', {})
    if observation.get('available'):
        lines += ['', 'Router resident/loading models: ' + (', '.join(
            f"{r['model']} ({r['state']})" for r in observation['models']) or 'none'), observation['note']]
    lines += ['', 'Check counts cover published evidence only; an in-flight model may not publish until it finishes.',
              'Use --details for every recorded model and --offline to skip the read-only router probe.']
    if data['error']:
        lines += ['', 'Recorded cause: ' + data['error']]
    inconclusive = [row for row in data['steps'] if row['status'] == 'inconclusive']
    inconclusive_names = {row['name'] for row in inconclusive}
    failures = [row for row in data['failed_steps'] if row['name'] not in inconclusive_names]
    for label, rows in (('Failed', failures), ('Inconclusive (not a model failure)', inconclusive),
                        ('Blocked (nothing ran)', data['blocked_steps'])):
        if rows:
            lines.append(f"{label}: " + ', '.join(
                f"{row['name']}(exit {row['exit_code']})" for row in rows))
    for row in data['steps']:
        if row['error']:
            lines.append(f"  {row['name']}: {row['error']}")
    if data['remaining']:
        lines.append('Remaining: ' + '; '.join(data['remaining']))
    lines += ['', 'Failures require repair/retest; finishing this pass is not qualification success.',
              'Host/process detail: bash scripts/local-model-status.sh --host-sharing',
              'Gate logs: verification/qualification-supervisor/<gate>.log; redirected runner paths in --json']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--watch', type=float, metavar='SECONDS')
    parser.add_argument('--details', action='store_true', help='show all recorded model/check outcomes')
    parser.add_argument('--offline', action='store_true', help='read artifacts only; do not query router residency')
    args = parser.parse_args()
    if args.watch is not None and (not 1 <= args.watch <= 3600):
        parser.error('--watch must be between 1 and 3600 seconds')
    while True:
        try:
            data = snapshot(args.state, live=not args.offline and args.state.resolve() == DEFAULT_STATE.resolve())
            print(json.dumps(data, indent=2) if args.json else render(data, args.details), flush=True)
        except (OSError, ValueError, TypeError) as exc:
            print(json.dumps({'error': str(exc)}) if args.json else f'Qualification status unavailable: {exc}', flush=True)
            return 1
        if args.watch is None:
            return 0
        time.sleep(args.watch)
        print('\n--- refreshed ' + datetime.now(timezone.utc).isoformat() + ' ---', flush=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
