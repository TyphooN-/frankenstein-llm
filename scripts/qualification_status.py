#!/usr/bin/env python3
"""Read-only qualification snapshot. Never starts, stops, or loads a model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / 'verification/qualification-supervisor/qualification-state.json'
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


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
        return result.timestamp() if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def snapshot(path, now=None, boot=None):
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
                         performance_log=step.get('performance_log')))
    counts = dict(Counter(row['status'] for row in rows))
    updated = timestamp(state.get('updated_at'))
    liveness = ('State-file observation only; running does not prove a live process'
                if not crossed_boot else
                f'State records boot {recorded_boot}; host is now on {boot}, '
                'so nothing it left running is live')
    return dict(status=state.get('status', 'unknown'), current_step=state.get('current_step'),
                total=len(rows), passed=counts.get('passed', 0), counts=counts,
                terminal_attempts=sum(counts.get(k, 0) for k in ('passed', 'failed', 'inconclusive')),
                state_age_seconds=None if updated is None else max(0, now - updated),
                steps=rows, eta='unknown: host waits and failed gates prevent a reliable completion estimate',
                scope='Recorded functional gates, not model count or optimization completeness',
                boot_id=recorded_boot, host_boot_id=boot, boot_crossed=crossed_boot,
                liveness=liveness)


def render(data):
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
    lines += ['', 'Failures require repair/retest; finishing this pass is not qualification success.',
              'Host/process detail: bash scripts/local-model-status.sh --host-sharing',
              'Gate logs: verification/qualification-supervisor/<gate>.log; redirected runner paths in --json']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--watch', type=float, metavar='SECONDS')
    args = parser.parse_args()
    if args.watch is not None and (not 1 <= args.watch <= 3600):
        parser.error('--watch must be between 1 and 3600 seconds')
    while True:
        try:
            data = snapshot(args.state)
            print(json.dumps(data, indent=2) if args.json else render(data), flush=True)
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
