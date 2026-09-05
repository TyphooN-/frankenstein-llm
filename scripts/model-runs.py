#!/usr/bin/env python3
"""User-operated functional mission and native llama.cpp benchmark runner.

Planning is the default. Never starts/stops services or builds dependencies.
"""
import argparse
import errno
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MISSION = ROOT / 'verification/mission-supervisor'
BIN = ROOT / 'upstream/llama.cpp/build/bin/llama-bench'
BUILD_NAMES = {'makepkg', 'make', 'ninja', 'cmake', 'cargo', 'rustc', 'clang',
               'clang++', 'cc1', 'cc1plus', 'gcc', 'g++', 'ld', 'ld.lld',
               'aria2c', 'curl', 'wget'}


def bounded(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f'must be between {low} and {high}')
        return number
    return parse


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    for name in ('qualify', 'benchmark'):
        s = sub.add_parser(name)
        s.add_argument('--config', default=str(ROOT / 'config/model-runs.json'))
        s.add_argument('--execute', action='store_true', help='run; otherwise print plan only')
        s.add_argument('--timeout', type=bounded(1, 172800), default=86400,
                       help='total run limit in seconds, including mission waiting')
        if name == 'benchmark':
            s.add_argument('--model', required=True, help='local GGUF path, first shard for split models')
            s.add_argument('--devices', default='ROCm0/ROCm1/ROCm2', help='native slash-separated benchmark device list')
            s.add_argument('--tensor-split', default='3/6/2', help='native slash-separated proportions')
            s.add_argument('--confirm-kernel', action='store_true',
                           help='I verified the intended kernel and authorize this benchmark run')
            s.add_argument('--threads', type=bounded(1, 1024), default=len(os.sched_getaffinity(0)))
            s.add_argument('--gpu-layers', type=bounded(0, 999), default=999,
                           help='GPU offload layer limit; use 0 for an explicit CPU baseline')
            s.add_argument('--prompt-tokens', type=bounded(1, 32768), default=512)
            s.add_argument('--generation-tokens', type=bounded(1, 8192), default=128)
            s.add_argument('--repetitions', type=bounded(1, 100), default=3)
    return p


def configured_args(argv):
    args = list(sys.argv[1:] if argv is None else argv)
    initial = parser().parse_args(args)
    cfg = json.loads(Path(initial.config).expanduser().read_text())
    if not isinstance(cfg, dict) or cfg.get('schema') != 1 or set(cfg) - {'schema', 'qualify', 'benchmark'}:
        raise ValueError('unknown config schema or keys')
    allowed = {'qualify': {'timeout'}, 'benchmark': {'timeout', 'threads', 'gpu_layers',
               'prompt_tokens', 'generation_tokens', 'repetitions', 'devices', 'tensor_split'}}
    for mode in ('qualify', 'benchmark'):
        values = cfg.get(mode, {})
        if not isinstance(values, dict) or set(values) - allowed[mode]:
            raise ValueError(f'unknown or unsafe {mode} config keys')
        if any(type(v) is not (str if k in {'devices', 'tensor_split'} else int) for k, v in values.items()):
            raise ValueError('config tuning value has wrong type')
    prefix = []
    for key, value in cfg.get(initial.mode, {}).items():
        prefix.extend(['--' + key.replace('_', '-'), str(value)])
    return parser().parse_args([args[0], *prefix, *args[1:]])


def command(a):
    if a.mode == 'qualify':
        return [sys.executable, str(MISSION / 'run_functional_mission.py')]
    if not re.fullmatch(r'ROCm[0-9]+(?:/ROCm[0-9]+)*', a.devices):
        raise ValueError('devices must be slash-separated ROCm device names')
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)*', a.tensor_split):
        raise ValueError('tensor-split must be slash-separated nonnegative numbers')
    if len(a.devices.split('/')) != len(a.tensor_split.split('/')) or not any(float(x) > 0 for x in a.tensor_split.split('/')):
        raise ValueError('tensor-split must match devices and have a positive share')
    return [str(BIN), '-m', str(Path(a.model).expanduser().resolve()),
            '-dev', a.devices, '-ts', a.tensor_split,
            '-t', str(a.threads), '-ngl', str(a.gpu_layers),
            '-p', str(a.prompt_tokens), '-n', str(a.generation_tokens),
            '-r', str(a.repetitions), '-o', 'json']


def process_blockers(proc=Path('/proc'), benchmark=False):
    blocked = []
    for p in proc.iterdir():
        if not p.name.isdecimal():
            continue
        try:
            raw = (p / 'stat').read_text()
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.ESRCH):
                continue
            raise RuntimeError(f'incomplete process inspection: {p.name}') from e
        end = raw.rfind(')')
        name = raw[raw.find('(') + 1:end]
        state = raw[end + 2:].split()[0]
        if state != 'Z' and (name in BUILD_NAMES or (benchmark and name.startswith('llama-'))):
            blocked.append(f'{p.name}:{name}')
    return sorted(blocked)


def storage_healthy(rc, text):
    return rc == 0 and text.strip() == 'all pools are healthy'


def preflight(a):
    result = subprocess.run(['zpool', 'status', '-x'], capture_output=True, text=True, timeout=15)
    if not storage_healthy(result.returncode, result.stdout):
        raise RuntimeError('ZFS not confirmed healthy. Inspect sudo zpool status -v; do not clear errors just to pass.')
    blocked = process_blockers(benchmark=a.mode == 'benchmark')
    if blocked:
        raise RuntimeError('active competing workload(s): ' + ', '.join(blocked[:12]))
    available = next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()
                     if l.startswith('MemAvailable:'))
    if available < 16 * 1024**3:
        raise RuntimeError('less than 16 GiB available RAM; this minimum is not a model-fit guarantee')
    if a.mode == 'benchmark':
        model = Path(a.model).expanduser().resolve()
        with model.open('rb') as f:
            if f.read(4) != b'GGUF':
                raise RuntimeError('model is not a GGUF file')
        if not BIN.is_file() or not os.access(BIN, os.X_OK):
            raise RuntimeError('llama-bench is not built; see docs/MODEL-RUNS.md')


def stop_child(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def execute(a, cmd, out):
    report = {'schema': 'frankenstein-operator-run/1', 'mode': a.mode,
              'command': cmd, 'status': 'running', 'exit_code': None,
              'benchmarking_requested': a.mode == 'benchmark',
              'kernel': os.uname().release,
              'kernel_build': Path('/proc/version').read_text().strip(),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    target = out / 'report.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    child = None
    old = {}
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old[sig] = signal.signal(sig, interrupted)
        with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
            report['exit_code'] = child.wait(timeout=a.timeout)
        report['status'] = 'passed' if report['exit_code'] == 0 else 'failed'
        if a.mode == 'benchmark' and report['exit_code'] == 0:
            # Preserve native machine-readable metrics, not invented proxy rates.
            data = json.loads((out / 'stdout.log').read_text())
            if not isinstance(data, list) or not data:
                raise ValueError('native benchmark returned no result rows')
            report['native_results'] = 'stdout.log'
    except (subprocess.TimeoutExpired, InterruptedError) as e:
        report.update(status='interrupted', error=str(e), exit_code=124 if isinstance(e, subprocess.TimeoutExpired) else 130)
    except Exception as e:
        report.update(status='failed', error=str(e), exit_code=1)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        if child is not None:
            try:
                stop_child(child)
            except (OSError, subprocess.TimeoutExpired) as e:
                report.update(status='cleanup-failed', cleanup_error=str(e), exit_code=1)
        temp = out / 'report.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        os.replace(temp, target)
    return report['exit_code'] if report['exit_code'] and report['exit_code'] > 0 else (0 if report['status'] == 'passed' else 1)


def main(argv=None):
    try:
        a = configured_args(argv)
        cmd = command(a)
    except (OSError, ValueError) as e:
        print(f'Configuration refused: {e}', file=sys.stderr)
        return 2
    if not a.execute:
        print(json.dumps({'plan_only': True, 'command': cmd, 'timeout': a.timeout}, indent=2))
        return 0
    if a.mode == 'benchmark' and not a.confirm_kernel:
        print('benchmark execution requires --confirm-kernel', file=sys.stderr)
        return 2
    lock = None
    try:
        # Qualification takes this lock inside the existing mission. Benchmarks
        # hold it here so neither lane can overlap the other.
        lock = (MISSION / 'mission.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight(a)
        if a.mode == 'qualify':
            lock.close()
            lock = None
            # Mission reacquires nonblocking; a race loses safely with exit 75.
        outroot = ROOT / 'logs/model-runs'
        outroot.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix=a.mode + '-', dir=outroot))
        print(f'Report directory: {out}', flush=True)
        return execute(a, cmd, out)
    except BlockingIOError:
        print('mission lock busy; stop the existing mission before running this command', file=sys.stderr)
        return 75
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
        print(f'Preflight refused: {e}', file=sys.stderr)
        return 75
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
