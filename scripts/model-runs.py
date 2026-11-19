#!/usr/bin/env python3
"""User-operated functional qualification and native llama.cpp benchmark runner.

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
import time

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / 'verification/qualification-supervisor'
BIN = ROOT / 'upstream/llama.cpp/build/bin/llama-bench'
BUILD_NAMES = {'makepkg', 'make', 'ninja', 'cmake', 'cargo', 'rustc', 'clang',
               'clang++', 'cc1', 'cc1plus', 'gcc', 'g++', 'ld', 'ld.lld',
               'aria2c', 'curl', 'wget'}
PROC = Path('/proc')
POLL_SECONDS = 0.1
GRACE_SECONDS = 10
ESCAPED = ('descendants that started another session (setsid) are not observable or '
           'signalled here; closing them needs separately admitted containment')


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
                       help='total run limit in seconds, including qualification waiting')
        if name == 'benchmark':
            s.add_argument('--model', required=True, help='local GGUF path, first shard for split models')
            s.add_argument('--bench-binary', type=lambda value: Path(value).expanduser().resolve(),
                           help='explicit local llama-bench executable; CLI-only, no build or installation')
            s.add_argument('--devices', default='ROCm0/ROCm1/ROCm2',
                           help='native slash-separated benchmark devices, or none with --gpu-layers 0')
            s.add_argument('--tensor-split', default='1/1/1',
                           help='native slash-separated proportions; the repository default '
                                'is the equal split llama-models.ini [*] configures')
            s.add_argument('--confirm-kernel', action='store_true',
                           help='I verified the intended kernel and authorize this benchmark run')
            s.add_argument('--threads', type=bounded(1, 1024), default=len(os.sched_getaffinity(0)))
            s.add_argument('--gpu-layers', type=bounded(0, 999), default=999,
                           help='GPU offload layer limit; use 0 for an explicit CPU baseline')
            s.add_argument('--prompt-tokens', type=bounded(1, 32768), default=512)
            s.add_argument('--generation-tokens', type=bounded(1, 8192), default=128)
            s.add_argument('--repetitions', type=bounded(1, 100), default=3)
            s.add_argument('--load-mode', choices=('mmap',),
                           help='explicit mmap load for a separately admitted diagnostic')
            s.add_argument('--lazy-mode', choices=('on',),
                           help='explicit lazy reads for source-marked tensors, when supported')
            s.add_argument('--no-repack', action='store_true', help='disable native weight repacking')
            s.add_argument('--no-warmup', action='store_true', help='skip native benchmark warmup')
            s.add_argument('--offline', action='store_true', help='disable native model network access')
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


def benchmark_binary(a):
    # Explicit paths are resolved once by argparse. Construction and preflight
    # consume that same path; the default remains the existing build target.
    return a.bench_binary if a.bench_binary is not None else BIN


def resolve_benchmark_model(a):
    return Path(a.model).expanduser().resolve()


def command(a, *, model_path=None):
    if a.mode == 'qualify':
        return [sys.executable, str(QUALIFICATION / 'run_qualification.py')]
    cpu_only = a.devices == 'none'
    if cpu_only and a.gpu_layers != 0:
        raise ValueError('CPU-only devices none requires --gpu-layers 0')
    if not cpu_only and not re.fullmatch(r'ROCm[0-9]+(?:/ROCm[0-9]+)*', a.devices):
        raise ValueError('devices must be slash-separated ROCm device names')
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)*', a.tensor_split):
        raise ValueError('tensor-split must be slash-separated nonnegative numbers')
    if (not cpu_only and len(a.devices.split('/')) != len(a.tensor_split.split('/'))) or not any(float(x) > 0 for x in a.tensor_split.split('/')):
        raise ValueError('tensor-split must match devices and have a positive share')
    if model_path is None:
        model_path = resolve_benchmark_model(a)
    cmd = [str(benchmark_binary(a)), '-m', str(model_path), '-dev', a.devices]
    if not cpu_only:
        cmd.extend(['-ts', a.tensor_split])
    cmd.extend(['-t', str(a.threads), '-ngl', str(a.gpu_layers),
                '-p', str(a.prompt_tokens), '-n', str(a.generation_tokens),
                '-r', str(a.repetitions), '-o', 'json'])
    if a.load_mode is not None:
        cmd.extend(['-lm', a.load_mode])
    if a.lazy_mode is not None:
        cmd.extend(['-lzm', a.lazy_mode])
    if a.no_repack:
        cmd.extend(['--repack', '0'])
    if a.no_warmup:
        cmd.append('--no-warmup')
    if a.offline:
        cmd.append('--offline')
    return cmd


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


def btrfs_healthy(rc, text):
    required = {'write_io_errs', 'read_io_errs', 'flush_io_errs',
                'corruption_errs', 'generation_errs'}
    devices = {}
    if rc != 0:
        return False
    for line in text.splitlines():
        match = re.fullmatch(r'\[(.+)\]\.([a-z_]+)\s+([0-9]+)', line.strip())
        if match is None:
            return False
        device, counter, value = match.groups()
        counters = devices.setdefault(device, set())
        if counter not in required or counter in counters or int(value) != 0:
            return False
        counters.add(counter)
    return bool(devices) and all(counters == required for counters in devices.values())


def storage_preflight(paths):
    checked = set()
    for path in paths:
        result = subprocess.run(
            ['findmnt', '--json', '--target', str(path),
             '--output', 'TARGET,SOURCE,FSTYPE,OPTIONS'],
            capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(f'cannot identify storage for {path}')
        try:
            mounts = json.loads(result.stdout)['filesystems']
            if not isinstance(mounts, list) or len(mounts) != 1:
                raise ValueError('expected exactly one containing mount')
            mount = mounts[0]
            target, source, fstype, options = (mount[key] for key in (
                'target', 'source', 'fstype', 'options'))
            if not all(isinstance(value, str) and value for value in (
                    target, source, fstype, options)) or not Path(target).is_absolute():
                raise ValueError('incomplete mount identity')
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f'invalid storage observation for {path}') from error
        flags = options.split(',')
        if ('rw' in flags) == ('ro' in flags):
            raise RuntimeError(f'invalid storage mount options: {target}')
        identity = (target, source, fstype)
        if identity in checked:
            continue
        if fstype == 'btrfs':
            result = subprocess.run(['btrfs', 'device', 'stats', '--check', target],
                                    capture_output=True, text=True, timeout=15)
            healthy = btrfs_healthy(result.returncode, result.stdout)
        elif fstype == 'zfs':
            result = subprocess.run(['zpool', 'status', '-x'],
                                    capture_output=True, text=True, timeout=15)
            healthy = storage_healthy(result.returncode, result.stdout)
        else:
            raise RuntimeError(f'unsupported storage filesystem: {fstype}')
        if not healthy:
            raise RuntimeError(f'{fstype} storage check failed for {target}; inspect the recorded filesystem errors')
        checked.add(identity)


def preflight(a, outroot=None, *, model_path=None):
    model_store = ROOT / 'models'
    if a.mode == 'benchmark':
        model_store = model_path if model_path is not None else resolve_benchmark_model(a)
    if outroot is None:
        outroot = (ROOT / 'logs/model-runs').resolve()
    output_store = outroot
    while not output_store.exists():
        parent = output_store.parent
        if parent == output_store:
            raise RuntimeError('cannot identify an existing output storage ancestor')
        output_store = parent
    storage_preflight([ROOT, model_store, output_store])
    blocked = process_blockers(benchmark=a.mode == 'benchmark')
    if blocked:
        raise RuntimeError('active competing workload(s): ' + ', '.join(blocked[:12]))
    available = next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()
                     if l.startswith('MemAvailable:'))
    if available < 16 * 1024**3:
        raise RuntimeError('less than 16 GiB available RAM; this minimum is not a model-fit guarantee')
    if a.mode == 'benchmark':
        with model_store.open('rb') as f:
            if f.read(4) != b'GGUF':
                raise RuntimeError('model is not a GGUF file')
        binary = benchmark_binary(a)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError('llama-bench is not built; see docs/MODEL-RUNS.md')


def stat_fields(entry):
    """(state, process group, session, num_threads); None once gone.

    A Z thread-group leader can still have live workers. Missing or unusable
    field 20 must not turn an incomplete observation into session emptiness.
    """
    try:
        raw = (entry / 'stat').read_text()
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.ESRCH):
            return None
        raise RuntimeError(f'incomplete owned-session inspection: {entry.name}') from e
    end = raw.rfind(')')
    fields = raw[end + 2:].split()
    if (end < 0 or len(fields) < 18 or fields[0] not in tuple('RSDZTtXxKWPI')
            or re.fullmatch(r'[0-9]+', fields[17]) is None or int(fields[17]) < 1):
        raise ValueError(f'unusable owned-session stat metadata: {entry.name}')
    return fields[0], int(fields[2]), int(fields[3]), int(fields[17])


def session_members(leader):
    """Live processes other than the leader in its session, as (pid, pgrp).

    The leader's process group is a subset; a member that moved to another
    group with setpgid is still in the session. Like process_blockers, only
    stat is read.
    """
    members = []
    for entry in PROC.iterdir():
        if entry.name.isdecimal() and int(entry.name) != leader:
            fields = stat_fields(entry)
            if (fields and fields[0] not in ('X', 'x') and fields[2] == leader
                    and (fields[0] != 'Z' or fields[3] > 1)):
                members.append((int(entry.name), fields[1]))
    return sorted(members)


def leader_exit(child, custody, consume=False):
    """The leader's exit status, or None while it runs.

    WNOWAIT leaves an exited leader a zombie, so its pid, process group and
    session ids cannot be reused while killpg and session_members rely on
    them; only consume=True reaps it. A status that is not delivered (ECHILD
    once another waiter took it) marks custody uncertain and is never
    inferred, unlike Popen.wait/poll, which report ECHILD as exit status 0.
    """
    try:
        result = os.waitid(os.P_PID, child.pid,
                           os.WEXITED | os.WNOHANG | (0 if consume else os.WNOWAIT))
        if result is None and consume:
            raise ChildProcessError('exited leader status was not delivered')
    except OSError as e:
        custody.update(state='uncertain', error=f'leader status not delivered: {e}')
        raise RuntimeError(custody['error']) from e
    if result is None:
        return None
    return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status


def signal_session(leader, members, signum):
    """Signal the leader's process group, then each session member outside it.

    killpg is exact while the leader is unreaped. Other members are signalled
    through a pidfd bound before their session is re-read, so a pid reused by
    an unrelated process is never signalled.
    """
    os.killpg(leader, signum)
    for pid, pgrp in members:
        if pgrp == leader:
            continue
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            fields = stat_fields(PROC / str(pid))
            if (fields and fields[0] not in ('X', 'x') and fields[2] == leader
                    and (fields[0] != 'Z' or fields[3] > 1)):
                signal.pidfd_send_signal(fd, signum)
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)


def stop_child(child, custody):
    """Close the owned session, then reap its leader.

    TERM, then KILL, each followed by a grace period, are sent only while the
    leader or another session member is alive. The leader is reaped only once
    it has exited and two consecutive session observations were empty; until then it pins the
    signalled ids. A reaped, uncertain or unresolved leader is never
    signalled again.
    """
    if custody['state'] == 'settled':
        return
    if custody['state'] != 'running':
        raise RuntimeError(f"custody {custody['state']}: nothing more is signalled")
    sent = custody.setdefault('signals_sent', [])
    code = members = scan_error = None
    empty_scans = 0
    for signum in (None, signal.SIGTERM, signal.SIGKILL):
        if signum is not None:
            name = signal.Signals(signum).name
            try:
                signal_session(child.pid, members or [], signum)
            except (OSError, RuntimeError, ValueError) as e:
                custody.update(state='unresolved', error=f'{name} not delivered: {e}')
                raise RuntimeError(custody['error']) from e
            sent.append({'signal': name, 'leader_exited': code is not None,
                         'members': [pid for pid, _ in members or []]})
        deadline = time.monotonic() + (GRACE_SECONDS if signum else 0)
        while True:
            code = leader_exit(child, custody)
            try:
                members, scan_error = session_members(child.pid), None
            except (OSError, RuntimeError, ValueError) as e:
                members, scan_error = None, e
            empty_scans = empty_scans + 1 if code is not None and members == [] else 0
            if code is not None and members == []:
                if empty_scans < 2:
                    # Catch a one-scan gap before consuming the identity pin.
                    # This is not atomic fork/setsid or descendant containment.
                    time.sleep(POLL_SECONDS)
                    continue
                custody['leader_returncode'] = code
                child.returncode = leader_exit(child, custody, consume=True)
                custody['state'] = 'settled'
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(POLL_SECONDS)
    custody.update(state='unresolved', leader_exited=code is not None,
                   members_remaining=None if members is None else [pid for pid, _ in members],
                   error='owned session not closed after SIGKILL' + (
                       f'; inspection failed: {scan_error}' if scan_error else ''))
    raise RuntimeError(custody['error'])


def publish(target, report):
    temp = target.with_name('report.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temp, target)


def execute(a, cmd, out, lease=None):
    cancelled = []
    lease_unconfirmed = 'held by parent; passing to the child is unconfirmed until the dispatch receipt'
    custody = {'state': 'dispatching', 'signals_received': cancelled,
               'lease': 'not held by this runner' if lease is None else lease_unconfirmed,
               'escaped_sessions': ESCAPED}
    report = {'schema': 'frankenstein-operator-run/1', 'mode': a.mode,
              'command': cmd, 'status': 'running', 'exit_code': None,
              'benchmarking_requested': a.mode == 'benchmark', 'custody': custody}
    target = out / 'report.json'
    child = None
    creation_attempted = False
    old = {}

    def deferred(signum, frame):
        # Record only. Raising here could abandon a child Popen has already
        # created (NAT-2) or skip cleanup and publication (NAT-3). Nothing is
        # blocked, so the child inherits no signal mask; exec resets handlers.
        cancelled.append(signum)

    try:
        for sig, handler in ((signal.SIGINT, deferred), (signal.SIGTERM, deferred),
                             (signal.SIGHUP, deferred), (signal.SIGCHLD, signal.SIG_DFL)):
            old[sig] = signal.signal(sig, handler)
        report.update(kernel=os.uname().release,
                      kernel_build=Path('/proc/version').read_text().strip(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        publish(target, report)
        if cancelled:
            raise InterruptedError(f'signal {cancelled[0]} before dispatch')
        passed = () if lease is None else (lease.fileno(),)
        if passed and passed[0] < 3:
            raise ValueError('lease descriptor must be at least 3')
        with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
            # Passing a descriptor is not proof a child/descendant keeps it.
            # Actual OFD custody and parent-death containment remain NAT-4.
            creation_attempted = True
            try:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr,
                                         start_new_session=True, pass_fds=passed)
            except OSError:
                # CPython's exec/chdir OSError path cleans up its child before
                # raising. Do not extend this negative to arbitrary failures.
                custody['state'] = 'not-dispatched'
                if lease is not None:
                    custody['lease'] = 'held by parent; not yet passed to a child'
                raise
            custody.update(state='running', pid=child.pid)
            if lease is not None:
                custody['lease'] = 'passed to the child; held only while a holder keeps the descriptor'
        publish(target, report)
        deadline = time.monotonic() + a.timeout
        while (code := leader_exit(child, custody)) is None:
            if cancelled:
                raise InterruptedError(f'signal {cancelled[0]}')
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd, a.timeout)
            time.sleep(POLL_SECONDS)
        report['exit_code'] = code
        report['status'] = 'passed' if code == 0 else 'failed'
        if a.mode == 'benchmark' and code == 0:
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
        try:
            if child is not None:
                try:
                    stop_child(child, custody)
                except RuntimeError as e:
                    report.update(status='cleanup-failed', cleanup_error=str(e), exit_code=1)
            elif not creation_attempted:
                custody['state'] = 'not-dispatched'
                if lease is not None:
                    custody['lease'] = 'held by parent; not yet passed to a child'
            elif custody['state'] != 'not-dispatched':
                custody.update(state='uncertain', error='child creation did not return a handle; dispatch is indeterminate')
                if lease is not None:
                    custody['lease'] = lease_unconfirmed
            try:
                publish(target, report)
            except OSError as e:
                report['publication_error'] = str(e)
                print(f'{target} could not be written; final run state follows:', file=sys.stderr)
                print(json.dumps(report), file=sys.stderr)
        finally:
            # Restored last: until the final report is out, a signal is only
            # recorded and cannot end this parent mid-cleanup (NAT-3).
            for sig, handler in old.items():
                signal.signal(sig, handler)
    if child is None and custody['state'] == 'not-dispatched':
        print(f"No child was dispatched: {report.get('error', report['status'])}", file=sys.stderr)
    if 'publication_error' in report:
        return 1
    return report['exit_code'] if report['exit_code'] and report['exit_code'] > 0 else (0 if report['status'] == 'passed' else 1)


def main(argv=None):
    try:
        a = configured_args(argv)
        # Validate the same resolved path that the eventual child will use.
        model_path = resolve_benchmark_model(a) if a.mode == 'benchmark' else None
        cmd = command(a, model_path=model_path)
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
        try:
            # Qualification takes this lock inside the existing qualification. Benchmarks
            # hold it here so neither lane can overlap the other.
            lock = (QUALIFICATION / 'qualification.lock').open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.mode == 'benchmark' and lock.fileno() < 3:
                raise RuntimeError('lease descriptor must be at least 3')
            outroot = (ROOT / 'logs/model-runs').resolve()
            preflight(a, outroot, model_path=model_path)
            if a.mode == 'qualify':
                lock.close()
                lock = None
                # Qualification reacquires nonblocking; a race loses safely with exit 75.
            outroot.mkdir(parents=True, exist_ok=True)
            out = Path(tempfile.mkdtemp(prefix=a.mode + '-', dir=outroot))
        except BlockingIOError:
            print('qualification lock busy; stop the existing qualification before running this command', file=sys.stderr)
            return 75
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
            print(f'Preflight refused: {e}', file=sys.stderr)
            return 75
        # No later failure is a wrapper preflight refusal. This print can fail
        # before execute has published any report; a child may itself exit 75.
        print(f'Report directory: {out}', flush=True)
        return execute(a, cmd, out, lease=lock)
    finally:
        # The benchmark descriptor was passed, not verified held: closing our
        # copy preserves exclusion only while another holder keeps its copy.
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
