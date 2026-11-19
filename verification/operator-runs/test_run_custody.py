"""Custody of the dispatched child, proved against a synthetic kernel.

Every effect is a fake: Popen, os.waitid, os.killpg, os.pidfd_open,
signal.pidfd_send_signal, signal.signal/pthread_sigmask, the monotonic clock,
the lease and a private /proc tree. No OS process, signal, wait or lease
operation happens. A pass proves model-runs' custody decisions against this
model only; Linux process-group, session, signal, lease and parent-death
behaviour still needs a separately admitted real-process stage.
"""
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from test_operator_runs import btrfs_stats, m, mount_observation, runnable_benchmark

PARENT = 4000   # this runner, inside the operator's session 3999
LEADER = 4100   # start_new_session child: pid == process group == session


class ParentKilled(BaseException):
    """The default action of a signal would terminate the Python parent here."""


class Proc:
    def __init__(self, pid, ppid, pgrp, sid, comm, ignore=(), unkillable=False, term_delay=None,
                 leader_zombie=False, num_threads=1, retain_zombie=False):
        self.pid, self.ppid, self.pgrp, self.sid, self.comm = pid, ppid, pgrp, sid, comm
        self.ignore, self.unkillable, self.term_delay = set(ignore), unkillable, term_delay
        self.status = None   # (si_code, si_status) once exited
        self.leader_zombie, self.num_threads = leader_zombie, num_threads
        self.retain_zombie = retain_zombie
        self.received = []


def group_member(pid=4101, **kw):
    return dict(pid=pid, ppid=LEADER, pgrp=LEADER, sid=LEADER, comm='worker) (x', **kw)


def moved_member(pid=4102, **kw):   # called setpgid: new group, same session
    return dict(pid=pid, ppid=LEADER, pgrp=pid, sid=LEADER, comm='setpgid', **kw)


def escaped(pid=4103, **kw):        # called setsid: new session
    return dict(pid=pid, ppid=LEADER, pgrp=pid, sid=pid, comm='setsid', **kw)


class Child:
    """The subprocess.Popen surface the caller uses. poll()/wait() reap like
    CPython 3.14, including Popen._try_wait/_internal_poll mapping ECHILD to
    returncode 0."""

    def __init__(self, kernel, proc):
        self.kernel, self.pid, self.returncode = kernel, proc.pid, None

    def _reap(self):
        proc = self.kernel.table.get(self.pid)
        if proc is not self.kernel.leader:
            self.returncode = 0
        elif proc.status is not None:
            self.kernel.reap(proc)
            code, status = proc.status
            self.returncode = status if code == os.CLD_EXITED else -status
        return self.returncode

    def poll(self):
        self.kernel.fire('first-query')
        self.kernel.events.append(('popen-poll', self.pid))
        return self.returncode if self.returncode is not None else self._reap()

    def wait(self, timeout=None):
        self.kernel.fire('first-query')
        self.kernel.events.append(('popen-wait', self.pid, timeout))
        deadline = None if timeout is None else self.kernel.now + timeout
        while self.returncode is None and self._reap() is None:
            due = [t[0] for t in self.kernel.timers if deadline is None or t[0] <= deadline]
            if not due:
                if deadline is None:
                    raise AssertionError('unbounded wait on a leader that never exits')
                self.kernel.advance(deadline)
                if self._reap() is None:
                    raise subprocess.TimeoutExpired('synthetic', timeout)
                break
            self.kernel.advance(min(due))
        return self.returncode


class Kernel:
    """Synthetic process table, clock, dispositions and lease for one run."""

    def __init__(self, tmp_path, monkeypatch, sigchld=signal.SIG_DFL):
        self.proc = tmp_path / 'private-proc'
        self.proc.mkdir()
        self.standin = tmp_path / 'pidfd-standin'
        self.standin.write_text('')
        self.table, self.events, self.timers, self.pidfds = {}, [], [], {}
        self.now, self.sequence = 0.0, 0
        self.handlers = {signal.SIGINT: signal.default_int_handler,
                         signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL,
                         signal.SIGCHLD: sigchld}
        self.original = dict(self.handlers)
        self.plan = {'leader': {}, 'members': []}
        self.hooks, self.popen_calls = {}, []
        self.leader = self.lease = self.native_rows = self.refusal = None
        self.lease_inheritance = True
        self.consumed_elsewhere = self.reuse_pid = False
        for pid, sid, comm in ((1, 1, 'systemd'), (PARENT, 3999, 'python3'),
                               (5200, 5200, 'llama-server')):
            self.add(Proc(pid, 0 if pid == 1 else 1, sid, sid, comm))
        read_text = Path.read_text

        def kernel_text(path, *args, **kwargs):
            if str(path) == '/proc/version':
                return 'Linux version synthetic-custody\n'
            if str(path) == '/proc/sys/kernel/random/boot_id':
                return 'synthetic-boot\n'
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', kernel_text)
        monkeypatch.setattr(m, 'PROC', self.proc, raising=False)
        monkeypatch.setattr(m, 'time', SimpleNamespace(monotonic=self.monotonic, sleep=self.sleep),
                            raising=False)
        monkeypatch.setattr(m.subprocess, 'Popen', self.popen)
        monkeypatch.setattr(m.os, 'waitid', self.waitid)
        monkeypatch.setattr(m.os, 'killpg', self.killpg)
        monkeypatch.setattr(m.os, 'pidfd_open', self.pidfd_open)
        monkeypatch.setattr(m.signal, 'pidfd_send_signal', self.pidfd_send_signal)
        monkeypatch.setattr(m.signal, 'signal', self.signal)
        monkeypatch.setattr(m.signal, 'pthread_sigmask', self.sigmask)
        replace = os.replace

        def observed_replace(source, target):
            if Path(target).name == 'report.json':
                status = json.loads(Path(source).read_text())['status']
                self.events.append(('publish', status))
                if status != 'running':
                    self.fire('final-publication')
            return replace(source, target)

        monkeypatch.setattr(m.os, 'replace', observed_replace)

    # process table and private /proc
    def write(self, proc):
        entry = self.proc / str(proc.pid)
        entry.mkdir(exist_ok=True)
        state = 'Z' if proc.status is not None or proc.leader_zombie else 'S'
        (entry / 'stat').write_text(f'{proc.pid} ({proc.comm}) {state} {proc.ppid} {proc.pgrp} '
                                    f'{proc.sid} 0 -1 4194560 0 0 0 0 1 0 0 0 20 0 {proc.num_threads} 0\n')

    def add(self, proc):
        self.table[proc.pid] = proc
        self.write(proc)
        return proc

    def remove(self, proc):
        del self.table[proc.pid]
        (self.proc / str(proc.pid) / 'stat').unlink()
        (self.proc / str(proc.pid)).rmdir()

    def reap(self, proc):
        self.remove(proc)
        self.events.append(('reaped', proc.pid))

    def exit(self, proc, code, status):
        if proc.status is not None:
            return
        proc.status = (code, status)
        proc.num_threads = 1
        self.close_lease(proc.pid)
        self.events.append(('exit', proc.pid, status if code == os.CLD_EXITED else -status))
        if proc is not self.leader:
            if proc.retain_zombie:
                self.write(proc)
            else:
                self.remove(proc)   # reaped promptly by its own parent or init
        elif self.handlers.get(signal.SIGCHLD) == signal.SIG_IGN or self.consumed_elsewhere:
            self.remove(proc)   # autoreaped, or consumed by another waiter
            self.events.append(('consumed-elsewhere', proc.pid))
            if self.reuse_pid:
                self.add(Proc(proc.pid, 1, proc.pid, proc.pid, 'unrelated'))
        else:
            self.write(proc)    # zombie keeps pid, group and session ids pinned

    def deliver(self, proc, sig):
        proc.received.append(sig)
        if proc.status is not None or proc.unkillable:
            return
        if sig == signal.SIGKILL:
            self.exit(proc, os.CLD_KILLED, sig)
        elif sig not in proc.ignore:
            if proc.term_delay:
                self.at(self.now + proc.term_delay, lambda: self.exit(proc, os.CLD_KILLED, sig))
            else:
                self.exit(proc, os.CLD_KILLED, sig)

    def live_owned(self):
        return sorted(pid for pid, proc in self.table.items()
                      if proc.status is None and proc.sid == LEADER)

    # clock and one-shot injection points
    def at(self, when, action):
        self.sequence += 1
        self.timers.append((when, self.sequence, action))

    def advance(self, until):
        while True:
            due = sorted((t for t in self.timers if t[0] <= until), key=lambda t: t[:2])
            if not due:
                break
            self.timers.remove(due[0])
            self.now = max(self.now, due[0][0])
            due[0][2]()
        self.now = max(self.now, until)

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.advance(self.now + seconds)

    def fire(self, name):
        action = self.hooks.pop(name, None)
        if action is not None:
            action()

    # parent dispositions
    def signal(self, sig, handler):
        previous = self.handlers.get(sig, signal.SIG_DFL)
        self.handlers[sig] = handler
        self.events.append(('disposition', sig))
        return previous

    def sigmask(self, how, mask):
        self.events.append(('sigmask', how, tuple(mask)))
        return set()

    def signal_parent(self, sig):
        self.events.append(('parent-signal', sig))
        handler = self.handlers.get(sig, signal.SIG_DFL)
        if handler == signal.SIG_IGN or (handler == signal.SIG_DFL and sig == signal.SIGCHLD):
            return
        if handler == signal.SIG_DFL:
            raise ParentKilled(sig)
        handler(sig, None)

    # process boundary
    def popen(self, cmd, **kwargs):
        self.popen_calls.append(dict(kwargs, cmd=cmd, handlers=dict(self.handlers)))
        self.events.append(('popen',))
        plan = self.plan['leader']
        self.leader = self.add(Proc(LEADER, PARENT, LEADER, LEADER, 'llama-bench', **{
            key: plan[key] for key in ('ignore', 'unkillable', 'term_delay') if key in plan}))
        for member in self.plan['members']:
            self.add(Proc(**member))
        if 'exit_at' in plan:
            self.at(plan['exit_at'], lambda: self.exit(self.leader, os.CLD_EXITED, plan.get('code', 0)))
        if self.native_rows is not None:
            kwargs['stdout'].write(self.native_rows)
            kwargs['stdout'].flush()
        if self.lease is not None and self.lease['fd'] in tuple(kwargs.get('pass_fds') or ()):
            fd = self.lease['fd']
            self.events.append(('lease-passed', LEADER, fd))
            # Passing is not holding: a failed inheritance or dup2 over a
            # standard output/error descriptor leaves no reference in child.
            collision = (fd == 1 and kwargs.get('stdout') is not None or
                         fd == 2 and kwargs.get('stderr') is not None)
            if self.lease_inheritance and not collision:
                self.inherit_lease(PARENT, LEADER)
                for member in self.plan['members']:
                    self.inherit_lease(LEADER, member['pid'])
        self.fire('inside-popen')
        return Child(self, self.leader)

    def waitid(self, idtype, pid, options):
        self.fire('first-query')
        assert idtype == os.P_PID and options & os.WEXITED
        self.events.append(('waitid', pid, bool(options & os.WNOWAIT)))
        proc = self.table.get(pid)
        if proc is None or proc is not self.leader:
            raise ChildProcessError(errno.ECHILD, 'No child processes')
        if proc.status is None:
            if options & os.WNOHANG:
                return None
            raise AssertionError('caller blocked waiting on a running leader')
        if not options & os.WNOWAIT:
            self.reap(proc)
        return os.waitid_result((pid, 0, signal.SIGCHLD, proc.status[1], proc.status[0]))

    def killpg(self, pgid, sig):
        self.events.append(('killpg', pgid, sig))
        if self.refusal is not None:
            raise self.refusal
        targets = [proc for proc in list(self.table.values()) if proc.pgrp == pgid]
        if not targets:
            raise ProcessLookupError(errno.ESRCH, 'No such process')
        for proc in targets:
            self.deliver(proc, sig)

    def pidfd_open(self, pid, flags=0):
        self.fire('pidfd-open')
        self.events.append(('pidfd_open', pid))
        proc = self.table.get(pid)
        if proc is None:
            raise ProcessLookupError(errno.ESRCH, 'No such process')
        fd = os.open(self.standin, os.O_RDONLY)   # private real descriptor so os.close works
        self.pidfds[fd] = proc
        return fd

    def pidfd_send_signal(self, fd, sig, siginfo=None, flags=0):
        proc = self.pidfds[fd]
        self.events.append(('pidfd_send_signal', proc.pid, sig))
        if self.table.get(proc.pid) is not proc:
            raise ProcessLookupError(errno.ESRCH, 'No such process')
        self.deliver(proc, sig)

    def signalled(self):
        return [event for event in self.events if event[0] in ('killpg', 'pidfd_send_signal')]

    def open_pidfds(self):
        standin = os.stat(self.standin)
        leaked = []
        for fd in self.pidfds:
            try:
                info = os.fstat(fd)
            except OSError:
                continue
            if (info.st_dev, info.st_ino) == (standin.st_dev, standin.st_ino):
                leaked.append(fd)
        return leaked

    # synthetic lease: never an OS flock
    def track_lease(self, monkeypatch, fd=None):
        # One synthetic open-file description; descriptor references are
        # explicit, independent of process life and of the pass_fds request.
        lease = {'fd': None, 'references': {}, 'ops': []}

        def flock(handle, operation):
            lease['ops'].append(operation)
            if operation != (m.fcntl.LOCK_EX | m.fcntl.LOCK_NB):
                raise AssertionError('lease operation other than nonblocking acquisition')
            if fd is not None:
                handle.fileno = lambda: fd
            lease['fd'] = handle.fileno()
            lease['references'][PARENT] = {lease['fd']}
            close = handle.close

            def dropped():
                if not handle.closed:
                    self.close_lease(PARENT)
                    self.events.append(('lease-closed-by-parent',))
                close()

            handle.close = dropped

        monkeypatch.setattr(m.fcntl, 'flock', flock)
        self.lease = lease
        return lease

    def inherit_lease(self, source, target):
        descriptors = self.lease['references'].get(source, set())
        if descriptors:
            self.lease['references'][target] = set(descriptors)
            self.events.append(('lease-inherited', source, target))

    def close_lease(self, pid):
        if self.lease is not None:
            self.lease['references'].pop(pid, None)

    def lease_held(self):
        return any(self.lease['references'].values())


NATIVE_ROWS = b'[{"model_type": "synthetic fixture", "n_prompt": 16, "avg_ts": 1.0}]\n'


def healthy_storage(monkeypatch):
    def observe(argv, **kwargs):
        text = json.dumps(mount_observation()) if argv[0] == 'findmnt' else btrfs_stats()
        return subprocess.CompletedProcess(argv, 0, text, '')
    monkeypatch.setattr(m.subprocess, 'run', observe)


def stderr_report(err):
    return [json.loads(line) for line in err.splitlines() if line.startswith('{')][-1]


def run_execute(kernel, tmp_path, timeout=30):
    a = m.parser().parse_args(['qualify', '--timeout', str(timeout)])
    out = tmp_path / 'run'
    out.mkdir()
    code = m.execute(a, ['synthetic-native', '--fixture'], out)
    return code, json.loads((out / 'report.json').read_text())


@pytest.mark.parametrize('moment', ['control', 'inside-popen', 'after-adoption'])
def test_nat2_cancellation_around_creation_keeps_exact_child_custody(tmp_path, monkeypatch, moment):
    kernel = Kernel(tmp_path, monkeypatch)
    kernel.plan['leader'] = {'exit_at': 1.0}
    if moment == 'inside-popen':
        # After fork/exec, before Popen returns (CPython reads its exec error
        # pipe there, so a raising Python handler would lose the object).
        kernel.hooks['inside-popen'] = lambda: kernel.signal_parent(signal.SIGTERM)
    elif moment == 'after-adoption':
        kernel.hooks['first-query'] = lambda: kernel.signal_parent(signal.SIGTERM)
    code, report = run_execute(kernel, tmp_path)
    live = kernel.live_owned()
    assert not live, f'NAT-2: created child {live} left running without custody'
    assert ('reaped', LEADER) in kernel.events, 'NAT-2: created child never reaped'
    if moment == 'control':
        assert (code, report['status']) == (0, 'passed')
        assert kernel.signalled() == []
    else:
        assert (code, report['status']) == (130, 'interrupted')
        assert kernel.signalled() == [('killpg', LEADER, signal.SIGTERM)]
    assert kernel.handlers == kernel.original, 'parent dispositions were not restored'
    assert not [event for event in kernel.events if event[0] == 'sigmask'], \
        'signal mask changed around dispatch; the child would inherit it'


@pytest.mark.parametrize('scenario', ['control', 'group-member', 'moved-member',
                                      'moved-member-pid-reused', 'member-ignores-term'])
def test_nat1_leader_exit_does_not_end_owned_session_custody(tmp_path, monkeypatch, scenario):
    kernel = Kernel(tmp_path, monkeypatch)
    expected, signalled = (0, 'passed'), [('killpg', LEADER, signal.SIGTERM)]
    if scenario == 'member-ignores-term':
        # Timeout path: the leader honours TERM, its group member ignores it.
        kernel.plan['members'] = [group_member(ignore={signal.SIGTERM})]
        code, report = run_execute(kernel, tmp_path, timeout=5)
        expected = (124, 'interrupted')
        signalled = [('killpg', LEADER, signal.SIGTERM), ('killpg', LEADER, signal.SIGKILL)]
    else:
        kernel.plan['leader'] = {'exit_at': 1.0}
        kernel.plan['members'] = {'control': [], 'group-member': [group_member()]}.get(
            scenario, [moved_member()])
        if scenario == 'control':
            signalled = []
        elif scenario == 'moved-member':
            signalled = signalled + [('pidfd_send_signal', 4102, signal.SIGTERM)]
        elif scenario == 'moved-member-pid-reused':
            def reuse():   # the member exits and an unrelated session leader takes its pid
                kernel.exit(kernel.table[4102], os.CLD_EXITED, 0)
                kernel.add(Proc(4102, 1, 4102, 4102, 'unrelated'))
            kernel.hooks['pidfd-open'] = reuse
        code, report = run_execute(kernel, tmp_path)
    live = kernel.live_owned()
    assert not live, f'NAT-1: owned session member(s) {live} outlived cleanup'
    assert (code, report['status']) == expected
    assert kernel.signalled() == signalled
    reaped = kernel.events.index(('reaped', LEADER))
    assert all(kernel.events.index(event) < reaped for event in kernel.signalled()), \
        'NAT-1: a signal relied on the leader pid after it was reaped'
    if scenario == 'moved-member-pid-reused':
        assert kernel.table[4102].received == [], 'NAT-1: signalled a reused pid'
    assert kernel.open_pidfds() == []


def test_nat1_escaped_session_is_never_signalled_or_claimed_closed(tmp_path, monkeypatch):
    kernel = Kernel(tmp_path, monkeypatch)
    kernel.plan['leader'] = {'exit_at': 1.0}
    kernel.plan['members'] = [group_member(), escaped()]
    code, report = run_execute(kernel, tmp_path)
    assert (code, report['status']) == (0, 'passed')
    assert kernel.table[4103].status is None and kernel.table[4103].received == [], \
        'NAT-1: a descendant outside the owned session was signalled'
    custody = report.get('custody') or {}
    assert custody.get('pid') == LEADER and custody.get('state') == 'settled'
    assert custody.get('leader_returncode') == 0
    assert custody.get('signals_sent') == [
        {'signal': 'SIGTERM', 'leader_exited': True, 'members': [4101]}]
    assert 'not observable' in custody.get('escaped_sessions', ''), \
        'NAT-1: report does not disclose that escaped sessions are outside custody'
    assert custody.get('lease') == 'not held by this runner'


@pytest.mark.parametrize('cause', ['inherited-sigchld-ignore', 'consumed-elsewhere-pid-reused'])
def test_nat1_uncertain_leader_status_is_never_inferred_or_resignalled(tmp_path, monkeypatch, cause):
    kernel = Kernel(tmp_path, monkeypatch,
                    sigchld=signal.SIG_IGN if cause == 'inherited-sigchld-ignore' else signal.SIG_DFL)
    kernel.plan['leader'] = {'exit_at': 1.0, 'code': 3}
    if cause == 'consumed-elsewhere-pid-reused':
        kernel.consumed_elsewhere = kernel.reuse_pid = True
    code, report = run_execute(kernel, tmp_path)
    if cause == 'inherited-sigchld-ignore':
        assert (code, report['status']) == (3, 'failed'), \
            f'NAT-1: inherited SIGCHLD=SIG_IGN turned exit 3 into {(code, report["status"])}'
        assert kernel.popen_calls[0]['handlers'][signal.SIGCHLD] == signal.SIG_DFL
        assert kernel.handlers == kernel.original
    else:
        assert (code, report['status']) == (1, 'cleanup-failed'), \
            f'NAT-1: uncertain leader status published as {(code, report["status"])}'
        assert kernel.signalled() == [], 'NAT-1: signalled a pid after its status left custody'
        assert kernel.table[LEADER].received == []
        assert report['custody']['state'] == 'uncertain'


@pytest.mark.parametrize('case', ['term', 'kill', 'refused', 'survives-kill'])
def test_nat1_timeout_term_kill_and_refusal_are_independent(tmp_path, monkeypatch, case):
    kernel = Kernel(tmp_path, monkeypatch)
    kernel.plan['leader'] = {'kill': {'ignore': {signal.SIGTERM}},
                             'survives-kill': {'unkillable': True}}.get(case, {})
    if case == 'refused':
        kernel.refusal = PermissionError(errno.EPERM, 'Operation not permitted')
    code, report = run_execute(kernel, tmp_path, timeout=5)
    sent = [event[2] for event in kernel.signalled()]
    if case in ('term', 'kill'):
        assert (code, report['status']) == (124, 'interrupted')
        assert sent == ([signal.SIGTERM] if case == 'term' else [signal.SIGTERM, signal.SIGKILL])
        assert ('reaped', LEADER) in kernel.events and kernel.live_owned() == []
    else:
        assert (code, report['status']) == (1, 'cleanup-failed')
        assert sent == ([signal.SIGTERM] if case == 'refused' else [signal.SIGTERM, signal.SIGKILL])
        assert ('reaped', LEADER) not in kernel.events, 'reaped without settled custody'
        assert kernel.live_owned() == [LEADER]


@pytest.mark.parametrize('moment,sig', [('cleanup', signal.SIGTERM), ('cleanup', signal.SIGINT),
                                        ('cleanup', signal.SIGHUP),
                                        ('final-publication', signal.SIGTERM)])
def test_nat3_cancellation_during_cleanup_and_publication_is_deferred(tmp_path, monkeypatch,
                                                                     moment, sig):
    kernel = Kernel(tmp_path, monkeypatch)
    if moment == 'cleanup':
        kernel.plan['leader'] = {'term_delay': 2.0}   # honours TERM after 2 s
        kernel.at(5.5, lambda: kernel.signal_parent(sig))   # inside the TERM grace period
        timeout = 5
    else:
        kernel.plan['leader'] = {'exit_at': 1.0}
        kernel.hooks['final-publication'] = lambda: kernel.signal_parent(sig)
        timeout = 30
    try:
        code, report = run_execute(kernel, tmp_path, timeout=timeout)
    except (ParentKilled, KeyboardInterrupt) as death:
        status = json.loads((tmp_path / 'run/report.json').read_text())['status']
        pytest.fail(f'NAT-3: {type(death).__name__} during {moment}; report left {status!r}')
    assert ('parent-signal', sig) in kernel.events
    assert kernel.live_owned() == [] and ('reaped', LEADER) in kernel.events
    if moment == 'cleanup':
        assert (code, report['status']) == (124, 'interrupted')
        assert report['custody']['signals_received'] == [sig]
    else:
        assert (code, report['status']) == (0, 'passed')
    final = max(i for i, event in enumerate(kernel.events)
                if event[0] == 'publish' and event[1] != 'running')
    restores = [i for i, event in enumerate(kernel.events) if event[0] == 'disposition'][-4:]
    assert min(restores) > final, 'NAT-3: dispositions restored before final publication'
    assert kernel.handlers == kernel.original


@pytest.mark.parametrize('case', ['benchmark-settled', 'benchmark-cleanup-failed', 'qualify'])
def test_nat4_lease_lifetime_follows_owned_child(tmp_path, monkeypatch, runnable_benchmark, case):
    args, attempts, root = runnable_benchmark
    healthy_storage(monkeypatch)
    kernel = Kernel(tmp_path, monkeypatch)
    lease = kernel.track_lease(monkeypatch)
    if case == 'qualify':
        (root / 'models').mkdir()
        args = ['qualify', '--config', str(root / 'settings.json'), '--execute']
        kernel.plan['leader'] = {'exit_at': 1.0}
    elif case == 'benchmark-settled':
        kernel.plan['leader'] = {'exit_at': 1.0}
        kernel.native_rows = NATIVE_ROWS
    else:
        kernel.plan['leader'] = {'unkillable': True}
        args = args + ['--timeout', '5']
    code = m.main(args)
    shared = tuple(kernel.popen_calls[0].get('pass_fds') or ())
    assert lease['ops'] == [m.fcntl.LOCK_EX | m.fcntl.LOCK_NB], 'lease mode changed or unlocked'
    closed = kernel.events.index(('lease-closed-by-parent',))
    if case == 'qualify':
        # Qualification reacquires the lease itself; this runner holds none.
        assert code == 0 and shared == ()
        assert closed < kernel.events.index(('popen',))
        return
    assert shared == (lease['fd'],), 'NAT-4: lease descriptor not shared with the child'
    final = max(i for i, event in enumerate(kernel.events)
                if event[0] == 'publish' and event[1] != 'running')
    assert closed > final, 'NAT-4: lease dropped before the final report'
    if case == 'benchmark-settled':
        assert code == 0 and closed > kernel.events.index(('reaped', LEADER))
        assert not kernel.lease_held()
    else:
        assert code == 1 and kernel.live_owned() == [LEADER]
        assert kernel.lease_held(), 'NAT-4: lease released while the owned child is alive'


def test_nat4_parent_death_after_adoption_leaves_owned_identity_and_held_lease(
        tmp_path, monkeypatch, runnable_benchmark):
    args, attempts, root = runnable_benchmark
    healthy_storage(monkeypatch)
    kernel = Kernel(tmp_path, monkeypatch)
    lease = kernel.track_lease(monkeypatch)
    kernel.plan['leader'] = {'exit_at': 1.0}
    kernel.native_rows = NATIVE_ROWS
    seen = {}

    def snapshot():
        # What a SIGKILL of this runner here would leave behind: the on-disk
        # report and what the child inherited. Real parent death, and that no
        # finally block runs then, is OS behaviour this model cannot prove.
        run = next((root / 'logs/model-runs').iterdir())
        seen['report'] = json.loads((run / 'report.json').read_text())
        seen['shared'] = tuple(kernel.popen_calls[0].get('pass_fds') or ())

    kernel.hooks['first-query'] = snapshot
    assert m.main(args) == 0
    custody = seen['report'].get('custody') or {}
    assert seen['report']['status'] == 'running'
    assert (custody.get('pid'), custody.get('state')) == (LEADER, 'running'), \
        'NAT-4: a dead parent would leave no owned child identity'
    assert seen['shared'] == (lease['fd'],), \
        'NAT-4: parent death would release the lease while the child runs'


@pytest.mark.parametrize('failure', ['kernel-identity', 'initial-report', 'dispatch-record',
                                     'final-report'])
def test_nat5_failures_after_the_run_directory_are_not_preflight_refusals(
        tmp_path, monkeypatch, runnable_benchmark, capsys, failure):
    args, attempts, root = runnable_benchmark
    healthy_storage(monkeypatch)
    kernel = Kernel(tmp_path, monkeypatch)
    kernel.track_lease(monkeypatch)
    kernel.plan['leader'] = {'exit_at': 1.0}
    kernel.native_rows = NATIVE_ROWS
    if failure == 'kernel-identity':
        read_text = Path.read_text

        def unreadable(path, *a, **kw):
            if str(path) == '/proc/version':
                raise OSError(errno.EIO, 'synthetic kernel identity read failure')
            return read_text(path, *a, **kw)

        monkeypatch.setattr(Path, 'read_text', unreadable)
    else:
        write_text = Path.write_text
        writes = []

        def report_write(path, data, *a, **kw):
            if path.name in ('report.json', 'report.tmp'):
                writes.append(json.loads(data)['status'])
                if {'initial-report': True, 'dispatch-record': len(writes) > 1,
                        'final-report': writes[-1] != 'running'}[failure]:
                    raise OSError(errno.EIO, 'synthetic report write failure')
            return write_text(path, data, *a, **kw)

        monkeypatch.setattr(Path, 'write_text', report_write)
    code = m.main(args)
    err = capsys.readouterr().err
    assert code != 75 and 'Preflight refused' not in err, \
        f'NAT-5: {failure} after the run directory was reported as preflight refusal {code}'
    assert code == 1
    runs = list((root / 'logs/model-runs').iterdir())
    assert len(runs) == 1
    if failure in ('kernel-identity', 'initial-report'):
        assert kernel.popen_calls == [], 'dispatched after a failure before dispatch'
        assert 'No child was dispatched' in err
        report = (json.loads((runs[0] / 'report.json').read_text())
                  if failure == 'kernel-identity' else stderr_report(err))
        assert (report['status'], report['custody']['state']) == ('failed', 'not-dispatched')
    else:
        assert len(kernel.popen_calls) == 1
        assert kernel.live_owned() == [] and ('reaped', LEADER) in kernel.events
        report = stderr_report(err)   # lifecycle truth survives the unwritable report
        assert report['custody']['state'] == 'settled' and 'publication_error' in report
        assert report['status'] == ('passed' if failure == 'final-report' else 'failed')


def test_repeat_and_late_cleanup_never_signal_a_settled_or_uncertain_leader(tmp_path, monkeypatch):
    kernel = Kernel(tmp_path, monkeypatch)
    kernel.plan['members'] = [group_member()]
    child = kernel.popen(['synthetic-native'])
    custody = {'state': 'running', 'pid': LEADER}
    m.stop_child(child, custody)
    assert custody['state'] == 'settled' and child.returncode == -signal.SIGTERM
    seen = len(kernel.events)
    m.stop_child(child, custody)
    for state in ('uncertain', 'unresolved'):
        with pytest.raises(RuntimeError, match=state):
            m.stop_child(child, {'state': state, 'pid': LEADER})
    assert kernel.events[seen:] == [], 'repeat or late cleanup signalled or waited again'


def test_src1_relative_symlinked_binary_is_resolved_once(tmp_path, monkeypatch, runnable_benchmark):
    args, attempts, root = runnable_benchmark
    real = root / 'private-build/bin/llama-bench'
    real.parent.mkdir(parents=True)
    real.write_text('synthetic executable identity only\n')
    real.chmod(0o700)
    (root / 'bench-link').symlink_to(real)
    monkeypatch.chdir(root)
    a = m.parser().parse_args(['benchmark', '--model', 'fixture.gguf', '--bench-binary', 'bench-link'])
    command = m.command(a)
    assert command[0] == str(real) and command[2] == str(root / 'fixture.gguf')
    checked = []
    access = os.access
    monkeypatch.setattr(m.os, 'access', lambda path, mode: checked.append(path) or access(path, mode))
    monkeypatch.setattr(m, 'storage_preflight', lambda paths: None)
    m.preflight(a)
    assert checked == [real] and attempts == []


def test_src3_real_config_keeps_gpu_defaults_and_cli_cpu_override(tmp_path):
    model = tmp_path / 'fixture.gguf'
    gpu = m.command(m.configured_args(['benchmark', '--model', str(model)]))
    assert gpu[gpu.index('-dev') + 1] == 'ROCm0/ROCm1/ROCm2'
    assert gpu[gpu.index('-ts') + 1] == '6/6/6' and gpu[gpu.index('-ngl') + 1] == '999'
    cpu_args = m.configured_args(['benchmark', '--model', str(model),
                                  '--devices', 'none', '--gpu-layers', '0'])
    cpu = m.command(cpu_args)
    assert cpu[cpu.index('-dev') + 1] == 'none' and '-ts' not in cpu
    assert cpu[cpu.index('-ngl') + 1] == '0'
    assert (cpu_args.timeout, cpu_args.tensor_split) == (3600, '6/6/6')
    with pytest.raises(ValueError, match='CPU-only'):
        m.command(m.configured_args(['benchmark', '--model', str(model), '--devices', 'none']))
