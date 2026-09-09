"""The serialized runners must never block their own exit on systemd.

Each GPU-exclusive gate stops ``llama-router.service`` for the duration of its
run and restarts it on the way out. The restart used to be synchronous and ran
from the shell's EXIT trap. That trap is reached while systemd is executing the
runner unit's *own* stop job, and every one of these units is ordered
``After=llama-router.service``, so the manager puts the qualification stop ahead
of the router start in the same transaction. The blocking ``systemctl start``
then waited for a job that was waiting for the trap to return.

Observed on 2026-09-09: stopping the legacy supervisor unit left the trap
(pid 116389) inside ``systemctl --user start llama-router.service`` while
``systemctl --user list-jobs`` showed the router start queued behind the
qualification stop, until the unit's 90s stop timeout expired and SIGKILL ended
it. A gate that dies that way loses its own exit status.

These tests execute the real ``restore_router``/``on_exit`` text lifted out of
each runner against a fake ``systemctl`` that reproduces the manager's
behaviour: a start without ``--no-block`` never returns. No systemd user
manager, no unit and no service is touched. ``test_fake_systemctl_reproduces...``
is the negative control that proves the fake can still catch a regression.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "tts": ROOT / "verification/tts-local/run_serialized.sh",
    "media": ROOT / "verification/generative-media/run_functional_serialized.sh",
    "computer-use": ROOT / "verification/computer-use-grounding/run_when_idle.sh",
}
# Preamble per runner: everything the extracted functions read, and stubs for
# the collaborators that are not under test here.
PRELUDE = {
    "tts": "",
    "media": "pid=\n",
    "computer-use": (
        'ROUTER=llama-router.service\n'
        'python_rc=0\nsignal_seen=\nARTIFACT=/nonexistent\n'
        'STARTED_AT=start\nartifact_state=missing\nartifact_pass=unknown\n'
        'log() { printf "%s\\n" "$*"; }\n'
        'write_result() { printf "write_result status=%s\\n" "$1"; }\n'),
}
FUNCTIONS = {"tts": ("restore_router", "on_exit"),
             "media": ("restore_router", "stop_comfy", "on_exit"),
             "computer-use": ("restore_router", "on_exit")}
# The fake manager. A start without --no-block is exactly the call that could
# not be dispatched during the runner unit's own stop, so it never returns.
FAKE_SYSTEMCTL = """#!/usr/bin/env bash
printf '%s\\n' "$*" >>"$FAKE_SYSTEMCTL_LOG"
if [ -n "${FAKE_SYSTEMCTL_FAIL:-}" ]; then exit 1; fi
for arg in "$@"; do
  if [ "$arg" = "--no-block" ]; then exit 0; fi
done
case " $* " in
  *" start "*|*" restart "*) exec sleep "${FAKE_SYSTEMCTL_BLOCK_SECONDS:-120}" ;;
esac
exit 0
"""
EXIT_STATUS = 42
RUN_TIMEOUT = 15
# The negative control is *expected* to hang, so it gets a short leash.
BLOCKED_TIMEOUT = 3


def extract(path: Path, name: str) -> str:
    """Lift one top-level shell function definition out of a runner, verbatim."""
    text = path.read_text()
    match = re.search(rf"^{re.escape(name)}\(\) \{{$.*?^\}}$", text, re.M | re.S)
    assert match, f"{path.name} defines no top-level {name}()"
    return match.group(0)


def harness(tmp_path: Path, runner: str) -> Path:
    body = "\n".join(extract(RUNNERS[runner], name) for name in FUNCTIONS[runner])
    script = tmp_path / f"{runner}-harness.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n"
        + PRELUDE[runner]
        + "router_was_active=1\nrouter_restored=0\nrouter_restore_queued=0\n"
        + body
        + f"\ntrap on_exit EXIT\nexit {EXIT_STATUS}\n")
    return script


def run(script: Path, tmp_path: Path, timeout: int = RUN_TIMEOUT,
        **env) -> subprocess.CompletedProcess:
    log = tmp_path / "systemctl-calls.log"
    log.touch()
    fake_dir = tmp_path / "bin"
    fake_dir.mkdir(exist_ok=True)
    fake = fake_dir / "systemctl"
    fake.write_text(FAKE_SYSTEMCTL)
    fake.chmod(0o755)
    environment = dict(os.environ,
                       PATH=f"{fake_dir}:/usr/bin:/bin",
                       FAKE_SYSTEMCTL_LOG=str(log), **env)
    child = subprocess.Popen(["/usr/bin/bash", str(script)], env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)
    try:
        out, _ = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the group: the fake manager's blocking start is a separate child.
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        pytest.fail(f"{script.name} did not exit within {timeout}s: the router "
                    f"restore blocked on systemd. systemctl calls: {log.read_text()!r}")
    result = subprocess.CompletedProcess(child.args, child.returncode, out, "")
    result.systemctl_calls = log.read_text().splitlines()  # type: ignore[attr-defined]
    return result


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_router_restore_never_waits_for_the_manager(tmp_path, runner):
    result = run(harness(tmp_path, runner), tmp_path)
    assert result.systemctl_calls, "the runner never asked for the router back"
    for call in result.systemctl_calls:
        assert "--no-block" in call, f"{runner} blocked on the manager: {call!r}"
        assert "start" in call and "llama-router.service" in call


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_exit_trap_preserves_the_gate_status(tmp_path, runner):
    result = run(harness(tmp_path, runner), tmp_path)
    assert result.returncode == EXIT_STATUS, result.stdout


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_restore_reports_a_request_and_never_readiness(tmp_path, runner):
    lowered = run(harness(tmp_path, runner), tmp_path).stdout.lower()
    assert "queue" in lowered, lowered
    for claim in ("router is ready", "router ready", "router restored",
                  "router is up", "router active", "router is running"):
        assert claim not in lowered, f"{runner} claimed unobserved readiness: {lowered!r}"


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_refused_restore_is_reported_and_does_not_change_the_status(tmp_path, runner):
    result = run(harness(tmp_path, runner), tmp_path, FAKE_SYSTEMCTL_FAIL="1")
    assert result.returncode == EXIT_STATUS, result.stdout
    assert "warning" in result.stdout.lower(), result.stdout


def test_the_router_is_left_alone_when_this_run_did_not_stop_it(tmp_path):
    script = harness(tmp_path, "computer-use")
    script.write_text(script.read_text().replace("router_was_active=1",
                                                 "router_was_active=0"))
    result = run(script, tmp_path)
    assert result.systemctl_calls == []
    assert result.returncode == EXIT_STATUS


def test_fake_systemctl_reproduces_the_blocking_start(tmp_path):
    """Negative control: without --no-block the harness must hang, or the
    tests above prove nothing."""
    script = tmp_path / "blocking-harness.sh"
    script.write_text("#!/usr/bin/env bash\nset -uo pipefail\n"
                      "restore() { systemctl --user start llama-router.service; }\n"
                      f"trap restore EXIT\nexit {EXIT_STATUS}\n")
    with pytest.raises(pytest.fail.Exception, match="did not exit"):
        run(script, tmp_path, timeout=BLOCKED_TIMEOUT, FAKE_SYSTEMCTL_BLOCK_SECONDS="120")


def test_a_signalled_tts_runner_keeps_its_signal_status(tmp_path):
    """The TTS runner maps SIGTERM to 143 and exits through the same trap."""
    body = "\n".join(extract(RUNNERS["tts"], name)
                     for name in ("restore_router", "on_exit", "interrupted"))
    script = tmp_path / "tts-signal-harness.sh"
    script.write_text("#!/usr/bin/env bash\nset -uo pipefail\n"
                      "router_was_active=1\n" + body
                      + "\ntrap on_exit EXIT\ntrap 'interrupted TERM 143' TERM\n"
                        "kill -TERM $$\nsleep 5\n")
    result = run(script, tmp_path)
    assert result.returncode == 143, result.stdout
    assert any("--no-block" in call for call in result.systemctl_calls)

