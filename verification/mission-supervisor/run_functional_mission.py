#!/usr/bin/env python3
"""Reboot-resumable, fail-closed functional qualification supervisor.

This mission deliberately records no token rate or comparative benchmark data.
Independent gates continue after a functional failure so one capability cannot
hide the status of every later one; the aggregate mission still fails closed.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path("/home/typhoon/git/frankenstein-llm")
HERE = ROOT / "verification" / "mission-supervisor"
STATE = HERE / "mission-state.json"
LOG = HERE / "mission.log"
LOCK = HERE / "mission.lock"
FOUNDATION = ROOT / "verification" / "local-coverage-foundation"
MIN_AVAILABLE = 32 << 30
MAX_LOAD = 6.0
POLL_SECONDS = 30


def quiet_timeout(raw: str | None) -> int:
    """Resolve the deadline for the quiet-host wait.

    ``wait_for_inputs`` is deliberately unbounded: it waits on ~136 GB of
    downloads that legitimately take days, and giving up on them would be wrong.
    The quiet-host wait is different. Its conditions are all things that should
    clear in minutes, so a wait that never ends means something is stuck -- a
    sidecar left resident by a failed gate used to do exactly that -- and an
    unbounded wait turns that into a unit which looks busy forever. Fail loudly
    instead, and fail loudly on an unusable override rather than silently
    substituting a default for it.
    """
    if raw is None or raw == "":
        return 6 * 3600
    try:
        seconds = int(raw)
    except ValueError as error:
        raise ValueError(
            f"HERMES_MISSION_QUIET_TIMEOUT must be a positive integer, got {raw!r}"
        ) from error
    if seconds <= 0:
        raise ValueError(
            f"HERMES_MISSION_QUIET_TIMEOUT must be a positive integer, got {seconds}")
    return seconds


QUIET_WAIT_TIMEOUT = quiet_timeout(os.environ.get("HERMES_MISSION_QUIET_TIMEOUT"))
UPSTREAM = (
    (FOUNDATION / "download-state.json", FOUNDATION / "downloads-complete.ok", "72134030730"),
    (FOUNDATION / "download-state-phase2.json", FOUNDATION / "downloads-phase2-complete.ok", "33184695056"),
    (FOUNDATION / "download-state-phase3.json", FOUNDATION / "downloads-phase3-complete.ok", "30987169046"),
)
STEPS = (
    ("router-reload-presets", ["/usr/bin/systemctl", "--user", "restart", "llama-router.service"]),
    ("router-models", [sys.executable, str(ROOT / "verification/router-functional/gate_router_models.py")]),
    ("embeddings", [sys.executable, str(FOUNDATION / "validators/gate_embeddings.py")]),
    ("reranker", [sys.executable, str(FOUNDATION / "validators/gate_reranker.py")]),
    ("fim", [sys.executable, str(FOUNDATION / "validators/gate_fim.py")]),
    ("computer-use-grounding", ["/usr/bin/bash", str(ROOT / "verification/computer-use-grounding/run_when_idle.sh")]),
    ("tts-asr-roundtrip", ["/usr/bin/bash", str(ROOT / "verification/tts-local/run_serialized.sh")]),
    ("repository-agent", ["/usr/bin/bash", str(ROOT / "verification/repository-agent/run_serialized.sh")]),
    ("generative-media-functional", ["/usr/bin/bash", str(ROOT / "verification/generative-media/run_functional_serialized.sh")]),
)
child: subprocess.Popen | None = None
stop_signal: int | None = None


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(value: dict) -> None:
    """Publish mission state durably; a resumed run reads it to skip passed steps."""
    temp = STATE.with_suffix(f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, STATE)
    directory = os.open(str(STATE.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)


def load_state() -> dict:
    try:
        value = json.loads(STATE.read_text(encoding="utf-8"))
        if value.get("schema") == "frankenstein-functional-mission/1":
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "schema": "frankenstein-functional-mission/1",
        "benchmarking_performed": False,
        "throughput_measured": False,
        "steps": {},
    }


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def command_of(proc: Path) -> tuple[str, str]:
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return "", ""
    try:
        cwd = str((proc / "cwd").resolve())
    except OSError:
        cwd = ""
    return command, cwd


def conflicts() -> list[dict[str, object]]:
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        command, cwd = command_of(proc)
        if not command:
            continue
        kernel = ("makepkg" in command or "/linux-tkg" in cwd) and proc.name != str(os.getpid())
        build = any(token in command for token in ("cmake --build", "ninja ", "cargo build", "cargo test"))
        transfer = "download_queue.py" in command
        inference = (
            ("llama-server" in command and "--models-preset" not in command)
            or ("tools/ComfyUI/main.py" in command)
            or ("gate_computer_use.py" in command)
            or ("gate_tts.py" in command)
        )
        if kernel or build or transfer or inference:
            found.append({"pid": int(proc.name), "command": command[:500], "cwd": cwd[:300]})
    return found


def phase_status(state_path: Path, stamp_path: Path, expected: str) -> tuple[str, str]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return "waiting", f"{state_path.name}: {type(error).__name__}"
    status = state.get("status")
    if status == "failed":
        return "failed", f"{state_path.name}: {state.get('error', 'unknown error')}"
    if status != "complete":
        return "waiting", f"{state_path.name}: {status}"
    try:
        stamp = stamp_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "waiting", f"{stamp_path.name}: missing"
    if stamp != expected:
        return "failed", f"{stamp_path.name}: expected {expected}, got {stamp!r}"
    return "ready", state_path.name


def wait_for_inputs(state: dict) -> None:
    last = None
    while True:
        statuses = [phase_status(*phase) for phase in UPSTREAM]
        failure = next((detail for status, detail in statuses if status == "failed"), None)
        if failure:
            state.update({"status": "failed", "error": failure, "updated_at": now()})
            atomic_json(state)
            raise RuntimeError(failure)
        if all(status == "ready" for status, _ in statuses):
            return
        detail = "; ".join(text for _, text in statuses)
        if detail != last:
            log(f"waiting for artifact queues: {detail}")
            last = detail
        state.update({"status": "waiting-artifacts", "updated_at": now()})
        atomic_json(state)
        time.sleep(POLL_SECONDS)


def wait_for_quiet(state: dict, timeout: int = QUIET_WAIT_TIMEOUT) -> None:
    quiet = 0
    last = None
    deadline = time.monotonic() + timeout
    while quiet < 2:
        active = conflicts()
        available = mem_available()
        load = os.getloadavg()[0]
        reasons = []
        if active:
            reasons.append(f"conflicts={active}")
        if available < MIN_AVAILABLE:
            reasons.append(f"MemAvailable={available}")
        if load > MAX_LOAD:
            reasons.append(f"load1={load:.2f}")
        if reasons:
            quiet = 0
            detail = "; ".join(reasons)
            if detail != last:
                log(f"waiting for safe host: {detail}")
                last = detail
        else:
            quiet += 1
            log(f"safe-host quiet poll {quiet}/2 MemAvailable={available} load1={load:.2f}")
        state.update({"status": "waiting-safe-host", "updated_at": now()})
        atomic_json(state)
        if quiet < 2:
            # Report the blockers observed in this poll, not "unknown": the whole
            # point of bounding the wait is to say what the host was busy with.
            if time.monotonic() >= deadline:
                state.update({"status": "failed", "updated_at": now()})
                atomic_json(state)
                raise RuntimeError(
                    f"host did not become quiet within {timeout}s; still blocked by: "
                    f"{'; '.join(reasons) or 'two consecutive quiet polls not yet observed'}")
            time.sleep(POLL_SECONDS)


def on_signal(signum: int, _frame) -> None:
    global stop_signal
    stop_signal = signum
    log(f"received signal {signum}")
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        return
    state = load_state()
    state.update({"status": "interrupted", "signal": signum, "updated_at": now()})
    atomic_json(state)
    raise SystemExit(128 + signum)


def run_step(name: str, command: list[str], state: dict) -> int:
    global child
    wait_for_quiet(state)
    step = {
        "command": command,
        "started_at": now(),
        "status": "running",
        "benchmarking_performed": False,
        "throughput_measured": False,
    }
    state["steps"][name] = step
    state.update({"status": "running", "current_step": name, "updated_at": now()})
    atomic_json(state)
    step_log = HERE / f"{name}.log"
    log(f"step start name={name}")
    with step_log.open("ab") as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        rc = child.wait()
    child = None
    step.update({"finished_at": now(), "exit_code": rc, "status": "passed" if rc == 0 else "failed"})
    state["updated_at"] = now()
    atomic_json(state)
    log(f"step end name={name} rc={rc}")
    return rc


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another mission supervisor owns the lock")
        return 75
    state = load_state()
    for stale in ("signal", "error", "failed_step", "failed_steps", "exit_code",
                  "interrupted_step", "step_exit_code", "step_status"):
        state.pop(stale, None)
    state.update({
        "status": "starting",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "kernel_release": os.uname().release,
        "kernel_build_signature": Path("/proc/version").read_text().strip(),
        "updated_at": now(),
    })
    atomic_json(state)
    wait_for_inputs(state)
    failures = []
    for name, command in STEPS:
        if state["steps"].get(name, {}).get("status") == "passed":
            log(f"step already passed; skipping name={name}")
            continue
        rc = run_step(name, command, state)
        # Order matters. A SIGTERM to this supervisor is forwarded to the running
        # step, which then exits non-zero -- so testing rc first recorded every
        # operator stop as "step X failed" and lost the fact that the mission was
        # interrupted at all. The signal is the more specific explanation, and a
        # non-zero exit under it is the stop rather than a verdict, so the step is
        # marked interrupted and the resume runs it again. A step that still
        # exited 0 really did pass; demoting that would discard a completed gate
        # and repeat hours of GPU work on the next boot.
        if stop_signal is not None:
            if rc != 0:
                state["steps"][name]["status"] = "interrupted"
            state.update({"status": "interrupted", "signal": stop_signal,
                          "interrupted_step": name, "step_exit_code": rc,
                          "step_status": state["steps"][name]["status"],
                          "updated_at": now()})
            atomic_json(state)
            return 128 + stop_signal
        if rc != 0:
            failures.append({"name": name, "exit_code": rc})
            state.update({"status": "running-with-failures", "failed_steps": failures,
                          "updated_at": now()})
            atomic_json(state)
    if failures:
        state.update({
            "status": "functional-foundation-incomplete",
            "current_step": None,
            "failed_steps": failures,
            "exit_code": 1,
            "remaining": [failure["name"] for failure in failures] + [
                "build bounded end-to-end computer control only after grounding passes",
            ],
            "updated_at": now(),
        })
        atomic_json(state)
        log(f"functional foundation incomplete; failed steps={failures}")
        return 1
    state.update({
        "status": "functional-foundation-complete",
        "current_step": None,
        "remaining": [
            "build bounded end-to-end computer control only after grounding passes",
        ],
        "updated_at": now(),
    })
    atomic_json(state)
    log("functional foundation complete; bounded computer-control integration remains")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - durable failure evidence is required
        HERE.mkdir(parents=True, exist_ok=True)
        state = load_state()
        state.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "updated_at": now()})
        atomic_json(state)
        log(f"fatal {type(error).__name__}: {error}")
        raise SystemExit(1)
