#!/usr/bin/env python3
"""Reboot-resumable, fail-closed functional qualification supervisor.

This mission deliberately records no token rate or comparative benchmark data.
The candidate-policy gate is a blocking prerequisite: no model-backed gate may
run unless it passes. Independent functional gates then continue after a failure
so one capability cannot hide the status of every later one; the aggregate
mission still fails closed.
"""
from __future__ import annotations

import fcntl
import hashlib
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
    (FOUNDATION / "download-state-phase4.json", FOUNDATION / "downloads-phase4-complete.ok", "98653778996"),
)
STEPS = (
    ("candidate-policy", [sys.executable, str(
        ROOT / "verification/candidate-qualification/gate_candidate_policy.py")]),
    ("wemm-embeddings", ["/usr/bin/bash", str(
        ROOT / "verification/candidate-qualification/run_wemm.sh")]),
    ("router-reload-presets", ["/usr/bin/systemctl", "--user", "restart", "llama-router.service"]),
    ("router-models", [sys.executable, str(ROOT / "verification/router-functional/gate_router_models.py")]),
    ("embeddings", [sys.executable, str(FOUNDATION / "validators/gate_embeddings.py")]),
    ("reranker", [sys.executable, str(FOUNDATION / "validators/gate_reranker.py")]),
    ("fim", [sys.executable, str(FOUNDATION / "validators/gate_fim.py")]),
    ("asr", [sys.executable, str(FOUNDATION / "validators/gate_asr.py")]),
    ("computer-use-grounding", ["/usr/bin/bash", str(ROOT / "verification/computer-use-grounding/run_when_idle.sh")]),
    ("tts-asr-roundtrip", ["/usr/bin/bash", str(ROOT / "verification/tts-local/run_serialized.sh")]),
    ("repository-agent", ["/usr/bin/bash", str(ROOT / "verification/repository-agent/run_serialized.sh")]),
    ("generative-media-functional", ["/usr/bin/bash", str(ROOT / "verification/generative-media/run_functional_serialized.sh")]),
)
child: subprocess.Popen | None = None
stop_signal: int | None = None


def mission_inputs_fingerprint() -> str:
    """Fingerprint source and promoted artifacts without re-hashing model weights."""
    digest = hashlib.sha256()
    source_paths = ("verification", "llama-models.ini", "scripts")
    for command in (
        ["git", "ls-files", "-s", "--", *source_paths],
        ["git", "diff", "--binary", "HEAD", "--", *source_paths],
    ):
        result = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, timeout=30)
        digest.update(result.stdout)

    # Promotion records are durable; file metadata additionally catches restored
    # or replaced bytes whose queue state was not rewritten.
    for state_path, stamp_path, _expected in UPSTREAM:
        for path in (state_path, stamp_path):
            digest.update(str(path).encode())
            try:
                digest.update(path.read_bytes())
            except OSError:
                # Not promoted yet. wait_for_inputs is what blocks on that; this
                # only has to change when the bytes do, and "absent" is a state
                # it can fingerprint rather than a reason to abort the mission
                # before it can report what it is waiting for.
                digest.update(b"\0absent\0")
    for queue_path in sorted(FOUNDATION.glob("download-queue*.json")):
        digest.update(queue_path.read_bytes())
        document = json.loads(queue_path.read_text(encoding="utf-8"))
        for artifact in document.get("artifacts", []):
            for entry in artifact.get("files", []):
                path = Path(entry["destination"])
                try:
                    stat = path.stat()
                    identity = (str(path), stat.st_size, stat.st_mtime_ns)
                except OSError:
                    identity = (str(path), None, None)
                digest.update(json.dumps(identity, separators=(",", ":")).encode())
    digest.update(json.dumps(STEPS, separators=(",", ":")).encode())
    return digest.hexdigest()


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
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, STATE)
    finally:
        # A write that failed part-way must not leave a partial document behind
        # for the next run's temp file to be confused with.
        temp.unlink(missing_ok=True)
    directory = os.open(str(STATE.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    except OSError:
        # Some filesystems refuse to sync a directory. The rename already
        # happened, so the state on disk is whole either way.
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


# Kernel task names as they appear in /proc/<pid>/comm. The kernel truncates
# that field to 15 characters, so a longer name is written here the way it
# actually arrives: "git-remote-https" is read back as "git-remote-http".
BUILD_COMMANDS = frozenset({
    "make", "gmake", "makepkg", "cmake", "ninja", "samu", "meson", "scons",
    "ccache", "sccache", "cc", "c++", "gcc", "g++", "clang", "clang++",
    "cc1", "cc1plus", "lto1", "collect2", "as", "ar", "ranlib", "strip",
    "objtool", "ld", "ld.bfd", "ld.gold", "ld.lld", "lld", "mold",
    "cargo", "rustc", "go", "nvcc", "cicc", "ptxas", "hipcc",
    "pacman", "paru", "yay", "dkms",
})
TRANSFER_COMMANDS = frozenset({
    "aria2c", "curl", "wget", "axel", "lftp", "rsync", "scp", "sftp",
    "git-lfs", "git-remote-http", "huggingface-cli", "hf",
})
# Anything that loads weights outside the router this mission manages itself.
INFERENCE_COMMANDS = frozenset({
    "ollama", "vllm", "sglang", "koboldcpp", "whisper-cli", "whisper-server",
    "sd-server", "stable-diffusio",
})
# Every llama.cpp binary shares this prefix, and the prefix survives truncation
# where the full names ("llama-perplexity", "llama-quantize") do not.
INFERENCE_PREFIX = "llama-"
# The one inference service the gates call rather than collide with. It is a
# systemd unit, so its cgroup path names it; llama-server run by hand does not.
ROUTER_UNIT = "llama-router.service"
KERNEL_TREE = "/linux-tkg"
DELETED_SUFFIX = " (deleted)"


def printable(raw: bytes | str) -> str:
    """Force kernel-supplied text to valid UTF-8.

    A task name is whatever bytes the process chose for it. Left alone, an
    undecodable one raises where it is least welcome -- not here, but later,
    when the reason the host is busy is written to the UTF-8 mission log.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "surrogateescape")
    return raw.decode("utf-8", "replace")


def process_metadata(proc: Path) -> tuple[str, str, str]:
    """Read conflict metadata without touching another task's address space.

    Linux implements ``/proc/<pid>/cmdline`` through ``access_remote_vm``. A
    task holding its mmap write lock can therefore wedge a supervisor that
    merely tries to inspect it. ``comm`` and ``cgroup`` are kernel metadata and
    ``cwd`` is read with ``readlink``; none of the three reads the target
    process's memory, and nothing here opens ``cmdline`` at all.

    Every read tolerates the task exiting underneath it. An empty command is the
    caller's signal that there was nothing left to classify.
    """
    try:
        command = printable((proc / "comm").read_bytes()).strip()
    except OSError:
        return "", "", ""
    try:
        cwd = printable(os.readlink(proc / "cwd"))
    except OSError:
        cwd = ""
    if cwd.endswith(DELETED_SUFFIX):
        # A cwd whose directory was removed reads back as "<path> (deleted)".
        # Left attached, the suffix hides a build still running in a tree that
        # was deleted out from under it.
        cwd = cwd[: -len(DELETED_SUFFIX)]
    try:
        cgroup = printable((proc / "cgroup").read_bytes())
    except OSError:
        cgroup = ""
    return command, cwd, cgroup


def under(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` itself or something inside it."""
    return bool(path) and (path == root or path.startswith(f"{root}/"))


def conflict_reason(command: str, cwd: str, cgroup: str) -> str | None:
    """Why this process collides with a serialized gate, or None if it does not.

    Classification is from kernel metadata only, so it is deliberately coarse
    and errs towards reporting: a build, a transfer or a second model server is
    named on the evidence of its task name alone, without confirming what it is
    working on. Waiting out a process that turned out to be harmless costs one
    poll; running a GPU gate next to a real one costs the gate.

    The managed router is the single exception, and it is excused by its cgroup
    rather than by its name: llama-server started by hand is still a conflict.
    """
    managed_router = ROUTER_UNIT in cgroup
    if KERNEL_TREE in cwd:
        return "kernel-build"
    if command in BUILD_COMMANDS:
        return "build"
    if command in TRANSFER_COMMANDS:
        return "transfer"
    if command.startswith(INFERENCE_PREFIX) or command in INFERENCE_COMMANDS:
        return None if managed_router else "inference"
    if command.startswith("python") and under(cwd, str(ROOT)):
        # A gate, a downloader or a test run out of this workspace. Which one it
        # is cannot be told from comm, and every one of them is a conflict.
        return None if managed_router else "workspace-python"
    return None


def conflicts(proc_root: Path = Path("/proc")) -> list[dict[str, object]]:
    """Every process whose work would collide with a serialized gate.

    ``proc_root`` is injectable so the classification can be exercised without
    inventing host processes. A proc root that is not there raises instead of
    returning nothing: "no conflicts" and "nowhere to look" are the same empty
    list, and only one of them means the host is quiet.
    """
    if not proc_root.is_dir():
        raise RuntimeError(
            f"process table {proc_root} is not readable; refusing to read an "
            "unreadable host as a quiet one")
    self_pid = os.getpid()
    found: list[dict[str, object]] = []
    pids = sorted(int(entry.name) for entry in proc_root.glob("[0-9]*")
                  if entry.name.isdigit())
    for pid in pids:
        if pid == self_pid:
            continue
        command, cwd, cgroup = process_metadata(proc_root / str(pid))
        if not command:
            # The task exited between listing the table and reading it.
            continue
        reason = conflict_reason(command, cwd, cgroup)
        if reason is not None:
            found.append({"pid": pid, "reason": reason, "command": command[:500],
                          "cwd": cwd[:300], "cgroup": cgroup.strip()[:300]})
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
                failure = (
                    f"host did not become quiet within {timeout}s; still blocked by: "
                    f"{'; '.join(reasons) or 'two consecutive quiet polls not yet observed'}")
                # Record the explanation before raising. The unit dies on this
                # exception and the outer handler may not get to run, so the
                # durable state has to already say why it gave up.
                state.update({"status": "failed", "error": failure, "updated_at": now()})
                atomic_json(state)
                raise RuntimeError(failure)
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
        "input_fingerprint": state["input_fingerprint"],
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
    input_fingerprint = mission_inputs_fingerprint()
    for stale in ("signal", "error", "failed_step", "failed_steps", "exit_code",
                  "interrupted_step", "step_exit_code", "step_status"):
        state.pop(stale, None)
    state.update({
        "status": "starting",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "kernel_release": os.uname().release,
        "kernel_build_signature": Path("/proc/version").read_text().strip(),
        "input_fingerprint": input_fingerprint,
        "updated_at": now(),
    })
    atomic_json(state)
    wait_for_inputs(state)

    policy_name, policy_command = STEPS[0]
    previous = state["steps"].get(policy_name, {})
    if (previous.get("status") == "passed"
            and previous.get("input_fingerprint") == input_fingerprint):
        log(f"step already passed; skipping name={policy_name}")
        policy_rc = 0
    else:
        if previous.get("status") == "passed":
            log(f"passed step inputs changed; rerunning name={policy_name}")
        policy_rc = run_step(policy_name, policy_command, state)
    if stop_signal is not None:
        if policy_rc != 0:
            state["steps"][policy_name]["status"] = "interrupted"
        state.update({"status": "interrupted", "signal": stop_signal,
                      "interrupted_step": policy_name, "step_exit_code": policy_rc,
                      "step_status": state["steps"][policy_name]["status"],
                      "updated_at": now()})
        atomic_json(state)
        return 128 + stop_signal
    if policy_rc != 0:
        for name, command in STEPS[1:]:
            state["steps"][name] = {
                "command": command,
                "input_fingerprint": input_fingerprint,
                "status": "blocked-policy",
                "blocked_by": policy_name,
                "updated_at": now(),
                "benchmarking_performed": False,
                "throughput_measured": False,
            }
        failure = {"name": policy_name, "exit_code": policy_rc}
        state.update({
            "status": "blocked-policy",
            "current_step": None,
            "failed_steps": [failure],
            "exit_code": 1,
            "remaining": [name for name, _command in STEPS[1:]],
            "updated_at": now(),
        })
        atomic_json(state)
        log(f"candidate policy blocked model-backed gates rc={policy_rc}")
        return 1

    failures = []
    for name, command in STEPS[1:]:
        previous = state["steps"].get(name, {})
        if (previous.get("status") == "passed"
                and previous.get("input_fingerprint") == input_fingerprint):
            log(f"step already passed; skipping name={name}")
            continue
        if previous.get("status") == "passed":
            log(f"passed step inputs changed; rerunning name={name}")
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
