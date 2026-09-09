#!/usr/bin/env python3
"""Reboot-resumable, fail-closed functional qualification supervisor.

This qualification deliberately records no token rate or comparative benchmark data.
The candidate-policy gate is a blocking prerequisite: no model-backed gate may
run unless it passes. Independent functional gates then continue after a failure
so one capability cannot hide the status of every later one; the aggregate
qualification still fails closed.
"""
from __future__ import annotations

from collections import Counter
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualification_cache import (Store, step_key, adopt_legacy_step, key_components,
                                 INCONCLUSIVE, EXIT_ADMISSION_REFUSED, EXIT_INCONCLUSIVE)

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "verification" / "qualification-supervisor"
STATE = HERE / "qualification-state.json"
LOG = HERE / "qualification.log"
LOCK = HERE / "qualification.lock"
FOUNDATION = ROOT / "verification" / "local-coverage-foundation"
TTS_PYTHON = ROOT / "venvs" / "tts" / "bin" / "python"
# Qwen3-ASR-1.7B declares Qwen3ASRForConditionalGeneration and model_type
# qwen3_asr, which the tts venv's transformers 4.57 does not register at all:
# gate_asr.py imports AutoModelForMultimodalLM from it and dies on ImportError
# before it reaches a GPU. venvs/asr carries transformers 5.16 and does have
# both. The two environments are kept apart because the TTS stack pins the
# older transformers, so the fix is to run the ASR gate in the venv built for
# it rather than to move either pin.
ASR_PYTHON = ROOT / "venvs" / "asr" / "bin" / "python"
SCRATCH = ROOT / "verification" / "tmp"
MIN_AVAILABLE = 32 << 30
POLL_SECONDS = 30
BLOCKER_LOG_REPEAT_SECONDS = 600
CONFLICT_SAMPLE_LIMIT = 8


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
            f"HERMES_QUALIFICATION_QUIET_TIMEOUT must be a positive integer, got {raw!r}"
        ) from error
    if seconds <= 0:
        raise ValueError(
            f"HERMES_QUALIFICATION_QUIET_TIMEOUT must be a positive integer, got {seconds}")
    return seconds


QUIET_WAIT_TIMEOUT = quiet_timeout(os.environ.get("HERMES_QUALIFICATION_QUIET_TIMEOUT"))
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
    ("asr", [str(ASR_PYTHON), str(FOUNDATION / "validators/gate_asr.py")]),
    ("computer-use-grounding", ["/usr/bin/bash", str(ROOT / "verification/computer-use-grounding/run_when_idle.sh")]),
    ("tts-asr-roundtrip", ["/usr/bin/bash", str(ROOT / "verification/tts-local/run_serialized.sh")]),
    ("repository-agent", ["/usr/bin/bash", str(ROOT / "verification/repository-agent/run_serialized.sh")]),
    ("generative-media-functional", ["/usr/bin/bash", str(ROOT / "verification/generative-media/run_functional_serialized.sh")]),
)
child: subprocess.Popen | None = None
stop_signal: int | None = None


# Re-verifying files that already exist rewrites these on every boot: they say
# when the downloader ran and how far it had got, not what it produced. Hashing
# them made every reboot change the fingerprint, which invalidated every gate
# that had already passed and restarted the whole qualification from the first step.
VOLATILE_QUEUE_KEYS = frozenset({
    "started_at", "completed_at", "first_started_at", "status", "files_complete"})


def durable_queue_state(raw: bytes) -> bytes:
    """The part of a download-state document that says what was downloaded.

    Provenance is unaffected: repository, revision and file counts still hash,
    the queue documents themselves are hashed whole, and each destination file
    still contributes its size and mtime. A failed or incomplete queue is caught
    by wait_for_inputs and by the stamp byte total, not by this digest.
    """
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        # Unparseable is still a change worth noticing; hash the bytes as given.
        return raw

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items()
                    if key not in VOLATILE_QUEUE_KEYS}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return json.dumps(strip(document), sort_keys=True, separators=(",", ":")).encode()


# Reading the checkout is a query about the host, not about a model, so a host
# that is too busy to answer it is a reason to wait rather than a qualification
# failure. ``git diff --binary HEAD`` walks every tracked file under
# ``verification``; while a download queue saturates the disk that regularly
# exceeded the original single 30-second attempt, and a concurrent git process
# makes it exit 128 over ``index.lock``. Both surfaced as an unhandled
# TimeoutExpired/CalledProcessError from the module-level handler, which recorded
# ``status: failed`` and ended the run -- twenty times on 2026-09-08, without a
# single gate having been attempted.
FINGERPRINT_ATTEMPTS = 4
FINGERPRINT_TIMEOUT_SECONDS = 180
FINGERPRINT_RETRY_SECONDS = 20


class InputsUnreadable(RuntimeError):
    """The qualification could not read its own inputs, so it measured nothing."""


def read_git_inputs(command: list[str]) -> bytes:
    """One read-only git query, retried, or a refusal that names the condition."""
    last: Exception | None = None
    for attempt in range(1, FINGERPRINT_ATTEMPTS + 1):
        try:
            return subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                                  timeout=FINGERPRINT_TIMEOUT_SECONDS).stdout
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                OSError) as error:
            last = error
            log(f"input fingerprint attempt {attempt}/{FINGERPRINT_ATTEMPTS}"
                f" failed name={command[1]} {type(error).__name__}")
            if attempt < FINGERPRINT_ATTEMPTS:
                time.sleep(FINGERPRINT_RETRY_SECONDS)
    raise InputsUnreadable(
        f"git {command[1]} failed {FINGERPRINT_ATTEMPTS} times: {type(last).__name__}")


def qualification_inputs_fingerprint() -> str:
    """Fingerprint source and promoted artifacts without re-hashing model weights."""
    digest = hashlib.sha256()
    source_paths = ("verification", "llama-models.ini", "scripts")
    for command in (
        ["git", "ls-files", "-s", "--", *source_paths],
        ["git", "diff", "--binary", "HEAD", "--", *source_paths],
    ):
        digest.update(read_git_inputs(command))

    # Promotion records are durable; file metadata additionally catches restored
    # or replaced bytes whose queue state was not rewritten.
    for state_path, stamp_path, _expected in UPSTREAM:
        for path in (state_path, stamp_path):
            digest.update(str(path).encode())
            try:
                raw = path.read_bytes()
            except OSError:
                # Not promoted yet. wait_for_inputs is what blocks on that; this
                # only has to change when the bytes do, and "absent" is a state
                # it can fingerprint rather than a reason to abort the qualification
                # before it can report what it is waiting for.
                digest.update(b"\0absent\0")
            else:
                digest.update(durable_queue_state(raw)
                              if path == state_path else raw)
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
    """Publish qualification state durably; a resumed run reads it to skip passed steps."""
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
        if value.get("schema") == "frankenstein-functional-qualification/1":
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "schema": "frankenstein-functional-qualification/1",
        "benchmarking_performed": False,
        "benchmark_performed": False,
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
# Anything that loads weights outside the router this qualification manages itself.
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
# Only these conflict classes actually collide with a serialized GPU gate on this
# desktop. cargo/rustc/makepkg in other trees, aria2 re-hash, and Chromium load
# are not kernel compiles and must not hold the qualification after a new kernel boot.
QUALIFICATION_BLOCKING_REASONS = frozenset({"kernel-build", "inference", "download-queue"})
# The systemd slice every download queue runs under. gate_router_models refuses
# to qualify beside one, so the qualification has to wait for it rather than start a
# gate that will immediately refuse.
DOWNLOAD_UNIT_PREFIX = "local-ai-model-downloads"


def printable(raw: bytes | str) -> str:
    """Force kernel-supplied text to valid UTF-8.

    A task name is whatever bytes the process chose for it. Left alone, an
    undecodable one raises where it is least welcome -- not here, but later,
    when the reason the host is busy is written to the UTF-8 qualification log.
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
    if DOWNLOAD_UNIT_PREFIX in cgroup:
        # Named by cgroup, before the command tests: a queue runs python3 and
        # aria2c, which would otherwise be classified transfer or
        # workspace-python and let the qualification walk into a refused gate.
        return "download-queue"
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
    """A matching completion stamp is readiness, even if a writer is re-checking.

    After reboot the downloader marks state ``running`` while it re-hashes files
    that already exist. The stamp is the durable byte-total proof; waiting for
    the re-hash to flip status back to complete would gate the qualification on work
    that is not a missing download.
    """
    try:
        stamp = stamp_path.read_text(encoding="utf-8").strip()
    except OSError:
        stamp = None
    if stamp == expected:
        return "ready", state_path.name
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return "waiting", f"{state_path.name}: {type(error).__name__}"
    status = state.get("status")
    if status == "failed":
        return "failed", f"{state_path.name}: {state.get('error', 'unknown error')}"
    if stamp is None:
        return "waiting", f"{stamp_path.name}: missing"
    return "failed", f"{stamp_path.name}: expected {expected}, got {stamp!r}"


def artifacts_in_flight() -> list[str]:
    """Queue destinations that are absent or the wrong size and being re-fetched.

    A completion stamp records that a queue finished once. It does not survive
    contact with this host: the queue verifies SHA-256 on promotion, quarantines
    a file that fails to ``<name>.bad-<stamp>`` and downloads it again, and the
    stamp still says complete throughout. On 2026-09-09 the candidate policy gate
    ran sixteen minutes before ``Gemma-4-12B-it-heretic-Q6_K.gguf`` was promoted,
    correctly reported it "missing or wrong size", and blocked all eleven
    model-backed gates behind a file that was simply still arriving.

    Only a destination with a ``.partial`` sibling counts: that is a transfer
    staging bytes right now. A file that is merely absent is a real problem and
    is left to the policy gate to report, so this never becomes a wait for
    something nobody is fetching.
    """
    waiting = []
    for queue_path in sorted(FOUNDATION.glob("download-queue*.json")):
        try:
            document = json.loads(queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for artifact in document.get("artifacts", []):
            for entry in artifact.get("files", []):
                destination = Path(entry.get("destination", ""))
                expected = entry.get("size")
                if not destination.name or not isinstance(expected, int):
                    continue
                try:
                    complete = destination.stat().st_size == expected
                except OSError:
                    complete = False
                if complete or not destination.with_name(destination.name + ".partial").exists():
                    continue
                waiting.append(str(destination))
    return sorted(waiting)


def wait_for_inputs(state: dict) -> None:
    last = None
    while True:
        statuses = [phase_status(*phase) for phase in UPSTREAM]
        failure = next((detail for status, detail in statuses if status == "failed"), None)
        if failure:
            state.update({"status": "failed", "error": failure, "updated_at": now()})
            atomic_json(state)
            raise RuntimeError(failure)
        in_flight = artifacts_in_flight()
        if all(status == "ready" for status, _ in statuses) and not in_flight:
            return
        if in_flight:
            statuses = statuses + [("re-fetching",
                                    f"{len(in_flight)} artifact(s) being re-fetched:"
                                    f" {in_flight[:4]}")]
        detail = "; ".join(text for _, text in statuses)
        if detail != last:
            log(f"waiting for artifact queues: {detail}")
            last = detail
        state.update({"status": "waiting-artifacts", "updated_at": now()})
        atomic_json(state)
        time.sleep(POLL_SECONDS)


def summarize_conflicts(active: list[dict], sample_limit: int = CONFLICT_SAMPLE_LIMIT) -> str:
    """Bound a potentially huge compiler fan-out for logs and timeout errors."""
    counts = Counter(str(item.get("reason", "unknown")) for item in active)
    by_reason = ",".join(f"{name}:{counts[name]}" for name in sorted(counts))
    sample = [
        f"{item.get('pid')}:{item.get('command')}:{item.get('reason')}"
        for item in active[:sample_limit]
    ]
    omitted = max(0, len(active) - len(sample))
    return (f"count={len(active)} by_reason={{{by_reason}}} "
            f"sample={sample} omitted={omitted}")


def wait_for_quiet(state: dict, timeout: int = QUIET_WAIT_TIMEOUT) -> None:
    quiet = 0
    last_key = None
    last_logged = float("-inf")
    deadline = time.monotonic() + timeout
    while quiet < 2:
        active = [item for item in conflicts()
                  if item.get("reason") in QUALIFICATION_BLOCKING_REASONS]
        available = mem_available()
        reasons = []
        if active:
            reasons.append(f"conflicts={summarize_conflicts(active)}")
        if available < MIN_AVAILABLE:
            reasons.append(f"MemAvailable={available}")
        if reasons:
            quiet = 0
            detail = "; ".join(reasons)
            key = (
                tuple(sorted({str(item.get("reason", "unknown")) for item in active})),
                available < MIN_AVAILABLE,
            )
            observed = time.monotonic()
            if key != last_key or observed - last_logged >= BLOCKER_LOG_REPEAT_SECONDS:
                log(f"waiting for safe host: {detail}")
                last_key = key
                last_logged = observed
        else:
            quiet += 1
            last_key = None
            log(f"safe-host quiet poll {quiet}/2 MemAvailable={available}")
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


def run_step(name: str, command: list[str], state: dict, before_start=None) -> int:
    global child
    wait_for_quiet(state)
    if before_start is not None:
        before_start()
    step = {
        "command": command,
        "input_fingerprint": state["input_fingerprint"],
        "started_at": now(),
        "status": "running",
        "benchmarking_performed": False,
        "benchmark_performed": False,
    }
    state["steps"][name] = step
    state.update({"status": "running", "current_step": name, "updated_at": now()})
    atomic_json(state)
    step_log = HERE / f"{name}.log"
    performance_path = HERE / f"{name}-{time.time_ns()}-{os.getpid()}.performance.jsonl"
    step["performance_log"] = str(performance_path)
    step["performance_kind"] = "passive-observations-not-a-benchmark"
    atomic_json(state)
    environment = dict(os.environ, QUALIFICATION_PERFORMANCE_PATH=str(performance_path))
    log(f"step start name={name}")
    with step_log.open("ab") as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
        rc = child.wait()
    child = None
    step.update({"finished_at": now(), "exit_code": rc, "status": exit_status(rc)})
    state["updated_at"] = now()
    atomic_json(state)
    log(f"step end name={name} rc={rc}")
    return rc


def prepare_child_environment() -> None:
    """Give every gate a writable temp dir under ProtectSystem=strict.

    systemd remounts /tmp read-only unless it is in ReadWritePaths. Torch
    probes tempfile.gettempdir() at import time and dies with FileNotFoundError
    if that probe cannot create a file. Point TMPDIR at a path the unit already
    allows, even when the unit file on disk has not been reloaded yet.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(SCRATCH)
    os.environ["TMP"] = str(SCRATCH)
    os.environ["TEMP"] = str(SCRATCH)


def qualification_key(name, command):
    try:
        return step_key(name, command)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def qualification_components(name, command):
    try:
        return key_components(step_key(name, command, details=True))
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def exit_status(rc):
    return {0: 'passed', EXIT_ADMISSION_REFUSED: 'blocked',
            EXIT_INCONCLUSIVE: 'inconclusive'}.get(rc, 'failed')


def reuse_step(name, command, state, force=False):
    key = qualification_key(name, command)
    cached = Store(HERE / 'qualification-cache').reuse(name, name, key, force)
    if cached is None:
        return False
    state['steps'][name] = dict(cached['step'], qualification_reused=True,
                               input_fingerprint=state['input_fingerprint'])
    atomic_json(state)
    log(f'qualification reused name={name}; no model execution')
    return True


def execute_qualification(name, command, state):
    store = Store(HERE / 'qualification-cache')
    previous = store.read(name, name)
    identity = {}

    def begin():
        # Admission waits must not revoke a completed qualification. Capture
        # inputs only once the host is admitted, immediately before dispatch.
        identity.update(key=qualification_key(name, command),
                        components=qualification_components(name, command))
        store.publish(name, name, identity['key'], components=identity['components'])

    rc = run_step(name, command, state, before_start=begin)
    key = identity.get('key')
    if rc == EXIT_ADMISSION_REFUSED:
        store.restore(name, name, previous)
        state['steps'][name].update(status='blocked', exit_code=rc)
        atomic_json(state)
        return rc
    outcome = exit_status(rc)
    if stop_signal is not None and rc != 0:
        # The operator stopped the qualification and the signal was forwarded to the
        # running gate, which then exited non-zero because it was stopped. That
        # run decided nothing about the model, so the receipt says so. Filing an
        # operator stop as a model failure is the same error as filing a refusal
        # as one -- it is fail-closed either way, but only one of them is true.
        outcome = INCONCLUSIVE
        state['steps'][name].update(status='interrupted', exit_code=rc)
        atomic_json(state)
    elif key is not None and qualification_key(name, command) != key:
        rc = EXIT_INCONCLUSIVE
        outcome = INCONCLUSIVE
        state['steps'][name].update(status='inconclusive', exit_code=rc,
                                    error='qualification inputs changed during execution')
        atomic_json(state)
    store.publish(name, name, key, {'pass': rc == 0, 'outcome': outcome,
                                  'step': state['steps'].get(name, {})},
                  components=identity.get('components'))
    return rc


def main(argv=()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', action='append', choices=[name for name, _ in STEPS])
    parser.add_argument('--model', action='append', help='router/repository preset')
    parser.add_argument('--requalify', action='store_true')
    parser.add_argument('--stability-runs', type=int, default=1)
    parser.add_argument('--plan', action='store_true', help='report gate reuse without starting workloads or writing state')
    parser.add_argument('--migrate-passes', action='store_true', help='import eligible legacy passes only; no workloads')
    args = parser.parse_args(argv)
    if not 1 <= args.stability_runs <= 100:
        parser.error('--stability-runs must be between 1 and 100')
    if args.stability_runs > 1:
        child_args = list(argv)
        if '--stability-runs' in child_args:
            index = child_args.index('--stability-runs')
            del child_args[index:index + 2]
        else:
            child_args = [x for x in child_args if not x.startswith('--stability-runs=')]
        if '--requalify' not in child_args:
            child_args.append('--requalify')
        for _ in range(args.stability_runs):
            rc = main(child_args)
            if rc:
                return rc
        return 0
    selected = set(args.gate or [name for name, _ in STEPS])
    if args.model:
        # Validate the *name*, which is what the operator typed. Stat-ing the
        # preset's weights here rejected a valid selection whenever the file was
        # mid-download, which is exactly when a targeted retest gets asked for.
        from qualification_cache import preset_values
        for model in args.model:
            try:
                preset_values(model)
            except (ValueError, OSError) as error:
                parser.error(str(error))
        selected &= {'router-models', 'repository-agent', 'candidate-policy', 'router-reload-presets'}
        if 'repository-agent' in selected and not set(args.model) <= {'heretic', 'qwen3-coder-next'}:
            selected.remove('repository-agent')
        if not selected - {'candidate-policy', 'router-reload-presets'}:
            parser.error('no applicable model gate selected')
    os.environ['HERMES_REQUALIFY'] = '1' if args.requalify else '0'
    os.environ['HERMES_QUALIFY_MODELS'] = json.dumps(args.model or [])
    if args.plan:
        store = Store(HERE / 'qualification-cache')
        print(json.dumps([{'gate': name, 'cache': store.explain(
            name, name, qualification_key(name, command),
            qualification_components(name, command), args.requalify), 'action':
            'reuse' if store.reuse(name, name, qualification_key(name, command), args.requalify)
            else ('per-model dispatch' if name in ('router-models', 'repository-agent') else 'run')}
            for name, command in STEPS if name in selected], indent=2))
        return 0
    HERE.mkdir(parents=True, exist_ok=True)
    prepare_child_environment()
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another qualification supervisor owns the lock")
        return 75
    state = load_state()
    imported = []
    if not args.requalify:
        for name, command in STEPS:
            if adopt_legacy_step(name, command, state, Store(HERE / 'qualification-cache')):
                imported.append(name)
                log(f'imported completed qualification name={name}; original evidence retained')
    if args.migrate_passes:
        print(json.dumps({'imported': imported, 'model_workloads_started': False}))
        return 0
    try:
        input_fingerprint = qualification_inputs_fingerprint()
    except InputsUnreadable as error:
        # Nothing was dispatched and no receipt was touched. Leave the previous
        # fingerprint in place so the next run compares against what actually
        # ran, and report a refusal rather than a qualification failure.
        state.update({"status": "inputs-unreadable", "current_step": None,
                      "error": str(error), "updated_at": now()})
        atomic_json(state)
        log(f"inputs unreadable; no gate was attempted: {error}")
        return EXIT_ADMISSION_REFUSED
    for stale in ("signal", "error", "failed_step", "failed_steps", "blocked_steps",
                  "exit_code", "interrupted_step", "step_exit_code", "step_status"):
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
                "benchmark_performed": False,
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
    blocked = []
    for name, command in STEPS[1:]:
        if name not in selected:
            continue
        previous = state["steps"].get(name, {})
        if reuse_step(name, command, state, args.requalify):
            continue
        receipt = Store(HERE / 'qualification-cache').read(name, name)
        if (not args.requalify and name not in ('router-models', 'repository-agent')
                and (receipt is None or (receipt.get('key') is None and receipt.get('status') == 'passed'))
                and previous.get("status") == "passed"
                and previous.get("input_fingerprint") == input_fingerprint):
            log(f"step already passed; skipping name={name}")
            continue
        if previous.get("status") == "passed":
            log(f"passed step inputs changed; rerunning name={name}")
        rc = execute_qualification(name, command, state)
        # Order matters. A SIGTERM to this supervisor is forwarded to the running
        # step, which then exits non-zero -- so testing rc first recorded every
        # operator stop as "step X failed" and lost the fact that the qualification was
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
        if rc == EXIT_ADMISSION_REFUSED:
            # The gate refused admission after this supervisor's own quiet-host
            # wait, so it never ran. It is outstanding work, not a failed gate:
            # listing it in failed_steps publishes a verdict nothing measured,
            # and the next pass has to attempt it again rather than repair it.
            blocked.append({"name": name, "exit_code": rc})
            state.update({"status": "running-with-failures" if failures
                          else "running-with-blocked-admission",
                          "blocked_steps": blocked, "updated_at": now()})
            atomic_json(state)
            continue
        if rc != 0:
            failures.append({"name": name, "exit_code": rc})
            state.update({"status": "running-with-failures", "failed_steps": failures,
                          "updated_at": now()})
            atomic_json(state)
    if failures or blocked:
        state.update({
            "status": "functional-foundation-incomplete" if failures else "admission-blocked",
            "current_step": None,
            "failed_steps": failures,
            "blocked_steps": blocked,
            "exit_code": 1 if failures else EXIT_ADMISSION_REFUSED,
            "remaining": [item["name"] for item in failures + blocked] + [
                "build bounded end-to-end computer control only after grounding passes",
            ],
            "updated_at": now(),
        })
        atomic_json(state)
        log(f"functional foundation incomplete; failed steps={failures}"
            f"; blocked steps={blocked}" if failures
            else f"admission blocked; nothing failed; blocked steps={blocked}")
        return 1 if failures else EXIT_ADMISSION_REFUSED
    state.update({
        "status": "selected-qualifications-complete" if args.gate or args.model else "functional-foundation-complete",
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
        raise SystemExit(main(sys.argv[1:]))
    except Exception as error:  # noqa: BLE001 - durable failure evidence is required
        HERE.mkdir(parents=True, exist_ok=True)
        state = load_state()
        state.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "updated_at": now()})
        atomic_json(state)
        log(f"fatal {type(error).__name__}: {error}")
        raise SystemExit(1)
