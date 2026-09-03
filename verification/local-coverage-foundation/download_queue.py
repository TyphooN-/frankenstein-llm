#!/usr/bin/env python3
"""Reboot-resumable, integrity-checked parallel downloader for a local AI queue."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
QUEUE = Path(os.environ.get("HERMES_DOWNLOAD_QUEUE", str(ROOT / "download-queue.json")))
STATE = Path(os.environ.get("HERMES_DOWNLOAD_STATE", str(ROOT / "download-state.json")))
LOCK = Path(os.environ.get("HERMES_DOWNLOAD_LOCK", str(ROOT / "download-queue.lock")))
STAMP = Path(os.environ.get("HERMES_DOWNLOAD_STAMP", str(ROOT / "downloads-complete.ok")))
LOG = Path(os.environ.get("HERMES_DOWNLOAD_LOG", str(ROOT / "downloads.log")))
DEFAULT_WORKERS = 16
MAX_WORKERS = 64
DEFAULT_CONNECTION_BUDGET = 32
MAX_CONNECTIONS_PER_FILE = 16
LOG_LOCK = threading.Lock()

PROGRAM = "download_queue.py"
USAGE = f"""usage: {PROGRAM}

Reboot-resumable, integrity-checked parallel downloader for the local AI queue.

Takes no arguments: the queue, state, lock, stamp and log paths are supplied by
the environment so that the systemd unit and an operator shell cannot disagree
about them. Run with no arguments to process the queue.

environment:
  HERMES_DOWNLOAD_QUEUE   queue definition   (default: {ROOT / "download-queue.json"})
  HERMES_DOWNLOAD_STATE   resumable state    (default: {ROOT / "download-state.json"})
  HERMES_DOWNLOAD_LOCK    single-writer lock (default: {ROOT / "download-queue.lock"})
  HERMES_DOWNLOAD_STAMP   completion stamp   (default: {ROOT / "downloads-complete.ok"})
  HERMES_DOWNLOAD_LOG     append-only log    (default: {ROOT / "downloads.log"})
  HERMES_DOWNLOAD_WORKERS concurrent files   (default: {DEFAULT_WORKERS}, max: {MAX_WORKERS})
  HERMES_DOWNLOAD_CONNECTION_BUDGET total HTTP connections (default: {DEFAULT_CONNECTION_BUDGET})

exit codes:
  0   queue complete, or --help
  1   a download failed
  2   bad usage
  75  another writer holds the lock"""


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    with LOG_LOCK:
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def fsync_parent(path: Path, tolerate_unsupported: bool = False) -> None:
    """Persist directory metadata after a rename."""
    directory = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(directory)
    except OSError:
        if not tolerate_unsupported:
            raise
    finally:
        os.close(directory)


def atomic_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text``, durably.

    The rename is atomic, but a rename alone only guarantees that a reader sees
    the old file or the new one -- not that the new one survives a power loss.
    This host has gone down mid-run before, and both files written through here
    are read after a reboot to decide what happens next, so the temp file and its
    directory are fsynced before the write is considered done.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # State remains readable on filesystems that reject directory fsync. Model
    # promotion below is stricter because downstream execution trusts its stamp.
    fsync_parent(path.parent, tolerate_unsupported=True)


def atomic_json(path: Path, value: object) -> None:
    """Publish the resumable state file; a restarted queue resumes from it."""
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, size: int, expected_sha: str | None) -> bool:
    try:
        if path.stat().st_size != size:
            return False
    except OSError:
        return False
    return expected_sha is None or sha256(path) == expected_sha


def durable_promote(partial: Path, destination: Path) -> None:
    """Persist verified bytes and their final name before publishing success.

    The completion state and stamp are separately fsynced. Without this ordering,
    a power loss could preserve those tiny files while losing dirty model pages or
    the rename they claim completed. If directory fsync fails after the rename,
    the queue fails closed; its next run revalidates the final destination.
    """
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    fsync_parent(destination.parent)


def promote_complete_partial(partial: Path, destination: Path, expected_sha: str | None) -> bool:
    """Promote a ``.partial`` that already holds every byte, or quarantine it.

    Called only when the partial's size already equals the expected size. A
    wrong digest at full size means the bytes are bad rather than incomplete, so
    resuming would never repair it; the file is moved aside and the caller
    restarts the transfer from zero.

    Returns whether the destination now holds verified bytes. The caller must
    not re-derive that by hashing the destination again: these files are tens of
    gigabytes and every worker thread would pay the second pass at once.
    """
    if expected_sha is not None:
        actual_sha = sha256(partial)
        if actual_sha != expected_sha:
            quarantine = partial.with_name(partial.name + f".bad-{actual_sha[:12]}")
            os.replace(partial, quarantine)
            log(f"quarantined complete-size partial with wrong sha256 {partial} -> {quarantine}")
            return False
    durable_promote(partial, destination)
    log(f"promoted complete partial {destination} bytes={destination.stat().st_size} "
        f"sha256={expected_sha or 'size-only'}")
    return True


def transfer_command(url: str, partial: Path, connections: int) -> list[str]:
    """Build a resumable transfer command; use ranges when aria2 is available."""
    if connections > 1 and shutil.which("aria2c"):
        return [
            "aria2c", "--continue=true", "--file-allocation=none",
            "--auto-file-renaming=false", "--allow-overwrite=true",
            f"--max-connection-per-server={connections}", f"--split={connections}",
            "--min-split-size=1M", "--max-tries=20", "--retry-wait=5",
            "--connect-timeout=30", "--timeout=60", "--summary-interval=30",
            "--console-log-level=notice", "--download-result=hide",
            "--dir", str(partial.parent), "--out", partial.name, url,
        ]
    return [
        "curl", "--fail", "--location", "--show-error", "--silent",
        "--retry", "20", "--retry-delay", "5", "--retry-all-errors",
        "--connect-timeout", "30", "--continue-at", "-",
        "--output", str(partial), url,
    ]


def download(repository: str, revision: str, repo_path: str, destination: Path, size: int,
             expected_sha: str | None, connections: int = 1) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if valid(destination, size, expected_sha):
        log(f"verified existing {destination} bytes={size}")
        return
    if destination.exists():
        quarantine = destination.with_name(destination.name + f".bad-{int(time.time())}")
        os.replace(destination, quarantine)
        log(f"quarantined invalid final {destination} -> {quarantine}")
    if partial.exists() and partial.stat().st_size > size:
        quarantine = partial.with_name(partial.name + f".bad-{int(time.time())}")
        os.replace(partial, quarantine)
        log(f"quarantined oversized partial {partial} -> {quarantine}")
    if partial.exists() and partial.stat().st_size == size:
        # A death between "curl finished" and "os.replace" leaves a partial that
        # is already the whole file. Resuming it asks for a byte range starting
        # at EOF; the server answers 416 and --fail turns that into a non-zero
        # exit, so the unit would restart into the same state forever without
        # ever transferring a byte. Verify and promote instead -- the same
        # handling glm53flash-local/download_and_verify.py already applies.
        if promote_complete_partial(partial, destination, expected_sha):
            return
    encoded_path = quote(repo_path, safe="/")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{encoded_path}?download=true"
    start = partial.stat().st_size if partial.exists() else 0
    command = transfer_command(url, partial, connections)
    log(f"download start key={repository}@{revision} file={repo_path} resume={start} "
        f"expected={size} transport={command[0]} connections={connections if command[0] == 'aria2c' else 1}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{command[0]} exited {result.returncode} for {repository}/{repo_path}")
    actual_size = partial.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"size mismatch for {partial}: expected {size}, got {actual_size}")
    if expected_sha is not None:
        actual_sha = sha256(partial)
        if actual_sha != expected_sha:
            quarantine = partial.with_name(partial.name + f".bad-{actual_sha[:12]}")
            os.replace(partial, quarantine)
            raise RuntimeError(f"SHA-256 mismatch for {repo_path}: expected {expected_sha}, got {actual_sha}")
    durable_promote(partial, destination)
    log(f"promoted {destination} bytes={size} sha256={expected_sha or 'size-only'}")


def configured_workers() -> int:
    raw = os.environ.get("HERMES_DOWNLOAD_WORKERS", str(DEFAULT_WORKERS))
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(f"HERMES_DOWNLOAD_WORKERS must be an integer, got {raw!r}") from error
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(
            f"HERMES_DOWNLOAD_WORKERS must be between 1 and {MAX_WORKERS}, got {workers}")
    return workers


def configured_connection_budget() -> int:
    raw = os.environ.get("HERMES_DOWNLOAD_CONNECTION_BUDGET", str(DEFAULT_CONNECTION_BUDGET))
    try:
        budget = int(raw)
    except ValueError as error:
        raise ValueError(
            f"HERMES_DOWNLOAD_CONNECTION_BUDGET must be an integer, got {raw!r}") from error
    if not 1 <= budget <= 256:
        raise ValueError(
            f"HERMES_DOWNLOAD_CONNECTION_BUDGET must be between 1 and 256, got {budget}")
    return budget


def prepare_jobs(queue: dict, state: dict[str, object]) -> list[tuple[str, dict, dict]]:
    """Build unique file jobs and initialize their artifact records.

    The queue lock provides one process-level owner. Destination uniqueness adds
    the per-file ownership invariant required for safe in-process concurrency.
    """
    jobs: list[tuple[str, dict, dict]] = []
    owners: dict[str, str] = {}
    artifacts = state["artifacts"]
    assert isinstance(artifacts, dict)
    for artifact in queue["artifacts"]:
        key = artifact["key"]
        files = artifact["files"]
        artifacts[key] = {
            "capability": artifact["capability"],
            "repository": artifact["repository"],
            "revision": artifact["revision"],
            "status": "running",
            "files_complete": 0,
            "files_total": len(files),
        }
        for item in files:
            destination = item["destination"]
            # Compare resolved paths, not the strings from the queue. Two entries
            # that differ only by "..", a doubled slash or a symlinked parent name
            # the same file and the same ".partial", so two workers would write
            # one destination and the size/digest check would score whichever
            # transfer happened to rename last.
            owner = os.path.realpath(destination)
            if owner in owners:
                raise ValueError(
                    f"duplicate destination in queue: {destination} resolves to "
                    f"{owner}, already claimed by {owners[owner]}")
            owners[owner] = key
            jobs.append((key, artifact, item))
    return jobs


def run_downloads(queue: dict, state: dict[str, object], workers: int,
                  connection_budget: int | None = None) -> None:
    """Download independent files concurrently; publish state in this thread."""
    jobs = prepare_jobs(queue, state)
    atomic_json(STATE, state)
    if not jobs:
        return
    active_workers = min(workers, len(jobs))
    budget = configured_connection_budget() if connection_budget is None else connection_budget
    per_file_connections = min(
        MAX_CONNECTIONS_PER_FILE, max(1, budget // active_workers))
    log(f"parallel queue start files={len(jobs)} workers={active_workers} "
        f"connection_budget={budget} connections_per_file={per_file_connections}")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=active_workers, thread_name_prefix="hf-file") as pool:
        pending = {
            pool.submit(
                download,
                artifact["repository"], artifact["revision"], item["repo_path"],
                Path(item["destination"]), int(item["size"]), item.get("sha256"),
                per_file_connections,
            ): (key, item["repo_path"])
            for key, artifact, item in jobs
        }
        artifacts = state["artifacts"]
        assert isinstance(artifacts, dict)
        for future in as_completed(pending):
            key, repo_path = pending[future]
            record = artifacts[key]
            try:
                future.result()
            except Exception as error:  # keep independent transfers useful
                record["status"] = "failed"
                record.setdefault("errors", []).append(
                    f"{repo_path}: {type(error).__name__}: {error}"[:1000])
                failures.append(f"{key}/{repo_path}: {type(error).__name__}: {error}")
            else:
                record["files_complete"] += 1
                if record["files_complete"] == record["files_total"]:
                    record["status"] = "complete"
                    record["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            atomic_json(STATE, state)
    if failures:
        raise RuntimeError(
            f"{len(failures)} file download(s) failed; first: {failures[0]}")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another queue writer owns the lock; exiting")
        return 75
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    state: dict[str, object] = {
        "schema": "hermes-hf-download-state/1",
        "queue_built_at": queue.get("built_at"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "running",
        "artifacts": {},
    }
    if STATE.exists():
        try:
            previous = json.loads(STATE.read_text(encoding="utf-8"))
            state["artifacts"] = previous.get("artifacts", {})
            state["first_started_at"] = previous.get("first_started_at", previous.get("started_at"))
        except Exception as error:
            log(f"ignoring unreadable prior state: {error!r}")
    state.setdefault("first_started_at", state["started_at"])
    atomic_json(STATE, state)
    try:
        run_downloads(queue, state, configured_workers())
        state["status"] = "complete"
        state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(STATE, state)
        # Downstream phases and the mission supervisor compare this stamp against
        # an exact byte total, so a torn write is not a retryable state: it reads
        # as a permanent mismatch against a queue that is in fact complete.
        atomic_text(STAMP, str(queue["total_bytes"]) + "\n")
        log(f"queue complete artifacts={len(queue['artifacts'])} bytes={queue['total_bytes']}")
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(STATE, state)
        log(f"fatal {error!r}")
        return 1


def parse_args(argv: list[str]) -> int | None:
    """Return an exit code to stop on, or None meaning "run the queue".

    Argument handling deliberately happens before anything else: this program
    takes a lock, rewrites state, appends to a log and moves multi-gigabyte
    files, so ``--help`` reaching ``main`` is not a cosmetic bug. Matching is
    exact -- no prefix abbreviation -- because a typo that resolves to a valid
    flag is the same class of accident.
    """
    if not argv:
        return None
    if argv in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    print(f"{PROGRAM}: unexpected argument(s): {' '.join(argv)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def run(argv: list[str]) -> int:
    """CLI entry point. Parse first; touch the queue only once parsing says to."""
    code = parse_args(argv)
    if code is not None:
        return code
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    return main()


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
