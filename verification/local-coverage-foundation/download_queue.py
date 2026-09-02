#!/usr/bin/env python3
"""Reboot-resumable, single-writer downloader for the local AI queue."""
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
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
QUEUE = Path(os.environ.get("HERMES_DOWNLOAD_QUEUE", str(ROOT / "download-queue.json")))
STATE = Path(os.environ.get("HERMES_DOWNLOAD_STATE", str(ROOT / "download-state.json")))
LOCK = Path(os.environ.get("HERMES_DOWNLOAD_LOCK", str(ROOT / "download-queue.lock")))
STAMP = Path(os.environ.get("HERMES_DOWNLOAD_STAMP", str(ROOT / "downloads-complete.ok")))
LOG = Path(os.environ.get("HERMES_DOWNLOAD_LOG", str(ROOT / "downloads.log")))

PROGRAM = "download_queue.py"
USAGE = f"""usage: {PROGRAM}

Reboot-resumable, single-writer downloader for the local AI queue.

Takes no arguments: the queue, state, lock, stamp and log paths are supplied by
the environment so that the systemd unit and an operator shell cannot disagree
about them. Run with no arguments to process the queue.

environment:
  HERMES_DOWNLOAD_QUEUE   queue definition   (default: {ROOT / "download-queue.json"})
  HERMES_DOWNLOAD_STATE   resumable state    (default: {ROOT / "download-state.json"})
  HERMES_DOWNLOAD_LOCK    single-writer lock (default: {ROOT / "download-queue.lock"})
  HERMES_DOWNLOAD_STAMP   completion stamp   (default: {ROOT / "downloads-complete.ok"})
  HERMES_DOWNLOAD_LOG     append-only log    (default: {ROOT / "downloads.log"})

exit codes:
  0   queue complete, or --help
  1   a download failed
  2   bad usage
  75  another writer holds the lock"""


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def download(repository: str, revision: str, repo_path: str, destination: Path, size: int, expected_sha: str | None) -> None:
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
    encoded_path = quote(repo_path, safe="/")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{encoded_path}?download=true"
    start = partial.stat().st_size if partial.exists() else 0
    log(f"download start key={repository}@{revision} file={repo_path} resume={start} expected={size}")
    command = [
        "curl", "--fail", "--location", "--show-error", "--silent",
        "--retry", "20", "--retry-delay", "5", "--retry-all-errors",
        "--connect-timeout", "30", "--continue-at", "-",
        "--output", str(partial), url,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl exited {result.returncode} for {repository}/{repo_path}")
    actual_size = partial.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"size mismatch for {partial}: expected {size}, got {actual_size}")
    if expected_sha is not None:
        actual_sha = sha256(partial)
        if actual_sha != expected_sha:
            quarantine = partial.with_name(partial.name + f".bad-{actual_sha[:12]}")
            os.replace(partial, quarantine)
            raise RuntimeError(f"SHA-256 mismatch for {repo_path}: expected {expected_sha}, got {actual_sha}")
    os.replace(partial, destination)
    log(f"promoted {destination} bytes={size} sha256={expected_sha or 'size-only'}")


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
        for artifact in queue["artifacts"]:
            key = artifact["key"]
            record = {
                "capability": artifact["capability"],
                "repository": artifact["repository"],
                "revision": artifact["revision"],
                "status": "running",
                "files_complete": 0,
                "files_total": len(artifact["files"]),
            }
            state["artifacts"][key] = record
            atomic_json(STATE, state)
            for item in artifact["files"]:
                download(
                    artifact["repository"], artifact["revision"], item["repo_path"],
                    Path(item["destination"]), int(item["size"]), item.get("sha256"),
                )
                record["files_complete"] += 1
                atomic_json(STATE, state)
            record["status"] = "complete"
            record["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            atomic_json(STATE, state)
        state["status"] = "complete"
        state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(STATE, state)
        STAMP.write_text(str(queue["total_bytes"]) + "\n", encoding="utf-8")
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
