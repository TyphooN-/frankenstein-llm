#!/usr/bin/env python3
"""Run the maintained download queues in order, under one owner.

Each of the four queues used to have its own systemd unit. The later three ran
waiter scripts that polled for the previous queue's completion stamp and then
exec'd the downloader, but once every stamp existed the waiting ordered nothing:
after each boot all four queues re-hashed the same disk at once. This runs them
one after another in a single process instead.

Every queue keeps the queue, state, lock, stamp and log file it always had. The
qualification supervisor, the capability ledger and the reconciliation scripts
read those names, and the state and stamp files record what earlier runs
downloaded, so renaming them would rewrite evidence rather than clarify it.
"""
from __future__ import annotations

from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import signal
import sys

import download_queue as dq

ROOT = Path(__file__).resolve().parent
PROGRAM = "download_service.py"
# What each queue installs and the suffix its files have always carried, in run
# order.
QUEUES = (
    ("core capabilities", ""),
    ("computer-use grounding", "-phase2"),
    ("image editing", "-phase3"),
    ("researched candidates", "-phase4"),
)
# The downloader's module-level paths, pointed at one queue at a time.
PATHS = ("queue", "state", "lock", "stamp", "log")

USAGE = "\n".join([
    f"usage: {PROGRAM} [--plan]",
    "",
    "With no arguments, runs the maintained download queues in one process.",
    "Every queue's lock is held for the whole run, so a hand-started",
    "download_queue.py for any of them exits 75 rather than racing it. Workers and",
    "connections come from HERMES_DOWNLOAD_WORKERS and",
    "HERMES_DOWNLOAD_CONNECTION_BUDGET, exactly as for download_queue.py.",
    "--plan prints ordered queue paths as JSON without reading manifests,",
    "taking locks, changing signal handlers, or starting downloads.",
    "",
    "queues, in order:",
    *(f"  {name:<23} download-queue{suffix}.json" for name, suffix in QUEUES),
    "",
    "exit codes:",
    "  0   every queue complete, or --help/--plan",
    "  1   a queue failed; the queues after it still ran",
    "  2   bad usage",
    "  75  another writer holds a queue lock; nothing was started",
])


def queue_plan(root: Path = ROOT) -> list[dict]:
    """Name the files each queue reads and writes, in run order. Touches nothing."""
    return [
        {
            "name": name,
            "queue": root / f"download-queue{suffix}.json",
            "state": root / f"download-state{suffix}.json",
            "lock": root / f"download-queue{suffix}.lock",
            "stamp": root / f"downloads{suffix}-complete.ok",
            "log": root / f"downloads{suffix}.log",
        }
        for name, suffix in QUEUES
    ]


def run_queue(row: dict) -> int:
    """Process one queue whose lock the caller holds, then restore the downloader."""
    saved = {key: getattr(dq, key.upper()) for key in PATHS}
    for key in PATHS:
        setattr(dq, key.upper(), row[key])
    try:
        return dq.process_queue()
    except Exception as error:  # an unreadable queue must not stop the next one
        try:
            dq.log(f"fatal {error!r}")
        except OSError as log_error:
            # A broken log path must not turn one failed queue into a stopped
            # service. Keep the original failure visible on the service stream.
            print(f"{PROGRAM}: fatal {error!r}; log unavailable: {log_error!r}",
                  file=sys.stderr, flush=True)
        return 1
    finally:
        for key, value in saved.items():
            setattr(dq, key.upper(), value)


def run(root: Path = ROOT) -> int:
    plan = queue_plan(root)
    with ExitStack() as held:
        # Take every lock before touching any queue. Holding them all for the
        # whole run is what makes this the only writer: a hand-started queue
        # cannot slip in between two queues, and a run that finds one already
        # taken leaves no state, stamp or log behind.
        for row in plan:
            handle = held.enter_context(row["lock"].open("a+"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"{PROGRAM}: another writer holds {row['lock'].name}; nothing started",
                      flush=True)
                return 75
        failed = []
        for row in plan:
            print(f"{PROGRAM}: {row['name']} queue ({row['queue'].name})", flush=True)
            code = run_queue(row)
            if code != 0:
                failed.append(f"{row['name']} exited {code}")
        if failed:
            print(f"{PROGRAM}: failed: {'; '.join(failed)}", flush=True)
            return 1
        print(f"{PROGRAM}: every queue complete", flush=True)
        return 0


def main(argv: list[str]) -> int:
    """Parse first, as download_queue.py does; only no arguments touches a queue."""
    if argv in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    if argv == ["--plan"]:
        print(json.dumps({
            "plan_only": True,
            "queues": [{key: str(value) for key, value in row.items()}
                       for row in queue_plan()],
        }, indent=2))
        return 0
    if argv:
        print(f"{PROGRAM}: unexpected argument(s): {' '.join(argv)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    return run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
