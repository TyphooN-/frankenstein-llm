"""Concurrency contracts for the resumable file downloader.

No network or model files are touched: download(), logging and state publication
are replaced with in-memory test doubles.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import cast
import unittest
from unittest import mock

import download_queue as dq


def artifact(key: str, count: int, offset: int = 0) -> dict:
    return {
        "key": key,
        "capability": "test",
        "repository": "owner/repo",
        "revision": "deadbeef",
        "files": [
            {"repo_path": f"{key}-{index}",
             "destination": f"/models/{key}-{index + offset}",
             "size": 10, "sha256": None}
            for index in range(count)
        ],
    }


def queue_with_files(count: int) -> dict:
    return {"artifacts": [artifact("fixture", count)]}


class WorkerConfigurationTests(unittest.TestCase):
    def test_default_is_high_enough_to_fill_a_fast_wan(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(16, dq.configured_workers())

    def test_explicit_worker_count_is_honored(self):
        with mock.patch.dict(os.environ, {"HERMES_DOWNLOAD_WORKERS": "32"}):
            self.assertEqual(32, dq.configured_workers())

    def test_invalid_worker_counts_fail_closed(self):
        for raw in ("0", "65", "many"):
            with self.subTest(raw=raw), \
                 mock.patch.dict(os.environ, {"HERMES_DOWNLOAD_WORKERS": raw}):
                with self.assertRaises(ValueError):
                    dq.configured_workers()

    def test_connection_budget_defaults_to_32_and_is_bounded(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(32, dq.configured_connection_budget())
        for raw in ("0", "257", "many"):
            with self.subTest(raw=raw), \
                 mock.patch.dict(os.environ, {"HERMES_DOWNLOAD_CONNECTION_BUDGET": raw}):
                with self.assertRaises(ValueError):
                    dq.configured_connection_budget()


class TransferCommandTests(unittest.TestCase):
    def test_aria2_uses_resumable_bounded_ranges(self):
        partial = Path("/models/artifact.partial")
        with mock.patch.object(dq.shutil, "which", return_value="/usr/bin/aria2c"):
            command = dq.transfer_command("https://example.invalid/file", partial, 16)
        self.assertEqual("aria2c", command[0])
        self.assertIn("--continue=true", command)
        self.assertIn("--max-connection-per-server=16", command)
        self.assertIn("--split=16", command)
        self.assertEqual(partial.name, command[command.index("--out") + 1])

    def test_curl_is_the_portable_single_connection_fallback(self):
        partial = Path("/models/artifact.partial")
        with mock.patch.object(dq.shutil, "which", return_value=None):
            command = dq.transfer_command("https://example.invalid/file", partial, 16)
        self.assertEqual("curl", command[0])
        self.assertIn("--continue-at", command)

    def test_curl_partial_without_aria_metadata_stays_on_curl(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "artifact.partial"
            partial.write_bytes(b"curl-prefix")
            with mock.patch.object(dq.shutil, "which", return_value="/usr/bin/aria2c"):
                command = dq.transfer_command("https://example.invalid/file", partial, 16)
        self.assertEqual("curl", command[0])

    def test_aria_partial_with_control_metadata_resumes_on_aria(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "artifact.partial"
            partial.write_bytes(b"sparse-ranges")
            dq.aria_control_path(partial).write_bytes(b"piece-map")
            with mock.patch.object(dq.shutil, "which", return_value="/usr/bin/aria2c"):
                command = dq.transfer_command("https://example.invalid/file", partial, 16)
        self.assertEqual("aria2c", command[0])


class ParallelQueueTests(unittest.TestCase):
    def run_queue(self, queue: dict, fake_download, workers: int = 4,
                  budget: int | None = None):
        state: dict[str, object] = {"artifacts": {}}
        with mock.patch.object(dq, "download", side_effect=fake_download), \
             mock.patch.object(dq, "atomic_json"), \
             mock.patch.object(dq, "log"):
            dq.run_downloads(queue, state, workers, connection_budget=budget)
        return state

    def test_independent_files_overlap_up_to_the_bound(self):
        lock = threading.Lock()
        active = 0
        peak = 0

        def fake_download(*_args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1

        state = self.run_queue(queue_with_files(8), fake_download, workers=4)
        self.assertEqual(4, peak)
        artifacts = cast(dict, state["artifacts"])
        self.assertEqual("complete", artifacts["fixture"]["status"])
        self.assertEqual(8, artifacts["fixture"]["files_complete"])

    def test_worker_count_is_clamped_to_file_count(self):
        threads: set[int] = set()
        barrier = threading.Barrier(3)

        def fake_download(*_args):
            threads.add(threading.get_ident())
            barrier.wait(timeout=2)

        self.run_queue(queue_with_files(3), fake_download, workers=16)
        self.assertEqual(3, len(threads))

    def test_connection_budget_is_shared_across_active_files(self):
        observed = []

        def fake_download(*args):
            observed.append(args[-1])

        self.run_queue(queue_with_files(4), fake_download, workers=16, budget=32)
        self.assertEqual([8] * 4, sorted(observed))

    def test_single_remaining_file_receives_the_per_file_maximum(self):
        observed = []

        def fake_download(*args):
            observed.append(args[-1])

        self.run_queue(queue_with_files(1), fake_download, workers=16, budget=32)
        self.assertEqual([16], observed)

    def test_duplicate_destination_is_rejected_before_work_starts(self):
        queue = queue_with_files(2)
        queue["artifacts"][0]["files"][1]["destination"] = "/models/fixture-0"
        with self.assertRaisesRegex(ValueError, "duplicate destination"):
            dq.prepare_jobs(queue, {"artifacts": {}})

    def test_destinations_that_only_look_different_are_still_one_owner(self):
        # Two workers writing one file share its ".partial" as well, so the
        # size/digest check would score whichever transfer renamed last. Paths
        # are compared resolved so "..", "//" and "./" cannot smuggle a second
        # owner past the guard.
        for alias in ("/models/../models/fixture-0", "/models//fixture-0",
                      "/models/./fixture-0"):
            with self.subTest(alias=alias):
                queue = queue_with_files(2)
                queue["artifacts"][0]["files"][1]["destination"] = alias
                with self.assertRaisesRegex(ValueError, "duplicate destination"):
                    dq.prepare_jobs(queue, {"artifacts": {}})

    def test_the_error_names_both_the_queue_entry_and_the_existing_owner(self):
        queue = queue_with_files(2)
        queue["artifacts"][0]["files"][1]["destination"] = "/models/../models/fixture-0"
        with self.assertRaises(ValueError) as caught:
            dq.prepare_jobs(queue, {"artifacts": {}})
        message = str(caught.exception)
        self.assertIn("/models/../models/fixture-0", message)
        self.assertIn("/models/fixture-0", message)
        self.assertIn("fixture", message)

    def test_distinct_destinations_are_all_scheduled(self):
        jobs = dq.prepare_jobs(queue_with_files(5), {"artifacts": {}})
        self.assertEqual(5, len(jobs))
        self.assertEqual({"fixture"}, {key for key, _, _ in jobs})

    def test_one_failure_does_not_discard_other_completed_transfers(self):
        completed = []

        def fake_download(_repo, _revision, repo_path, *_rest):
            if repo_path == "fixture-1":
                raise RuntimeError("fixture failure")
            completed.append(repo_path)

        state: dict[str, object] = {"artifacts": {}}
        with mock.patch.object(dq, "download", side_effect=fake_download), \
             mock.patch.object(dq, "atomic_json"), \
             mock.patch.object(dq, "log"):
            with self.assertRaisesRegex(RuntimeError, "1 file download"):
                dq.run_downloads(queue_with_files(4), state, workers=4)
        self.assertEqual(3, len(completed))
        record = cast(dict, state["artifacts"])["fixture"]
        self.assertEqual("failed", record["status"])
        self.assertEqual(3, record["files_complete"])
        self.assertEqual(1, len(record["errors"]))
    def test_a_failing_artifact_does_not_hold_back_an_independent_one(self):
        queue = {"artifacts": [artifact("alpha", 3), artifact("beta", 3, offset=10)]}

        def fake_download(_repo, _revision, repo_path, *_rest):
            if repo_path == "alpha-1":
                raise RuntimeError("fixture failure")

        state: dict[str, object] = {"artifacts": {}}
        with mock.patch.object(dq, "download", side_effect=fake_download), \
             mock.patch.object(dq, "atomic_json"), \
             mock.patch.object(dq, "log"):
            with self.assertRaises(RuntimeError):
                dq.run_downloads(queue, state, workers=6)
        artifacts = cast(dict, state["artifacts"])
        self.assertEqual("failed", artifacts["alpha"]["status"])
        self.assertEqual("complete", artifacts["beta"]["status"])
        self.assertIn("completed_at", artifacts["beta"])
        self.assertNotIn("errors", artifacts["beta"])


class SingleWriterTests(unittest.TestCase):
    """One process owns the queue; inside it, one thread owns the state file."""

    def test_state_is_published_only_from_the_thread_that_runs_the_queue(self):
        # Concurrent atomic_json calls share one ".tmp" path, so two writers
        # would race the same temp file and could publish a half-built state.
        writers: list[int] = []
        barrier = threading.Barrier(4)

        def fake_download(*_args):
            barrier.wait(timeout=5)

        state: dict[str, object] = {"artifacts": {}}
        with mock.patch.object(dq, "download", side_effect=fake_download), \
             mock.patch.object(dq, "atomic_json",
                               side_effect=lambda *_a: writers.append(threading.get_ident())), \
             mock.patch.object(dq, "log"):
            dq.run_downloads(queue_with_files(8), state, workers=4)
        self.assertGreater(len(writers), 1, "state was never published")
        self.assertEqual({threading.get_ident()}, set(writers))

    def test_every_file_reaches_a_terminal_count_exactly_once(self):
        state: dict[str, object] = {"artifacts": {}}
        with mock.patch.object(dq, "download"), \
             mock.patch.object(dq, "atomic_json"), \
             mock.patch.object(dq, "log"):
            dq.run_downloads(queue_with_files(32), state, workers=16)
        record = cast(dict, state["artifacts"])["fixture"]
        self.assertEqual(32, record["files_complete"])
        self.assertEqual(32, record["files_total"])
        self.assertEqual("complete", record["status"])


class ConcurrentLogTests(unittest.TestCase):
    """Sixteen workers append to one log; no line may be torn or interleaved."""

    LINE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[-+]\d{4} worker-\d+ line-\d+$")

    def test_lines_from_many_threads_stay_whole(self):
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "downloads.log"
            with mock.patch.object(dq, "LOG", log_path), \
                 mock.patch("builtins.print"):
                def spam(worker: int) -> None:
                    for index in range(40):
                        dq.log(f"worker-{worker} line-{index}")

                threads = [threading.Thread(target=spam, args=(worker,)) for worker in range(16)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=30)
            lines = log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(16 * 40, len(lines))
        malformed = [line for line in lines if not self.LINE.match(line)]
        self.assertEqual([], malformed[:5], "log lines were interleaved")
        messages = {line.split(" ", 1)[1] for line in lines}
        expected = {f"worker-{worker} line-{index}"
                    for worker in range(16) for index in range(40)}
        self.assertEqual(expected, messages, "a log line was lost or duplicated")


if __name__ == "__main__":
    unittest.main()
