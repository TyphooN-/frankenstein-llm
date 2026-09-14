"""Contracts for the download service that replaced the per-queue phase units.

No network and no models directory. Each queue is built in a temporary directory
around a tiny file that already has the right size and digest, so the real
downloader only verifies it; the transfer entry point is replaced so that any
attempt to fetch is recorded.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import download_queue as dq
import download_service as ds

HERE = Path(__file__).resolve().parent
UNITS = HERE.parents[1] / "services" / "systemd"
DOWNLOADER_PATHS = ("QUEUE", "STATE", "LOCK", "STAMP", "LOG")


class ServiceRunTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "foundation"
        self.root.mkdir()
        weights = Path(temp.name) / "models"
        weights.mkdir()
        for name, suffix in ds.QUEUES:
            payload = name.encode()
            destination = weights / f"queue{suffix}.bin"
            destination.write_bytes(payload)
            (self.root / f"download-queue{suffix}.json").write_text(json.dumps({
                "total_bytes": len(payload),
                "artifacts": [{
                    "key": f"fixture{suffix}", "capability": "test",
                    "repository": "owner/repo", "revision": "pinned",
                    "files": [{"repo_path": destination.name,
                               "destination": str(destination),
                               "size": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest()}],
                }],
            }))
        self.before = {name: getattr(dq, name) for name in DOWNLOADER_PATHS}
        transfers = mock.patch.object(dq.subprocess, "run")
        self.transfers = transfers.start()
        self.addCleanup(transfers.stop)
        quiet = mock.patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def plan(self):
        return ds.queue_plan(self.root)

    def assertDownloaderRestored(self):
        self.assertEqual(self.before, {name: getattr(dq, name) for name in DOWNLOADER_PATHS})

    def test_every_queue_is_verified_in_order_and_stamped(self):
        order = []
        process = dq.process_queue

        def recording():
            order.append(dq.QUEUE.name)
            return process()

        with mock.patch.object(dq, "process_queue", recording):
            self.assertEqual(0, ds.run(self.root))
        self.assertEqual([f"download-queue{suffix}.json" for _, suffix in ds.QUEUES], order)
        for row in self.plan():
            with self.subTest(queue=row["name"]):
                total = json.loads(row["queue"].read_text())["total_bytes"]
                self.assertEqual("complete", json.loads(row["state"].read_text())["status"])
                self.assertEqual(f"{total}\n", row["stamp"].read_text())
                self.assertIn("verified existing", row["log"].read_text())
        self.transfers.assert_not_called()
        self.assertDownloaderRestored()

    def test_a_held_queue_lock_starts_nothing(self):
        for row in self.plan():
            with self.subTest(held=row["lock"].name), row["lock"].open("a+") as other:
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(dq, "process_queue") as process:
                    self.assertEqual(75, ds.run(self.root))
                process.assert_not_called()
        for row in self.plan():
            for key in ("state", "stamp", "log"):
                self.assertFalse(row[key].exists(), row[key])
        self.transfers.assert_not_called()

    def test_a_failed_queue_does_not_stop_the_queues_after_it(self):
        failing = self.plan()[1]
        queue = json.loads(failing["queue"].read_text())
        Path(queue["artifacts"][0]["files"][0]["destination"]).unlink()
        self.transfers.return_value = mock.Mock(returncode=22)
        self.assertEqual(1, ds.run(self.root))
        self.transfers.assert_called_once()
        for row in self.plan():
            with self.subTest(queue=row["name"]):
                status = json.loads(row["state"].read_text())["status"]
                if row["name"] == failing["name"]:
                    self.assertEqual("failed", status)
                    self.assertFalse(row["stamp"].exists())
                else:
                    self.assertEqual("complete", status)
                    self.assertTrue(row["stamp"].exists())
        self.assertDownloaderRestored()

    def test_a_log_failure_does_not_stop_later_queues(self):
        broken = self.plan()[0]
        broken["queue"].write_text("{not json")
        # A directory at the log path causes a real write failure, without
        # replacing the downloader or depending on filesystem permissions.
        broken["log"].mkdir()
        self.assertEqual(1, ds.run(self.root))
        for row in self.plan()[1:]:
            self.assertEqual("complete", json.loads(row["state"].read_text())["status"])
        self.assertDownloaderRestored()
        self.transfers.assert_not_called()

    def test_an_unreadable_queue_is_logged_and_the_rest_still_run(self):
        broken = self.plan()[0]
        broken["queue"].write_text("{not json")
        self.assertEqual(1, ds.run(self.root))
        self.assertIn("fatal", broken["log"].read_text())
        self.assertFalse(broken["state"].exists())
        self.assertFalse(broken["stamp"].exists())
        for row in self.plan()[1:]:
            with self.subTest(queue=row["name"]):
                self.assertEqual("complete", json.loads(row["state"].read_text())["status"])
        self.assertDownloaderRestored()


class PlanTests(unittest.TestCase):
    def test_every_queue_keeps_its_historical_file_names(self):
        self.assertEqual([
            ("download-queue.json", "download-state.json", "download-queue.lock",
             "downloads-complete.ok", "downloads.log"),
            ("download-queue-phase2.json", "download-state-phase2.json",
             "download-queue-phase2.lock", "downloads-phase2-complete.ok", "downloads-phase2.log"),
            ("download-queue-phase3.json", "download-state-phase3.json",
             "download-queue-phase3.lock", "downloads-phase3-complete.ok", "downloads-phase3.log"),
            ("download-queue-phase4.json", "download-state-phase4.json",
             "download-queue-phase4.lock", "downloads-phase4-complete.ok", "downloads-phase4.log"),
        ], [tuple(row[key].name for key in ds.PATHS) for row in ds.queue_plan()])
        for row in ds.queue_plan():
            self.assertTrue(row["queue"].is_file(), row["queue"])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.runs = []
        patcher = mock.patch.object(ds, "run", lambda: self.runs.append("run") or 0)
        patcher.start()
        self.addCleanup(patcher.stop)
        # main() ignores SIGHUP on the run path; put it back for the test process.
        self.addCleanup(signal.signal, signal.SIGHUP, signal.getsignal(signal.SIGHUP))

    def test_help_and_bad_arguments_never_reach_the_queues(self):
        with mock.patch("builtins.print"):
            for flag in ("-h", "--help"):
                self.assertEqual(0, ds.main([flag]))
            for argv in (["--bogus"], ["--hel"], ["--help", "--help"], ["positional"],
                         ["--plan", "extra"], ["--plan", "--help"], ["--plan", "--plan"]):
                with self.subTest(argv=argv):
                    self.assertEqual(2, ds.main(argv))
        self.assertEqual([], self.runs)

    def test_plan_reports_ordered_paths_without_side_effects(self):
        saved = {key: getattr(dq, key) for key in DOWNLOADER_PATHS}
        with (mock.patch("builtins.print") as printed,
              mock.patch("builtins.open", side_effect=AssertionError("file access")),
              mock.patch.object(Path, "open", side_effect=AssertionError("path access")),
              mock.patch.object(fcntl, "flock", side_effect=AssertionError("lock access")),
              mock.patch.object(signal, "signal", side_effect=AssertionError("signal change")),
              mock.patch.object(dq, "process_queue", side_effect=AssertionError("download")),
              mock.patch.object(dq, "log", side_effect=AssertionError("log write"))):
            self.assertEqual(0, ds.main(["--plan"]))
        printed.assert_called_once()
        plan = json.loads(printed.call_args.args[0])
        self.assertEqual({"plan_only", "queues"}, set(plan))
        self.assertIs(plan["plan_only"], True)
        self.assertEqual([
            {key: str(value) for key, value in row.items()} for row in ds.queue_plan()
        ], plan["queues"])
        self.assertEqual([], self.runs)
        self.assertEqual(saved, {key: getattr(dq, key) for key in DOWNLOADER_PATHS})

    def test_no_arguments_runs_the_service(self):
        self.assertEqual(0, ds.main([]))
        self.assertEqual(["run"], self.runs)

    def test_import_installs_no_signal_handler(self):
        source = Path(ds.__file__).read_text()
        self.assertNotIn("signal.signal", source.split("def main(", 1)[0])


class ServiceInventoryTests(unittest.TestCase):
    """One unit runs every maintained queue, and it keeps the phase units' sandbox."""

    def unit(self, name):
        settings = {}
        for line in (UNITS / name).read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and not line.startswith("#"):
                settings.setdefault(key, []).append(value)
        return settings

    def test_one_sandboxed_download_unit_runs_the_service(self):
        self.assertEqual(["local-ai-model-downloads.service"],
                         sorted(path.name for path in UNITS.glob("local-ai-model-downloads*")))
        unit = self.unit("local-ai-model-downloads.service")
        self.assertEqual(1, len(unit["ExecStart"]))
        self.assertTrue(unit["ExecStart"][0].endswith(
            "/verification/local-coverage-foundation/download_service.py"), unit["ExecStart"])
        for key, value in (("NoNewPrivileges", "true"), ("PrivateTmp", "true"),
                           ("ProtectSystem", "strict"), ("ProtectHome", "read-only")):
            with self.subTest(setting=key):
                self.assertEqual([value], unit.get(key))
        writable = unit["ReadWritePaths"][0].split()
        self.assertEqual({"models", "verification/local-coverage-foundation"},
                         {path.split("/frankenstein-llm/", 1)[1] for path in writable})

    def test_no_waiter_script_or_phase_unit_ordering_remains(self):
        self.assertEqual([], sorted(path.name for path in HERE.glob("run_download_phase*.py")))
        after = self.unit("local-ai-qualification.service")["After"][0].split()
        self.assertIn("local-ai-model-downloads.service", after)
        self.assertEqual([], [name for name in after if "downloads-phase" in name])


if __name__ == "__main__":
    unittest.main()
