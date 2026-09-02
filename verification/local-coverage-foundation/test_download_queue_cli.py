"""CLI contract for the single-writer downloader.

The queue takes a lock, rewrites state, appends to a log and moves
multi-gigabyte files. Before this contract existed ``--help`` fell straight
through to ``main`` and did all of that, so the tests below are written to prove
absence of side effects, not just the exit code.

Nothing here runs the real queue: ``main`` is replaced by a recorder, and every
path constant is redirected into a temporary directory that is asserted to stay
empty.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import download_queue as dq

SIDE_EFFECT_PATHS = ("QUEUE", "STATE", "LOCK", "STAMP", "LOG")


class CliContractTests(unittest.TestCase):
    """--help and bad usage must not reach main, and must leave no trace."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp.name)
        self.calls = []
        # Every path the queue would write to is redirected here. If the CLI
        # touches anything, it lands in this directory and the assertion fails.
        patches = {name: self.sandbox / f"{name.lower()}.file"
                   for name in SIDE_EFFECT_PATHS}
        for name, path in patches.items():
            patcher = mock.patch.object(dq, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        main_patch = mock.patch.object(
            dq, "main", lambda: self.calls.append("main") or 0)
        main_patch.start()
        self.addCleanup(main_patch.stop)
        # run() installs a SIGHUP handler on the dispatch path; put it back so a
        # passing test cannot change the behaviour of the process running it.
        original = signal.getsignal(signal.SIGHUP)
        self.addCleanup(signal.signal, signal.SIGHUP, original)

    def tearDown(self):
        self.temp.cleanup()

    def assertNoSideEffects(self):
        self.assertEqual([], sorted(p.name for p in self.sandbox.iterdir()),
                         "the CLI created a file it should never have touched")
        self.assertEqual([], self.calls, "main() was reached")

    def test_help_prints_usage_exits_zero_and_touches_nothing(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag), mock.patch("builtins.print") as printed:
                self.assertEqual(0, dq.run([flag]))
                printed.assert_called_once()
                self.assertIn("usage:", printed.call_args.args[0])
            self.assertNoSideEffects()

    def test_unknown_arguments_exit_nonzero_and_touch_nothing(self):
        for argv in (["--bogus"], ["--hel"], ["-h", "--extra"], ["--help=x"],
                     ["positional"], ["--help", "--help"], ["-"], [""]):
            with self.subTest(argv=argv), mock.patch("builtins.print"):
                code = dq.run(argv)
            self.assertNotEqual(0, code, f"{argv} was accepted")
            self.assertEqual(2, code)
            self.assertNoSideEffects()

    def test_no_arguments_dispatches_main(self):
        self.assertEqual(0, dq.run([]))
        self.assertEqual(["main"], self.calls)

    def test_dispatch_returns_the_exit_code_main_reports(self):
        with mock.patch.object(dq, "main", return_value=75):
            self.assertEqual(75, dq.run([]))

    def test_parse_args_reports_run_only_for_no_arguments(self):
        self.assertIsNone(dq.parse_args([]))
        with mock.patch("builtins.print"):
            self.assertEqual(0, dq.parse_args(["--help"]))
            self.assertEqual(2, dq.parse_args(["--bogus"]))

    def test_help_text_documents_the_environment_contract(self):
        for name in SIDE_EFFECT_PATHS:
            self.assertIn(f"HERMES_DOWNLOAD_{name}", dq.USAGE)


class RealEntryPointTests(unittest.TestCase):
    """Exercise the actual __main__ dispatch, never with zero arguments.

    Zero arguments is the one invocation that legitimately processes the queue,
    so it is proven above with main() replaced rather than executed here.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp.name)
        self.env = dict(os.environ)
        self.env.update({
            f"HERMES_DOWNLOAD_{name}": str(self.sandbox / f"{name.lower()}.file")
            for name in SIDE_EFFECT_PATHS
        })

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *argv):
        return subprocess.run(
            [sys.executable, str(Path(dq.__file__).resolve()), *argv],
            capture_output=True, text=True, timeout=60, env=self.env,
            cwd=self.sandbox)

    def test_help_exits_zero_and_writes_no_file(self):
        result = self.invoke("--help")
        self.assertEqual(0, result.returncode, result.stderr[-400:])
        self.assertIn("usage:", result.stdout)
        self.assertIn("Takes no arguments", result.stdout)
        self.assertEqual([], list(self.sandbox.iterdir()))

    def test_unknown_argument_exits_two_and_writes_no_file(self):
        result = self.invoke("--start-everything")
        self.assertEqual(2, result.returncode)
        self.assertIn("unexpected argument", result.stderr)
        self.assertEqual("", result.stdout)
        self.assertEqual([], list(self.sandbox.iterdir()))

    def test_help_does_not_read_the_real_queue(self):
        """The redirected queue path does not exist, so a reader would crash."""
        result = self.invoke("--help")
        self.assertEqual(0, result.returncode)
        self.assertNotIn("Traceback", result.stderr)


class ProductionQueueUntouchedTests(unittest.TestCase):
    """The real workspace queue state must be unchanged by running these tests."""

    def test_real_state_and_stamp_are_not_written(self):
        root = Path(dq.__file__).resolve().parent
        before = {}
        for name in ("download-state.json", "downloads-complete.ok",
                     "download-queue.lock", "downloads.log"):
            path = root / name
            before[name] = path.stat().st_mtime_ns if path.exists() else None
        with mock.patch("builtins.print"):
            dq.run(["--help"])
            dq.run(["--bogus"])
        for name, mtime in before.items():
            path = root / name
            now = path.stat().st_mtime_ns if path.exists() else None
            self.assertEqual(mtime, now, f"{name} was modified")

    def test_module_import_installs_no_signal_handler(self):
        """Importing for a test must not change the process's SIGHUP disposition."""
        source = Path(dq.__file__).read_text()
        head = source.split("def log(", 1)[0]
        self.assertNotIn("signal.signal", head,
                         "SIGHUP is installed at import time, which leaks into "
                         "any process that merely imports this module")


if __name__ == "__main__":
    unittest.main()
