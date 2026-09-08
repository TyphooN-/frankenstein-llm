"""Contracts for the bounded local repository-agent gate. No models, no GPU.

The tests that matter here are the negative ones. A gate that hands an
unreviewed model ``write_file`` and ``run_tests`` is only as good as what it
refuses, so each boundary is exercised from the failing side: a write outside
the allowlist, a symlinked target, an escaped path, a rewritten oracle, a
sandbox that will not start, and a green run that a later write invalidates.

Anything that would otherwise depend on the host -- whether bwrap is installed,
what it prints, whether a filesystem can fsync a directory -- is pinned with a
double, so the same verdict comes out on any machine. The cases that do run real
Bubblewrap are the ones whose whole point is that the boundary is real.
"""
from __future__ import annotations

import contextlib
import json
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
import unittest.mock

import gate_repo_agent as gate

ORACLE_SOURCE = (
    "import unittest\nfrom calculator import add\n"
    "class T(unittest.TestCase):\n"
    "    def test_add(self): self.assertEqual(7, add(3, 4))\n"
)
NEUTERED_ORACLE_SOURCE = (
    "import unittest\n"
    "class T(unittest.TestCase):\n"
    "    def test_nothing(self): pass\n"
)


class RepoAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "calculator.py").write_text("def add(a,b): return a-b\n")
        (self.root / "util.py").write_text("def identity(value): return value\n")
        (self.root / gate.ORACLE_FILENAME).write_text(ORACLE_SOURCE)

    def tearDown(self):
        self.temp.cleanup()

    @contextlib.contextmanager
    def pinned_sandbox(self, **run_behaviour):
        """Pin what running the sandbox does, including that it exists at all.

        ``os.access`` is pinned alongside ``subprocess.run`` on purpose: on a
        machine with no bwrap installed, every one of these cases would pass for
        the wrong reason by refusing at the access check before reaching the
        behaviour under test.
        """
        with unittest.mock.patch.object(gate.os, "access", return_value=True), \
             unittest.mock.patch.object(gate.subprocess, "run", **run_behaviour):
            yield

    def canned_sandbox(self, returncode: int, stdout: str = "", stderr: str = ""):
        """The sandbox ran and the oracle reached this exit code."""
        return self.pinned_sandbox(return_value=subprocess.CompletedProcess(
            gate.sandbox_command(self.root), returncode, stdout=stdout, stderr=stderr))

    def test_path_escape_is_refused(self):
        for path in ("../outside", "/etc/passwd", "sub/../../outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                gate.safe_path(self.root, path)

    def test_write_allowlist_is_enforced(self):
        with self.assertRaises(ValueError):
            gate.execute_tool(self.root, "write_file", {"path": "secret.txt", "content": "x"})

    def test_nested_allowlisted_basename_is_not_an_authorized_path(self):
        (self.root / "nested").mkdir()
        (self.root / "nested/calculator.py").write_text("pass\n")
        with self.assertRaises(ValueError):
            gate.execute_tool(self.root, "write_file",
                              {"path": "nested/calculator.py", "content": "pass\n"})

    def test_symlink_is_not_a_writable_target(self):
        (self.root / "util.py").write_text("pass\n")
        (self.root / "calculator.py").unlink()
        (self.root / "calculator.py").symlink_to("util.py")
        with self.assertRaises(ValueError):
            gate.execute_tool(self.root, "write_file",
                              {"path": "calculator.py", "content": "pass\n"})

    def test_util_module_is_writable_for_multi_file_repairs(self):
        result = gate.execute_tool(
            self.root, "write_file",
            {"path": "util.py", "content": "def identity(value): return value\n"},
        )
        self.assertEqual("util.py", result["path"])

    def test_write_targets_must_already_exist_as_regular_files(self):
        """Allowlisted is not the same as present: create-by-write is not a repair.

        Both of these used to surface as an OSError from ``lstat``, which the
        agent loop reports as an opaque tool error instead of the refusal it is.
        """
        (self.root / "util.py").unlink()
        with self.subTest(target="missing"), self.assertRaises(ValueError):
            gate.execute_tool(self.root, "write_file",
                              {"path": "util.py", "content": "pass\n"})
        (self.root / "util.py").mkdir()
        with self.subTest(target="directory"), self.assertRaises(ValueError):
            gate.execute_tool(self.root, "write_file",
                              {"path": "util.py", "content": "pass\n"})

    def test_a_non_string_path_is_refused_rather_than_raising_a_type_error(self):
        for path in (None, 17, ["calculator.py"]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                gate.execute_tool(self.root, "write_file",
                                  {"path": path, "content": "pass\n"})

    def test_read_file_refuses_a_file_over_the_cap(self):
        # Candidate code writes into this same workspace from inside the
        # sandbox, so the cap has to hold against a file the gate did not write.
        (self.root / "util.py").write_text("x" * (gate.MAX_FILE_BYTES + 1))
        with self.assertRaises(ValueError):
            gate.execute_tool(self.root, "read_file", {"path": "util.py"})

    def test_read_file_refuses_a_symlink_that_leaves_the_workspace(self):
        (self.root / "util.py").unlink()
        (self.root / "util.py").symlink_to("/etc/passwd")
        with self.assertRaises(ValueError):
            gate.execute_tool(self.root, "read_file", {"path": "util.py"})

    def test_a_refused_write_leaves_the_previous_bytes_and_no_debris(self):
        """A write is all-or-nothing, so a green oracle keeps describing real bytes."""
        original = (self.root / "calculator.py").read_text()
        with self.assertRaises(ValueError):
            gate.execute_tool(
                self.root, "write_file",
                {"path": "calculator.py", "content": "x" * (gate.MAX_FILE_BYTES + 1)})
        self.assertEqual(original, (self.root / "calculator.py").read_text())
        self.assertEqual([], sorted(self.root.glob(".staged-*")))

    def test_oracle_is_not_writable(self):
        """The candidate may not edit the tests it is being judged by."""
        with self.assertRaises(ValueError):
            gate.execute_tool(self.root, "write_file",
                              {"path": gate.ORACLE_FILENAME, "content": "pass\n"})

    def test_run_tests_reports_a_tampered_oracle(self):
        oracle = gate.oracle_digest(self.root)
        (self.root / "calculator.py").write_text("def add(a, b): return a + b\n")
        clean = gate.execute_tool(self.root, "run_tests", {}, oracle=oracle)
        self.assertEqual(0, clean["returncode"])
        self.assertTrue(clean["oracle_intact"])

        (self.root / gate.ORACLE_FILENAME).write_text(NEUTERED_ORACLE_SOURCE)
        tampered = gate.execute_tool(self.root, "run_tests", {}, oracle=oracle)
        self.assertEqual(0, tampered["returncode"])
        self.assertFalse(tampered["oracle_intact"])

    def test_a_deleted_oracle_is_reported_as_tampering_not_as_a_tool_error(self):
        """Removing the suite is tampering under a different name.

        This has to stay a verdict rather than an exception: an exception is
        returned to the model as a retryable tool error, and ``oracle_tampered``
        would never be set on a run that destroyed what judges it.
        """
        oracle = gate.oracle_digest(self.root)
        (self.root / gate.ORACLE_FILENAME).unlink()
        with self.canned_sandbox(0, stderr="OK\n"):
            result = gate.execute_tool(self.root, "run_tests", {}, oracle=oracle)
        self.assertEqual(0, result["returncode"])
        self.assertFalse(result["oracle_intact"])

    def test_a_sandbox_setup_failure_is_not_reported_as_a_failing_suite(self):
        """bwrap failing to build the namespace is the host, not the candidate."""
        with self.canned_sandbox(
                1, stderr="bwrap: Creating new namespace failed: Operation not permitted\n"):
            with self.assertRaises(gate.SandboxUnavailable):
                gate.execute_tool(self.root, "run_tests", {})

    def test_a_red_suite_is_a_verdict_and_not_a_sandbox_abort(self):
        """The other half of that rule: a genuinely failing oracle still counts."""
        with self.canned_sandbox(
                1, stderr="FAIL: test_add\nRan 1 test in 0.001s\n\nFAILED (failures=1)\n"):
            result = gate.execute_tool(self.root, "run_tests", {},
                                       oracle=gate.oracle_digest(self.root))
        self.assertEqual(1, result["returncode"])
        self.assertTrue(result["oracle_intact"])

    def test_a_sandbox_that_cannot_be_executed_aborts_rather_than_retrying(self):
        # os.access said yes and the exec then failed: the boundary went away
        # between the check and the run, which is not a candidate mistake.
        with self.pinned_sandbox(side_effect=OSError(8, "Exec format error")):
            with self.assertRaises(gate.SandboxUnavailable):
                gate.execute_tool(self.root, "run_tests", {})

    def run_oracle_with_probe(self, probe: str) -> dict:
        """Run the oracle with hostile candidate code in front of a real repair.

        The probe sits at module scope in calculator.py, which is where a real
        candidate's code gets its chance: the oracle imports it, so the probe
        runs inside the sandbox before a single assertion is evaluated. The
        repair itself is correct, so anything that escapes shows up as a file
        the probe managed to create rather than as a red suite.
        """
        (self.root / "calculator.py").write_text(
            probe + "\n\ndef add(a, b): return a + b\n")
        result = gate.execute_tool(
            self.root, "run_tests", {}, oracle=gate.oracle_digest(self.root))
        self.assertEqual(0, result["returncode"], result["stderr"])
        self.assertTrue(result["oracle_intact"])
        return result

    def test_sandboxed_oracle_cannot_read_host_files(self):
        host = tempfile.TemporaryDirectory()
        self.addCleanup(host.cleanup)
        secret = Path(host.name) / "host-secret.txt"
        secret.write_text("must-not-cross-sandbox")
        self.run_oracle_with_probe(
            "from pathlib import Path\n"
            f"try:\n    leaked = Path({str(secret)!r}).read_text()\n"
            "except OSError:\n    pass\n"
            "else:\n    Path('host-file-leak').write_text(leaked)\n"
        )
        self.assertFalse((self.root / "host-file-leak").exists())

    def test_sandboxed_oracle_cannot_reach_host_loopback_services(self):
        """The router itself listens on host loopback; the candidate must not."""
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        self.run_oracle_with_probe(
            "from pathlib import Path\nimport socket\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1', {port}), timeout=0.5).close()\n"
            "except OSError:\n    pass\n"
            "else:\n    Path('host-loopback-leak').write_text('connected')\n"
        )
        self.assertFalse((self.root / "host-loopback-leak").exists())

    def test_sandboxed_oracle_cannot_see_or_signal_host_processes(self):
        """A private PID namespace: the gate's own process is not even addressable."""
        self.run_oracle_with_probe(
            "import json, os\nfrom pathlib import Path\n"
            "Path('visible-pids').write_text(json.dumps(\n"
            "    sorted(int(e) for e in os.listdir('/proc') if e.isdigit())))\n"
            f"try:\n    os.kill({os.getpid()}, 0)\n"
            "except OSError:\n    pass\n"
            "else:\n    Path('host-process-leak').write_text('signalled')\n"
        )
        self.assertFalse((self.root / "host-process-leak").exists())
        visible = json.loads((self.root / "visible-pids").read_text())
        # Only the sandbox's own init and the oracle it spawned are addressable.
        # An empty list would mean the probe never read a /proc at all, which is
        # not the thing being proven, so it fails rather than passing vacuously.
        self.assertTrue(visible, "the probe saw no /proc entries at all")
        self.assertLessEqual(len(visible), 2, visible)
        self.assertLess(max(visible), 3, visible)

    def test_a_missing_sandbox_refuses_instead_of_running_on_the_host(self):
        """There is no unisolated fallback: no sandbox means no run at all."""
        (self.root / "calculator.py").write_text(
            "from pathlib import Path\n"
            "Path('candidate-code-ran').write_text('unsandboxed')\n"
            "def add(a, b): return a + b\n"
        )
        with unittest.mock.patch.object(gate, "SANDBOX", Path("/nonexistent/bwrap")):
            with self.assertRaises(gate.SandboxUnavailable):
                gate.execute_tool(self.root, "run_tests", {})
            with self.assertRaises(SystemExit):
                gate.require_sandbox()
        self.assertFalse((self.root / "candidate-code-ran").exists())

    def test_sandbox_timeout_aborts_the_agent_instead_of_becoming_a_tool_error(self):
        replies = iter([{
            "choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "run_tests", "arguments": "{}"},
            }]}}],
        }])
        expired = subprocess.TimeoutExpired(gate.sandbox_command(self.root), 30)
        with self.pinned_sandbox(side_effect=expired):
            with self.assertRaises(gate.SandboxUnavailable):
                gate.run_agent(self.root, lambda _messages: next(replies))

    def test_a_tool_error_still_returns_to_the_model(self):
        """Only infrastructure aborts. An ordinary refusal is the model's to fix."""
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "write_file",
                             "arguments": json.dumps({"path": "secret.txt",
                                                      "content": "x"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "understood"}}]},
        ])
        result = gate.run_agent(self.root, lambda _messages: next(replies))
        self.assertFalse(result["events"][0]["ok"])
        self.assertIn("ValueError", result["events"][0]["error"])
        self.assertFalse(result["pass"])

    def test_a_malformed_tool_call_does_not_end_the_qualification(self):
        # A reply the router shaped badly is the model getting it wrong, not the
        # boundary failing, so it comes back as a tool error like any other.
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [
                {"id": "1", "type": "function"}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "sorry"}}]},
        ])
        result = gate.run_agent(self.root, lambda _messages: next(replies))
        self.assertFalse(result["events"][0]["ok"])
        self.assertFalse(result["pass"])

    def test_the_sandbox_profile_keeps_every_isolation_flag(self):
        options = gate.sandbox_identity(self.root)["options"]
        for flag in ("--unshare-all", "--unshare-user", "--disable-userns",
                     "--cap-drop", "--clearenv", "--die-with-parent",
                     "--new-session", "--ro-bind", "--proc", "--tmpfs", "--size"):
            with self.subTest(flag=flag):
                self.assertIn(flag, options)
        self.assertNotIn("--share-net", options)

    def test_the_sandbox_binds_only_the_workspace_and_runs_only_the_oracle(self):
        """The argv is the boundary; asserting the flag names alone would miss it.

        ``sandbox_identity`` records option names, so a bind that pointed
        somewhere else entirely would leave the recorded profile unchanged.
        """
        argv = gate.sandbox_command(self.root)
        self.assertEqual(str(gate.SANDBOX), argv[0])
        self.assertEqual(
            ["/usr/bin/python3", "-m", "unittest", "-v", gate.ORACLE_FILENAME],
            argv[-5:])
        self.assertEqual(1, argv.count("--bind"), "exactly one writable bind")
        bind = argv.index("--bind")
        self.assertEqual([str(self.root), gate.SANDBOX_WORKSPACE], argv[bind + 1:bind + 3])
        self.assertEqual(gate.SANDBOX_WORKSPACE, argv[argv.index("--chdir") + 1])
        self.assertEqual(["/usr"], [argv[index + 1] for index, item in enumerate(argv)
                                    if item == "--ro-bind"])
        # --clearenv has to precede every --setenv: bwrap applies them in order,
        # so the reverse would discard the environment it was meant to pin.
        self.assertLess(argv.index("--clearenv"), argv.index("--setenv"))
        # The tmpfs is sized, so candidate code cannot page the host out through
        # the one writable filesystem it does not need.
        self.assertEqual(str(gate.SANDBOX_TMPFS_BYTES), argv[argv.index("--size") + 1])
        self.assertEqual("/tmp", argv[argv.index("--tmpfs") + 1])

    def test_unknown_tool_is_refused(self):
        with self.assertRaises(ValueError):
            gate.execute_tool(self.root, "shell", {"command": "id"})

    def test_agent_requires_passing_tests(self):
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "calculator.py"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "calculator.py", "content": "def add(a,b): return a+b\n"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        ])
        result = gate.run_agent(self.root, lambda _messages: next(replies))
        self.assertFalse(result["pass"])
        self.assertFalse(result["saw_passing_tests"])

    def test_agent_passes_only_after_real_test_tool_success(self):
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "calculator.py", "content": "def add(a,b): return a+b\n"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "tests pass"}}]},
        ])
        result = gate.run_agent(self.root, lambda _messages: next(replies))
        self.assertTrue(result["pass"])
        self.assertTrue(result["saw_passing_tests"])

    def test_write_after_green_tests_invalidates_the_verdict(self):
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "calculator.py", "content": "def add(a,b): return a+b\n"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "3", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "calculator.py", "content": "def add(a,b): return a-b\n"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        ])
        result = gate.run_agent(self.root, lambda _messages: next(replies))
        self.assertFalse(result["pass"])
        self.assertFalse(result["saw_passing_tests"])

    def test_a_refused_write_after_green_tests_keeps_the_verdict(self):
        """Only a *completed* write ends the generation the green run described.

        The write is atomic, so a refused one leaves the exact bytes the oracle
        already passed on. Invalidating there would make the gate unable to
        qualify anything a model fumbled once on the way to a correct repair.
        """
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "calculator.py", "content": "def add(a,b): return a+b\n"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "3", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "secret.txt", "content": "x"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        ])
        result = gate.run_agent(self.root, lambda _messages: next(replies))
        self.assertFalse(result["events"][2]["ok"])
        self.assertTrue(result["saw_passing_tests"])
        self.assertTrue(result["pass"])

    def test_tampered_oracle_never_qualifies(self):
        """A green run against a rewritten oracle is not a repair."""
        def call(_messages):
            # The harness stands in for a model that neutered the test suite by
            # some route other than write_file, then ran the suite.
            (self.root / gate.ORACLE_FILENAME).write_text(NEUTERED_ORACLE_SOURCE)
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [
                {"id": "1", "type": "function",
                 "function": {"name": "run_tests", "arguments": "{}"}}]}}]}

        result = gate.run_agent(self.root, call)
        self.assertTrue(result["oracle_tampered"])
        self.assertFalse(result["saw_passing_tests"])
        self.assertFalse(result["pass"])

    def test_clean_repair_records_the_oracle_digest(self):
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "calculator.py", "content": "def add(a,b): return a+b\n"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "tests pass"}}]},
        ])
        result = gate.run_agent(self.root, lambda _messages: next(replies))
        self.assertTrue(result["pass"])
        self.assertFalse(result["oracle_tampered"])
        self.assertEqual(gate.oracle_digest(self.root), result["oracle_sha256"])


class EvidenceLifecycleTests(unittest.TestCase):
    """A verdict on disk outlives the run, so superseding one has to be exact."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        (self.root / "calculator.py").write_text("def add(a,b): return a-b\n")
        (self.root / "util.py").write_text("def identity(value): return value\n")
        (self.root / gate.ORACLE_FILENAME).write_text(ORACLE_SOURCE)
        self.evidence = Path(self.temp.name) / "evidence"
        self.path = self.evidence / "gate-repo-agent-heretic.json"
        self.patch = unittest.mock.patch.object(gate, "EVIDENCE", self.evidence)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temp.cleanup)

    def stale_pass(self) -> None:
        """The verdict a previous run left behind, which no rerun may inherit."""
        gate.publish_evidence(self.path, {
            "gate": "local-repository-agent", "model": "heretic",
            "run_id": "old", "pass": True, "interrupted": False,
            "status": "complete", "finished_at": "2026-09-04T08:00:00-0400",
        })

    def test_http_error_publishes_a_terminal_failure(self):
        import urllib.error
        from email.message import Message
        self.stale_pass()

        def fail(_messages):
            raise urllib.error.HTTPError('http://127.0.0.1:8080', 500, 'error', Message(), None)

        result = gate.qualify(self.root, 'heretic', fail)
        current = json.loads(self.path.read_text())
        self.assertFalse(result['pass'])
        self.assertEqual('failed', current['status'])
        self.assertFalse(current['interrupted'])
        self.assertEqual(500, current['http_status'])
        self.assertIn('finished_at', current)
        self.assertNotEqual('old', current['run_id'])

    def test_new_attempt_invalidates_an_older_pass_before_router_failure(self):
        self.stale_pass()

        def fail(_messages):
            raise RuntimeError("router unavailable")

        with self.assertRaises(RuntimeError):
            gate.qualify(self.root, "heretic", fail)
        current = json.loads(self.path.read_text())
        self.assertFalse(current["pass"])
        self.assertTrue(current["interrupted"])
        self.assertEqual("running", current["status"])
        self.assertNotEqual("old", current["run_id"])

    def test_a_sandbox_abort_leaves_the_interrupted_record_rather_than_the_old_pass(self):
        # Aborting is the right answer to a broken boundary, but it must not be
        # the route by which yesterday's pass survives into today's ledger.
        self.stale_pass()

        def abort(_messages):
            raise gate.SandboxUnavailable("bwrap could not create a namespace")

        with self.assertRaises(gate.SandboxUnavailable):
            gate.qualify(self.root, "heretic", abort)
        current = json.loads(self.path.read_text())
        self.assertFalse(current["pass"])
        self.assertTrue(current["interrupted"])
        self.assertNotEqual("old", current["run_id"])

    def test_a_completed_run_supersedes_the_previous_verdict(self):
        self.stale_pass()
        result = gate.qualify(self.root, "heretic", lambda _messages: {
            "choices": [{"message": {"role": "assistant", "content": "nothing to do"}}]})
        current = json.loads(self.path.read_text())
        self.assertFalse(result["pass"])
        self.assertFalse(current["pass"])
        self.assertEqual("complete", current["status"])
        self.assertFalse(current["interrupted"])
        self.assertNotEqual("old", current["run_id"])

    def test_the_verdict_and_its_directory_entry_are_both_fsynced(self):
        # A rename is atomic but not durable. Syncing the file and not the
        # directory lets the rename evaporate on power loss, which resurrects
        # exactly the stale pass this whole lifecycle exists to retire.
        synced = []
        real_fsync = os.fsync

        def recording(descriptor):
            synced.append(os.fstat(descriptor).st_mode & 0o170000)
            real_fsync(descriptor)

        with unittest.mock.patch.object(gate.os, "fsync", recording):
            gate.publish_evidence(self.path, {"gate": "local-repository-agent",
                                              "pass": False})
        self.assertIn(0o100000, synced, "the verdict file was never fsynced")
        self.assertIn(0o040000, synced, "the directory entry was never fsynced")

    def test_a_filesystem_that_refuses_directory_fsync_still_publishes(self):
        def only_dir_fails(descriptor):
            if os.fstat(descriptor).st_mode & 0o170000 == 0o040000:
                raise OSError("directory fsync unsupported")

        with unittest.mock.patch.object(gate.os, "fsync", only_dir_fails):
            gate.publish_evidence(self.path, {"gate": "local-repository-agent",
                                              "pass": True})
        self.assertTrue(json.loads(self.path.read_text())["pass"])

    def test_a_failed_publish_leaves_the_previous_verdict_and_no_debris(self):
        self.stale_pass()
        with unittest.mock.patch.object(gate.os, "replace",
                                        side_effect=OSError("no space left on device")):
            with self.assertRaises(OSError):
                gate.publish_evidence(self.path, {"gate": "local-repository-agent",
                                                  "pass": False})
        self.assertEqual(["gate-repo-agent-heretic.json"],
                         sorted(item.name for item in self.evidence.iterdir()))
        self.assertTrue(json.loads(self.path.read_text())["pass"])

    def test_publishing_replaces_the_verdict_and_leaves_no_partial_file(self):
        """Reboot semantics: exactly one artifact, and it is a whole document."""
        gate.publish_evidence(self.path, {"gate": "local-repository-agent", "pass": False})
        gate.publish_evidence(self.path, {"gate": "local-repository-agent", "pass": True})
        self.assertEqual(["gate-repo-agent-heretic.json"],
                         sorted(item.name for item in self.evidence.iterdir()))
        self.assertTrue(json.loads(self.path.read_text())["pass"])

    def test_evidence_names_the_sandbox_that_isolated_the_run(self):
        """Both the in-progress and the final record state their own trust boundary."""
        captured: list[dict] = []

        def call(_messages):
            captured.append(json.loads(self.path.read_text()))
            return {"choices": [{"message": {"role": "assistant", "content": "no"}}]}

        result = gate.qualify(self.root, "heretic", call)
        for record in (captured[0], result, json.loads(self.path.read_text())):
            self.assertEqual("bubblewrap", record["sandbox"]["tool"])
            self.assertEqual(gate.SANDBOX_WORKSPACE, record["sandbox"]["workspace"])
            self.assertIn("--unshare-all", record["sandbox"]["options"])

    def test_completed_evidence_has_a_ledger_accepted_timestamp(self):
        replies = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "calculator.py", "content": "def add(a,b): return a+b\n"})}}]}}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        ])
        result = gate.qualify(self.root, "heretic", lambda _messages: next(replies))
        self.assertTrue(result["pass"])

        ledger_source = (Path(__file__).parents[1]
                         / "local-coverage-foundation/build_capability_ledger.py")
        spec = importlib.util.spec_from_file_location("repo_agent_ledger_test", ledger_source)
        assert spec is not None and spec.loader is not None
        ledger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ledger)
        record = ledger.read_evidence(self.path, "local-repository-agent")
        state, reasons = ledger.classify([], [record])
        self.assertEqual(ledger.STATE_QUALIFIED, state, reasons)
        self.assertFalse(record["throughput_measured"])


class ModelSelectionTests(unittest.TestCase):
    """This gate hands out executable tools, so it is picky about who gets them."""

    def test_the_incumbent_and_the_reviewed_candidate_are_both_admitted(self):
        for model in ("heretic", "qwen3-coder-next"):
            with self.subTest(model=model):
                self.assertEqual(model, gate.resolve_model(model))

    def test_an_unlisted_preset_is_refused(self):
        for model in ("ridge", "phr00ty", "", "../heretic"):
            with self.subTest(model=model):
                with self.assertRaises(SystemExit):
                    gate.resolve_model(model)

    def test_a_low_privilege_candidate_can_never_be_selected(self):
        # Gemma-4 Heretic is admitted elsewhere as a multimodal reader. Routing
        # it here would hand ablated weights write_file and run_tests.
        for model in ("gemma4-heretic", "gemma4-heretic-vision"):
            with self.subTest(model=model):
                with self.assertRaises(SystemExit) as refused:
                    gate.resolve_model(model)
                self.assertIn("low-privilege", str(refused.exception))

    def test_the_selected_model_reaches_the_router_payload(self):
        captured: dict = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        def fake_urlopen(request, timeout=None):
            captured["payload"] = json.loads(request.data)
            return FakeResponse()

        with unittest.mock.patch.object(gate.urllib.request, "urlopen", fake_urlopen):
            gate.router_call([{"role": "user", "content": "hi"}], "qwen3-coder-next")
        self.assertEqual("qwen3-coder-next", captured["payload"]["model"])
        self.assertEqual(gate.TOOLS, captured["payload"]["tools"])


if __name__ == "__main__":
    unittest.main()
