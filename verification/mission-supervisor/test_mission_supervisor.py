"""Contracts for the serialized qualification supervisor. No models, no GPU.

Three behaviours decide whether this supervisor can be trusted to run unattended
across reboots and operator stops:

* the quiet-host wait is bounded, and says what the host was busy with;
* mission state reaches the disk durably, because the resume decision after a
  reboot is made from it;
* an operator stop is classified as an interruption rather than a step failure,
  without discarding a step that genuinely passed.

Everything the supervisor would touch outside the process -- /proc, systemd, the
step subprocess, sleeping -- is replaced with a double. The module is imported
with STATE and LOG redirected into a sandbox so importing it can never write to
the real mission state.

Run: python3 -m unittest discover -s . -p 'test_*.py' -v
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings

SOURCE = Path(__file__).resolve().parent / "run_functional_mission.py"

# One conflict entry in the shape conflicts() actually produces. Shared so the
# wait tests and the discovery tests cannot drift apart on what an entry is.
BUSY = [{"pid": 4242, "reason": "inference", "command": "llama-server",
         "cwd": "/tmp", "cgroup": "0::/user.slice/hand.scope"}]


def load_supervisor(sandbox: Path):
    """Import the supervisor with its state and log redirected into ``sandbox``."""
    spec = importlib.util.spec_from_file_location("supervisor_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.HERE = sandbox
    module.STATE = sandbox / "mission-state.json"
    module.LOG = sandbox / "mission.log"
    module.LOCK = sandbox / "mission.lock"
    return module


class SupervisorTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.supervisor = load_supervisor(self.sandbox)
        self.addCleanup(setattr, self.supervisor, "stop_signal", None)


class QuietTimeoutConfigurationTests(SupervisorTestCase):
    def test_unset_and_empty_use_the_six_hour_default(self):
        for raw in (None, ""):
            with self.subTest(raw=raw):
                self.assertEqual(6 * 3600, self.supervisor.quiet_timeout(raw))

    def test_an_explicit_override_is_honoured(self):
        self.assertEqual(900, self.supervisor.quiet_timeout("900"))

    def test_unusable_overrides_fail_closed_rather_than_silently_defaulting(self):
        # Falling back to the default here would hide a typo in the unit file and
        # quietly restore the unbounded-feeling six-hour wait the operator meant
        # to shorten.
        for raw in ("0", "-1", "soon", "3.5"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    self.supervisor.quiet_timeout(raw)


class QuietWaitTests(SupervisorTestCase):
    """The quiet-host wait must end, and must explain why it gave up."""

    def arrange(self, polls, available=64 << 30, load=0.1):
        """Feed ``conflicts()`` one entry per poll; everything else stays quiet."""
        self.slept = []
        self.clock = [0.0]

        def fake_sleep(seconds):
            self.slept.append(seconds)
            self.clock[0] += seconds

        conflicts = mock.Mock(side_effect=list(polls))
        self.log = mock.Mock()
        return (
            mock.patch.object(self.supervisor, "conflicts", conflicts),
            mock.patch.object(self.supervisor, "mem_available", return_value=available),
            mock.patch.object(self.supervisor.os, "getloadavg", return_value=(load, load, load)),
            mock.patch.object(self.supervisor.time, "sleep", fake_sleep),
            mock.patch.object(self.supervisor.time, "monotonic", lambda: self.clock[0]),
            mock.patch.object(self.supervisor, "log", self.log),
        )

    def run_wait(self, polls, timeout, state=None, **kwargs):
        state = {"steps": {}} if state is None else state
        patches = self.arrange(polls, **kwargs)
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.supervisor.wait_for_quiet(state, timeout=timeout)
        return state

    def test_two_consecutive_quiet_polls_release_the_wait(self):
        state = self.run_wait([[], []], timeout=3600)
        self.assertEqual("waiting-safe-host", state["status"])
        self.assertEqual([self.supervisor.POLL_SECONDS], self.slept)

    def test_a_busy_poll_resets_the_quiet_counter(self):
        # One quiet poll is not enough: a gate that has just released the GPU can
        # look quiet for a moment while the next one is still starting.
        self.run_wait([[], BUSY, [], []], timeout=3600)
        self.assertEqual(3, len(self.slept))

    def test_a_permanently_busy_host_fails_instead_of_waiting_forever(self):
        with self.assertRaises(RuntimeError) as caught:
            self.run_wait([BUSY] * 40, timeout=120)
        self.assertIn("did not become quiet within 120s", str(caught.exception))

    def test_the_failure_names_what_the_host_was_busy_with(self):
        with self.assertRaises(RuntimeError) as caught:
            self.run_wait([BUSY] * 40, timeout=60)
        message = str(caught.exception)
        self.assertIn("conflicts=", message)
        self.assertIn("4242", message)
        self.assertIn("inference", message)
        self.assertNotIn("unknown", message)

    def test_large_compiler_fanout_is_bounded_and_reports_omissions(self):
        active = [
            {"pid": number, "reason": "build", "command": "cc1plus",
             "cwd": "/tmp/build", "cgroup": "test"}
            for number in range(1000, 1200)
        ]
        summary = self.supervisor.summarize_conflicts(active)
        self.assertIn("count=200", summary)
        self.assertIn("build:200", summary)
        self.assertIn("omitted=192", summary)
        self.assertLess(len(summary), 600)

    def test_pid_churn_in_one_blocker_class_does_not_spam_the_log(self):
        polls = []
        for number in range(10):
            polls.append([{"pid": 5000 + number, "reason": "inference",
                           "command": "llama-server", "cwd": "", "cgroup": "test"}])
        with self.assertRaises(RuntimeError):
            self.run_wait(polls, timeout=60)
        waiting = [call for call in self.log.call_args_list
                   if "waiting for safe host" in call.args[0]]
        self.assertEqual(1, len(waiting))

    def test_low_memory_is_reported_as_the_blocker(self):
        with self.assertRaises(RuntimeError) as caught:
            self.run_wait([[]] * 40, timeout=60, available=1 << 30, load=99.0)
        message = str(caught.exception)
        self.assertIn("MemAvailable=", message)
        self.assertNotIn("load1=", message)

    def test_desktop_load_and_unrelated_builds_do_not_block_the_wait(self):
        cargo = [{"pid": 99, "reason": "build", "command": "rustc",
                  "cwd": "/home/typhoon/git/xiphercash", "cgroup": "test"}]
        state = self.run_wait([cargo, cargo], timeout=3600, load=99.0)
        self.assertEqual("waiting-safe-host", state["status"])

    def test_kernel_tree_builds_still_block(self):
        busy = [{"pid": 7, "reason": "kernel-build", "command": "bash",
                 "cwd": "/home/typhoon/git/linux-tkg/src", "cgroup": "test"}]
        with self.assertRaises(RuntimeError) as caught:
            self.run_wait([busy] * 40, timeout=60)
        self.assertIn("kernel-build", str(caught.exception))

    def test_the_timeout_is_recorded_in_state_before_it_raises(self):
        # The unit dies on this exception, so the durable record of why has to be
        # written first; otherwise the state file still says "waiting-safe-host".
        state = {"steps": {}}
        with self.assertRaises(RuntimeError):
            self.run_wait([BUSY] * 40, timeout=60, state=state)
        self.assertEqual("failed", state["status"])
        published = json.loads(self.supervisor.STATE.read_text())
        self.assertEqual("failed", published["status"])
        self.assertIn("did not become quiet", published["error"])


class ConflictDiscoveryTests(SupervisorTestCase):
    """Conflict discovery must never read another process's address space.

    Every case runs against an injected proc root, so the classification is
    exercised without needing the host to be busy in any particular way -- and
    without the suite depending on which processes happen to be running.
    """

    def setUp(self):
        super().setUp()
        self.proc = self.sandbox / "proc"
        self.proc.mkdir()
        # Fake PIDs start above this process's own, which conflicts() skips.
        # Reusing it by accident would silently defeat half of these tests.
        self.pids = itertools.count(os.getpid() + 1)

    def process(self, comm: str, cwd="/tmp", cgroup="0::/user.slice/test.scope\n",
                argv: str = "sleep\x0060\x00", pid: int | None = None) -> int:
        """Write one fake /proc entry and return its pid.

        ``argv`` is what ``/proc/<pid>/cmdline`` would have said, and it always
        disagrees with ``comm``. That disagreement is the point: a regression
        back to reading cmdline changes the verdict and fails a test here,
        rather than hiding behind a blocking read that never returns.
        """
        pid = next(self.pids) if pid is None else pid
        entry = self.proc / str(pid)
        entry.mkdir()
        (entry / "comm").write_text(f"{comm}\n")
        (entry / "cgroup").write_text(cgroup)
        (entry / "cwd").symlink_to(str(cwd))
        (entry / "cmdline").write_text(argv)
        return pid

    def found(self) -> list[dict]:
        return self.supervisor.conflicts(self.proc)

    def pids_found(self) -> list[int]:
        return [entry["pid"] for entry in self.found()]

    def test_the_reported_command_is_the_kernel_task_name_not_the_cmdline(self):
        pid = self.process("cc1plus", cwd="/tmp/build", argv="sleep\x0060\x00")
        found = self.found()
        self.assertEqual([pid], [entry["pid"] for entry in found])
        self.assertEqual("cc1plus", found[0]["command"])
        self.assertEqual("build", found[0]["reason"])

    def test_a_quiet_task_name_is_not_rescued_by_a_busy_cmdline(self):
        # The deterministic other half: reading cmdline would report this one.
        self.process("sleep", cwd="/tmp", argv="cc1plus\x00-O2\x00")
        self.assertEqual([], self.pids_found())

    def test_managed_router_is_excluded_but_direct_server_is_a_conflict(self):
        self.process("llama-server", cgroup="0::/user.slice/llama-router.service\n")
        direct = self.process("llama-server")
        self.assertEqual([direct], self.pids_found())

    def test_llama_helpers_are_conflicts_under_their_truncated_task_names(self):
        # /proc/<pid>/comm is 15 characters, so "llama-perplexity" never arrives
        # whole. Matching the family prefix is what survives that.
        for comm in ("llama-bench", "llama-cli", "llama-perplexi", "llama-embedding"):
            with self.subTest(comm=comm):
                pid = self.process(comm)
                self.assertIn(pid, self.pids_found())

    def test_builds_and_transfers_are_detected_by_task_name_alone(self):
        for comm, reason in (("gcc", "build"), ("g++", "build"), ("cc1", "build"),
                             ("ld.lld", "build"), ("makepkg", "build"),
                             ("ninja", "build"), ("rustc", "build"),
                             ("aria2c", "transfer"), ("rsync", "transfer"),
                             ("huggingface-cli", "transfer"),
                             ("git-remote-http", "transfer")):
            with self.subTest(comm=comm):
                pid = self.process(comm)
                match = [entry for entry in self.found() if entry["pid"] == pid]
                self.assertEqual([reason], [entry["reason"] for entry in match])

    def test_a_kernel_tree_build_is_a_conflict_whatever_the_task_name(self):
        # The kernel tree is recognised wherever it is checked out, and by the
        # directory rather than the task name: most of a kernel build is shells,
        # and the ones that are not are already gone by the next poll.
        pid = self.process("bash", cwd=f"/home/typhoon/git{self.supervisor.KERNEL_TREE}/src")
        self.assertEqual([pid], self.pids_found())
        self.assertEqual("kernel-build", self.found()[0]["reason"])

    def test_python_in_repository_is_a_fail_closed_conflict(self):
        inside = self.process("python3", cwd=self.supervisor.ROOT / "verification")
        self.process("python3", cwd="/tmp/unrelated")
        self.assertEqual([inside], self.pids_found())

    def test_a_deleted_working_directory_still_places_the_process(self):
        # /proc reports a removed cwd as "<path> (deleted)". A build that just
        # had its tree rm -rf'd out from under it is still running.
        pid = self.process("python3", cwd="/tmp/gone")
        (self.proc / str(pid) / "cwd").unlink()
        (self.proc / str(pid) / "cwd").symlink_to(
            f"{self.supervisor.ROOT}/verification (deleted)")
        self.assertEqual([pid], self.pids_found())

    def test_an_idle_host_reports_nothing(self):
        for comm in ("systemd", "firefox", "sshd", "kworker/0:1", "sleep"):
            self.process(comm, cwd="/home/typhoon")
        self.assertEqual([], self.pids_found())

    def test_the_supervisor_does_not_report_itself(self):
        self.process("python3", cwd=self.supervisor.ROOT, pid=os.getpid())
        self.assertEqual([], self.pids_found())

    def test_a_task_that_exits_mid_scan_is_skipped_rather_than_raising(self):
        vanished = self.proc / str(next(self.pids))
        vanished.mkdir()  # listed by the scan, with nothing left to read
        busy = self.process("cc1plus")
        self.assertEqual([busy], self.pids_found())

    def test_an_unreadable_working_directory_does_not_lose_the_process(self):
        pid = self.process("aria2c")
        (self.proc / str(pid) / "cwd").unlink()
        self.assertEqual([pid], self.pids_found())
        self.assertEqual("", self.found()[0]["cwd"])

    def test_a_task_name_that_is_not_utf8_does_not_crash_the_scan(self):
        # comm is whatever bytes the process chose. An undecodable one used to
        # raise out of the poll, which is a wedged mission rather than a report.
        odd = self.proc / str(next(self.pids))
        odd.mkdir()
        (odd / "comm").write_bytes(b"\xff\xfe-broken\n")
        (odd / "cgroup").write_text("0::/user.slice/test.scope\n")
        (odd / "cwd").symlink_to("/tmp")
        busy = self.process("cc1plus")
        self.assertEqual([busy], self.pids_found())

    def test_a_non_numeric_proc_entry_is_ignored(self):
        odd = self.proc / "9nvidia"
        odd.mkdir()
        (odd / "comm").write_text("cc1plus\n")
        self.assertEqual([], self.pids_found())

    def test_an_unreadable_process_table_is_not_a_quiet_host(self):
        # Returning [] here would report "the host is idle" on the strength of
        # having been unable to look at it.
        with self.assertRaises(RuntimeError):
            self.supervisor.conflicts(self.sandbox / "no-such-proc")

    def test_reported_entries_carry_the_metadata_the_operator_needs(self):
        pid = self.process("llama-server", cwd="/tmp", cgroup="0::/user.slice/hand.scope\n")
        entry = self.found()[0]
        self.assertEqual(
            {"pid": pid, "reason": "inference", "command": "llama-server",
             "cwd": "/tmp", "cgroup": "0::/user.slice/hand.scope"},
            entry)


class DurableStateTests(SupervisorTestCase):
    """A resumed mission skips passed steps based on this file."""

    def test_input_fingerprint_changes_when_a_promoted_artifact_changes(self):
        foundation = self.sandbox / "foundation"
        foundation.mkdir()
        state_path = foundation / "download-state.json"
        stamp_path = foundation / "complete.ok"
        state_path.write_text('{"status":"complete"}')
        stamp_path.write_text("4")
        weight = self.sandbox / "weight.bin"
        weight.write_bytes(b"1234")
        queue = foundation / "download-queue.json"
        queue.write_text(json.dumps({"artifacts": [{"files": [{
            "destination": str(weight), "size": 4}]}]}))
        completed = mock.Mock(stdout=b"source-generation")
        with mock.patch.object(self.supervisor, "ROOT", self.sandbox), \
             mock.patch.object(self.supervisor, "FOUNDATION", foundation), \
             mock.patch.object(self.supervisor, "UPSTREAM", ((state_path, stamp_path, "4"),)), \
             mock.patch.object(self.supervisor, "STEPS", (("step", ["/bin/true"]),)), \
             mock.patch.object(self.supervisor.subprocess, "run", return_value=completed):
            first = self.supervisor.mission_inputs_fingerprint()
            stat = weight.stat()
            os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            second = self.supervisor.mission_inputs_fingerprint()
        self.assertNotEqual(first, second)

    def test_upstream_artifacts_that_have_not_landed_yet_are_fingerprintable(self):
        # main() fingerprints before wait_for_inputs, so a queue that has not
        # promoted anything yet must not crash the run that exists to wait for
        # it. Absent still has to be distinguishable from present.
        foundation = self.sandbox / "foundation"
        foundation.mkdir()
        state_path = foundation / "download-state.json"
        stamp_path = foundation / "complete.ok"
        completed = mock.Mock(stdout=b"source-generation")
        with mock.patch.object(self.supervisor, "ROOT", self.sandbox), \
             mock.patch.object(self.supervisor, "FOUNDATION", foundation), \
             mock.patch.object(self.supervisor, "UPSTREAM", ((state_path, stamp_path, "4"),)), \
             mock.patch.object(self.supervisor, "STEPS", (("step", ["/bin/true"]),)), \
             mock.patch.object(self.supervisor.subprocess, "run", return_value=completed):
            absent = self.supervisor.mission_inputs_fingerprint()
            state_path.write_text('{"status":"complete"}')
            stamp_path.write_text("4")
            present = self.supervisor.mission_inputs_fingerprint()
        self.assertNotEqual(absent, present)


class ArtifactsInFlightTests(SupervisorTestCase):
    """A completion stamp does not survive a quarantine and re-download.

    The queue verifies SHA-256 on promotion, moves a mismatching file to
    ``<name>.bad-<stamp>`` and fetches it again; the stamp says complete the
    whole time. On 2026-09-09 the policy gate ran sixteen minutes before
    ``Gemma-4-12B-it-heretic-Q6_K.gguf`` finished arriving, reported it missing,
    and blocked eleven model-backed gates behind it.
    """

    def queue(self, size=7, present=None, partial=False):
        destination = self.sandbox / "weights.gguf"
        if present is not None:
            destination.write_bytes(b"x" * present)
        if partial:
            destination.with_name(destination.name + ".partial").write_bytes(b"x")
        (self.sandbox / "download-queue-fixture.json").write_text(json.dumps(
            {"artifacts": [{"files": [{"destination": str(destination), "size": size}]}]}))
        return mock.patch.object(self.supervisor, "FOUNDATION", self.sandbox)

    def in_flight(self, **kwargs):
        with self.queue(**kwargs):
            return self.supervisor.artifacts_in_flight()

    def test_a_short_file_with_a_partial_sibling_is_in_flight(self):
        self.assertEqual([str(self.sandbox / "weights.gguf")],
                         self.in_flight(present=3, partial=True))

    def test_an_absent_file_with_a_partial_sibling_is_in_flight(self):
        self.assertEqual([str(self.sandbox / "weights.gguf")],
                         self.in_flight(present=None, partial=True))

    def test_a_complete_file_is_not_in_flight(self):
        self.assertEqual([], self.in_flight(present=7, partial=True))

    def test_an_absent_file_nobody_is_fetching_is_not_waited_for(self):
        # A permanently missing artifact is a real problem for the policy gate
        # to report. Waiting on it would replace a clear failure with a hang.
        self.assertEqual([], self.in_flight(present=None, partial=False))

    def test_an_entry_without_a_declared_size_is_ignored(self):
        destination = self.sandbox / "weights.gguf"
        destination.with_name(destination.name + ".partial").write_bytes(b"x")
        (self.sandbox / "download-queue-fixture.json").write_text(json.dumps(
            {"artifacts": [{"files": [{"destination": str(destination)}]}]}))
        with mock.patch.object(self.supervisor, "FOUNDATION", self.sandbox):
            self.assertEqual([], self.supervisor.artifacts_in_flight())

    def test_an_unreadable_queue_is_skipped_rather_than_fatal(self):
        (self.sandbox / "download-queue-broken.json").write_text("{not json")
        with mock.patch.object(self.supervisor, "FOUNDATION", self.sandbox):
            self.assertEqual([], self.supervisor.artifacts_in_flight())

    def test_the_wait_holds_while_an_artifact_is_being_re_fetched(self):
        answers = [["/models/a.gguf"], []]
        slept = []
        with mock.patch.object(self.supervisor, "phase_status",
                               return_value=("ready", "queue")), \
             mock.patch.object(self.supervisor, "artifacts_in_flight",
                               side_effect=lambda: answers.pop(0)), \
             mock.patch.object(self.supervisor, "atomic_json"), \
             mock.patch.object(self.supervisor, "log") as log, \
             mock.patch.object(self.supervisor.time, "sleep", slept.append):
            state = {}
            self.supervisor.wait_for_inputs(state)
        self.assertEqual([self.supervisor.POLL_SECONDS], slept)
        self.assertEqual("waiting-artifacts", state["status"])
        self.assertIn("being re-fetched", log.call_args[0][0])

    def test_a_ready_queue_with_nothing_in_flight_does_not_wait(self):
        with mock.patch.object(self.supervisor, "phase_status",
                               return_value=("ready", "queue")), \
             mock.patch.object(self.supervisor, "artifacts_in_flight", return_value=[]), \
             mock.patch.object(self.supervisor.time, "sleep",
                               side_effect=AssertionError("must not wait")):
            self.supervisor.wait_for_inputs({})


class PhaseStatusTests(SupervisorTestCase):
    def test_a_matching_stamp_is_ready_while_state_says_running(self):
        state_path = self.sandbox / "download-state.json"
        stamp_path = self.sandbox / "complete.ok"
        state_path.write_text('{"status":"running"}')
        stamp_path.write_text("4\n")
        self.assertEqual(
            ("ready", "download-state.json"),
            self.supervisor.phase_status(state_path, stamp_path, "4"))

    def test_a_missing_stamp_waits(self):
        state_path = self.sandbox / "download-state.json"
        stamp_path = self.sandbox / "complete.ok"
        state_path.write_text('{"status":"complete"}')
        self.assertEqual(
            ("waiting", "complete.ok: missing"),
            self.supervisor.phase_status(state_path, stamp_path, "4"))

    def test_a_wrong_stamp_fails(self):
        state_path = self.sandbox / "download-state.json"
        stamp_path = self.sandbox / "complete.ok"
        state_path.write_text('{"status":"complete"}')
        stamp_path.write_text("3\n")
        status, detail = self.supervisor.phase_status(state_path, stamp_path, "4")
        self.assertEqual("failed", status)
        self.assertIn("expected 4", detail)

    def test_state_round_trips_and_leaves_no_temp_file(self):
        self.supervisor.atomic_json({"status": "running", "steps": {}})
        self.assertEqual({"status": "running", "steps": {}},
                         json.loads(self.supervisor.STATE.read_text()))
        self.assertEqual(["mission-state.json"],
                         sorted(p.name for p in self.sandbox.iterdir()))

    def test_a_failed_publish_leaves_no_half_written_document_behind(self):
        # The next run's temp file must not be able to inherit a partial one, and
        # nothing must be left that looks like state to a human reading the dir.
        self.supervisor.atomic_json({"status": "running", "steps": {}})
        with mock.patch.object(self.supervisor.os, "replace",
                               side_effect=OSError("no space left on device")):
            with self.assertRaises(OSError):
                self.supervisor.atomic_json({"status": "complete"})
        self.assertEqual(["mission-state.json"],
                         sorted(p.name for p in self.sandbox.iterdir()))
        self.assertEqual("running",
                         json.loads(self.supervisor.STATE.read_text())["status"])

    def test_the_write_is_fsynced_before_the_rename(self):
        # A rename is atomic but not durable. Without the fsync a power loss can
        # publish the new name over content that never reached the disk, and the
        # supervisor would resume from a state it never actually wrote.
        synced = []
        real_fsync = os.fsync

        def recording_fsync(fd):
            synced.append(fd)
            real_fsync(fd)

        with mock.patch.object(self.supervisor.os, "fsync", recording_fsync):
            self.supervisor.atomic_json({"status": "complete"})
        self.assertGreaterEqual(len(synced), 1, "mission state was never fsynced")

    def test_a_filesystem_that_refuses_directory_fsync_still_publishes(self):
        def only_dir_fails(fd):
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("directory fsync unsupported")

        with mock.patch.object(self.supervisor.os, "fsync", only_dir_fails):
            self.supervisor.atomic_json({"status": "complete"})
        self.assertEqual({"status": "complete"},
                         json.loads(self.supervisor.STATE.read_text()))

    def test_a_truncated_state_file_falls_back_to_a_fresh_mission(self):
        self.supervisor.STATE.write_text('{"schema": "frankenstein-func')
        state = self.supervisor.load_state()
        self.assertEqual("frankenstein-functional-mission/1", state["schema"])
        self.assertEqual({}, state["steps"])

    def test_state_from_another_schema_is_not_resumed(self):
        self.supervisor.STATE.write_text(json.dumps({"schema": "something/9", "steps": {"a": 1}}))
        self.assertEqual({}, self.supervisor.load_state()["steps"])


class InputFingerprintResilienceTests(SupervisorTestCase):
    """A checkout too busy to read is a host condition, not a mission failure.

    ``git diff --binary HEAD`` over ``verification`` is the mission's own input
    query. Under sixteen concurrent downloads it exceeded a single 30-second
    attempt, and a concurrent git process made it exit 128; both escaped as
    unhandled exceptions and ended the run before any gate started.
    """

    def arrange(self, results):
        """Feed ``subprocess.run`` one outcome per call: bytes, or an exception."""
        self.slept = []
        outcomes = list(results)

        def fake_run(command, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return subprocess.CompletedProcess(command, 0, stdout=outcome, stderr=b"")

        return (
            mock.patch.object(self.supervisor.subprocess, "run", fake_run),
            mock.patch.object(self.supervisor.time, "sleep", self.slept.append),
            mock.patch.object(self.supervisor, "log"),
        )

    def run_fingerprint(self, results):
        for patch in self.arrange(results):
            patch.start()
            self.addCleanup(patch.stop)
        return self.supervisor.mission_inputs_fingerprint()

    def test_a_slow_checkout_is_retried_rather_than_fatal(self):
        timeout = subprocess.TimeoutExpired(["git", "diff"], 180)
        value = self.run_fingerprint([b"ls-files", timeout, timeout, b"diff"])
        self.assertRegex(value, r"^[0-9a-f]{64}$")
        self.assertEqual([self.supervisor.FINGERPRINT_RETRY_SECONDS] * 2, self.slept)

    def test_a_contended_index_is_retried_rather_than_fatal(self):
        conflict = subprocess.CalledProcessError(128, ["git", "diff"])
        self.assertRegex(self.run_fingerprint([b"ls-files", conflict, b"diff"]),
                         r"^[0-9a-f]{64}$")

    def test_an_unreadable_checkout_refuses_instead_of_raising_out(self):
        timeout = subprocess.TimeoutExpired(["git", "ls-files"], 180)
        with self.assertRaises(self.supervisor.InputsUnreadable):
            self.run_fingerprint([timeout] * self.supervisor.FINGERPRINT_ATTEMPTS)
        self.assertEqual(self.supervisor.FINGERPRINT_ATTEMPTS - 1, len(self.slept))

    def test_the_retry_budget_is_bounded(self):
        self.assertGreaterEqual(self.supervisor.FINGERPRINT_ATTEMPTS, 2)
        self.assertLessEqual(self.supervisor.FINGERPRINT_ATTEMPTS, 8)
        self.assertLessEqual(self.supervisor.FINGERPRINT_TIMEOUT_SECONDS, 600)

    def test_main_waits_instead_of_recording_a_mission_failure(self):
        self.supervisor.STATE.write_text(json.dumps({
            "schema": "frankenstein-functional-mission/1", "steps": {},
            "input_fingerprint": "earlier", "status": "functional-foundation-complete"}))
        with mock.patch.object(self.supervisor, "mission_inputs_fingerprint",
                               side_effect=self.supervisor.InputsUnreadable("git diff failed")), \
             mock.patch.object(self.supervisor, "log"), \
             mock.patch.object(self.supervisor.fcntl, "flock"), \
             mock.patch.object(self.supervisor, "adopt_legacy_step", return_value=False), \
             mock.patch.object(self.supervisor, "run_step",
                               side_effect=AssertionError("no gate may be attempted")), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            code = self.supervisor.main()
        state = json.loads(self.supervisor.STATE.read_text())
        self.assertEqual(self.supervisor.EXIT_ADMISSION_REFUSED, code)
        self.assertEqual("inputs-unreadable", state["status"])
        self.assertEqual("earlier", state["input_fingerprint"],
                         "an unread fingerprint must not overwrite the one that ran")
        self.assertIn("git diff failed", state["error"])


class InterruptedStepClassificationTests(SupervisorTestCase):
    """An operator stop is not a step verdict."""

    def drive(self, exit_codes, signal_after=None, fingerprint="fixture-v1"):
        """Run main() over fake steps, raising ``signal_after`` steps' worth of work.

        ``exit_codes`` is one code per step. Setting ``signal_after`` to an index
        sets ``stop_signal`` once that step's process has exited, which is what a
        forwarded SIGTERM looks like from main()'s point of view.
        """
        names = [f"step-{index}" for index in range(len(exit_codes))]
        steps = tuple((name, ["/bin/true"]) for name in names)
        calls = []
        # Each real run is a fresh process, so a resume never inherits the
        # previous run's signal. Emulate that rather than leaking it across
        # drive() calls inside one test.
        self.supervisor.stop_signal = None

        def fake_run_step(name, command, state, before_start=None):
            if before_start:
                before_start()
            index = names.index(name)
            calls.append(name)
            rc = exit_codes[index]
            state["steps"][name] = {
                # The real run_step classifies by exit code, so 75 and 76 are not
                # "failed" here either; a fake that flattened them would let the
                # supervisor's blocked/inconclusive handling go untested.
                "command": command, "status": self.supervisor.exit_status(rc),
                "exit_code": rc, "input_fingerprint": state["input_fingerprint"],
            }
            if signal_after == index:
                self.supervisor.stop_signal = 15
            return rc

        with mock.patch.object(self.supervisor, "STEPS", steps), \
             mock.patch.object(self.supervisor, "run_step", fake_run_step), \
             mock.patch.object(self.supervisor, "mission_inputs_fingerprint",
                               return_value=fingerprint), \
             mock.patch.object(self.supervisor, "wait_for_inputs"), \
             mock.patch.object(self.supervisor, "log"), \
             mock.patch.object(self.supervisor.fcntl, "flock"), \
             warnings.catch_warnings():
            # main() holds the queue lock open for the lifetime of the process on
            # purpose -- closing it would drop the flock -- so the interpreter's
            # unclosed-file warning is expected here rather than a leak.
            warnings.simplefilter("ignore", ResourceWarning)
            code = self.supervisor.main()
        return code, json.loads(self.supervisor.STATE.read_text()), calls

    def test_a_clean_run_completes_every_step(self):
        code, state, calls = self.drive([0, 0, 0])
        self.assertEqual(0, code)
        self.assertEqual("functional-foundation-complete", state["status"])
        self.assertEqual(["step-0", "step-1", "step-2"], calls)

    def test_failed_gates_self_heal_on_next_invocation(self):
        code, _, _ = self.drive([0, 1, 0, 2])
        self.assertEqual(1, code)
        code, state, calls = self.drive([0, 0, 0, 0])
        self.assertEqual(["step-1", "step-3"], calls)
        self.assertEqual(0, code)
        self.assertEqual("functional-foundation-complete", state["status"])
        self.assertNotIn("failed_steps", state)

    def test_persistent_failure_is_attempted_once_per_invocation(self):
        self.drive([0, 1, 0])
        code, state, calls = self.drive([0, 1, 0])
        self.assertEqual(["step-1"], calls)
        self.assertEqual(1, code)
        self.assertEqual("functional-foundation-incomplete", state["status"])

    def test_failed_policy_recovers_before_previously_blocked_gates(self):
        self.drive([1, 0, 0])
        code, state, calls = self.drive([0, 0, 0])
        self.assertEqual(["step-0", "step-1", "step-2"], calls)
        self.assertEqual(0, code)
        self.assertEqual("functional-foundation-complete", state["status"])

    def test_a_genuine_step_failure_is_reported_as_a_failure(self):
        code, state, calls = self.drive([0, 3, 0])
        self.assertEqual(1, code)
        self.assertEqual("functional-foundation-incomplete", state["status"])
        self.assertEqual([{"name": "step-1", "exit_code": 3}], state["failed_steps"])
        self.assertEqual(["step-0", "step-1", "step-2"], calls,
                         "one functional failure hid independent gate results")

    def test_a_refused_gate_is_outstanding_work_not_a_failed_gate(self):
        """75 means the gate declined to start, so no verdict exists to record.

        Recording it in ``failed_steps`` published a failure for a model that was
        never dispatched, and made a busy host indistinguishable from a broken
        one in the only summary an operator reads.
        """
        code, state, calls = self.drive([0, 75, 0])
        self.assertEqual(self.supervisor.EXIT_ADMISSION_REFUSED, code)
        self.assertEqual("admission-blocked", state["status"])
        self.assertEqual([], state["failed_steps"])
        self.assertEqual([{"name": "step-1", "exit_code": 75}], state["blocked_steps"])
        self.assertEqual("blocked", state["steps"]["step-1"]["status"])
        self.assertEqual(["step-0", "step-1", "step-2"], calls,
                         "a refusal must not stop the gates that follow it")
        self.assertIn("step-1", state["remaining"])

    def test_a_real_failure_outranks_a_refusal_in_the_same_pass(self):
        code, state, _ = self.drive([0, 75, 1])
        self.assertEqual(1, code)
        self.assertEqual("functional-foundation-incomplete", state["status"])
        self.assertEqual([{"name": "step-2", "exit_code": 1}], state["failed_steps"])
        self.assertEqual([{"name": "step-1", "exit_code": 75}], state["blocked_steps"])

    def test_a_refused_gate_is_attempted_again_on_the_next_invocation(self):
        self.drive([0, 75, 0])
        code, state, calls = self.drive([0, 0, 0])
        self.assertEqual(["step-1"], calls)
        self.assertEqual(0, code)
        self.assertEqual("functional-foundation-complete", state["status"])
        self.assertNotIn("blocked_steps", state)

    def test_an_inconclusive_gate_is_still_reported_as_incomplete(self):
        code, state, _ = self.drive([0, 76, 0])
        self.assertEqual(1, code)
        self.assertEqual("functional-foundation-incomplete", state["status"])
        self.assertEqual([{"name": "step-1", "exit_code": 76}], state["failed_steps"])

    def test_multiple_functional_failures_are_aggregated_after_policy_passes(self):
        code, state, calls = self.drive([0, 2, 4])
        self.assertEqual(1, code)
        self.assertEqual(
            [{"name": "step-1", "exit_code": 2}, {"name": "step-2", "exit_code": 4}],
            state["failed_steps"],
        )
        self.assertEqual(["step-0", "step-1", "step-2"], calls)

    def test_policy_failure_blocks_every_later_command(self):
        code, state, calls = self.drive([7, 0, 0])
        self.assertEqual(1, code)
        self.assertEqual("blocked-policy", state["status"])
        self.assertEqual(["step-0"], calls)
        self.assertEqual([{"name": "step-0", "exit_code": 7}], state["failed_steps"])
        for name in ("step-1", "step-2"):
            with self.subTest(name=name):
                self.assertEqual("blocked-policy", state["steps"][name]["status"])
                self.assertEqual("step-0", state["steps"][name]["blocked_by"])

    def test_a_forwarded_signal_is_an_interruption_not_a_step_failure(self):
        # SIGTERM to the supervisor is forwarded to the child, so the child exits
        # non-zero. Reading that as "the gate failed" would record a false verdict
        # about a model that was never actually judged.
        code, state, calls = self.drive([0, 143], signal_after=1)
        self.assertEqual(128 + 15, code)
        self.assertEqual("interrupted", state["status"])
        self.assertEqual(15, state["signal"])
        self.assertEqual("step-1", state["interrupted_step"])
        self.assertEqual(143, state["step_exit_code"])
        self.assertEqual("interrupted", state["steps"]["step-1"]["status"])
        self.assertNotIn("failed_step", state)
        self.assertEqual(["step-0", "step-1"], calls)

    def test_a_step_that_still_exited_zero_keeps_its_pass(self):
        # A stop that lands after the gate already succeeded must not throw the
        # result away: the resume would spend hours re-running a passed gate.
        code, state, _ = self.drive([0, 0], signal_after=1)
        self.assertEqual(128 + 15, code)
        self.assertEqual("interrupted", state["status"])
        self.assertEqual("passed", state["steps"]["step-1"]["status"])
        self.assertEqual("step-1", state["interrupted_step"])

    def test_an_interrupted_step_is_retried_on_resume_and_passed_ones_are_not(self):
        code, first, _ = self.drive([0, 143], signal_after=1)
        self.assertEqual(128 + 15, code)
        code, second, calls = self.drive([0, 0])
        self.assertEqual(0, code)
        self.assertEqual(["step-1"], calls, "resume did not skip the passed step")
        self.assertEqual("functional-foundation-complete", second["status"])

    def test_passed_steps_are_retried_when_source_or_artifact_inputs_change(self):
        code, _, calls = self.drive([0, 0], fingerprint="generation-a")
        self.assertEqual(0, code)
        self.assertEqual(["step-0", "step-1"], calls)

        code, state, calls = self.drive([0, 0], fingerprint="generation-b")
        self.assertEqual(0, code)
        self.assertEqual(["step-0", "step-1"], calls)
        self.assertEqual("generation-b", state["input_fingerprint"])
        self.assertTrue(all(
            step["input_fingerprint"] == "generation-b"
            for step in state["steps"].values()
        ))

    def test_a_resumed_run_clears_the_previous_interruption_markers(self):
        # Stale "interrupted_step"/"signal" keys on a mission that later completes
        # would describe a stop that is no longer true.
        self.drive([0, 143], signal_after=1)
        _, state, _ = self.drive([0, 0])
        for stale in ("signal", "interrupted_step", "step_exit_code", "step_status",
                      "failed_step", "exit_code", "error"):
            with self.subTest(key=stale):
                self.assertNotIn(stale, state)


class MissionPolicyTests(SupervisorTestCase):
    """The mission records no performance data, by construction."""

    def test_state_declares_that_no_benchmarking_happened(self):
        state = self.supervisor.load_state()
        self.assertFalse(state["benchmarking_performed"])
        self.assertFalse(state["throughput_measured"])

    def test_candidate_policy_precedes_wemm_and_model_loads(self):
        names = [name for name, _command in self.supervisor.STEPS]
        self.assertEqual("candidate-policy", names[0])
        self.assertIn("wemm-embeddings", names)
        self.assertLess(names.index("candidate-policy"), names.index("wemm-embeddings"))
        self.assertLess(names.index("wemm-embeddings"), names.index("router-models"))

    def test_dedicated_asr_gate_precedes_tts_roundtrip(self):
        steps = dict(self.supervisor.STEPS)
        names = list(steps)
        self.assertIn("asr", steps)
        # The interpreter itself is asserted by
        # test_asr_gate_runs_in_the_venv_that_can_import_its_model; this step
        # only has to be the dedicated ASR gate, ahead of the round trip.
        self.assertEqual(str(self.supervisor.ASR_PYTHON), steps["asr"][0])
        self.assertTrue(steps["asr"][-1].endswith("validators/gate_asr.py"))
        self.assertLess(names.index("asr"), names.index("tts-asr-roundtrip"))

        ledger_source = (self.supervisor.FOUNDATION / "build_capability_ledger.py")
        spec = importlib.util.spec_from_file_location("capability_ledger_under_test", ledger_source)
        assert spec is not None and spec.loader is not None
        ledger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ledger)
        self.assertEqual((ledger.EVIDENCE / "gate-asr.json",),
                         ledger.CAPABILITY_EVIDENCE["asr"])
        self.assertEqual("asr", ledger.EXPECTED_EVIDENCE_GATES[
            ledger.EVIDENCE / "gate-asr.json"])

    def test_mission_unit_gives_miopen_a_writable_cache(self):
        """MIOpen must not be pointed at $HOME while ProtectHome is read-only.

        MIOpen takes a lock beside its user database before reading it. Under
        ProtectHome=read-only that create fails, and MIOpen surfaces it as
        miopenStatusUnknownError from whatever convolution ran first -- which is
        how one sandbox setting failed wemm-embeddings, computer-use-grounding
        and generative-media-functional at once on 2026-09-07, each looking like
        an unrelated model bug. Both paths have to land somewhere ReadWritePaths
        already covers.
        """
        unit = (self.supervisor.ROOT / "services/systemd/local-ai-functional-mission.service").read_text()
        self.assertIn("ProtectHome=read-only", unit)
        writable = unit.split("ReadWritePaths=", 1)[1].splitlines()[0].split()
        for key in ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
            with self.subTest(variable=key):
                line = next((row for row in unit.splitlines()
                             if row.startswith(f"Environment={key}=")), None)
                self.assertIsNotNone(line, f"{key} is not set for the mission")
                target = line.split("=", 2)[2]
                self.assertFalse(
                    target.startswith(str(Path.home()) + "/."),
                    f"{key} points into the read-only home at {target}")
                self.assertTrue(
                    any(target == root or target.startswith(root + "/")
                        for root in writable),
                    f"{key} at {target} is not under any ReadWritePaths entry")
                self.assertIn(f"mkdir -p {target}",
                              unit.replace("/cache", "/cache").replace(
                                  "ExecStartPre=/usr/bin/", ""),
                              f"{key} is never created before the mission runs")

    def test_asr_gate_runs_in_the_venv_that_can_import_its_model(self):
        """The ASR gate needs venvs/asr, not the TTS one.

        Qwen3-ASR declares model_type qwen3_asr, which only the newer
        transformers in venvs/asr registers. Running the gate under the TTS venv
        fails at import, before any GPU work, and reads in the mission log as an
        ASR capability failure rather than as the wrong interpreter.
        """
        step = dict(self.supervisor.STEPS)["asr"]
        self.assertEqual(str(self.supervisor.ASR_PYTHON), step[0])
        self.assertNotEqual(str(self.supervisor.TTS_PYTHON), step[0])
        self.assertIn("venvs/asr", step[0])

    def test_tts_roundtrip_spawns_the_asr_venv_not_its_own(self):
        """gate_tts.py runs under the TTS venv and must not reuse it for ASR.

        The round trip loads the same Qwen3-ASR weights the ASR gate does, so
        sys.executable -- the TTS interpreter -- cannot import them either.
        """
        gate = (self.supervisor.ROOT / "verification/tts-local/gate_tts.py").read_text()
        self.assertIn('ASR_PYTHON = ROOT.parents[1] / "venvs" / "asr" / "bin" / "python"', gate)
        self.assertIn('[str(ASR_PYTHON), str(ROOT / "asr_roundtrip.py")', gate)
        self.assertNotIn('[sys.executable, str(ROOT / "asr_roundtrip.py")', gate)

    def test_mission_unit_keeps_a_writable_temp_dir(self):
        unit = (self.supervisor.ROOT / "services/systemd/local-ai-functional-mission.service").read_text()
        self.assertIn("ProtectSystem=strict", unit)
        self.assertIn("Environment=TMPDIR=/home/typhoon/git/frankenstein-llm/verification/tmp", unit)
        writable = unit.split("ReadWritePaths=", 1)[1].splitlines()[0]
        for path in ("/tmp", "/var/tmp", "/venvs", "/tools", "/verification"):
            with self.subTest(path=path):
                self.assertIn(path, writable)


class DownloadQueueConflictTests(SupervisorTestCase):
    """A running download queue confounds the router gate, so the mission waits.

    Before ``download-queue`` was a waited class the mission started
    ``router-models`` while the queues were re-verifying ~130 GB of files that
    already existed. ``gate_router_models`` refuses a confounded qualification,
    so the step ended in under a second and was recorded as nine model failures
    -- on every boot, because the queue units start at boot too.
    """

    CGROUP = ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
              "local-ai-model-downloads-phase4.service\n")

    def test_a_queue_process_is_named_by_its_cgroup_not_its_command(self):
        for command in ("python3", "aria2c"):
            with self.subTest(command=command):
                self.assertEqual("download-queue", self.supervisor.conflict_reason(
                    command, str(self.supervisor.ROOT), self.CGROUP))

    def test_the_mission_waits_for_a_download_queue(self):
        self.assertIn("download-queue", self.supervisor.MISSION_BLOCKING_REASONS)

    def test_the_router_gate_and_the_mission_name_the_same_units(self):
        gate = (self.supervisor.ROOT
                / "verification/router-functional/gate_router_models.py").read_text()
        self.assertIn(f'DOWNLOAD_UNIT_PREFIX = "{self.supervisor.DOWNLOAD_UNIT_PREFIX}"',
                      gate)

    def test_a_transfer_outside_a_queue_unit_still_does_not_hold_the_mission(self):
        # aria2 re-hash by hand and a browser download are not queue units.
        # Blocking the mission on those is what the narrow waited set avoids.
        self.assertEqual("transfer", self.supervisor.conflict_reason(
            "aria2c", "/tmp", "0::/user.slice/hand.scope\n"))
        self.assertNotIn("transfer", self.supervisor.MISSION_BLOCKING_REASONS)


class DownloadStateFingerprintTests(SupervisorTestCase):
    """Re-verifying an existing download must not invalidate a passed gate.

    The queue units run at every boot and rewrite their state document with new
    run timestamps even when every file is already present and verified. That
    document is hashed into the mission input fingerprint, so hashing the
    timestamps meant every boot changed the fingerprint, every passed gate was
    re-run, and the mission could never converge.
    """

    DOCUMENT = {
        "schema": "hermes-hf-download-state/1",
        "queue_built_at": "2026-09-02T10:50:00-0400",
        "first_started_at": "2026-09-03T14:42:01-0400",
        "started_at": "2026-09-08T11:55:37-0400",
        "completed_at": "2026-09-08T11:56:48-0400",
        "status": "complete",
        "artifacts": {
            "image-editing": {
                "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
                "revision": "984166f60a9b1fcede5e9b9287b7a7aebc050010",
                "files_total": 1,
                "files_complete": 1,
                "completed_at": "2026-09-08T11:56:48-0400",
                "status": "complete",
            },
        },
    }

    def digest(self, document):
        return self.supervisor.durable_queue_state(json.dumps(document).encode())

    def variant(self, **changes):
        document = json.loads(json.dumps(self.DOCUMENT))
        artifact = changes.pop("artifact", {})
        document.update(changes)
        document["artifacts"]["image-editing"].update(artifact)
        return document

    def test_a_rerun_that_only_reverifies_hashes_the_same(self):
        rerun = self.variant(started_at="2026-09-08T12:40:00-0400",
                             completed_at="2026-09-08T12:47:00-0400",
                             artifact={"completed_at": "2026-09-08T12:47:00-0400"})
        self.assertEqual(self.digest(self.DOCUMENT), self.digest(rerun))

    def test_progress_partway_through_a_rerun_hashes_the_same(self):
        partway = self.variant(status="running",
                               artifact={"status": "running", "files_complete": 0})
        self.assertEqual(self.digest(self.DOCUMENT), self.digest(partway))

    def test_key_order_does_not_change_the_digest(self):
        self.assertEqual(self.digest(self.DOCUMENT),
                         self.digest(dict(reversed(list(self.DOCUMENT.items())))))

    def test_what_was_downloaded_still_changes_the_digest(self):
        for artifact in ({"revision": "0" * 40}, {"files_total": 2},
                         {"repository": "someone-else/weights"}):
            with self.subTest(artifact=artifact):
                self.assertNotEqual(self.digest(self.DOCUMENT),
                                    self.digest(self.variant(artifact=artifact)))

    def test_an_unparseable_document_is_hashed_as_given(self):
        # Silently ignoring bytes that will not parse would hide a real change.
        self.assertEqual(b"{not json",
                         self.supervisor.durable_queue_state(b"{not json"))

    def test_the_whole_fingerprint_survives_a_reverify(self):
        state = self.sandbox / "download-state.json"
        stamp = self.sandbox / "downloads-complete.ok"
        foundation = self.sandbox / "foundation"
        foundation.mkdir()
        state.write_text(json.dumps(self.DOCUMENT))
        stamp.write_text("72134030730")
        with mock.patch.object(self.supervisor, "UPSTREAM",
                               ((state, stamp, "72134030730"),)), \
                mock.patch.object(self.supervisor, "FOUNDATION", foundation), \
                mock.patch.object(self.supervisor.subprocess, "run",
                                  return_value=mock.Mock(stdout=b"tree")):
            before = self.supervisor.mission_inputs_fingerprint()
            state.write_text(json.dumps(self.variant(
                started_at="2026-09-08T12:40:00-0400",
                completed_at="2026-09-08T12:47:00-0400")))
            self.assertEqual(before, self.supervisor.mission_inputs_fingerprint())
            # The stamp is the durable byte total; a change there must still count.
            stamp.write_text("72134030731")
            self.assertNotEqual(before, self.supervisor.mission_inputs_fingerprint())


if __name__ == "__main__":
    unittest.main()
