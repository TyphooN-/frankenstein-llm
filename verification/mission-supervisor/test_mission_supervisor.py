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
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

SOURCE = Path(__file__).resolve().parent / "run_functional_mission.py"


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
        return (
            mock.patch.object(self.supervisor, "conflicts", conflicts),
            mock.patch.object(self.supervisor, "mem_available", return_value=available),
            mock.patch.object(self.supervisor.os, "getloadavg", return_value=(load, load, load)),
            mock.patch.object(self.supervisor.time, "sleep", fake_sleep),
            mock.patch.object(self.supervisor.time, "monotonic", lambda: self.clock[0]),
            mock.patch.object(self.supervisor, "log"),
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
        busy = [{"pid": 1, "command": "llama-server", "cwd": ""}]
        self.run_wait([[], busy, [], []], timeout=3600)
        self.assertEqual(3, len(self.slept))

    def test_a_permanently_busy_host_fails_instead_of_waiting_forever(self):
        busy = [{"pid": 4242, "command": "llama-server --model heretic", "cwd": ""}]
        with self.assertRaises(RuntimeError) as caught:
            self.run_wait([busy] * 40, timeout=120)
        self.assertIn("did not become quiet within 120s", str(caught.exception))

    def test_the_failure_names_what_the_host_was_busy_with(self):
        busy = [{"pid": 4242, "command": "llama-server --model heretic", "cwd": ""}]
        with self.assertRaises(RuntimeError) as caught:
            self.run_wait([busy] * 40, timeout=60)
        message = str(caught.exception)
        self.assertIn("conflicts=", message)
        self.assertIn("4242", message)
        self.assertNotIn("unknown", message)

    def test_low_memory_and_high_load_are_reported_as_the_blockers(self):
        with self.assertRaises(RuntimeError) as caught:
            self.run_wait([[]] * 40, timeout=60, available=1 << 30, load=99.0)
        message = str(caught.exception)
        self.assertIn("MemAvailable=", message)
        self.assertIn("load1=99.00", message)

    def test_the_timeout_is_recorded_in_state_before_it_raises(self):
        # The unit dies on this exception, so the durable record of why has to be
        # written first; otherwise the state file still says "waiting-safe-host".
        state = {"steps": {}}
        busy = [{"pid": 7, "command": "cmake --build .", "cwd": ""}]
        with self.assertRaises(RuntimeError):
            self.run_wait([busy] * 40, timeout=60, state=state)
        self.assertEqual("failed", state["status"])
        self.assertEqual("failed", json.loads(self.supervisor.STATE.read_text())["status"])


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

    def test_state_round_trips_and_leaves_no_temp_file(self):
        self.supervisor.atomic_json({"status": "running", "steps": {}})
        self.assertEqual({"status": "running", "steps": {}},
                         json.loads(self.supervisor.STATE.read_text()))
        self.assertEqual(["mission-state.json"],
                         sorted(p.name for p in self.sandbox.iterdir()))

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

        def fake_run_step(name, command, state):
            index = names.index(name)
            calls.append(name)
            rc = exit_codes[index]
            state["steps"][name] = {
                "command": command, "status": "passed" if rc == 0 else "failed",
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

    def test_a_genuine_step_failure_is_reported_as_a_failure(self):
        code, state, calls = self.drive([0, 3, 0])
        self.assertEqual(1, code)
        self.assertEqual("functional-foundation-incomplete", state["status"])
        self.assertEqual([{"name": "step-1", "exit_code": 3}], state["failed_steps"])
        self.assertEqual(["step-0", "step-1", "step-2"], calls,
                         "one functional failure hid independent gate results")

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

    def test_an_active_transfer_or_build_counts_as_a_conflict(self):
        markers = ("download_queue.py", "cmake --build", "llama-server")
        source = SOURCE.read_text()
        for marker in markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, source)

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


if __name__ == "__main__":
    unittest.main()
