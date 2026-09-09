"""Unit tests for the shared admission-gate primitives. No server, no GPU.

These cover the properties that decide whether a sidecar gate's verdict means
anything: a clean-unload check must fail when it could not measure, cleanup must
run whether or not the gate reached its own unload step, and cleanup must never
raise out of the ``finally`` that calls it.

Run: python3 -m unittest discover -s . -p 'test_*.py' -v
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gatelib  # noqa: E402

GIB = 1 << 30
TOLERANCE = 256 << 20


class VramResidueTests(unittest.TestCase):
    def test_clean_release_passes(self):
        verdict = gatelib.unload_verdict({"card0": GIB, "card1": 0},
                                         {"card0": GIB, "card1": 0}, TOLERANCE)
        self.assertTrue(verdict["pass"], verdict["problems"])
        self.assertEqual({"card0": 0, "card1": 0}, verdict["vram_residue_bytes"])

    def test_retained_memory_fails(self):
        verdict = gatelib.unload_verdict({"card0": 0}, {"card0": 2 * GIB}, TOLERANCE)
        self.assertFalse(verdict["pass"])
        self.assertIn("still held", " ".join(verdict["problems"]))

    def test_unreadable_after_reading_is_not_a_release(self):
        # -1 is "sysfs did not answer". Arithmetic on it produces a hugely
        # negative residue that clears any tolerance, which is how "we could not
        # measure" used to be scored as "it released cleanly".
        verdict = gatelib.unload_verdict({"card0": 4 * GIB}, {"card0": -1}, TOLERANCE)
        self.assertFalse(verdict["pass"])
        self.assertEqual(["card0"], verdict["unreadable_cards"])
        self.assertEqual({}, verdict["vram_residue_bytes"])

    def test_unreadable_baseline_is_not_a_release(self):
        verdict = gatelib.unload_verdict({"card0": -1}, {"card0": 8 * GIB}, TOLERANCE)
        self.assertFalse(verdict["pass"])
        self.assertEqual(["card0"], verdict["unreadable_cards"])

    def test_missing_card_in_the_after_reading_is_not_a_release(self):
        verdict = gatelib.unload_verdict({"card0": 0, "card2": 0}, {"card0": 0}, TOLERANCE)
        self.assertFalse(verdict["pass"])
        self.assertEqual(["card2"], verdict["unreadable_cards"])

    def test_no_measurable_card_fails_rather_than_passing_vacuously(self):
        verdict = gatelib.unload_verdict({}, {}, TOLERANCE)
        self.assertFalse(verdict["pass"])
        self.assertIn("usable before/after", " ".join(verdict["problems"]))

    def test_one_bad_card_fails_even_when_the_others_are_clean(self):
        verdict = gatelib.unload_verdict(
            {"card0": 0, "card1": 0, "card2": 0},
            {"card0": 0, "card1": 4 * GIB, "card2": 0}, TOLERANCE)
        self.assertFalse(verdict["pass"])


class InProcessUnloadTests(unittest.TestCase):
    """Card-level VRAM cannot decide the model's release for an in-process gate.

    The sidecar gates stop a unit, so the process holding the weights exits and
    the card reading is a reading about the model. The TTS, grounding and vision
    gates load the model inside their own process, which keeps its HIP context,
    compiled kernels and BLAS workspaces until it exits. That residue was scored
    as "VRAM still held after unload", and the three gates had each answered it
    with a different constant -- 256, 512 and 768 MiB.

    The allocator reading answers the question the section actually asks, and it
    answers it exactly: a live tensor is a failed unload at any size.
    """

    CLEAN = {"cuda:0": {"allocated_bytes": 0, "reserved_bytes": 0}}

    def test_clean_allocator_does_not_waive_unattributed_device_residue(self):
        verdict = gatelib.unload_verdict(
            {"card0": 27_836_416, "card1": 404_557_824, "card2": 2_195_812_352},
            {"card0": 525_914_112, "card1": 704_323_584, "card2": 2_493_698_048},
            TOLERANCE,
            allocator={f"cuda:{index}": {"allocated_bytes": 0, "reserved_bytes": 0}
                       for index in range(3)})
        self.assertFalse(verdict["pass"])
        self.assertEqual(498_077_696, verdict["unattributed_device_bytes"]["card0"])
        self.assertNotIn("runtime_context_bytes", verdict)
        self.assertIn("release remains unproven", " ".join(verdict["problems"]))

    def test_a_live_tensor_fails_even_under_the_card_tolerance(self):
        verdict = gatelib.unload_verdict(
            {"card0": 0}, {"card0": 1024}, TOLERANCE,
            allocator={"cuda:0": {"allocated_bytes": 4096, "reserved_bytes": 2 << 20}})
        self.assertFalse(verdict["pass"])
        self.assertIn("model memory still allocated after unload",
                      " ".join(verdict["problems"]))

    def test_reserved_but_unallocated_memory_is_still_a_failed_unload(self):
        verdict = gatelib.unload_verdict(
            {"card0": 0}, {"card0": 0}, TOLERANCE,
            allocator={"cuda:0": {"allocated_bytes": 0, "reserved_bytes": 8 << 20}})
        self.assertFalse(verdict["pass"])

    def test_an_unreadable_card_still_fails_with_a_clean_allocator(self):
        # The allocator says this process released everything; that is not a
        # claim about a card whose sysfs node stopped answering.
        verdict = gatelib.unload_verdict({"card0": -1}, {"card0": 0}, TOLERANCE,
                                         allocator=self.CLEAN)
        self.assertFalse(verdict["pass"])
        self.assertEqual(["card0"], verdict["unreadable_cards"])

    def test_no_measurable_card_still_fails_with_a_clean_allocator(self):
        verdict = gatelib.unload_verdict({}, {}, TOLERANCE, allocator=self.CLEAN)
        self.assertFalse(verdict["pass"])

    def test_an_unreadable_allocator_leaves_the_card_check_deciding(self):
        for report in (None, {}, "unavailable", {"cuda:0": {"allocated_bytes": 0}},
                       {"cuda:0": {"allocated_bytes": -1, "reserved_bytes": 0}},
                       {"cuda:0": None}):
            with self.subTest(report=report):
                self.assertIsNone(gatelib.allocator_outstanding(report))
                verdict = gatelib.unload_verdict({"card0": 0}, {"card0": 2 * GIB},
                                                 TOLERANCE, allocator=report)
                self.assertFalse(verdict["pass"])
                self.assertNotIn("allocator_bytes", verdict)

    def test_the_sidecar_callers_are_unchanged_by_the_new_argument(self):
        clean = gatelib.unload_verdict({"card0": GIB}, {"card0": GIB}, TOLERANCE)
        held = gatelib.unload_verdict({"card0": 0}, {"card0": 2 * GIB}, TOLERANCE)
        self.assertTrue(clean["pass"])
        self.assertFalse(held["pass"])
        for verdict in (clean, held):
            self.assertNotIn("allocator_bytes", verdict)
            self.assertNotIn("runtime_context_bytes", verdict)

    def test_the_allocator_report_never_raises_when_torch_is_absent(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertIsNone(gatelib.allocator_report())

    def test_the_in_process_gates_pass_the_allocator_reading(self):
        """The fix is worth nothing if the gates keep scoring the proxy."""
        root = Path(__file__).resolve().parents[3]
        for gate in ("tts-local/gate_tts.py",
                     "computer-use-grounding/gate_computer_use.py"):
            with self.subTest(gate=gate):
                source = (root / "verification" / gate).read_text()
                self.assertRegex(source, r"unload_verdict\([^)]*allocator=")


class EnsureUnloadedTests(unittest.TestCase):
    """Cleanup must stop the sidecar on every path out of a gate."""

    def systemd_stub(self, active_after_stop=False):
        calls = []
        states = iter(["active"] if active_after_stop else ["inactive"])

        def fake(*args):
            calls.append(args)
            stdout = ""
            if args[0] == "is-active":
                stdout = next(states, "active" if active_after_stop else "inactive")
            return mock.Mock(returncode=0, stdout=stdout, stderr="")

        return fake, calls

    def test_success_path_verdict_is_left_alone(self):
        summary = {"unload": {"unit": "u", "pass": True}}
        fake, calls = self.systemd_stub()
        with mock.patch.object(gatelib, "systemd", fake):
            gatelib.ensure_unloaded(summary, "u", {"card0": 0})
        self.assertEqual([], calls, "cleanup stopped an already-unloaded sidecar")
        self.assertTrue(summary["unload"]["pass"])

    def test_failed_gate_still_stops_the_sidecar(self):
        summary = {"pass": False, "error": "GateFailure: dimension mismatch"}
        fake, calls = self.systemd_stub()
        with mock.patch.object(gatelib, "systemd", fake), \
             mock.patch.object(gatelib.time, "sleep"), \
             mock.patch.object(gatelib, "vram_used", return_value={"card0": 0}):
            gatelib.ensure_unloaded(summary, "llama-sidecar@embeddings.service",
                                    {"card0": 0})
        self.assertIn(("stop", "llama-sidecar@embeddings.service"), calls)
        self.assertEqual("cleanup", summary["unload"]["ran_as"])
        self.assertFalse(summary["unload"]["pass"])
        self.assertFalse(summary["unload"]["still_active"])

    def test_cleanup_never_reports_a_pass(self):
        # An unload check that never ran is not a passing unload check, however
        # clean the VRAM happens to look afterwards.
        summary: dict = {"pass": False}
        fake, _ = self.systemd_stub()
        with mock.patch.object(gatelib, "systemd", fake), \
             mock.patch.object(gatelib.time, "sleep"), \
             mock.patch.object(gatelib, "vram_used", return_value={"card0": 0}):
            gatelib.ensure_unloaded(summary, "u", {"card0": 0})
        self.assertFalse(summary["unload"]["pass"])
        self.assertTrue(summary["unload"]["vram_release"]["pass"])

    def test_cleanup_without_a_baseline_still_records_the_stop(self):
        summary: dict = {"pass": False}
        fake, calls = self.systemd_stub()
        with mock.patch.object(gatelib, "systemd", fake), \
             mock.patch.object(gatelib.time, "sleep"), \
             mock.patch.object(gatelib, "vram_used", return_value={"card0": 0}):
            gatelib.ensure_unloaded(summary, "u", None)
        self.assertIn(("stop", "u"), calls)
        self.assertIsNone(summary["unload"]["vram_release"])

    def test_cleanup_records_a_failure_instead_of_raising_out_of_a_finally(self):
        # Gates call this from a finally block. An exception escaping there would
        # replace the gate's own failure and skip record(), so the run would end
        # with no evidence file at all -- the one outcome a gate may not produce.
        summary: dict = {"pass": False}
        with mock.patch.object(gatelib, "systemd",
                               side_effect=FileNotFoundError("systemctl")):
            gatelib.ensure_unloaded(summary, "u", {"card0": 0})
        self.assertFalse(summary["unload"]["pass"])
        self.assertEqual("cleanup", summary["unload"]["ran_as"])
        self.assertIn("FileNotFoundError", summary["unload"]["cleanup_error"])

    def test_a_sidecar_that_will_not_stop_is_recorded_as_still_active(self):
        fake, _ = self.systemd_stub(active_after_stop=True)
        summary: dict = {"pass": False}
        with mock.patch.object(gatelib, "systemd", fake), \
             mock.patch.object(gatelib.time, "sleep"), \
             mock.patch.object(gatelib, "vram_used", return_value={"card0": 0}):
            gatelib.ensure_unloaded(summary, "u", {"card0": 0})
        self.assertTrue(summary["unload"]["still_active"])
        self.assertIsNone(summary["unload"]["vram_release"])

    def test_cleanup_waits_for_the_driver_to_hand_the_memory_back(self):
        # The unit going inactive is not the same instant the VRAM is free.
        # Sampling once at that moment reported residue that was about to clear.
        fake, _ = self.systemd_stub()
        readings = [{"card0": 6 << 30}, {"card0": 2 << 30}, {"card0": 0}]
        summary: dict = {"pass": False}
        with mock.patch.object(gatelib, "systemd", fake), \
             mock.patch.object(gatelib.time, "sleep"), \
             mock.patch.object(gatelib, "vram_used", side_effect=readings):
            gatelib.ensure_unloaded(summary, "u", {"card0": 0})
        self.assertTrue(summary["unload"]["vram_release"]["pass"])
        self.assertFalse(summary["unload"]["pass"], "cleanup still is not a passing check")


class SidecarGateContractTests(unittest.TestCase):
    """Every sidecar gate must route cleanup through the shared helper."""

    GATES = ("gate_embeddings.py", "gate_reranker.py", "gate_fim.py", "gate_ocr.py")

    def test_each_gate_stops_its_sidecar_in_a_finally_block(self):
        for name in self.GATES:
            source = (Path(__file__).resolve().parent / name).read_text()
            with self.subTest(gate=name):
                self.assertIn("ensure_unloaded(summary, UNIT, baseline)", source)
                self.assertIn("    finally:\n        ensure_unloaded(", source)


class ResidueArithmeticContractTests(unittest.TestCase):
    """No gate may score a clean unload with its own subtraction.

    Open-coded ``after[card] - baseline[card]`` is the shape of the bug this
    module exists to remove: an unreadable card reads -1, the difference goes
    hugely negative, and it clears any tolerance. Every gate that judges a
    release therefore goes through ``unload_verdict``.
    """

    REPOSITORY = Path("/home/typhoon/git/frankenstein-llm")
    SCORING_GATES = (
        "verification/local-coverage-foundation/validators/gate_asr.py",
        "verification/local-coverage-foundation/validators/gate_vision_grounding.py",
        "verification/tts-local/gate_tts.py",
        "verification/computer-use-grounding/gate_computer_use.py",
    )
    OPEN_CODED = re.compile(
        r"(?:released|settled|after)\[\s*c(?:ard)?\s*\]\s*-\s*baseline")

    def test_scoring_gates_use_the_shared_verdict(self):
        for relative in self.SCORING_GATES:
            source = (self.REPOSITORY / relative).read_text()
            with self.subTest(gate=relative):
                self.assertIn("unload_verdict(", source)
                self.assertIsNone(self.OPEN_CODED.search(source),
                                  "gate scores VRAM residue with its own subtraction")

    def test_the_router_gate_reports_a_missing_card_rather_than_assuming_it(self):
        source = (self.REPOSITORY / "verification/router-functional/gate_router_models.py").read_text()
        # The router keeps its own sampler, which omits unreadable cards instead
        # of reporting -1, so it must test membership rather than defaulting.
        self.assertIn('if card not in after["vram_used_bytes"]', source)
        self.assertNotIn('.get(card, base)', source)


if __name__ == "__main__":
    unittest.main()


class ExitAfterVerdictTests(unittest.TestCase):
    """The process must leave on the gate's verdict, not on ROCm's teardown.

    ROCm's HSA runtime segfaults in its own exit teardown on this host. On boot
    1669f3ad the ASR gate transcribed the fixture exactly, proved a clean unload,
    wrote ``"pass": true`` to gate-asr.json, and *then* died with SIGSEGV -- and
    the qualification recorded exit -11 as the gate's answer. These tests pin the two
    properties that keep that from happening again without softening anything:
    the code the gate computed is the code the process returns, and stdout that
    the gate printed is not lost to the bypassed finalization.
    """

    REPOSITORY = Path("/home/typhoon/git/frankenstein-llm")
    ROCM_GATES = (
        "verification/local-coverage-foundation/validators/gate_asr.py",
        "verification/computer-use-grounding/gate_computer_use.py",
    )

    def run_child(self, code: int) -> "tuple[int, str]":
        import subprocess
        program = (
            "import sys;"
            f"sys.path.insert(0, {str(self.REPOSITORY / 'verification/local-coverage-foundation/validators')!r});"
            "import gatelib;"
            "print('verdict written');"
            f"gatelib.exit_after_verdict({code})"
        )
        done = subprocess.run([sys.executable, "-c", program],
                              capture_output=True, text=True, timeout=60)
        return done.returncode, done.stdout

    def test_a_pass_leaves_the_process_with_zero(self):
        returncode, stdout = self.run_child(0)
        self.assertEqual(0, returncode)
        self.assertIn("verdict written", stdout)

    def test_a_failure_code_is_preserved_exactly(self):
        # The helper must not launder a failure into a pass; it only stops a
        # teardown fault from overwriting whichever code the gate decided on.
        for code in (1, 3, 5):
            with self.subTest(code=code):
                returncode, _ = self.run_child(code)
                self.assertEqual(code, returncode)

    def test_buffered_output_is_flushed_before_the_process_ends(self):
        # os._exit skips the flush that interpreter shutdown would have done, so
        # the helper has to do it. gate_tts.py reads a sibling gate's stdout to
        # score intelligibility; silently truncating it would be a new failure.
        returncode, stdout = self.run_child(0)
        self.assertEqual(0, returncode)
        self.assertEqual("verdict written\n", stdout)

    def test_rocm_gates_do_not_exit_through_interpreter_finalization(self):
        for relative in self.ROCM_GATES:
            source = (self.REPOSITORY / relative).read_text()
            with self.subTest(gate=relative):
                self.assertIn("exit_after_verdict(main())", source)
                self.assertNotIn("raise SystemExit(main())", source)
