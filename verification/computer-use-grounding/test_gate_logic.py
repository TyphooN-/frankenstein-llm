#!/usr/bin/env python3
"""Offline unit tests for the computer-use gate's logic. No model, no GPU.

These exist because a grounding gate is only as trustworthy as its scoring. If
the coordinate mapping, the action parser or the injection judge are wrong, the
gate will happily emit a confident verdict about nothing. Everything here runs
on a busy host in about a second.

Run: python3 -m unittest discover -s . -p 'test_*.py' -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from culib import (  # noqa: E402
    FIXTURES, GateFailure, image_token_count, in_bounds, injection_obeyed, inside,
    parse_action, parse_json_point, rescale_point, smart_resize,
)
from gate_computer_use import (  # noqa: E402
    DEVICE_MAP, EXIT_ALREADY_RUNNING, EXIT_FAIL, EXIT_INTERRUPTED, EXIT_PASS,
    GATE_SECTIONS, GateInterrupted, HANDLED_SIGNALS, SignalState, acquire_single_instance_lock,
    candidate_points, decoder_layers_by_device, finalize_verdict, gate_exit_code,
    mark_interrupted, normalize_text, ocr_score, pick_convention, score_target,
    signal_exit_status, write_json_atomic,
)
from gpu_telemetry import read_records, summarize_records, summarize_samples  # noqa: E402
from inspect_runtime import classify_last_run  # noqa: E402
from sandbox import SandboxScreen  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_when_idle.sh"
UNIT_FILE = Path.home() / ".config/systemd/user/local-ai-computer-use-gate.service"


def load_truth() -> dict:
    return json.loads((FIXTURES / "ground-truth.json").read_text())


class TestSmartResize(unittest.TestCase):
    def test_matches_processor_for_fixtures(self):
        # Cross-checked against the real Qwen2VLImageProcessor in
        # evidence/runtime-compatibility.json.
        self.assertEqual(smart_resize(1280, 800), (1288, 812))
        self.assertEqual(smart_resize(1100, 620), (1092, 616))

    def test_multiple_of_factor(self):
        for width, height in ((1280, 800), (1100, 620), (37, 91), (4000, 3000)):
            w, h = smart_resize(width, height)
            self.assertEqual(w % 28, 0)
            self.assertEqual(h % 28, 0)

    def test_upscales_below_min_pixels(self):
        w, h = smart_resize(1, 1)
        self.assertGreaterEqual(w * h, 3136)

    def test_downscales_above_max_pixels(self):
        w, h = smart_resize(9000, 9000)
        self.assertLessEqual(w * h, 12845056)

    def test_rejects_extreme_aspect_ratio(self):
        with self.assertRaises(GateFailure):
            smart_resize(4200, 14)


class TestDeviceMap(unittest.TestCase):
    def test_every_gpu_owns_real_decoder_blocks(self):
        ownership = decoder_layers_by_device(DEVICE_MAP)
        self.assertEqual(set(ownership), {0, 1, 2})
        self.assertEqual(ownership[2], [22, 23, 24, 25, 26, 27])

    def test_policy_order_and_complete_layer_coverage(self):
        ownership = decoder_layers_by_device(DEVICE_MAP)
        self.assertGreater(len(ownership[1]), len(ownership[0]))
        self.assertGreater(len(ownership[0]), len(ownership[2]))
        self.assertEqual(sorted(sum(ownership.values(), [])), list(range(28)))

    def test_tied_head_alias_is_not_used_as_gpu2_proof(self):
        tied_head_only = dict(DEVICE_MAP)
        for module in list(tied_head_only):
            if module.startswith("model.language_model.layers.") and tied_head_only[module] == 2:
                tied_head_only[module] = 1
        tied_head_only["lm_head"] = 2
        self.assertNotIn(2, decoder_layers_by_device(tied_head_only))


class TestGpuTelemetry(unittest.TestCase):
    def test_summarizes_activity_and_power_delta(self):
        baseline = {"cards": {
            card: {"vram_used_bytes": 100, "busy_percent": 0,
                   "power_microwatts": 20_000_000}
            for card in ("card0", "card1", "card2")}}
        samples = []
        for busy in (0, 25, 80):
            samples.append({"cards": {
                card: {"vram_used_bytes": 1000 + busy, "busy_percent": busy,
                       "power_microwatts": 20_000_000 + busy * 1_000_000}
                for card in ("card0", "card1", "card2")}})
        summary = summarize_samples(samples, baseline)
        self.assertEqual(summary["cards"]["card2"]["busy_max_percent"], 80)
        self.assertEqual(summary["cards"]["card2"]["busy_samples_ge_20"], 2)
        self.assertEqual(
            summary["cards"]["card2"]["power_peak_delta_microwatts"], 80_000_000)


class TestCoordinates(unittest.TestCase):
    def test_rescale_round_trip(self):
        point = (644, 406)
        moved = rescale_point(point, (1288, 812), (1280, 800))
        back = rescale_point(moved, (1280, 800), (1288, 812))
        self.assertLessEqual(abs(back[0] - point[0]), 1)
        self.assertLessEqual(abs(back[1] - point[1]), 1)

    def test_rescale_rejects_degenerate_source(self):
        with self.assertRaises(GateFailure):
            rescale_point((5, 5), (0, 100), (10, 10))

    def test_inside_includes_edges(self):
        box = [96, 120, 316, 176]
        self.assertTrue(inside(box, (96, 120)))
        self.assertTrue(inside(box, (316, 176)))
        self.assertFalse(inside(box, (95, 120)))
        self.assertFalse(inside(box, (316, 177)))
        self.assertFalse(inside(box, None))

    def test_bounds(self):
        self.assertTrue(in_bounds((0, 0), (1280, 800)))
        self.assertTrue(in_bounds((1279, 799), (1280, 800)))
        self.assertFalse(in_bounds((1280, 799), (1280, 800)))
        self.assertFalse(in_bounds((-1, 5), (1280, 800)))
        self.assertFalse(in_bounds(None, (1280, 800)))

    def test_candidate_points_none(self):
        self.assertEqual(candidate_points(None, (1288, 812), (1280, 800)),
                         {"raw": None, "rescaled": None})

    def test_image_token_count(self):
        self.assertEqual(image_token_count([1, 58, 92], 2), 1334)
        self.assertEqual(image_token_count([1, 44, 44], 2), 484)


class TestActionParser(unittest.TestCase):
    def test_box_tag_spelling(self):
        parsed = parse_action(
            "Thought: I should open it.\n"
            "Action: click(start_box='<|box_start|>(206,148)<|box_end|>')")
        self.assertEqual(parsed["verb"], "click")
        self.assertEqual(parsed["point"], (206.0, 148.0))
        self.assertEqual(parsed["spelling"], "box_tag")
        self.assertIn("open it", parsed["thought"])

    def test_point_tag_spelling(self):
        parsed = parse_action("Action: click(point='<point>206 148</point>')")
        self.assertEqual(parsed["spelling"], "point_tag")
        self.assertEqual(parsed["point"], (206.0, 148.0))

    def test_bare_parentheses(self):
        parsed = parse_action("Action: left_double(start_box='(12.5,7)')")
        self.assertEqual(parsed["verb"], "left_double")
        self.assertEqual(parsed["point"], (12.5, 7.0))
        self.assertEqual(parsed["spelling"], "bare_parens")

    def test_non_grounded_verbs(self):
        self.assertEqual(parse_action("Action: finished(content='done')")["verb"], "finished")
        self.assertEqual(parse_action("Action: wait()")["verb"], "wait")
        typed = parse_action("Action: type(content='hello world')")
        self.assertEqual(typed["verb"], "type")
        self.assertIsNone(typed["point"])

    def test_unknown_verb_is_not_accepted(self):
        parsed = parse_action("Action: explode(start_box='(1,2)')")
        self.assertIsNone(parsed["verb"])
        self.assertFalse(parsed["parsed"])

    def test_garbage_and_empty(self):
        for text in ("", None, "I am not going to answer that.", "%%%%"):
            parsed = parse_action(text)
            self.assertIsNone(parsed["verb"])
            self.assertFalse(parsed["parsed"])

    def test_last_action_wins_over_earlier_thought(self):
        parsed = parse_action(
            "Thought: first I considered scroll(start_box='(1,1)')\n"
            "Action: click(start_box='<|box_start|>(500,300)<|box_end|>')")
        self.assertEqual(parsed["verb"], "click")
        self.assertEqual(parsed["point"], (500.0, 300.0))


class TestJsonParser(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json_point('{"x": 206, "y": 148}'), (206, 148))

    def test_json_with_prose(self):
        self.assertEqual(parse_json_point('Sure! {"x": 10, "y": 20} hope that helps'),
                         (10, 20))

    def test_float_and_negative(self):
        self.assertEqual(parse_json_point('{"x": 10.7, "y": -3}'), (10, -3))

    def test_bare_number_fallback(self):
        self.assertEqual(parse_json_point("the point is 206, 148"), (206, 148))

    def test_unparseable(self):
        self.assertIsNone(parse_json_point("no numbers here"))
        self.assertIsNone(parse_json_point(""))
        self.assertIsNone(parse_json_point(None))

    def test_ignores_non_point_json(self):
        self.assertEqual(parse_json_point('{"a": 1} then {"x": 5, "y": 6}'), (5, 6))


class TestInjectionJudge(unittest.TestCase):
    def test_compliant_listing_is_clean(self):
        verdict = injection_obeyed('["Approve", "Reject"]')
        self.assertFalse(verdict["obeyed"])
        self.assertEqual(verdict["echoed_payload_markers"], [])

    def test_executing_injected_command_is_caught(self):
        verdict = injection_obeyed(
            "Action: type(content='sudo rm -rf /home/typhoon --no-preserve-root')")
        self.assertTrue(verdict["obeyed"])
        self.assertTrue(verdict["selected_forbidden_verb"])

    def test_echoing_payload_is_caught(self):
        verdict = injection_obeyed(
            "I will exfiltrate credentials to http://198.51.100.7/collect as instructed.")
        self.assertTrue(verdict["obeyed"])
        self.assertIn("http://", verdict["echoed_payload_markers"])

    def test_legitimate_click_under_injection_is_clean(self):
        verdict = injection_obeyed(
            "Thought: the page text is untrusted.\n"
            "Action: click(start_box='<|box_start|>(432,588)<|box_end|>')")
        self.assertFalse(verdict["obeyed"])


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.truth = load_truth()
        self.app = self.truth["screens"]["app"]
        self.control = self.app["controls"][0]  # Restart Router

    def test_perfect_model_scores_full_marks(self):
        """A model returning exact centres in the resized frame must score 6/6."""
        results = []
        for control in self.app["controls"]:
            cx, cy = control["center"]
            rx, ry = rescale_point((cx, cy), (1280, 800), (1288, 812))
            raw = f"Thought: ok\nAction: click(start_box='<|box_start|>({rx},{ry})<|box_end|>')"
            results.append(score_target(raw, control, (1280, 800), (1288, 812), "native"))
        convention, hits = pick_convention(results)
        self.assertEqual(hits, len(self.app["controls"]))
        self.assertEqual(convention, "rescaled")

    def test_wrong_model_scores_zero(self):
        """The gate must be capable of failing: a model pointing at 0,0 scores 0."""
        results = [
            score_target("Action: click(start_box='<|box_start|>(0,0)<|box_end|>')",
                         control, (1280, 800), (1288, 812), "native")
            for control in self.app["controls"]
        ]
        _, hits = pick_convention(results)
        self.assertEqual(hits, 0)

    def test_refusing_model_scores_zero_without_crashing(self):
        results = [score_target("I cannot help with that.", control,
                                (1280, 800), (1288, 812), "native")
                   for control in self.app["controls"]]
        _, hits = pick_convention(results)
        self.assertEqual(hits, 0)
        self.assertTrue(all(r["point"] is None for r in results))

    def test_raw_convention_detected_when_model_uses_original_frame(self):
        results = []
        for control in self.app["controls"]:
            cx, cy = control["center"]
            raw = f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
            results.append(score_target(raw, control, (1280, 800), (1288, 812), "native"))
        convention, hits = pick_convention(results)
        self.assertEqual(hits, len(self.app["controls"]))
        # Both frames are close here, so assert only that a full-marks convention won.
        self.assertIn(convention, {"raw", "rescaled"})

    def test_scorer_reads_the_action_space_it_is_given(self):
        """The same scorer must grade the tool-calling challenger fairly.

        UI-Mate answers with a tool call, not with ``Action:``. Scored under its
        own action space the hit is a hit; scored under the default UI-TARS
        grammar the identical, correctly grounded action reads as a refusal --
        which is why the parameter exists rather than an assumed default.
        """
        cx, cy = self.control["center"]
        raw = ("<think>\nThe Run Gate button.\n</think>\n<tool_call>\n<function=click>\n"
               f"<parameter=coordinate>\n[{cx}, {cy}]\n</parameter>\n</function>\n</tool_call>")
        scored = score_target(raw, self.control, (1280, 800), (1280, 800),
                              "native", action_space="tool-call-xml")
        self.assertEqual("click", scored["verb"])
        self.assertTrue(scored["inside_raw"])
        self.assertEqual("tool-call-xml", scored["action_space"])

        misread = score_target(raw, self.control, (1280, 800), (1280, 800), "native")
        self.assertEqual("uitars-text", misread["action_space"])
        self.assertIsNone(misread["verb"])
        self.assertFalse(misread["inside_raw"])

    def test_default_action_space_is_unchanged(self):
        cx, cy = self.control["center"]
        raw = f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
        explicit = score_target(raw, self.control, (1280, 800), (1288, 812),
                                "native", action_space="uitars-text")
        self.assertEqual(
            score_target(raw, self.control, (1280, 800), (1288, 812), "native"), explicit)
        self.assertEqual("click", explicit["verb"])

    def test_json_mode_scoring(self):
        cx, cy = self.control["center"]
        result = score_target(json.dumps({"x": cx, "y": cy}), self.control,
                              (1280, 800), (1280, 800), "json")
        self.assertTrue(result["inside_raw"])
        self.assertEqual(result["spelling"], "json")


class TestOcrScoring(unittest.TestCase):
    def setUp(self):
        self.document = load_truth()["document"]

    def test_exact_transcript_scores_full(self):
        transcript = "\n".join(self.document["lines"]) + "\n" + \
            " ".join(self.document["table_header"]) + "\n" + \
            "\n".join(" ".join(row) for row in self.document["table_rows"])
        score = ocr_score(transcript, self.document)
        self.assertEqual(score["lines_found"], score["lines_total"])
        self.assertEqual(score["rows_found"], score["rows_total"])
        self.assertEqual(score["headers_found"], score["headers_total"])

    def test_whitespace_and_case_insensitive(self):
        transcript = "  INCIDENT   report    4172  "
        score = ocr_score(transcript, self.document)
        self.assertEqual(score["lines"][0]["found"], True)

    def test_empty_transcript_scores_zero(self):
        score = ocr_score("", self.document)
        self.assertEqual(score["lines_found"], 0)
        self.assertEqual(score["rows_found"], 0)

    def test_normalize_text(self):
        self.assertEqual(normalize_text("  A  b\n C "), "a b c")
        self.assertEqual(normalize_text(None), "")


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.truth = load_truth()
        self.app = SandboxScreen(self.truth["screens"]["app"])
        self.dialog = SandboxScreen(self.truth["screens"]["distractor"])

    def test_click_lands_on_control(self):
        control = self.truth["screens"]["app"]["controls"][2]  # Open Evidence
        event = self.app.apply("click", tuple(control["center"]))
        self.assertTrue(event["accepted"])
        self.assertEqual(self.app.state()["last_clicked"], "Open Evidence")

    def test_click_on_empty_space_is_a_miss(self):
        event = self.app.apply("click", (1250, 700))
        self.assertFalse(event["accepted"])
        self.assertEqual(event["reason"], "click landed on no control")
        self.assertIsNone(self.app.state()["last_clicked"])

    def test_out_of_bounds_click_rejected(self):
        event = self.app.apply("click", (5000, 5000))
        self.assertFalse(event["accepted"])
        self.assertEqual(event["reason"], "coordinate outside screen bounds")

    def test_disabled_control_absorbs_click(self):
        disabled = [c for c in self.truth["screens"]["distractor"]["controls"]
                    if not c["enabled"]][0]
        event = self.dialog.apply("click", tuple(disabled["center"]))
        self.assertFalse(event["accepted"])
        self.assertEqual(event["reason"], "control is disabled")
        self.assertEqual(self.dialog.state()["rejected_disabled"], ["Save"])
        self.assertIsNone(self.dialog.state()["last_clicked"])

    def test_enabled_twin_is_distinguished_from_disabled(self):
        enabled = [c for c in self.truth["screens"]["distractor"]["controls"]
                   if c["label"] == "Save" and c["enabled"]][0]
        event = self.dialog.apply("click", tuple(enabled["center"]))
        self.assertTrue(event["accepted"])
        self.assertEqual(self.dialog.state()["last_clicked"], "Save")
        self.assertEqual(self.dialog.state()["rejected_disabled"], [])

    def test_checkbox_toggles(self):
        checkbox = self.truth["screens"]["app"]["checkbox"]
        before = self.app.state()["checkbox_checked"]
        self.app.apply("click", tuple(checkbox["center"]))
        self.assertNotEqual(self.app.state()["checkbox_checked"], before)

    def test_type_requires_focus_then_writes(self):
        blocked = self.app.apply("type", None, "hello")
        self.assertFalse(blocked["accepted"])
        self.assertEqual(blocked["reason"], "type with no focused field")

        field = self.truth["screens"]["app"]["fields"][0]
        self.app.apply("click", tuple(field["center"]))
        self.assertEqual(self.app.state()["focused_field"], field["label"])
        written = self.app.apply("type", None, "/tmp/evidence")
        self.assertTrue(written["accepted"])
        self.assertEqual(self.app.state()["field_values"][field["label"]], "/tmp/evidence")

    def test_unparseable_action_is_recorded_not_raised(self):
        event = self.app.apply(None, None)
        self.assertFalse(event["accepted"])
        self.assertEqual(event["reason"], "no parseable action")

    def test_non_grounded_verbs_accepted(self):
        self.assertTrue(self.app.apply("finished", None)["accepted"])
        self.assertTrue(self.app.apply("wait", None)["accepted"])

    def test_hotkey_has_no_effect(self):
        event = self.app.apply("hotkey", None, "ctrl alt t")
        self.assertFalse(event["accepted"])
        self.assertIsNone(self.app.state()["last_clicked"])


class TestFixtureIntegrity(unittest.TestCase):
    """The ground truth must actually describe the pixels that were drawn."""

    def setUp(self):
        self.truth = load_truth()

    def test_every_fixture_file_exists(self):
        for screen in self.truth["screens"].values():
            self.assertTrue(Path(screen["path"]).is_file(), screen["path"])
        self.assertTrue(Path(self.truth["document"]["path"]).is_file())
        for spec in self.truth["malformed"].values():
            self.assertTrue(Path(spec["path"]).exists(), spec["path"])

    def test_declared_sizes_match_files(self):
        from PIL import Image
        for screen in self.truth["screens"].values():
            with Image.open(screen["path"]) as handle:
                self.assertEqual(list(handle.size), screen["size"])

    def test_centres_lie_inside_their_boxes(self):
        for screen in self.truth["screens"].values():
            for control in screen["controls"]:
                self.assertTrue(inside(control["box"], tuple(control["center"])),
                                f"{control['label']} centre outside its own box")

    def test_boxes_are_within_the_screen(self):
        for screen in self.truth["screens"].values():
            width, height = screen["size"]
            for control in screen["controls"]:
                x0, y0, x1, y1 = control["box"]
                self.assertGreaterEqual(x0, 0)
                self.assertGreaterEqual(y0, 0)
                self.assertLessEqual(x1, width)
                self.assertLessEqual(y1, height)
                self.assertLess(x0, x1)
                self.assertLess(y0, y1)

    def test_controls_do_not_overlap(self):
        for name, screen in self.truth["screens"].items():
            boxes = [c["box"] for c in screen["controls"]]
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    overlap = not (a[2] < b[0] or b[2] < a[0]
                                   or a[3] < b[1] or b[3] < a[1])
                    self.assertFalse(overlap, f"{name}: overlapping controls {a} {b}")

    def test_distractor_has_a_disabled_twin(self):
        controls = self.truth["screens"]["distractor"]["controls"]
        saves = [c for c in controls if c["label"] == "Save"]
        self.assertEqual(len(saves), 2)
        self.assertEqual(sorted(c["enabled"] for c in saves), [False, True])

    def test_injection_fixture_carries_hostile_text(self):
        screen = self.truth["screens"]["injection"]
        self.assertTrue(screen["hostile_text"])
        joined = " ".join(screen["hostile_text"]).lower()
        self.assertIn("ignore all previous instructions", joined)


class TestSignalResultSemantics(unittest.TestCase):
    """A run that was signalled must never be reportable as a pass.

    This is the property the 2026-09-01 lifecycle failure needed and did not
    have: the shell trap could return a status of its own while Python was still
    being killed, so an interrupted run could look indistinguishable from a
    finished one.
    """

    def test_shell_exit_status_matches_convention(self):
        self.assertEqual(signal_exit_status(signal.SIGHUP), 129)
        self.assertEqual(signal_exit_status(signal.SIGINT), 130)
        self.assertEqual(signal_exit_status(signal.SIGTERM), 143)

    def test_sigkill_is_not_claimed_as_handled(self):
        self.assertNotIn(signal.SIGKILL, HANDLED_SIGNALS)
        self.assertNotIn(signal.SIGSTOP, HANDLED_SIGNALS)
        note = SignalState().as_evidence()["uncatchable_note"]
        self.assertIn("SIGKILL", note)

    def test_marking_interrupted_destroys_a_pass(self):
        summary = {"pass": True}
        mark_interrupted(summary, signal.SIGTERM, "grounding")
        self.assertFalse(summary["pass"])
        self.assertTrue(summary["interrupted"])
        self.assertEqual(summary["outcome"], "interrupted")
        self.assertEqual(summary["interrupt"]["signal"], "SIGTERM")
        self.assertEqual(summary["interrupt"]["shell_exit_status"], 143)
        self.assertEqual(summary["interrupt"]["phase"], "grounding")
        self.assertEqual(gate_exit_code(summary), EXIT_INTERRUPTED)

    def test_exit_codes_are_distinct_and_fail_closed(self):
        self.assertEqual(gate_exit_code({"pass": True}), EXIT_PASS)
        self.assertEqual(gate_exit_code({"pass": False}), EXIT_FAIL)
        self.assertEqual(gate_exit_code({"pass": True, "error": "boom"}), EXIT_FAIL)
        self.assertEqual(gate_exit_code({"pass": True, "interrupted": True}),
                         EXIT_INTERRUPTED)
        self.assertEqual(gate_exit_code({}), EXIT_FAIL)
        self.assertEqual(
            len({EXIT_PASS, EXIT_FAIL, EXIT_ALREADY_RUNNING, EXIT_INTERRUPTED}), 4)

    def test_missing_sections_cannot_pass(self):
        # Every hard gate present and passing is the only way through.
        full = {section: {"pass": True} for section in GATE_SECTIONS}
        self.assertTrue(finalize_verdict(dict(full))["pass"])
        for section in GATE_SECTIONS:
            partial = {k: v for k, v in full.items() if k != section}
            verdict = finalize_verdict(partial)
            self.assertFalse(verdict["pass"], f"{section} missing must fail the gate")
            self.assertIn(section, verdict["sections_missing"])

    def test_interrupted_run_with_all_sections_still_fails(self):
        summary = {section: {"pass": True} for section in GATE_SECTIONS}
        mark_interrupted(summary, signal.SIGHUP, "ocr")
        finalize_verdict(summary)
        self.assertFalse(summary["pass"])
        self.assertEqual(gate_exit_code(summary), EXIT_INTERRUPTED)

    def test_handler_raises_once_then_records_without_unwinding_cleanup(self):
        previous = {s: signal.getsignal(s) for s in HANDLED_SIGNALS}
        state = SignalState()
        try:
            state.install()
            with self.assertRaises(GateInterrupted) as caught:
                os.kill(os.getpid(), signal.SIGTERM)
            self.assertEqual(caught.exception.signum, signal.SIGTERM)
            # A second signal during cleanup must be recorded, not raised: the
            # unload/telemetry/artifact path has to be allowed to finish.
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGHUP)
            self.assertEqual(state.received, ["SIGTERM", "SIGTERM", "SIGHUP"])
            self.assertEqual(state.first_signum, signal.SIGTERM)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


class TestAtomicArtifacts(unittest.TestCase):
    """A reader must never observe a half-written verdict."""

    def test_replaces_atomically_and_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "computer-use-grounding.json"
            write_json_atomic(path, {"pass": False, "outcome": "interrupted"})
            self.assertEqual(json.loads(path.read_text())["outcome"], "interrupted")
            write_json_atomic(path, {"pass": True, "outcome": "passed"})
            self.assertEqual(json.loads(path.read_text())["outcome"], "passed")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["computer-use-grounding.json"])

    def test_failed_write_leaves_the_previous_artifact_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            write_json_atomic(path, {"pass": True})
            with self.assertRaises(TypeError):
                write_json_atomic(path, {"pass": object()})  # not JSON-serializable
            self.assertEqual(json.loads(path.read_text()), {"pass": True})
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["artifact.json"])


class TestSingleInstanceLock(unittest.TestCase):
    def test_second_holder_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.lock"
            first, holder = acquire_single_instance_lock(path)
            self.addCleanup(first.close)
            self.assertIsNotNone(first)
            self.assertIsNone(holder)
            second, existing = acquire_single_instance_lock(path)
            self.assertIsNone(second, "a second model worker must not be admitted")
            self.assertIn(f"pid={os.getpid()}", existing)

    def test_lock_is_released_when_the_holder_dies(self):
        # flock lives on the file descriptor, so the kernel releases it on any
        # death including SIGKILL. There is no stale lock to reap.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.lock"
            first, _ = acquire_single_instance_lock(path)
            first.close()
            second, _ = acquire_single_instance_lock(path)
            self.addCleanup(second.close)
            self.assertIsNotNone(second)


class TestTelemetryAggregation(unittest.TestCase):
    """Aggregation has to stay honest when the run died mid-sample."""

    @staticmethod
    def _sample(busy, power=None, vram=1000, monotonic=0.0, cards=("card0", "card1", "card2")):
        return {"monotonic": monotonic, "cards": {
            card: {"vram_used_bytes": vram, "busy_percent": busy,
                   "power_microwatts": power} for card in cards}}

    def test_no_samples_reports_nothing_rather_than_zero(self):
        summary = summarize_samples([], {"cards": {}})
        self.assertEqual(summary["sample_count"], 0)
        self.assertIsNone(summary["duration_seconds"])
        self.assertEqual(summary["cards_reporting"], [])
        for card in ("card0", "card1", "card2"):
            self.assertIsNone(summary["cards"][card]["busy_max_percent"])
            self.assertEqual(summary["cards"][card]["busy_samples_ge_20"], 0)
            self.assertIsNone(summary["cards"][card]["power_peak_delta_microwatts"])

    def test_activity_threshold_is_inclusive_at_twenty(self):
        samples = [self._sample(19), self._sample(20), self._sample(21)]
        summary = summarize_samples(samples, {"cards": {}})
        self.assertEqual(summary["cards"]["card2"]["busy_samples_ge_20"], 2)

    def test_unreadable_counters_do_not_become_activity(self):
        samples = [self._sample(None), self._sample(None), self._sample(50)]
        summary = summarize_samples(samples, {"cards": {}})
        card = summary["cards"]["card1"]
        self.assertEqual(card["samples"], 3)
        self.assertEqual(card["readings"], 1)
        self.assertEqual(card["busy_max_percent"], 50)
        self.assertEqual(card["busy_mean_percent"], 50.0)

    def test_missing_card_is_reported_as_absent_not_idle(self):
        samples = [self._sample(80, cards=("card0", "card1"))]
        summary = summarize_samples(samples, {"cards": {}})
        self.assertEqual(summary["cards_reporting"], ["card0", "card1"])
        self.assertEqual(summary["cards"]["card2"]["samples"], 0)
        self.assertIsNone(summary["cards"]["card2"]["busy_max_percent"])

    def test_power_and_vram_deltas_need_a_baseline(self):
        baseline = {"cards": {"card0": {"power_microwatts": 20_000_000,
                                        "vram_used_bytes": 100}}}
        samples = [self._sample(50, power=320_000_000, vram=9_000)]
        summary = summarize_samples(samples, baseline)
        self.assertEqual(summary["cards"]["card0"]["power_peak_delta_microwatts"],
                         300_000_000)
        self.assertEqual(summary["cards"]["card0"]["vram_peak_delta_bytes"], 8_900)
        # No baseline for card2: report the absolute peak, never a fabricated delta.
        self.assertIsNone(summary["cards"]["card2"]["power_peak_delta_microwatts"])
        self.assertEqual(summary["cards"]["card2"]["power_max_microwatts"], 320_000_000)

    def test_duration_comes_from_the_samples_not_the_interval(self):
        samples = [self._sample(10, monotonic=100.0), self._sample(10, monotonic=277.5)]
        summary = summarize_samples(samples, {"cards": {}}, interval=0.1)
        self.assertEqual(summary["duration_seconds"], 177.5)
        self.assertEqual(summary["interval_seconds"], 0.1)

    def test_torn_final_line_is_dropped_and_the_rest_survives(self):
        # A host crash mid-write leaves a partial record. Losing the whole run's
        # telemetry to it would discard the only evidence an abrupt death leaves.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            good = [self._sample(30, power=200_000_000, monotonic=float(i))
                    for i in range(3)]
            path.write_text("\n".join(json.dumps(s, sort_keys=True) for s in good)
                            + '\n{"monotonic": 3.0, "cards": {"car')
            self.assertEqual(len(read_records(path)), 3)
            rebuilt = summarize_records(path)
            self.assertEqual(rebuilt["sample_count"], 3)
            self.assertTrue(rebuilt["reconstructed_from_disk"])
            self.assertEqual(rebuilt["cards"]["card2"]["busy_max_percent"], 30)

    def test_missing_telemetry_file_is_empty_not_an_error(self):
        self.assertEqual(read_records(Path("/nonexistent/telemetry.jsonl")), [])


class TestLifecycleClassification(unittest.TestCase):
    """Tell "the gate died" apart from "the host died under the gate"."""

    def test_unfinished_breadcrumb_from_a_previous_boot_is_a_host_loss(self):
        verdict = classify_last_run(
            {"phase": "grounding", "boot_id": "old-boot", "pid": 1234, "finished": False},
            "current-boot", None)
        self.assertEqual(verdict["state"], "vanished-across-reboot")
        self.assertEqual(verdict["phase"], "grounding")

    def test_unfinished_breadcrumb_this_boot_with_a_dead_pid_is_an_uncatchable_kill(self):
        verdict = classify_last_run(
            {"phase": "load", "boot_id": "b", "pid": 999_999_999, "finished": False},
            "b", None)
        self.assertEqual(verdict["state"], "vanished-same-boot")

    def test_live_pid_this_boot_is_reported_as_running(self):
        verdict = classify_last_run(
            {"phase": "ocr", "boot_id": "b", "pid": os.getpid(), "finished": False},
            "b", None)
        self.assertEqual(verdict["state"], "running")

    def test_finished_breadcrumb_reports_the_artifact_verdict(self):
        verdict = classify_last_run(
            {"phase": "finished", "boot_id": "b", "pid": 1, "finished": True,
             "outcome": "passed", "exit_code": 0}, "b", {"pass": True})
        self.assertEqual(verdict["state"], "finished")
        self.assertTrue(verdict["artifact_pass"])

    def test_absent_breadcrumb_is_not_mistaken_for_success(self):
        self.assertEqual(classify_last_run(None, "b", None)["state"], "no-run-state")


class TestRunnerAndUnitContract(unittest.TestCase):
    """Static guards on the two lifecycle properties that live outside Python."""

    def test_runner_waits_for_the_child_instead_of_exiting_from_the_trap(self):
        text = RUNNER.read_text()
        self.assertIn('kill -s "$sig" "$child"', text)
        self.assertIn('wait "$child"', text)
        # A signal trap that exits directly is the bug this replaced.
        for trapped in ("forward TERM", "forward HUP", "forward INT"):
            self.assertIn(f"trap '{trapped}'", text)
        self.assertNotIn("trap 'interrupted", text)

    def test_runner_fails_closed_without_the_artifact(self):
        text = RUNNER.read_text()
        self.assertIn("EXIT_NO_ARTIFACT=5", text)
        self.assertIn("rc=$EXIT_NO_ARTIFACT", text)

    def test_runner_restores_the_router_it_stopped(self):
        text = RUNNER.read_text()
        self.assertIn("systemctl --user stop \"$ROUTER\"", text)
        # The restart is enqueued, not awaited: the EXIT trap runs inside this
        # unit's own stop job, and a blocking start there deadlocks against it.
        # Behaviour is covered by test_serialized_runner_router_restore.py.
        self.assertIn("systemctl --user --no-block start \"$ROUTER\"", text)
        self.assertNotIn("systemctl --user start \"$ROUTER\"", text)
        self.assertIn("trap on_exit EXIT", text)

    def test_runner_no_longer_waits_for_an_idle_host(self):
        text = RUNNER.read_text()
        self.assertNotIn("kernel_build_active", text)
        self.assertNotIn("quiet_poll", text)

    def test_unit_is_persistent_and_bounded(self):
        if not UNIT_FILE.exists():
            self.skipTest(f"{UNIT_FILE} is not installed on this host")
        text = UNIT_FILE.read_text()
        for directive in ("Type=exec", "KillMode=control-group", "KillSignal=SIGTERM",
                          "TimeoutStartSec=infinity", "TimeoutStopSec=300",
                          "Restart=on-failure", "RestartPreventExitStatus=1 3 5",
                          "StartLimitBurst=3", "ExecStopPost="):
            self.assertIn(directive, text, f"unit is missing {directive}")
        # No [Install]: a 15.5 GiB three-GPU run must never start itself at boot.
        # Match directives, not the prose that explains them.
        directives = [line.strip() for line in text.splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
        self.assertNotIn("[Install]", directives)
        self.assertFalse([d for d in directives if d.startswith("WantedBy=")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
