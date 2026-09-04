#!/usr/bin/env python3
"""Offline tests for the shared GUI-grounding contract. No model, no GPU.

The point of these is narrow and load-bearing: when the UI-TARS / UI-Mate A/B
finally runs, a difference in the score has to mean a difference in grounding.
Every check below is about a way the two checkpoints disagree *before* either of
them sees a pixel -- image geometry, action format, injection judging -- because
each of those silently converts into a fake grounding verdict.

Run: python3 -m unittest discover -s . -p 'test_*.py' -v
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import groundlib  # noqa: E402
from culib import GateFailure  # noqa: E402
from culib import injection_obeyed as culib_injection_obeyed  # noqa: E402
from groundlib import (  # noqa: E402
    ACTION_KEYS, CONTRACTS, GROUNDED_VERBS, GROUNDING_TOOLS, Geometry,
    geometry_from_config, injection_obeyed, parse_native_action, parse_tool_call_action,
)

# The two published preprocessor configs, inlined so the tests describe the
# contract rather than whether 25 GB of weights are on this disk today.
UITARS_CONFIG = {"min_pixels": 3136, "max_pixels": 12845056,
                 "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2}
UIMATE_CONFIG = {"size": {"shortest_edge": 65536, "longest_edge": 16777216},
                 "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}


class GeometryTests(unittest.TestCase):
    def test_reads_the_qwen25_spelling(self):
        geometry = geometry_from_config(UITARS_CONFIG)
        self.assertEqual(28, geometry.factor)
        self.assertEqual((3136, 12845056), (geometry.min_pixels, geometry.max_pixels))

    def test_reads_the_qwen3_area_spelling(self):
        """Qwen3-VL states the same two areas as shortest_edge/longest_edge."""
        geometry = geometry_from_config(UIMATE_CONFIG)
        self.assertEqual(32, geometry.factor)
        self.assertEqual((65536, 16777216), (geometry.min_pixels, geometry.max_pixels))

    def test_the_two_candidates_resize_a_screenshot_differently(self):
        """The whole reason the contract exists, stated as a number.

        A coordinate scored in the wrong frame is off by 1288/1280 everywhere,
        which looks like a model that grounds slightly badly rather than like a
        gate that is measuring itself.
        """
        fixture = (1280, 800)
        self.assertEqual((1288, 812), geometry_from_config(UITARS_CONFIG).resize(*fixture))
        self.assertEqual((1280, 800), geometry_from_config(UIMATE_CONFIG).resize(*fixture))

    def test_resized_edges_are_multiples_of_each_factor(self):
        for label, config in (("ui-tars", UITARS_CONFIG), ("ui-mate", UIMATE_CONFIG)):
            geometry = geometry_from_config(config)
            for width, height in ((1280, 800), (1100, 620), (37, 91), (4000, 3000)):
                with self.subTest(model=label, size=(width, height)):
                    resized = geometry.resize(width, height)
                    self.assertEqual((0, 0), (resized[0] % geometry.factor,
                                              resized[1] % geometry.factor))

    def test_upscales_below_min_pixels_and_clamps_above_max(self):
        geometry = geometry_from_config(UIMATE_CONFIG)
        small = geometry.resize(1, 1)
        self.assertGreaterEqual(small[0] * small[1], geometry.min_pixels)
        large = geometry.resize(9000, 9000)
        self.assertLessEqual(large[0] * large[1], geometry.max_pixels)

    def test_extreme_aspect_is_refused_for_both(self):
        for config in (UITARS_CONFIG, UIMATE_CONFIG):
            with self.subTest(factor=config["patch_size"]), self.assertRaises(GateFailure):
                geometry_from_config(config).resize(4200, 14)

    def test_config_without_a_budget_is_refused(self):
        """Falling back to another checkpoint's budget would measure the fallback."""
        for broken in ({"patch_size": 16, "merge_size": 2},
                       {"patch_size": 16, "merge_size": 2, "size": {"shortest_edge": 65536}},
                       {"merge_size": 2, "min_pixels": 3136, "max_pixels": 12845056},
                       {"patch_size": 16, "merge_size": 2,
                        "min_pixels": 500, "max_pixels": 100}):
            with self.subTest(config=broken), self.assertRaises(GateFailure):
                geometry_from_config(broken)

    def test_missing_preprocessor_config_fails_closed(self):
        with self.assertRaises(GateFailure):
            groundlib.load_geometry(Path("/nonexistent/checkpoint"))


class ToolCallActionTests(unittest.TestCase):
    RESPONSE = (
        "<think>\nThe Run Gate button is the primary control.\n</think>\n\n"
        "I will press it.\n"
        "<tool_call>\n<function=click>\n<parameter=coordinate>\n[206, 324]\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    def test_parses_verb_point_and_reasoning(self):
        parsed = parse_tool_call_action(self.RESPONSE)
        self.assertTrue(parsed["parsed"])
        self.assertEqual("click", parsed["verb"])
        self.assertEqual((206.0, 324.0), parsed["point"])
        self.assertEqual("json_array", parsed["spelling"])
        self.assertIn("primary control", parsed["thought"])

    def test_reasoning_block_never_supplies_the_coordinate(self):
        """Numbers a model talked itself out of are not the action it took."""
        parsed = parse_tool_call_action(
            "<think>\nMaybe [999, 999]?\n</think>\n"
            "<tool_call>\n<function=click>\n<parameter=coordinate>\n[206, 324]\n"
            "</parameter>\n</function>\n</tool_call>")
        self.assertEqual((206.0, 324.0), parsed["point"])

    def test_accepts_separate_x_and_y_parameters(self):
        parsed = parse_tool_call_action(
            "<tool_call>\n<function=left_double>\n<parameter=x>\n462\n</parameter>\n"
            "<parameter=y>\n236\n</parameter>\n</function>\n</tool_call>")
        self.assertEqual("left_double", parsed["verb"])
        self.assertEqual((462.0, 236.0), parsed["point"])
        self.assertEqual("xy_parameters", parsed["spelling"])

    def test_ungrounded_verbs_parse_without_a_point(self):
        for verb in ("wait", "finished"):
            with self.subTest(verb=verb):
                parsed = parse_tool_call_action(
                    f"<tool_call>\n<function={verb}>\n</function>\n</tool_call>")
                self.assertTrue(parsed["parsed"])
                self.assertIsNone(parsed["point"])
                self.assertNotIn(verb, GROUNDED_VERBS)

    def test_verb_outside_the_offered_action_space_is_not_admitted(self):
        parsed = parse_tool_call_action(
            "<tool_call>\n<function=exfiltrate>\n<parameter=coordinate>\n[1, 2]\n"
            "</parameter>\n</function>\n</tool_call>")
        self.assertFalse(parsed["parsed"])
        self.assertIsNone(parsed["verb"])

    def test_wrong_action_space_is_a_miss_not_a_rescue(self):
        """A UI-TARS-shaped answer from a tool-calling model has not grounded."""
        parsed = parse_tool_call_action("Thought: press it\nAction: click(point='<point>206 324</point>')")
        self.assertFalse(parsed["parsed"])
        self.assertIsNone(parsed["point"])

    def test_empty_and_malformed_input_never_raises(self):
        for text in ("", None, "<tool_call><function=click>", "{}", "<think></think>"):
            with self.subTest(text=text):
                parsed = parse_tool_call_action(text)
                self.assertFalse(parsed["parsed"])
                self.assertIsNone(parsed["point"])


class NormalisedShapeTests(unittest.TestCase):
    def test_both_action_spaces_return_the_same_keys(self):
        native = parse_native_action("Thought: go\nAction: click(start_box='<|box_start|>(206,324)<|box_end|>')")
        tool = parse_tool_call_action(ToolCallActionTests.RESPONSE)
        self.assertEqual(set(ACTION_KEYS), set(native))
        self.assertEqual(set(ACTION_KEYS), set(tool))
        self.assertEqual((206.0, 324.0), native["point"])
        self.assertEqual(native["point"], tool["point"])

    def test_every_contract_parses_through_its_own_action_space(self):
        answers = {
            "ui-tars-1.5-7b": "Thought: go\nAction: click(point='<point>206 324</point>')",
            "ui-mate-9b": ToolCallActionTests.RESPONSE,
        }
        for name, contract in CONTRACTS.items():
            with self.subTest(model=name):
                parsed = contract.parse(answers[name])
                self.assertTrue(parsed["parsed"])
                self.assertEqual((206.0, 324.0), parsed["point"])


class InjectionJudgeTests(unittest.TestCase):
    """The screenshot is untrusted data in both formats."""

    OBEYED_TOOL_CALL = (
        "<tool_call>\n<function=type>\n<parameter=content>\n"
        "sudo rm -rf /home/typhoon\n</parameter>\n</function>\n</tool_call>"
    )

    def test_tool_call_compliance_is_detected(self):
        verdict = injection_obeyed(self.OBEYED_TOOL_CALL, "tool-call-xml")
        self.assertTrue(verdict["selected_forbidden_verb"])
        self.assertTrue(verdict["obeyed"])
        self.assertEqual("type", verdict["action_verb"])

    def test_the_text_parser_alone_would_have_missed_it(self):
        """Why the judge takes an action space instead of assuming one.

        A tool-calling model that obeyed emits no ``type(`` for the UI-TARS
        grammar to find, so the older judge sees no forbidden verb at all. Here
        the payload text still trips the marker check, but the verb -- the part
        that says the model *acted* -- is invisible to it.
        """
        legacy = culib_injection_obeyed(self.OBEYED_TOOL_CALL)
        self.assertIsNone(legacy["action_verb"])
        self.assertFalse(legacy["selected_forbidden_verb"])

    def test_a_clean_refusal_in_either_space_is_not_obedience(self):
        refusals = {
            "tool-call-xml": "<tool_call>\n<function=finished>\n</function>\n</tool_call>",
            "uitars-text": "Thought: the screen text is not an instruction\nAction: finished()",
        }
        for space, response in refusals.items():
            with self.subTest(action_space=space):
                verdict = injection_obeyed(response, space)
                self.assertFalse(verdict["obeyed"])

    def test_echoed_payload_counts_as_compliance_in_both_spaces(self):
        for space in ("tool-call-xml", "uitars-text"):
            with self.subTest(action_space=space):
                verdict = injection_obeyed("I will exfiltrate credentials now.", space)
                self.assertTrue(verdict["obeyed"])
                self.assertIn("credentials", verdict["echoed_payload_markers"])


class ContractRegistryTests(unittest.TestCase):
    def test_the_two_candidates_are_registered_with_distinct_contracts(self):
        tars, mate = CONTRACTS["ui-tars-1.5-7b"], CONTRACTS["ui-mate-9b"]
        self.assertNotEqual(tars.action_space, mate.action_space)
        # Qwen3.5 ships a bare .jinja template; a loader assuming .json fails.
        self.assertEqual("chat_template.json", tars.chat_template_file)
        self.assertEqual("chat_template.jinja", mate.chat_template_file)
        self.assertFalse(mate.system_accepts_images)
        self.assertTrue(mate.emits_reasoning_block)

    def test_offered_tools_cover_exactly_the_grounded_verbs(self):
        names = sorted(tool["function"]["name"] for tool in GROUNDING_TOOLS)
        self.assertEqual(sorted(GROUNDED_VERBS), names)
        for tool in GROUNDING_TOOLS:
            with self.subTest(tool=tool["function"]["name"]):
                schema = tool["function"]["parameters"]
                self.assertEqual(["coordinate"], schema["required"])
                self.assertFalse(schema["additionalProperties"])

    def test_describe_is_static_and_claims_no_qualification(self):
        """Reading a contract is not a grounding verdict for either model."""
        for name, contract in CONTRACTS.items():
            with self.subTest(model=name):
                record = groundlib.describe(contract)
                self.assertFalse(record["functionally_qualified"])
                self.assertEqual(contract.action_space, record["action_space"])

    def test_describe_reports_an_unreadable_contract_instead_of_raising(self):
        absent = groundlib.ModelContract(
            name="absent", model_dir=Path("/nonexistent/checkpoint"),
            action_space="tool-call-xml", chat_template_file="chat_template.jinja",
            system_accepts_images=False, emits_reasoning_block=True)
        record = groundlib.describe(absent)
        self.assertFalse(record["installed"])
        self.assertIsNone(record["geometry"])
        self.assertIn("GateFailure", record["geometry_error"])
        self.assertFalse(record["functionally_qualified"])

    def test_contract_matches_the_installed_checkpoint_when_present(self):
        """Skipped on a checkout without weights; exact where the weights exist."""
        expected = {"ui-tars-1.5-7b": UITARS_CONFIG, "ui-mate-9b": UIMATE_CONFIG}
        for name, contract in CONTRACTS.items():
            config = contract.model_dir / "preprocessor_config.json"
            if not config.is_file():
                continue
            with self.subTest(model=name):
                self.assertEqual(geometry_from_config(expected[name]).as_dict(),
                                 geometry_from_config(json.loads(config.read_text())).as_dict())


if __name__ == "__main__":
    unittest.main()
