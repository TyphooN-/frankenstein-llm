#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("gate_router_models.py")
spec = importlib.util.spec_from_file_location("gate_router_models", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RouterFunctionalGateTests(unittest.TestCase):
    def test_model_metadata_uses_supported_collection_endpoint(self) -> None:
        payload = {"data": [{"id": "ridge", "status": {"value": "loaded"}}]}
        with patch.object(module, "http_json", return_value=payload) as request:
            self.assertEqual(payload["data"][0], module.model_metadata("ridge"))
        request.assert_called_once_with("/models")

    def test_model_metadata_fails_when_model_is_absent(self) -> None:
        with patch.object(module, "http_json", return_value={"data": []}):
            with self.assertRaises(KeyError):
                module.model_metadata("missing")

    def test_chat_discards_timing_and_usage_metadata(self) -> None:
        raw = {
            "choices": [{"message": {"role": "assistant", "content": "PONG"}}],
            "timings": {"predicted_per_second": 99},
            "usage": {"completion_tokens": 1},
        }
        with patch.object(module, "http_json", return_value=raw):
            result = module.chat("ridge", "ping")
        self.assertEqual(result, {"message": {"role": "assistant", "content": "PONG"}})
        self.assertNotIn("timings", result)
        self.assertNotIn("usage", result)

    def test_vision_gate_requires_exact_fixture_answer_and_unloads(self) -> None:
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
            {"mem_available_bytes": 40 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 4 << 30}},
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ]
        captured: list[dict] = []

        def fake_chat(model: str, content: list[dict], **_kwargs) -> dict:
            captured.extend(content)
            return {"message": {"content": "Run Gate"}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "http_json", return_value={
                "data": [{"id": module.VISION_MODELS[0]}],
            }),
            patch.object(module, "unload") as unload,
        ):
            result = module.check_vision_model(module.VISION_MODELS[0])

        self.assertTrue(result["pass"])
        self.assertTrue(captured[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        unload.assert_called_once_with(module.VISION_MODELS[0])


class PrivilegeProfileTests(unittest.TestCase):
    """A preset policy refuses tools to must never be sent a tools payload."""

    def test_incumbent_chat_presets_are_still_judged_on_tool_use(self):
        for model in ("ridge", "heretic", "obliterated", "fable", "phr00ty"):
            with self.subTest(model=model):
                self.assertEqual(module.CORE_CHECKS, module.checks_for(model))

    def test_the_repository_agent_candidate_keeps_the_tool_contract(self):
        self.assertIn("tool_call", module.checks_for("qwen3-coder-next"))

    def test_the_low_privilege_candidate_is_never_offered_tools(self):
        for model in ("gemma4-heretic", "gemma4-heretic-vision"):
            with self.subTest(model=model):
                self.assertEqual(module.READER_CHECKS, module.checks_for(model))
                self.assertNotIn("tool_call", module.checks_for(model))

    def test_unknown_and_non_chat_presets_are_not_offered_tools(self):
        for model in ("unknown-alias", "qwen3-embedding-8b", "qwen25-coder-7b-fim"):
            with self.subTest(model=model):
                self.assertEqual(module.READER_CHECKS, module.checks_for(model))

    def test_a_reader_run_sends_no_tools_and_claims_no_tool_capability(self):
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
            {"mem_available_bytes": 40 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 4 << 30}},
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ]
        offered: list[object] = []

        def fake_chat(model, prompt, *, schema=None, tools=None):
            offered.append(tools)
            content = "PONG" if schema is None else '{"status": "ready", "count": 3}'
            return {"message": {"content": content}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "http_json", return_value={"data": [{"id": "gemma4-heretic"}]}),
            patch.object(module, "unload"),
        ):
            result = module.check_model("gemma4-heretic")

        self.assertTrue(result["pass"])
        self.assertFalse(result["tools_offered"])
        self.assertEqual([None, None], offered, "a tools payload reached a low-privilege preset")
        self.assertNotIn("tool_call", result["checks"])
        self.assertIn("tool_policy", result)

    def test_a_missing_required_check_still_fails_the_model(self):
        # The required set is compared exactly, so a section that never ran
        # cannot be mistaken for one that passed.
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ]

        def fake_chat(model, prompt, *, schema=None, tools=None):
            if schema is not None:
                raise RuntimeError("router dropped the request")
            return {"message": {"content": "PONG"}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "http_json", return_value={"data": [{"id": "gemma4-heretic"}]}),
            patch.object(module, "unload"),
        ):
            result = module.check_model("gemma4-heretic")

        self.assertFalse(result["pass"])
        self.assertNotIn("structured_output", result["checks"])


class ModelCoverageTests(unittest.TestCase):
    """The presets this gate qualifies must exist in the router preset file."""

    PRESETS = Path("/home/typhoon/git/frankenstein-llm/llama-models.ini").read_text()

    def test_every_gated_preset_is_declared(self):
        for model in list(module.CHAT_MODELS) + list(module.VISION_MODELS):
            with self.subTest(model=model):
                self.assertIn(f"[{model}]", self.PRESETS)

    def test_the_fim_preset_is_not_displaced_by_the_coder_candidate(self):
        # Qwen3-Coder-Next is a repository-agent candidate, not a completion
        # model. Losing the FIM preset would silently remove /infill coverage.
        self.assertIn("[qwen25-coder-7b-fim]", self.PRESETS)
        self.assertIn("[qwen3-coder-next]", self.PRESETS)


class ReleaseAccountingTests(unittest.TestCase):
    """Eviction has to be proven, not assumed from a reading that went missing."""

    def release(self, before: dict, after: dict) -> list[str]:
        result: dict = {"problems": []}
        with patch.object(module, "sample", return_value=after):
            module.record_release(result, before)
        self.assertEqual(after, result["after_unload"])
        return result["problems"]

    @staticmethod
    def state(vram: dict[str, int], available: int = 50 << 30, swap: int = 0) -> dict:
        return {"mem_available_bytes": available, "swap_used_bytes": swap,
                "vram_used_bytes": vram}

    def test_a_clean_release_records_no_problem(self):
        self.assertEqual([], self.release(self.state({"card0": 0, "card1": 0}),
                                          self.state({"card0": 0, "card1": 0})))

    def test_retained_vram_is_reported(self):
        problems = self.release(self.state({"card0": 0}), self.state({"card0": 8 << 30}))
        self.assertEqual(1, len(problems))
        self.assertIn("retained", problems[0])

    def test_a_card_that_stopped_reporting_is_not_a_release(self):
        # vram_used() drops a card whose sysfs node stopped answering. Falling
        # back to the baseline for it produced a zero delta, so "we lost the
        # reading" scored exactly like "it gave the memory back".
        problems = self.release(self.state({"card0": 0, "card1": 0}), self.state({"card0": 0}))
        self.assertEqual(1, len(problems))
        self.assertIn("card1", problems[0])
        self.assertIn("unproven", problems[0])

    def test_a_baseline_with_no_cards_at_all_fails(self):
        problems = self.release(self.state({}), self.state({}))
        self.assertTrue(any("eviction is unproven" in problem for problem in problems))

    def test_ram_not_returned_is_reported(self):
        problems = self.release(self.state({"card0": 0}, available=50 << 30),
                                self.state({"card0": 0}, available=40 << 30))
        self.assertTrue(any("available RAM" in problem for problem in problems))

    def test_swap_growth_is_reported(self):
        problems = self.release(self.state({"card0": 0}, swap=0),
                                self.state({"card0": 0}, swap=8 << 30))
        self.assertTrue(any("swap grew" in problem for problem in problems))

    def test_tolerances_absorb_ordinary_noise(self):
        problems = self.release(
            self.state({"card0": 0}, available=50 << 30, swap=0),
            self.state({"card0": module.VRAM_TOLERANCE}, available=50 << 30, swap=0))
        self.assertEqual([], problems)


if __name__ == "__main__":
    unittest.main()
