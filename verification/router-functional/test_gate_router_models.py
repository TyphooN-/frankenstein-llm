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
            patch.object(module, "http_json", return_value={"id": module.VISION_MODEL}),
            patch.object(module, "unload") as unload,
        ):
            result = module.check_vision_model()

        self.assertTrue(result["pass"])
        self.assertTrue(captured[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        unload.assert_called_once_with(module.VISION_MODEL)


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
