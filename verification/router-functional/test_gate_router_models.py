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


if __name__ == "__main__":
    unittest.main()
