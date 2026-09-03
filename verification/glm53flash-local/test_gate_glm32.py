"""Offline telemetry contracts for the GLM 32K functional gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parent / "gate_glm32.py"
SPEC = importlib.util.spec_from_file_location("gate_glm32", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class VramTelemetryTests(unittest.TestCase):
    def test_all_required_cards_must_be_readable(self):
        self.assertEqual([], module.unreadable_cards(
            {"card0": 1, "card1": 2, "card2": 3}))
        self.assertEqual(["card1", "card2"], module.unreadable_cards(
            {"card0": 1, "card1": -1}))

    def test_vram_uses_the_shared_fail_closed_reader(self):
        readings = {"card0": 10, "card1": 20, "card2": -1}
        with mock.patch.object(module, "vram_used", return_value=readings):
            self.assertEqual(readings, module.vram())

    def test_unload_verdict_rejects_missing_after_reading(self):
        verdict = module.unload_verdict(
            {"card0": 0, "card1": 0, "card2": 0},
            {"card0": 0, "card1": 0},
            module.VRAM_TOLERANCE,
        )
        self.assertFalse(verdict["pass"])
        self.assertEqual(["card2"], verdict["unreadable_cards"])


if __name__ == "__main__":
    unittest.main()
