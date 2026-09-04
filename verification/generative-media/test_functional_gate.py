#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).with_name("functional_gate.py")
spec = importlib.util.spec_from_file_location("functional_gate", MODULE_PATH)
functional_gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(functional_gate)


class WorkflowTests(unittest.TestCase):
    def assert_no_performance_fields(self, workflow: dict) -> None:
        serialized = repr(workflow).lower()
        self.assertNotIn("tokens/sec", serialized)
        self.assertNotIn("throughput", serialized)
        self.assertNotIn("benchmark", serialized)

    def test_z_image_graph_has_real_output(self) -> None:
        workflow = functional_gate.z_image_workflow()
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], "z_image_turbo_bf16.safetensors")
        self.assertEqual(workflow["10"]["class_type"], "SaveImage")
        self.assertEqual(workflow["10"]["inputs"]["images"], ["9", 0])
        self.assert_no_performance_fields(workflow)

    def test_music_graph_uses_installed_aio_checkpoint(self) -> None:
        workflow = functional_gate.music_workflow()
        self.assertEqual(workflow["1"]["inputs"]["ckpt_name"], "ace_step_1.5_turbo_aio.safetensors")
        self.assertEqual(workflow["8"]["class_type"], "SaveAudio")
        self.assertGreaterEqual(workflow["2"]["inputs"]["duration"], 8)
        self.assert_no_performance_fields(workflow)

    def test_edit_graph_uses_int8_convrot_and_lightning(self) -> None:
        workflow = functional_gate.edit_workflow("fixture.png")
        self.assertEqual(workflow["3"]["inputs"]["unet_name"], "qwen_image_edit_2511_int8_convrot.safetensors")
        self.assertEqual(workflow["6"]["inputs"]["lora_name"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        self.assertEqual(workflow["14"]["inputs"]["steps"], 4)
        self.assertIn("red square", workflow["9"]["inputs"]["prompt"])
        self.assert_no_performance_fields(workflow)

    def test_flux2_klein_graph_uses_installed_native_checkpoint(self) -> None:
        workflow = functional_gate.flux2_klein_workflow()
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], "flux-2-klein-4b.safetensors")
        self.assertEqual(workflow["2"]["inputs"]["clip_name"], "qwen_3_4b.safetensors")
        self.assertEqual(workflow["2"]["inputs"]["type"], "flux2")
        self.assertEqual(workflow["9"]["class_type"], "SaveImage")
        self.assert_no_performance_fields(workflow)


if __name__ == "__main__":
    unittest.main()
