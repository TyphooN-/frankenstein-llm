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


def test_comfy_bootstrap_selects_backend_in_server_process():
    from types import SimpleNamespace
    import runpy
    configure = runpy.run_path(str(MODULE_PATH.with_name('comfy_runtime.py')))['configure']
    calls = []
    def original(a, b):
        calls.append((a, b))
        return 17
    torch = SimpleNamespace(version=SimpleNamespace(hip='7.2'),
        backends=SimpleNamespace(cuda=SimpleNamespace(preferred_blas_library=calls.append)))
    eager = SimpleNamespace(_int8_matmul_accumulate=original)
    configure(torch, eager)
    x = SimpleNamespace(device=SimpleNamespace(type='cpu'))
    assert eager._int8_matmul_accumulate(x, x) == 17
    assert calls == ['cublas', (x, x)]


def test_int8_tiling_preserves_integer_accumulation():
    import runpy
    import pytest
    torch = pytest.importorskip('torch')
    accumulate = runpy.run_path(str(MODULE_PATH.with_name('comfy_runtime.py')))['int8_accumulate']
    torch.manual_seed(17)
    for m, k, n in ((3, 2049, 7), (257, 17, 1025), (0, 16, 7), (3, 0, 7)):
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8)
        b = torch.randint(-128, 128, (k, n), dtype=torch.int8)
        torch.testing.assert_close(accumulate(torch, a, b).long(), a.long() @ b.long(), rtol=0, atol=0)
    a = torch.full((1, 2049), -128, dtype=torch.int8)
    b = torch.full((2049, 1), -128, dtype=torch.int8)
    torch.testing.assert_close(accumulate(torch, a, b).long(), a.long() @ b.long(), rtol=0, atol=0)
    with pytest.raises(ValueError, match='INT8'):
        accumulate(torch, a.float(), b)
    with pytest.raises(ValueError, match='compatible'):
        accumulate(torch, a, b.T)


def test_comfy_runner_does_not_double_remap_devices():
    source = MODULE_PATH.with_name('run_functional_serialized.sh').read_text()
    assert 'unset DISPLAY HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES' in source
    assert 'export ROCR_VISIBLE_DEVICES=GPU-a21e268c0b0a73d7' in source
    assert '--cuda-device 1' not in source
    assert '"$DIR/comfy_runtime.py"' in source


def test_failure_records_workflow_without_claiming_completion(monkeypatch, tmp_path):
    import copy
    reports = []
    monkeypatch.setattr(functional_gate, 'EVIDENCE', tmp_path)
    monkeypatch.setattr(functional_gate, 'OUTPUTS', tmp_path / 'outputs')
    monkeypatch.setattr(functional_gate, 'atomic_write', lambda value: reports.append(copy.deepcopy(value)))
    def failure(_):
        raise RuntimeError('backend unavailable')
    monkeypatch.setattr(functional_gate, 'submit', failure)
    assert functional_gate.main() == 1
    assert reports[-1]['failed_workflow'] == 'image-generation'
    assert reports[-1]['current_workflow'] is None
    assert reports[-1]['workflows'] == {}
    assert len(reports[-1]['planned_workflows']) == 3


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
