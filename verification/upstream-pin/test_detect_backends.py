"""Backend and GPU-target detection for the llama.cpp build (ADR 0009)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import detect_backends  # noqa: E402


def kfd(root: Path, versions: list[int]) -> Path:
    for index, version in enumerate(versions):
        node = root / str(index)
        node.mkdir(parents=True)
        (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {version}\n")
    return root


def smi(stdout: str = "", returncode: int = 0):
    def run(argv, **_):
        return subprocess.CompletedProcess(argv, returncode, stdout, "driver not loaded")
    return run


class HipTargetTests(unittest.TestCase):
    def test_target_version_decodes_to_the_gfx_name(self):
        self.assertEqual("gfx1030", detect_backends.gfx_name(100300))
        self.assertEqual("gfx1101", detect_backends.gfx_name(110001))
        self.assertEqual("gfx90a", detect_backends.gfx_name(90010))
        self.assertIsNone(detect_backends.gfx_name(0))

    def test_cpu_nodes_are_skipped_and_targets_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = kfd(Path(tmp), [0, 100300, 100300, 110000])
            self.assertEqual(["gfx1030", "gfx1100"], detect_backends.hip_targets(root))

    def test_no_kfd_topology_is_no_targets(self):
        self.assertEqual([], detect_backends.hip_targets(Path("/nonexistent/kfd")))


class CudaArchitectureTests(unittest.TestCase):
    def test_compute_capabilities_become_cmake_architectures(self):
        archs, reason = detect_backends.cuda_architectures(smi("8.6\n7.5\n8.6\n"))
        self.assertEqual(["75", "86"], archs)
        self.assertIsNone(reason)

    def test_a_failing_driver_is_no_devices_with_a_reason(self):
        archs, reason = detect_backends.cuda_architectures(smi(returncode=9))
        self.assertEqual([], archs)
        self.assertIn("exited 9", reason)


class AutoSelectionTests(unittest.TestCase):
    def detect(self, versions, smi_out, env=None):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, env or {}, clear=False):
            for name in ("HIP_GPU_TARGETS", "CUDA_ARCHITECTURES"):
                if name not in (env or {}):
                    os.environ.pop(name, None)
            return detect_backends.detect(kfd(Path(tmp), versions), smi(smi_out))

    def test_mixed_vendor_host_builds_both(self):
        detected = self.detect([100300], "8.6\n")
        self.assertEqual(["hip", "cuda"], detect_backends.auto(detected))

    def test_nvidia_only_host(self):
        self.assertEqual(["cuda"], detect_backends.auto(self.detect([0], "6.1\n")))

    def test_no_gpu_falls_back_to_cpu(self):
        self.assertEqual(["cpu"], detect_backends.auto(self.detect([0], "")))

    def test_overrides_replace_detection(self):
        detected = self.detect([100300], "8.6\n",
                               {"HIP_GPU_TARGETS": "gfx1100;gfx1030", "CUDA_ARCHITECTURES": "89"})
        self.assertEqual(["gfx1100", "gfx1030"], detected["hip"]["targets"])
        self.assertEqual("HIP_GPU_TARGETS", detected["hip"]["source"])
        self.assertEqual(["89"], detected["cuda"]["architectures"])

    def test_separator_only_override_is_not_a_silent_cpu_fallback(self):
        with self.assertRaisesRegex(ValueError, "CUDA_ARCHITECTURES"):
            self.detect([], "", {"CUDA_ARCHITECTURES": "; , ;"})

    def test_override_targets_are_deduplicated_in_order(self):
        detected = self.detect([], "", {"CUDA_ARCHITECTURES": "86,75;86"})
        self.assertEqual(["86", "75"], detected["cuda"]["architectures"])


class NvccDiscoveryTests(unittest.TestCase):
    def test_path_has_priority(self):
        with mock.patch.object(detect_backends.shutil, "which", return_value="/custom/nvcc"):
            self.assertEqual("/custom/nvcc", detect_backends.find_nvcc())

    def test_standard_nvidia_install_is_found_without_path(self):
        with mock.patch.object(detect_backends.shutil, "which", return_value=None), \
                mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(os, "access", return_value=True):
            self.assertEqual("/usr/local/cuda/bin/nvcc", detect_backends.find_nvcc())

    def test_non_executable_toolkit_is_not_available(self):
        with mock.patch.object(detect_backends.shutil, "which", return_value=None), \
                mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(os, "access", return_value=False):
            self.assertIsNone(detect_backends.find_nvcc())


if __name__ == "__main__":
    unittest.main()
