"""Exercise the real build shell in a disposable host/toolchain fixture.

The fake compiler records configuration; these are orchestration tests, not
claims that CUDA/HIP binaries were compiled or run on the reference host.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


class BackendBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / "upstream/llama.cpp").mkdir(parents=True)
        (self.root / "upstream/llama.cpp/.git").touch()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "cmake.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        FIXTURE=str(self.root))
        shutil.copyfile(REPO / "scripts/build-llama-cpp.sh",
                        self.root / "scripts/build-llama-cpp.sh")
        shutil.copyfile(REPO / "upstream/llama-cpp.lock.json",
                        self.root / "upstream/llama-cpp.lock.json")
        # These fake-toolchain fixtures cover backend orchestration; patch
        # enforcement has independent real-Git tests in test_llama_patches.py.
        fixture_lock = json.loads((self.root / "upstream/llama-cpp.lock.json").read_text())
        fixture_lock["local_patches"] = []
        (self.root / "upstream/llama-cpp.lock.json").write_text(json.dumps(fixture_lock))
        shutil.copyfile(REPO / "scripts/verify_llama_patches.py",
                        self.root / "scripts/verify_llama_patches.py")
        self.tool("git", '''import json, os, sys
from pathlib import Path
root = Path(os.environ['FIXTURE'])
if 'rev-parse' in sys.argv:
    print(json.loads((root / 'upstream/llama-cpp.lock.json').read_text())['commit'])
elif 'status' in sys.argv:
    sys.exit(int(os.environ.get('FAIL_GIT_STATUS', '0')))
''')
        self.tool("nproc", "print(2)\n")
        self.tool("hipconfig", '''import os, sys
sys.exit(7) if os.environ.get('FAIL_HIPCONFIG') else print('/fixture/rocm')
''')
        self.tool("glslc", "pass\n")
        self.tool("cmake", '''import json, os, sys
from pathlib import Path
root = Path(os.environ['FIXTURE'])
with (root / 'cmake.jsonl').open('a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if '--build' in sys.argv:
    folder = Path(sys.argv[sys.argv.index('--build') + 1]) / 'bin'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('llama-server', 'llama-cli', 'llama-quantize', 'llama-gguf'):
        p = folder / name
        p.write_text('#!/bin/sh\\nexit 0\\n')
        p.chmod(0o755)
''')
        self.detected = {
            "hip": {"targets": [], "toolchain": True},
            "cuda": {"architectures": [], "nvcc": "/fixture/nvcc", "reason": "no device"},
        }
        self.detector = self.root / "scripts/detect_backends.py"
        self.detector.write_text('''import json, os
from pathlib import Path
root = Path(os.environ['FIXTURE'])
with (root / 'detections').open('a') as f:
    f.write('detected\\n')
print((root / 'detected.json').read_text())
''')

    def tool(self, name, body):
        p = self.bin / name
        p.write_text("#!/usr/bin/env python3\n" + body)
        p.chmod(0o755)

    def build(self, *args):
        (self.root / "detected.json").write_text(json.dumps(self.detected))
        return subprocess.run(["bash", str(self.root / "scripts/build-llama-cpp.sh"), *args],
                              env=self.env, capture_output=True, text=True, timeout=15)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_auto_cpu_and_explicit_vulkan(self):
        for args, directory, flag in [((), "build-cpu", None),
                                       (("vulkan",), "build-vulkan", "-DGGML_VULKAN=ON")]:
            with self.subTest(args=args):
                result = self.build(*args)
                self.assertEqual(0, result.returncode, result.stderr)
                configure = self.calls()[-2]
                self.assertTrue(configure[configure.index("-B") + 1].endswith(directory))
                for backend in ("HIP", "CUDA", "VULKAN"):
                    self.assertIn(f"-DGGML_{backend}=OFF", configure)
                if flag:
                    self.assertIn(flag, configure)

    def test_auto_mixed_vendor_uses_one_snapshot_and_separate_directories(self):
        self.detected["hip"]["targets"] = ["gfx1030"]
        self.detected["cuda"]["architectures"] = ["86"]
        result = self.build()
        self.assertEqual(0, result.returncode, result.stderr)
        hip, _, cuda, _ = self.calls()
        self.assertIn("-DGPU_TARGETS=gfx1030", hip)
        self.assertIn("-DGGML_HIP=ON", hip)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", cuda)
        self.assertIn("-DCMAKE_CUDA_COMPILER=/fixture/nvcc", cuda)
        self.assertNotEqual(hip[hip.index("-B") + 1], cuda[cuda.index("-B") + 1])
        self.assertEqual("detected\n", (self.root / "detections").read_text())

    def test_no_device_and_unknown_backend_refuse_before_cmake(self):
        for backend in ("hip", "cuda", "wrong", "auto cpu"):
            with self.subTest(backend=backend):
                self.assertNotEqual(0, self.build(backend).returncode)
                self.assertEqual([], self.calls())

    def test_missing_cuda_toolchain_refuses_before_cmake(self):
        self.detected["cuda"].update(architectures=["86"], nvcc=None)
        result = self.build("cuda")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("no nvcc", result.stderr)
        self.assertEqual([], self.calls())

    def test_failed_detection_cannot_report_success(self):
        self.detector.write_text("raise SystemExit(9)\n")
        self.assertEqual(9, self.build().returncode)
        self.assertEqual([], self.calls())

    def test_invalid_lock_cannot_report_success(self):
        (self.root / "upstream/llama-cpp.lock.json").write_text("not json")
        self.assertNotEqual(0, self.build().returncode)
        self.assertEqual([], self.calls())

    def test_patch_verifier_refusal_stops_before_cmake(self):
        (self.root / "scripts/verify_llama_patches.py").write_text("raise SystemExit(17)\n")
        self.assertEqual(17, self.build("cpu").returncode)
        self.assertEqual([], self.calls())

    def test_git_status_error_is_not_a_clean_worktree(self):
        self.env["FAIL_GIT_STATUS"] = "8"
        self.assertEqual(8, self.build("cpu").returncode)
        self.assertEqual([], self.calls())

    def test_hipconfig_error_refuses_before_cmake(self):
        self.env["FAIL_HIPCONFIG"] = "1"
        self.detected["hip"]["targets"] = ["gfx1030"]
        self.assertEqual(7, self.build("hip").returncode)
        self.assertEqual([], self.calls())


if __name__ == "__main__":
    unittest.main()
