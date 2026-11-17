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
        (self.root / "scripts/build-llama-cpp.sh").chmod(0o755)
        shutil.copyfile(REPO / "scripts/build-model-bench.sh",
                        self.root / "scripts/build-model-bench.sh")
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
    build = Path(sys.argv[sys.argv.index('--build') + 1]).resolve()
    calls = [json.loads(line) for line in (root / 'cmake.jsonl').read_text().splitlines()]
    configured = {Path(call[call.index('-B') + 1]).resolve()
                  for call in calls if '-B' in call}
    if build not in configured:
        raise SystemExit('unconfigured fixture build directory: ' + str(build))
    folder = build / 'bin'
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

    def test_all_backends_use_nproc_despite_inherited_parallelism(self):
        self.detected["hip"]["targets"] = ["gfx1030"]
        self.detected["cuda"]["architectures"] = ["86"]
        self.env["CMAKE_BUILD_PARALLEL_LEVEL"] = "2"
        self.env["CARGO_BUILD_JOBS"] = "2"
        self.env["MAKEFLAGS"] = "-j2"
        for jobs in (7, 44):
            with self.subTest(nproc=jobs):
                self.tool("nproc", f"print({jobs})\n")
                previous = len(self.calls())
                result = self.build("cpu", "hip", "cuda", "vulkan")
                self.assertEqual(0, result.returncode, result.stderr)
                builds = [call for call in self.calls()[previous:] if "--build" in call]
                self.assertEqual(4, len(builds))
                self.assertEqual([str(jobs)] * 4,
                                 [call[call.index("--parallel") + 1] for call in builds])

    def test_benchmark_builder_also_uses_nproc(self):
        # This legacy helper targets the HIP layout, not a CPU-only checkout.
        self.detected["hip"]["targets"] = ["gfx1030"]
        (self.root / "detected.json").write_text(json.dumps(self.detected))
        self.env["CMAKE_BUILD_PARALLEL_LEVEL"] = "2"
        for jobs in (7, 44):
            with self.subTest(nproc=jobs):
                self.tool("nproc", f"print({jobs})\n")
                previous = len(self.calls())
                result = subprocess.run(
                    ["bash", str(self.root / "scripts/build-model-bench.sh")],
                    env=self.env, capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                calls = self.calls()[previous:]
                configured = [call for call in calls if "-B" in call]
                self.assertEqual(1, len(configured))
                self.assertIn("-DGGML_HIP=ON", configured[0])
                directory = configured[0][configured[0].index("-B") + 1]
                builds = [call for call in calls if "--build" in call]
                self.assertEqual([directory, directory],
                                 [call[call.index("--build") + 1] for call in builds])
                self.assertEqual(2, len(builds))
                self.assertEqual([str(jobs), str(jobs)],
                                 [call[call.index("--parallel") + 1] for call in builds])
                self.assertIn("llama-bench", builds[-1])

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


class BackendModelContractTests(unittest.TestCase):
    """In-process fake-toolchain checks; never run a shell/compiler/server.

    The full BackendBuildTests shell matrix is separate. These controls execute
    the exact generated Python CMake stand-in and inspect the actual benchmark
    test's fixture at its mocked dispatch boundary.
    """

    def setUp(self):
        self.case = BackendBuildTests("test_benchmark_builder_also_uses_nproc")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def cmake(self, *args):
        from unittest import mock

        tool = self.case.bin / "cmake"
        body = tool.read_bytes()
        with (
            mock.patch.dict(os.environ, self.case.env, clear=True),
            mock.patch("sys.argv", [str(tool), *map(str, args)]),
        ):
            # Compile these source bytes directly, without a subprocess or pyc.
            exec(compile(body, str(tool), "exec"), {"__name__": "fixture_cmake"})

    def test_configured_directory_positive_control(self):
        directory = self.case.root / "upstream/llama.cpp/build-cpu"
        self.cmake("-S", self.case.root / "upstream/llama.cpp", "-B", directory)
        self.cmake("--build", directory)
        self.assertEqual([str(directory)],
                         [call[call.index("--build") + 1]
                          for call in self.case.calls() if "--build" in call])
        self.assertTrue((directory / "bin/llama-server").is_file())

    def test_unconfigured_directory_is_rejected(self):
        directory = self.case.root / "upstream/llama.cpp/build"
        with self.assertRaisesRegex(SystemExit, "unconfigured fixture build directory"):
            self.cmake("--build", directory)
        self.assertFalse((directory / "bin").exists())

    def test_configuring_cpu_does_not_configure_hip_directory(self):
        cpu = self.case.root / "upstream/llama.cpp/build-cpu"
        hip = self.case.root / "upstream/llama.cpp/build"
        self.cmake("-S", self.case.root / "upstream/llama.cpp", "-B", cpu)
        with self.assertRaisesRegex(SystemExit, "unconfigured fixture build directory"):
            self.cmake("--build", hip, "--target", "llama-bench")
        self.assertFalse((hip / "bin").exists())

    def test_actual_parallelism_fixture_declares_hip_before_dispatch(self):
        from unittest import mock

        class CheckedFixture(BaseException):
            pass

        def inspect_before_dispatch(*args, **kwargs):
            detected = json.loads((self.case.root / "detected.json").read_text())
            self.assertEqual(["gfx1030"], detected["hip"]["targets"],
                             "benchmark positive fixture must explicitly select HIP")
            self.assertEqual([], detected["cuda"]["architectures"])
            raise CheckedFixture

        with mock.patch("subprocess.run", side_effect=inspect_before_dispatch) as dispatch:
            with self.assertRaises(CheckedFixture):
                self.case.test_benchmark_builder_also_uses_nproc()
        dispatch.assert_called_once()


    def test_each_backend_directory_requires_its_own_configuration(self):
        directories = [self.case.root / "upstream/llama.cpp" / name
                       for name in ("build-cpu", "build", "build-cuda", "build-vulkan")]
        for directory in directories:
            with self.subTest(directory=directory.name):
                self.cmake("-S", self.case.root / "upstream/llama.cpp", "-B", directory)
                self.cmake("--build", directory, "--target", "llama-bench")
                self.assertTrue((directory / "bin/llama-server").is_file())
        calls = self.case.calls()
        self.assertEqual([str(path) for path in directories],
                         [call[call.index("-B") + 1] for call in calls if "-B" in call])
        self.assertEqual([str(path) for path in directories],
                         [call[call.index("--build") + 1]
                          for call in calls if "--build" in call])



if __name__ == "__main__":
    unittest.main()
