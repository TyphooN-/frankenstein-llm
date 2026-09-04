from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import media_policy as policy
import preflight


def healthy_probe(blockers=()):
    """A synthetic host where every artifact and node contract is present.

    The report is built from this instead of from the real filesystem so the
    assertions below describe the gate's logic rather than whether 20 GB of
    weights happen to be on this disk today.
    """
    return {
        "inventory": {
            "files": {
                key: {"path": str(path), "exists": True,
                      "bytes": policy.REQUIRED_BYTES[key],
                      "expected_bytes": policy.REQUIRED_BYTES[key], "ok": True}
                for key, path in policy.ARTIFACTS.items()
            },
            "problems": [],
            "pass": True,
        },
        "nodes": [
            {"name": name, "path": str(path), "exists": True,
             "required_node_ids": list(policy.REQUIRED_NODE_IDS[name]),
             "missing_node_ids": [], "pass": True}
            for name, path in policy.REQUIRED_NODE_FILES.items()
        ],
        "blockers": list(blockers),
        "comfy_python_present": True,
        "comfy_main_present": True,
        "extra_paths_present": True,
    }


class MediaPolicyTests(unittest.TestCase):
    def test_display_gpu_is_excluded_and_v620_is_primary(self):
        env = policy.comfy_env()
        self.assertEqual("0,1", env["HIP_VISIBLE_DEVICES"])
        self.assertNotIn("2", env["HIP_VISIBLE_DEVICES"].split(","))
        argv = policy.comfy_argv()
        self.assertEqual("1", argv[argv.index("--cuda-device") + 1])
        self.assertEqual("127.0.0.1", argv[argv.index("--listen") + 1])

    def test_inventory_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(b"bad")
            with mock.patch.dict(policy.ARTIFACTS, {"z-image-unet": path}, clear=True), \
                 mock.patch.dict(policy.REQUIRED_BYTES, {"z-image-unet": 42}, clear=True):
                result = policy.artifact_inventory()
        self.assertFalse(result["pass"])
        self.assertEqual(3, result["files"]["z-image-unet"]["bytes"])

    def test_node_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nodes.py"
            path.write_text("TextEncodeZImageOmni\n")
            result = preflight.inspect_node_contract("z", path, ("TextEncodeZImageOmni", "Missing"))
        self.assertFalse(result["pass"])
        self.assertEqual(["Missing"], result["missing_node_ids"])

    def test_kernel_build_blocks_functional_readiness_not_static_pass(self):
        report = preflight.build_report(healthy_probe(["pid=1 kernel-build"]))
        self.assertTrue(report["pass"])
        self.assertFalse(report["functional_gate_ready_now"])
        self.assertFalse(report["throughput_measured"])
        self.assertEqual(["pid=1 kernel-build"], report["kernel_build_blockers"])

    def test_idle_host_is_functionally_ready(self):
        report = preflight.build_report(healthy_probe())
        self.assertTrue(report["pass"])
        self.assertTrue(report["functional_gate_ready_now"])

    def test_missing_artifact_fails_the_static_gate(self):
        probe = healthy_probe()
        probe["inventory"] = {"files": {}, "problems": ["z-image-unet: missing or wrong size"],
                              "pass": False}
        report = preflight.build_report(probe)
        self.assertFalse(report["pass"])
        self.assertFalse(report["functional_gate_ready_now"])

    def test_missing_runtime_pieces_fail_the_static_gate(self):
        for key, expected in (("comfy_python_present", "ComfyUI venv interpreter missing"),
                              ("comfy_main_present", "ComfyUI main.py missing"),
                              ("extra_paths_present", "extra model paths config missing")):
            probe = healthy_probe()
            probe[key] = False
            with self.subTest(missing=key):
                report = preflight.build_report(probe)
                self.assertFalse(report["pass"])
                self.assertIn(expected, report["problems"])

    def test_failed_node_contract_fails_the_static_gate(self):
        probe = healthy_probe()
        probe["nodes"][0] = {**probe["nodes"][0], "pass": False,
                             "missing_node_ids": ["TextEncodeZImageOmni"]}
        report = preflight.build_report(probe)
        self.assertFalse(report["pass"])

    def test_default_report_still_reads_the_real_host(self):
        """Injection must not become a way to bypass the environment check."""
        sentinel = healthy_probe(["pid=99 kernel-build"])
        with mock.patch.object(preflight, "probe_environment",
                               return_value=sentinel) as probed:
            report = preflight.build_report()
        probed.assert_called_once_with()
        self.assertEqual(["pid=99 kernel-build"], report["kernel_build_blockers"])

    def test_probe_environment_reports_every_field_the_report_needs(self):
        probe = preflight.probe_environment()
        self.assertEqual(sorted(healthy_probe()), sorted(probe))


class WorkflowClaimTests(unittest.TestCase):
    """A passing static preflight must not read as functional coverage."""

    def test_no_workflow_claims_functional_proof(self):
        report = preflight.build_report(healthy_probe())
        self.assertEqual([], report["workflows_functionally_proven"])
        for name, claim in report["workflow_claims"].items():
            with self.subTest(workflow=name):
                self.assertFalse(claim["functionally_proven"])

    def test_image_editing_weights_are_declared_but_not_functionally_proven(self):
        claim = policy.WORKFLOW_CLAIMS["image-editing"]
        self.assertTrue(claim["artifacts_present"])
        self.assertFalse(claim["functionally_proven"])
        self.assertIn("qwen-image-edit-unet", claim["artifacts"])
        self.assertIn("flux2-klein-unet", claim["artifacts"])

    def test_image_generation_artifacts_are_declared_and_inventoried(self):
        claim = policy.WORKFLOW_CLAIMS["image-generation"]
        self.assertTrue(claim["artifacts_present"])
        for key in claim["artifacts"]:
            with self.subTest(artifact=key):
                self.assertIn(key, policy.ARTIFACTS)
                self.assertIn(key, policy.REQUIRED_BYTES)

    def test_every_claimed_artifact_is_a_known_artifact(self):
        for name, claim in policy.WORKFLOW_CLAIMS.items():
            for key in claim["artifacts"]:
                with self.subTest(workflow=name, artifact=key):
                    self.assertIn(key, policy.ARTIFACTS)

    def test_extra_paths_include_flux2_klein_layout(self):
        text = policy.EXTRA_PATHS.read_text()
        self.assertIn("flux2-klein-4b", text)
        self.assertIn("flux2-klein-4b/vae", text)
        self.assertIn("flux2-klein-4b/tokenizer", text)


if __name__ == "__main__":
    unittest.main()
