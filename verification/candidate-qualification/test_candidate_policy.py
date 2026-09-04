#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_policy as policy  # noqa: E402
import gate_candidate_policy as gate  # noqa: E402
import wemm_remote_code_review as review  # noqa: E402


class CandidatePolicyTests(unittest.TestCase):
    def test_manifest_backs_every_downloaded_candidate(self):
        for name, candidate in policy.CANDIDATES.items():
            with self.subTest(name=name):
                artifact = policy.artifact_for(name)
                if candidate["artifact_key"] is None:
                    self.assertIsNone(artifact)
                else:
                    self.assertIsNotNone(artifact)
                    assert artifact is not None
                    self.assertEqual(candidate["artifact_key"], artifact["key"])
                    self.assertTrue(artifact["revision"])
                    self.assertTrue(artifact["files"])

    def test_privileges_fail_closed(self):
        self.assertTrue(policy.tool_grant_allowed("qwen3-coder-next"))
        for name in ("gemma4-heretic", "ui-mate-9b", "wemm-embedding-2b",
                     "flux2-klein-4b", "qwen3-asr-1.7b", "unknown"):
            with self.subTest(name=name):
                self.assertFalse(policy.tool_grant_allowed(name))

    def test_preset_privilege_covers_the_router_ini_and_fails_closed(self):
        for preset in ("ridge", "heretic", "obliterated", "obliterated-vision",
                       "fable", "phr00ty", "qwen3-coder-next"):
            with self.subTest(preset=preset):
                self.assertTrue(policy.tool_grant_allowed_for_preset(preset))
        for preset in ("gemma4-heretic", "gemma4-heretic-vision",
                       "qwen3-embedding-8b", "qwen3-reranker-8b",
                       "qwen25-coder-7b-fim", "unknown-alias"):
            with self.subTest(preset=preset):
                self.assertFalse(policy.tool_grant_allowed_for_preset(preset))

    def test_actors_cannot_be_abliterated(self):
        self.assertFalse(policy.abliteration_allowed("qwen3-coder-next"))
        self.assertFalse(policy.abliteration_allowed("ui-mate-9b"))
        self.assertFalse(policy.abliteration_allowed("unknown"))
        self.assertTrue(policy.abliteration_allowed("gemma4-heretic"))

    def test_control_requires_positive_grounding(self):
        self.assertFalse(policy.control_allowed("ui-mate-9b", None))
        self.assertFalse(policy.control_allowed("ui-mate-9b", False))
        self.assertTrue(policy.control_allowed("ui-mate-9b", True))
        self.assertFalse(policy.control_allowed("gemma4-heretic", True))

    def test_remote_code_requires_positive_review(self):
        self.assertFalse(policy.remote_code_execution_allowed("wemm-embedding-2b", None))
        self.assertFalse(policy.remote_code_execution_allowed("wemm-embedding-2b", False))
        self.assertTrue(policy.remote_code_execution_allowed("wemm-embedding-2b", True))
        self.assertTrue(policy.remote_code_execution_allowed("ui-mate-9b", None))
        self.assertFalse(policy.remote_code_execution_allowed("unknown", True))

    def test_vector_spaces_are_separate(self):
        self.assertEqual([], policy.index_conflicts())
        merged = dict(policy.MULTIMODAL_INDEX, database=policy.TEXT_INDEX["database"])
        self.assertTrue(policy.index_conflicts(multimodal_index=merged))

    def test_phase_one_really_satisfies_phase_four_asr(self):
        self.assertEqual([], policy.already_satisfied_conflicts())

    def test_missing_prerequisite_blocks_admission(self):
        blockers = policy.admission_blockers("wemm-embedding-2b", {})
        self.assertTrue(any("no verdict" in item for item in blockers))
        self.assertTrue(any("not approved" in item for item in blockers))
        self.assertEqual([], policy.admission_blockers(
            "wemm-embedding-2b", {"wemm-remote-code-review": True}))

    def test_inventory_size_check_uses_manifest_without_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"abc")
            queue = root / "queue.json"
            queue.write_text(json.dumps({"artifacts": [{
                "key": "x", "repository": "r", "revision": "rev", "capability": "test",
                "files": [{"destination": str(model), "size": 3}],
            }]}))
            self.assertTrue(policy.installed_inventory(queue)["pass"])
            model.write_bytes(b"ab")
            self.assertFalse(policy.installed_inventory(queue)["pass"])


class RemoteCodeReviewTests(unittest.TestCase):
    def test_current_pinned_snapshot_matches_review(self):
        report = review.scan()
        self.assertTrue(report["pass"], report["problems"])
        self.assertTrue(review.execution_permitted(report))
        self.assertIn("patch_sglang_video.py", report["never_execute"])
        self.assertNotIn("patch_sglang_video.py", report["approved_for_import"])

    def test_changed_or_unreviewed_code_revokes_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "approved.py").write_text("safe = True\n")
            reviewed = {
                "approved.py": {
                    "sha256": review.digest(root / "approved.py"),
                    "verdict": review.APPROVED,
                    "imported_by": "test",
                    "finding": "test fixture",
                }
            }
            document = root / "review.md"
            document.write_text("reviewed\n")
            with mock.patch.object(review, "REVIEWED_FILES", reviewed), \
                 mock.patch.object(review, "REVIEW_DOCUMENT", document):
                self.assertTrue(review.scan(root)["pass"])
                (root / "surprise.py").write_text("print('unexpected')\n")
                report = review.scan(root)
                self.assertFalse(report["pass"])
                self.assertFalse(review.execution_permitted(report))
                self.assertEqual(["surprise.py"], report["unreviewed_executable_files"])


class CandidateRuntimeLockTests(unittest.TestCase):
    def test_lock_pins_required_candidate_packages_without_cuda_stack(self):
        lock = (HERE / "requirements.lock").read_text()
        self.assertIn("transformers==5.5.4", lock)
        self.assertIn("qwen-vl-utils==0.0.14", lock)
        self.assertIn("accelerate==1.14.0", lock)
        for forbidden in ("nvidia-", "cuda-bindings==", "cuda-toolkit==", "triton=="):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lock)

    def test_setup_reuses_host_rocm_torch(self):
        script = (HERE / "setup_runtime.sh").read_text()
        self.assertIn("--system-site-packages", script)
        self.assertNotIn("pip install torch", script)
        self.assertIn("torch.version.hip", script)
        self.assertIn("uv venv --python 3.14 --system-site-packages", script)
        self.assertIn('need 3.14', script)


class CandidateGateTests(unittest.TestCase):
    def test_evaluate_aggregates_failures(self):
        inventory = {"pass": False, "problems": ["wrong size"], "artifacts": {}}
        remote = {"pass": False, "problems": ["changed code"]}
        with mock.patch.object(policy, "installed_inventory", return_value=inventory), \
             mock.patch.object(policy, "already_satisfied_conflicts", return_value=["asr drift"]), \
             mock.patch.object(policy, "index_conflicts", return_value=["index collision"]), \
             mock.patch.object(review, "scan", return_value=remote):
            report = gate.evaluate()
        self.assertFalse(report["pass"])
        self.assertEqual(
            ["wrong size", "asr drift", "index collision", "changed code"],
            report["problems"])
        self.assertFalse(report["model_inference_performed"])
        self.assertFalse(report["benchmarking_performed"])

    def test_atomic_writer_publishes_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            gate.write_atomic({"pass": True}, path)
            self.assertEqual({"pass": True}, json.loads(path.read_text()))
            self.assertEqual([], list(path.parent.glob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
