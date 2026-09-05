from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import functional_gate
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
        "graph_resolution": policy.resolve_graphs(
            functional_gate.workflow_graphs(),
            paths=policy.search_paths(),
            exists=lambda path: str(path) in {str(p) for p in policy.ARTIFACTS.values()},
        ),
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


class KernelBuildDetectionTests(unittest.TestCase):
    """The needles, against a fixture /proc rather than against this host."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.proc = Path(self.temporary.name)

    def process(self, pid: str, argv: str, root: Path | None = None) -> Path:
        entry = (root or self.proc) / pid
        entry.mkdir()
        (entry / "cmdline").write_bytes(argv.replace(" ", "\0").encode())
        return entry

    def fresh_proc(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_a_kernel_build_is_seen_whatever_job_count_it_chose(self):
        # The point of dropping the "-j45 LLVM=1" literal: parallelism derived
        # from nproc differs per machine, and the old needle only knew one.
        for argv in ("make -j45 LLVM=1", "make -j", "make -j8", "make -j128 LLVM=1"):
            with self.subTest(argv=argv):
                root = self.fresh_proc()
                self.process("4242", argv, root)
                self.assertEqual(
                    ["pid=4242 kernel-build"],
                    policy.host_exclusive_blockers(root),
                )

    def test_the_link_stage_is_seen_when_no_make_is_left(self):
        self.process("7", "sh -c ./link-vmlinux.sh")
        self.assertEqual(["pid=7 kernel-build"], policy.host_exclusive_blockers(self.proc))

    def test_an_idle_host_reports_nothing(self):
        self.process("11", "sleep 600")
        self.process("12", "hyprland")
        self.assertEqual([], policy.host_exclusive_blockers(self.proc))

    def test_non_numeric_and_unreadable_entries_are_skipped(self):
        (self.proc / "self").mkdir()
        (self.proc / "meminfo").write_text("MemAvailable: 1 kB\n")
        (self.proc / "99").mkdir()  # no cmdline at all
        self.assertEqual([], policy.host_exclusive_blockers(self.proc))

    def test_the_scan_stops_at_the_first_blocker(self):
        # Bounded on purpose: one blocker already withholds readiness, and this
        # walk happens on a host that is by definition busy.
        for pid in ("101", "102", "103"):
            self.process(pid, "make -j LLVM=1")
        self.assertEqual(1, len(policy.host_exclusive_blockers(self.proc)))


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

    def test_extra_paths_register_the_flux2_klein_subfolders(self):
        """The publisher layout is only usable if each subfolder joins its category."""
        paths = {category: [str(path) for path in entries]
                 for category, entries in policy.search_paths().items()}
        base = str(policy.ROOT / "models/comfy")
        self.assertIn(f"{base}/flux2-klein-4b", paths["diffusion_models"])
        self.assertIn(f"{base}/flux2-klein-4b/text_encoder", paths["text_encoders"])
        self.assertIn(f"{base}/flux2-klein-4b/vae", paths["vae"])
        # The flat tree stays first so the pre-existing lanes keep resolving.
        self.assertEqual(f"{base}/vae", paths["vae"][0])


def declared_only(path) -> bool:
    """Existence oracle in which exactly the inventoried artifacts are installed."""
    return str(path) in {str(known) for known in policy.ARTIFACTS.values()}


class GraphResolutionTests(unittest.TestCase):
    """A pinned graph names bare filenames; ComfyUI turns those into paths.

    These run against the tracked search paths and the tracked inventory rather
    than against this host's disk, so they keep working on a checkout with no
    weights in it -- which is every checkout except this one.
    """

    def resolve(self, graphs=None):
        return policy.resolve_graphs(
            functional_gate.workflow_graphs() if graphs is None else graphs,
            paths=policy.search_paths(), exists=declared_only)

    def test_every_pinned_graph_reference_is_a_declared_artifact(self):
        self.assertEqual([], policy.graph_problems(self.resolve()))

    def test_flux2_graph_reuses_the_shared_qwen3_encoder(self):
        """The editing lane really does depend on a generation-lane artifact."""
        flux2 = [entry for entry in self.resolve() if entry["graph"] == "flux2-klein"]
        self.assertEqual(1, len(flux2))
        by_input = {reference["input"]: reference for reference in flux2[0]["references"]}
        self.assertEqual("z-image-clip", by_input["clip_name"]["artifact_key"])
        self.assertEqual("flux2-klein-vae", by_input["vae_name"]["artifact_key"])
        self.assertIn("z-image-clip", policy.WORKFLOW_CLAIMS["image-editing"]["artifacts"])

    def test_graph_references_cover_every_loader_input(self):
        graph = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "k.safetensors"}},
            "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "l.safetensors"}},
            "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "out"}},
        }
        references = policy.graph_references(graph)
        self.assertEqual(
            ["diffusion_models", "text_encoders", "vae", "checkpoints", "loras"],
            [reference["category"] for reference in references])

    def test_missing_file_is_reported_not_ignored(self):
        graph = {"1": {"class_type": "UNETLoader",
                       "inputs": {"unet_name": "never_downloaded.safetensors"}}}
        problems = policy.graph_problems(
            self.resolve({"image-generation": {"synthetic": graph}}))
        self.assertEqual(1, len(problems))
        self.assertIn("not found under diffusion_models", problems[0])

    def test_uninventoried_file_is_reported(self):
        """A file ComfyUI can load but nothing verified is not admissible input."""
        stray = policy.ROOT / "models/comfy/vae/unverified.safetensors"
        graph = {"1": {"class_type": "VAELoader",
                       "inputs": {"vae_name": "unverified.safetensors"}}}
        resolution = policy.resolve_graphs(
            {"image-generation": {"synthetic": graph}}, paths=policy.search_paths(),
            exists=lambda path: str(path) == str(stray))
        problems = policy.graph_problems(resolution)
        self.assertEqual(1, len(problems))
        self.assertIn("resolves to an uninventoried file", problems[0])

    def test_undeclared_artifact_is_reported(self):
        """Regression guard: a claim may not load weights it does not declare."""
        graph = {"1": {"class_type": "CheckpointLoaderSimple",
                       "inputs": {"ckpt_name": "ace_step_1.5_turbo_aio.safetensors"}}}
        problems = policy.graph_problems(
            self.resolve({"image-generation": {"synthetic": graph}}))
        self.assertEqual(
            ["image-generation: graph loads undeclared artifacts ['ace-step-1.5-aio']"],
            problems)

    def test_same_basename_in_two_search_paths_is_ambiguous(self):
        """ComfyUI takes the first hit; a second hit is a silent coin flip."""
        base = policy.ROOT / "models/comfy"
        both = {str(base / "vae/ae.safetensors"),
                str(base / "flux2-klein-4b/vae/ae.safetensors")}
        graph = {"1": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}}
        resolution = policy.resolve_graphs(
            {"image-generation": {"synthetic": graph}}, paths=policy.search_paths(),
            exists=lambda path: str(path) in both)
        self.assertTrue(resolution[0]["references"][0]["ambiguous"])
        self.assertIn("resolves to 2 files", policy.graph_problems(resolution)[0])

    def test_unresolvable_graph_fails_the_static_preflight(self):
        probe = healthy_probe()
        probe["graph_resolution"] = self.resolve(
            {"image-generation": {"synthetic": {"1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "never_downloaded.safetensors"}}}}})
        report = preflight.build_report(probe)
        self.assertFalse(report["pass"])
        self.assertFalse(report["functional_gate_ready_now"])

    def test_resolution_is_not_functional_proof(self):
        report = preflight.build_report(healthy_probe())
        self.assertTrue(report["pass"])
        self.assertEqual([], report["workflows_functionally_proven"])
        self.assertTrue(report["static_readiness_is_not_functional_qualification"])


if __name__ == "__main__":
    unittest.main()
