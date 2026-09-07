#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("gate_router_models.py")
spec = importlib.util.spec_from_file_location("gate_router_models", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RouterFunctionalGateTests(unittest.TestCase):
    def test_model_metadata_uses_supported_collection_endpoint(self) -> None:
        payload = {"data": [{"id": "ridge", "status": {"value": "loaded"}}]}
        with patch.object(module, "http_json", return_value=payload) as request:
            self.assertEqual(payload["data"][0], module.model_metadata("ridge"))
        request.assert_called_once_with("/models")

    def test_model_metadata_fails_when_model_is_absent(self) -> None:
        with patch.object(module, "http_json", return_value={"data": []}):
            with self.assertRaises(KeyError):
                module.model_metadata("missing")

    def test_chat_discards_timing_and_usage_metadata(self) -> None:
        # finish_reason is carried because it is functional evidence -- it
        # separates "the model stopped on its own" from "the reply was truncated
        # at max_tokens" when a tool call is missing. Timing and token counts
        # stay out: this gate makes no throughput claim.
        raw = {
            "choices": [{"message": {"role": "assistant", "content": "PONG"},
                         "finish_reason": "stop"}],
            "timings": {"predicted_per_second": 99},
            "usage": {"completion_tokens": 1},
        }
        with patch.object(module, "http_json", return_value=raw):
            result = module.chat("ridge", "ping")
        self.assertEqual(result, {"message": {"role": "assistant", "content": "PONG"},
                                  "finish_reason": "stop"})
        self.assertNotIn("timings", result)
        self.assertNotIn("usage", result)

    def test_a_missing_tool_call_records_what_the_model_said_instead(self) -> None:
        # An empty tool_calls list is not self-explanatory. obliterated and
        # phr00ty produced exactly this on boot 1669f3ad because their chat
        # templates have no tools branch, so the function signatures never
        # reached the model; the artifact could not tell that apart from a model
        # that saw the tools and chose prose.
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ]

        def fake_chat(model: str, prompt, **kwargs) -> dict:
            if kwargs.get("tools"):
                return {"message": {"content": "I would look up ticket 4172 for you."},
                        "finish_reason": "stop"}
            if kwargs.get("schema"):
                return {"message": {"content": '{"status": "ready", "count": 3}'}}
            return {"message": {"content": "PONG"}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "checks_for", return_value=module.CORE_CHECKS),
            patch.object(module, "model_metadata", return_value={"id": "obliterated"}),
            patch.object(module, "template_caps", return_value={
                "chat_template_length": 506,
                "chat_template_caps": {"supports_tools": False,
                                       "supports_tool_calls": False}}),
            patch.object(module, "unload"),
        ):
            result = module.check_model("obliterated")

        tool_check = result["checks"]["tool_call"]
        self.assertFalse(tool_check["pass"])
        self.assertEqual([], tool_check["tool_calls"])
        self.assertEqual("I would look up ticket 4172 for you.", tool_check["content"])
        self.assertEqual("stop", tool_check["finish_reason"])
        self.assertEqual(506, tool_check["template"]["chat_template_length"])
        self.assertFalse(result["pass"])
        # The cause is named, not left to be re-diagnosed from an empty list.
        self.assertTrue(any("does not support them" in problem
                            for problem in result["problems"]), result["problems"])

    def test_a_working_tool_call_is_not_failed_by_a_capability_false_negative(self) -> None:
        # The caps come from llama.cpp introspecting the template. If that
        # introspection is wrong about a preset that demonstrably emitted a
        # well-formed native call, the gate must not fail the model for the tool
        # it just used correctly. The guard is a diagnosis for a missing call,
        # not an independent way to fail a present one.
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ] * 3

        def fake_chat(model: str, prompt, **kwargs) -> dict:
            if kwargs.get("tools"):
                return {"message": {"content": "", "tool_calls": [
                    {"function": {"name": "lookup_ticket",
                                  "arguments": '{"ticket_id": 4172}'}}]},
                        "finish_reason": "tool_calls"}
            if kwargs.get("schema"):
                return {"message": {"content": '{"status": "ready", "count": 3}'}}
            return {"message": {"content": "PONG"}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "checks_for", return_value=module.CORE_CHECKS),
            patch.object(module, "model_metadata", return_value={"id": "obliterated"}),
            patch.object(module, "template_caps", return_value={
                "chat_template_caps": {"supports_tools": False}}),
            patch.object(module, "unload"),
        ):
            result = module.check_model("obliterated")

        self.assertTrue(result["checks"]["tool_call"]["pass"])
        self.assertTrue(result["pass"], result["problems"])
        self.assertEqual([], result["problems"])

    def test_an_unreadable_props_probe_does_not_fail_a_working_tool_call(self) -> None:
        # The capability probe is a diagnosis, not the verdict. If /props cannot
        # be read the gate still decides on the call the model actually made,
        # and records why the probe is missing.
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ] * 3

        def fake_chat(model: str, prompt, **kwargs) -> dict:
            if kwargs.get("tools"):
                return {"message": {"content": "", "tool_calls": [
                    {"function": {"name": "lookup_ticket",
                                  "arguments": '{"ticket_id": 4172}'}}]},
                        "finish_reason": "tool_calls"}
            if kwargs.get("schema"):
                return {"message": {"content": '{"status": "ready", "count": 3}'}}
            return {"message": {"content": "PONG"}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "checks_for", return_value=module.CORE_CHECKS),
            patch.object(module, "model_metadata", return_value={"id": "ridge"}),
            patch.object(module, "template_caps",
                         return_value={"error": "URLError: refused"}),
            patch.object(module, "unload"),
        ):
            result = module.check_model("ridge")

        self.assertTrue(result["pass"], result["problems"])
        self.assertEqual("URLError: refused",
                         result["checks"]["tool_call"]["template"]["error"])

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
            patch.object(module, "http_json", return_value={
                "data": [{"id": module.VISION_MODELS[0]}],
            }),
            patch.object(module, "unload") as unload,
        ):
            result = module.check_vision_model(module.VISION_MODELS[0])

        self.assertTrue(result["pass"])
        self.assertTrue(captured[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        unload.assert_called_once_with(module.VISION_MODELS[0])


class LateConflictIntegrationTests(unittest.TestCase):
    def test_text_and_vision_reject_late_build_and_still_unload(self):
        conflict = {"pid": 123, "reason": "build", "command": "clang", "cwd": "/build"}
        state = {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0,
                 "vram_used_bytes": {"card0": 0}}
        for vision in (False, True):
            with self.subTest(vision=vision):
                def chat(_model, _prompt, **kwargs):
                    if vision:
                        return {"message": {"content": "Run Gate"}}
                    content = '{"status":"ready","count":3}' if kwargs.get("schema") else "PONG"
                    return {"message": {"content": content}}

                with (
                    patch.object(module, "sample", return_value=state),
                    patch.object(module, "blocked_workloads", side_effect=[[], [conflict]]) as probe,
                    patch.object(module, "chat", side_effect=chat),
                    patch.object(module, "checks_for", return_value=module.READER_CHECKS),
                    patch.object(module, "model_metadata", return_value={"id": "candidate"}),
                    patch.object(module, "unload") as unload,
                ):
                    gate = module.check_vision_model if vision else module.check_model
                    result = gate("candidate")
                self.assertEqual(2, probe.call_count)
                self.assertTrue(all(check["pass"] for check in result["checks"].values()))
                self.assertFalse(result["pass"])
                self.assertEqual([conflict], result["post_load_conflicts"])
                self.assertTrue(any("confounded qualification" in p for p in result["problems"]))
                unload.assert_called_once_with("candidate")


class PrivilegeProfileTests(unittest.TestCase):
    """A preset policy refuses tools to must never be sent a tools payload."""

    def test_incumbent_chat_presets_are_still_judged_on_tool_use(self):
        for model in ("ridge", "heretic", "obliterated", "fable", "phr00ty"):
            with self.subTest(model=model):
                self.assertEqual(module.CORE_CHECKS, module.checks_for(model))

    def test_the_repository_agent_candidate_keeps_the_tool_contract(self):
        self.assertIn("tool_call", module.checks_for("qwen3-coder-next"))

    def test_the_low_privilege_candidate_is_never_offered_tools(self):
        for model in ("gemma4-heretic", "gemma4-heretic-vision"):
            with self.subTest(model=model):
                self.assertEqual(module.READER_CHECKS, module.checks_for(model))
                self.assertNotIn("tool_call", module.checks_for(model))

    def test_unknown_and_non_chat_presets_are_not_offered_tools(self):
        for model in ("unknown-alias", "qwen3-embedding-8b", "qwen25-coder-7b-fim"):
            with self.subTest(model=model):
                self.assertEqual(module.READER_CHECKS, module.checks_for(model))

    def test_a_reader_run_sends_no_tools_and_claims_no_tool_capability(self):
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
            {"mem_available_bytes": 40 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 4 << 30}},
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ]
        offered: list[object] = []

        def fake_chat(model, prompt, *, schema=None, tools=None):
            offered.append(tools)
            content = "PONG" if schema is None else '{"status": "ready", "count": 3}'
            return {"message": {"content": content}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "http_json", return_value={"data": [{"id": "gemma4-heretic"}]}),
            patch.object(module, "unload"),
        ):
            result = module.check_model("gemma4-heretic")

        self.assertTrue(result["pass"])
        self.assertFalse(result["tools_offered"])
        self.assertEqual([None, None], offered, "a tools payload reached a low-privilege preset")
        self.assertNotIn("tool_call", result["checks"])
        self.assertIn("tool_policy", result)

    def test_a_missing_required_check_still_fails_the_model(self):
        # The required set is compared exactly, so a section that never ran
        # cannot be mistaken for one that passed.
        samples = [
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
            {"mem_available_bytes": 50 << 30, "swap_used_bytes": 0, "vram_used_bytes": {"card0": 0}},
        ]

        def fake_chat(model, prompt, *, schema=None, tools=None):
            if schema is not None:
                raise RuntimeError("router dropped the request")
            return {"message": {"content": "PONG"}}

        with (
            patch.object(module, "sample", side_effect=samples),
            patch.object(module, "blocked_workloads", return_value=[]),
            patch.object(module, "chat", side_effect=fake_chat),
            patch.object(module, "http_json", return_value={"data": [{"id": "gemma4-heretic"}]}),
            patch.object(module, "unload"),
        ):
            result = module.check_model("gemma4-heretic")

        self.assertFalse(result["pass"])
        self.assertNotIn("structured_output", result["checks"])


class ModelCoverageTests(unittest.TestCase):
    """The presets this gate qualifies must exist in the router preset file."""

    PRESETS = Path("/home/typhoon/git/frankenstein-llm/llama-models.ini").read_text()

    def test_every_gated_preset_is_declared(self):
        for model in list(module.CHAT_MODELS) + list(module.VISION_MODELS):
            with self.subTest(model=model):
                self.assertIn(f"[{model}]", self.PRESETS)

    def test_the_fim_preset_is_not_displaced_by_the_coder_candidate(self):
        # Qwen3-Coder-Next is a repository-agent candidate, not a completion
        # model. Losing the FIM preset would silently remove /infill coverage.
        self.assertIn("[qwen25-coder-7b-fim]", self.PRESETS)
        self.assertIn("[qwen3-coder-next]", self.PRESETS)


class ReleaseAccountingTests(unittest.TestCase):
    """Eviction has to be proven, not assumed from a reading that went missing."""

    def test_late_competitor_preserves_existing_failures(self):
        conflict = {"pid": 123, "reason": "build", "command": "clang", "cwd": "/build"}
        result = {"problems": ["existing failure"]}
        with patch.object(module, "blocked_workloads", return_value=[conflict]):
            module.record_late_conflicts(result)
        self.assertEqual([conflict], result["post_load_conflicts"])
        self.assertEqual("existing failure", result["problems"][0])
        self.assertIn("confounded qualification", result["problems"][1])

    def test_quiet_post_load_check_adds_no_failure(self):
        result = {"problems": []}
        with patch.object(module, "blocked_workloads", return_value=[]):
            module.record_late_conflicts(result)
        self.assertEqual([], result["post_load_conflicts"])
        self.assertEqual([], result["problems"])

    def release(self, before: dict, after: dict) -> list[str]:
        result: dict = {"problems": []}
        with patch.object(module, "sample", return_value=after):
            module.record_release(result, before)
        self.assertEqual(after, result["after_unload"])
        return result["problems"]

    @staticmethod
    def state(vram: dict[str, int], available: int = 50 << 30, swap: int = 0,
              arc: int = 8 << 30, arc_min: int = 4 << 30) -> dict:
        reclaimable = max(0, arc - arc_min)
        return {"mem_available_bytes": available, "swap_used_bytes": swap,
                "vram_used_bytes": vram, "zfs_arc_bytes": arc,
                "zfs_arc_min_bytes": arc_min,
                "zfs_arc_reclaimable_bytes": reclaimable,
                "reclaimable_ram_bytes": available + reclaimable}

    def test_a_clean_release_records_no_problem(self):
        self.assertEqual([], self.release(self.state({"card0": 0, "card1": 0}),
                                          self.state({"card0": 0, "card1": 0})))

    def test_retained_vram_is_reported(self):
        problems = self.release(self.state({"card0": 0}), self.state({"card0": 8 << 30}))
        self.assertEqual(1, len(problems))
        self.assertIn("retained", problems[0])

    def test_a_card_that_stopped_reporting_is_not_a_release(self):
        # vram_used() drops a card whose sysfs node stopped answering. Falling
        # back to the baseline for it produced a zero delta, so "we lost the
        # reading" scored exactly like "it gave the memory back".
        problems = self.release(self.state({"card0": 0, "card1": 0}), self.state({"card0": 0}))
        self.assertEqual(1, len(problems))
        self.assertIn("card1", problems[0])
        self.assertIn("unproven", problems[0])

    def test_a_baseline_with_no_cards_at_all_fails(self):
        problems = self.release(self.state({}), self.state({}))
        self.assertTrue(any("eviction is unproven" in problem for problem in problems))

    def test_ram_not_returned_is_reported(self):
        problems = self.release(self.state({"card0": 0}, available=50 << 30),
                                self.state({"card0": 0}, available=40 << 30))
        self.assertTrue(any("reclaimable RAM" in problem for problem in problems))

    def test_ram_absorbed_by_the_zfs_arc_is_not_charged_to_the_model(self):
        # The whole shortfall moved into the ARC, which the kernel gives back
        # under pressure. This is what actually happened to phr00ty on boot
        # 1669f3ad: MemAvailable fell 2.25 GiB across its check while the ARC was
        # filling toward its 32 GiB cap, and the gate called it a leak.
        problems = self.release(
            self.state({"card0": 0}, available=50 << 30, arc=8 << 30),
            self.state({"card0": 0}, available=40 << 30, arc=18 << 30))
        self.assertEqual([], problems)

    def test_an_anonymous_leak_still_fails_when_the_arc_is_accounted(self):
        # The correction must not become an excuse. Here the ARC did not move, so
        # the missing 10 GiB is memory the model did not give back.
        problems = self.release(
            self.state({"card0": 0}, available=50 << 30, arc=8 << 30),
            self.state({"card0": 0}, available=40 << 30, arc=8 << 30))
        self.assertTrue(any("reclaimable RAM" in problem for problem in problems))

    def test_a_leak_larger_than_the_arc_growth_still_fails(self):
        # ARC grew 2 GiB, but 10 GiB went missing: the 8 GiB the ARC cannot
        # explain is still over tolerance and is still the model's.
        problems = self.release(
            self.state({"card0": 0}, available=50 << 30, arc=8 << 30),
            self.state({"card0": 0}, available=40 << 30, arc=10 << 30))
        self.assertTrue(any("reclaimable RAM" in problem for problem in problems))

    def test_arc_below_its_floor_is_never_credited(self):
        # Only the part above c_min is reclaimable; an ARC at its floor may not be
        # counted as free memory just because it is large.
        self.assertEqual(0, module.arc_reclaimable({"size": 4 << 30, "c_min": 4 << 30}))
        self.assertEqual(1 << 30, module.arc_reclaimable({"size": 5 << 30, "c_min": 4 << 30}))

    def test_a_host_without_zfs_credits_nothing(self):
        # arcstats is absent, so the correction contributes zero and the check
        # falls back to exactly the MemAvailable comparison it used before.
        self.assertEqual(0, module.arc_reclaimable({"size": -1, "c_min": -1}))

    def test_ram_recovery_is_recorded_even_when_it_passes(self):
        result: dict = {"problems": []}
        after = self.state({"card0": 0}, available=40 << 30, arc=18 << 30)
        with patch.object(module, "sample", return_value=after):
            module.record_release(result, self.state({"card0": 0}, available=50 << 30,
                                                     arc=8 << 30))
        recovery = result["ram_recovery"]
        self.assertEqual(0, recovery["reclaimable_delta_bytes"])
        self.assertEqual(10 << 30, recovery["mem_available_delta_bytes"])
        self.assertEqual(10 << 30, recovery["zfs_arc_delta_bytes"])
        self.assertTrue(recovery["arc_accounted"])

    def test_swap_growth_is_reported(self):
        problems = self.release(self.state({"card0": 0}, swap=0),
                                self.state({"card0": 0}, swap=8 << 30))
        self.assertTrue(any("swap grew" in problem for problem in problems))

    def test_tolerances_absorb_ordinary_noise(self):
        problems = self.release(
            self.state({"card0": 0}, available=50 << 30, swap=0),
            self.state({"card0": module.VRAM_TOLERANCE}, available=50 << 30, swap=0))
        self.assertEqual([], problems)


class BlockedWorkloadTests(unittest.TestCase):
    """Contended-host detection, against a fixture /proc rather than this host.

    Nothing here writes a ``cmdline`` file: reading one is what the scan must
    never do, so a fixture that offered one would let a regression pass.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.proc = Path(self.temporary.name)

    def process(self, pid: str, command: str, cgroup: str = "0::/user.slice",
                cwd: str | None = None) -> None:
        entry = self.proc / pid
        entry.mkdir()
        (entry / "comm").write_text(command + "\n")
        (entry / "cgroup").write_text(cgroup + "\n")
        if cwd is not None:
            (entry / "cwd").symlink_to(cwd)

    def reasons(self) -> list[tuple[int, object]]:
        return [(entry["pid"], entry["reason"])
                for entry in module.blocked_workloads(self.proc)]

    def test_a_quiet_host_blocks_nothing(self):
        self.process("10", "hyprland")
        self.process("11", "zsh")
        self.process("12", "llama-server",
                     cgroup="0::/user.slice/.../llama-router.service")
        self.process("13", "cargo")
        self.process("14", "rustc")
        self.assertEqual([], self.reasons())

    def test_unmanaged_inference_blocks(self):
        self.process("12", "llama-server")
        self.assertEqual([(12, "inference")], self.reasons())

    def test_kernel_tree_builds_block(self):
        self.process("20", "make", cwd="/home/typhoon/git/linux-tkg/src")
        self.assertEqual([(20, "kernel-build")], self.reasons())

    def test_unrelated_compilers_and_transfers_do_not_block(self):
        for pid, command in (("20", "makepkg"), ("21", "cmake"), ("22", "ninja"),
                             ("23", "cargo"), ("24", "rustc"), ("25", "clang"),
                             ("30", "aria2c"), ("31", "curl")):
            self.process(pid, command)
        self.assertEqual([], self.reasons())

    def test_the_download_queue_is_found_by_its_unit_not_its_task_name(self):
        self.process("40", "python3",
                     cgroup="0::/user.slice/.../local-ai-model-downloads-phase4.service")
        self.assertEqual([(40, "download-queue")], self.reasons())

    def test_the_supervisor_that_launched_this_gate_is_not_a_blocker(self):
        """Every gate in this workspace is a "python3"; blocking on that name
        would make the mission refuse every run it started itself."""
        self.process("50", "python3", cgroup="0::/user.slice/.../local-ai-functional-mission.service")
        self.process("51", "python3")
        self.assertEqual([], self.reasons())

    def test_the_scanning_process_does_not_report_itself(self):
        self.process(str(os.getpid()), "llama-server")
        self.assertEqual([], self.reasons())

    def test_a_task_that_exited_mid_scan_is_not_a_workload(self):
        (self.proc / "60").mkdir()  # no comm: gone between listing and reading
        self.process("61", "llama-server")
        self.assertEqual([(61, "inference")], self.reasons())

    def test_findings_are_ordered_so_two_scans_compare(self):
        for pid in ("900", "9", "90"):
            self.process(pid, "llama-server")
        self.assertEqual([9, 90, 900], [pid for pid, _ in self.reasons()])

    def test_cmdline_is_never_opened(self):
        self.process("70", "llama-server")
        original = Path.read_bytes

        def guarded(path, *args, **kwargs):
            if path.name == "cmdline":
                raise AssertionError("reading target address space is forbidden")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_bytes", guarded):
            self.assertEqual([(70, "inference")], self.reasons())

    def test_refusal_matches_what_the_mission_waits_for(self):
        """A functional gate must not refuse cargo the supervisor would let through."""
        source = (Path(__file__).resolve().parents[1]
                  / "mission-supervisor" / "run_functional_mission.py")
        spec = importlib.util.spec_from_file_location("supervisor_workload_names", source)
        assert spec is not None and spec.loader is not None
        supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supervisor)
        self.assertEqual(supervisor.MISSION_BLOCKING_REASONS,
                         module.MISSION_BLOCKING_REASONS)


if __name__ == "__main__":
    unittest.main()
