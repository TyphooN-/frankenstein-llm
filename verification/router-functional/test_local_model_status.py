#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MODULE = SCRIPTS / "local_model_status.py"
SUPERVISOR = ROOT / "verification/mission-supervisor/run_functional_mission.py"
# Running the script puts scripts/ on sys.path; loading it out of band has to
# arrange the same thing, or its shared display module is not importable.
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("local_model_status", MODULE)
assert SPEC is not None and SPEC.loader is not None
status = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = status
SPEC.loader.exec_module(status)


def mission_statuses() -> set[str]:
    """Every literal the supervisor writes to the *mission's* own ``status``.

    Read from ``state.update({...})`` calls specifically. A plain text search for
    ``"status":`` also collects the per-step vocabulary -- ``passed``, ``failed``,
    ``blocked-policy`` -- which lives in ``state["steps"][name]`` and answers a
    different question, so classifying against it would be a false comparison.
    """
    tree = ast.parse(SUPERVISOR.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "update" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "state" or not node.args:
            continue
        argument = node.args[0]
        if not isinstance(argument, ast.Dict):
            continue
        for key, value in zip(argument.keys, argument.values):
            if not (isinstance(key, ast.Constant) and key.value == "status"):
                continue
            # A status chosen by a conditional is still a status the supervisor
            # writes. Reading only plain constants let two of them reach the
            # status reader unclassified, which is exactly the drift this test
            # exists to catch.
            for candidate in ([value.body, value.orelse]
                              if isinstance(value, ast.IfExp) else [value]):
                if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                    found.add(candidate.value)
    assert found, "no mission statuses found in the supervisor"
    return found


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        return json.dumps(self.payload).encode()


class LocalModelStatusTests(unittest.TestCase):
    def test_collects_bounded_summary_without_model_arguments(self):
        replies = iter([
            FakeResponse({"status": "ok"}),
            FakeResponse({
                "data": [
                    {
                        "id": "zeta",
                        "status": {"value": "unloaded", "args": ["--secret-like-noise"]},
                        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    },
                    {
                        "id": "alpha",
                        "status": {"value": "loaded"},
                        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                    },
                ]
            }),
        ])
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)) as request:
            result = status.collect_status(2.0)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["endpoint"], "http://127.0.0.1:8080")
        self.assertEqual(result["model_count"], 2)
        self.assertEqual(result["loaded_models"], ["alpha"])
        self.assertEqual([model["id"] for model in result["models"]], ["alpha", "zeta"])
        self.assertNotIn("args", result["models"][1])
        self.assertFalse(result["loads_models"])

    def test_unhealthy_router_fails_before_models_request(self):
        with patch.object(status.urllib.request, "urlopen", return_value=FakeResponse({"status": "loading"})) as request:
            with self.assertRaisesRegex(status.StatusError, "unhealthy"):
                status.collect_status()
        self.assertEqual(request.call_count, 1)

    def test_malformed_model_payload_fails_closed(self):
        replies = iter([FakeResponse({"status": "ok"}), FakeResponse({"data": [{}]})])
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            with self.assertRaisesRegex(status.StatusError, "malformed model entry"):
                status.collect_status()

    def test_timeout_is_strictly_bounded(self):
        for value in (0, -1, status.MAX_TIMEOUT + 0.1):
            with self.subTest(value=value), self.assertRaises(status.StatusError):
                status.fetch_json("/health", value)

    def test_response_body_is_bounded_before_json_decoding(self):
        class OversizedResponse(FakeResponse):
            def read(self, size=-1):
                self.requested_size = size
                return b"x" * size

        response = OversizedResponse(None)
        with patch.object(status.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(status.StatusError, "response exceeds"):
                status.fetch_json("/models")
        self.assertEqual(response.requested_size, status.MAX_RESPONSE_BYTES + 1)

    def _models_reply(self, *entries):
        return iter([
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [
                {
                    "id": name,
                    "status": {"value": value},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                }
                for name, value in entries
            ]}),
        ])

    def test_only_states_that_hold_a_child_process_count_as_resident(self):
        # llama.cpp v0.4.0 answers /models with six states. "sleeping" is a live
        # child that idled and "loading" is one starting up: both occupy the
        # router's single resident slot. "downloaded" only means weights reached
        # local disk, and the pre-v0.4.0 rule -- resident is anything except
        # "unloaded" -- reported that as a loaded model.
        replies = self._models_reply(
            ("cached", "downloaded"),
            ("fetching", "downloading"),
            ("idle", "sleeping"),
            ("starting", "loading"),
            ("live", "loaded"),
            ("cold", "unloaded"),
        )
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            result = status.collect_status()
        self.assertEqual(["idle", "live", "starting"], result["loaded_models"])
        self.assertEqual(6, result["model_count"])

    def test_status_vocabulary_is_partitioned_and_matches_the_pinned_runtime(self):
        self.assertEqual(frozenset(), status.RESIDENT_STATUSES & status.VACANT_STATUSES)
        self.assertEqual(
            {"downloading", "downloaded", "unloaded", "loading", "loaded", "sleeping"},
            set(status.KNOWN_STATUSES),
        )

    def test_status_outside_the_pinned_vocabulary_fails_closed(self):
        # A state this pin does not define could sit on either side of the
        # resident boundary, so guessing would silently misreport the slot.
        replies = self._models_reply(("mystery", "hibernating"))
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            with self.assertRaisesRegex(status.StatusError, "unknown status"):
                status.collect_status()

    def test_text_render_does_not_dump_router_launch_arguments(self):
        summary = {
            "endpoint": status.BASE_URL,
            "health": "ok",
            "model_count": 1,
            "status_counts": {"unloaded": 1},
            "loaded_models": [],
            "models": [{
                "id": "heretic",
                "status": "unloaded",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            }],
        }
        rendered = status.render_text(summary)
        self.assertIn("unloaded [text -> text]", rendered)
        self.assertIn("[compatibility alias: heretic]", rendered)
        self.assertNotIn("--model", rendered)

    def test_a_router_id_is_resolved_to_the_weight_file_it_loads(self):
        # "heretic" is the alias; RVN-Q6_K-multilingual-mtp.gguf is the file.
        replies = self._models_reply(("heretic", "loaded"), ("ridge", "unloaded"))
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            result = status.collect_status()
        by_id = {model["id"]: model for model in result["models"]}
        self.assertEqual(
            "RVN-Q6_K-multilingual-mtp.gguf (General-purpose Hermes agent and tool use)",
            by_id["heretic"]["description"])
        self.assertEqual(
            ["RVN-Q6_K-multilingual-mtp.gguf (General-purpose Hermes agent and tool use)"
             "  [compatibility alias: heretic]"],
            result["loaded_descriptions"])
        rendered = status.render_text(result)
        # Identity leads every line; the router's own key trails it, labelled.
        self.assertIn("  RVN-Q6_K-multilingual-mtp.gguf (General-purpose", rendered)
        self.assertIn("Loaded: RVN-Q6_K-multilingual-mtp.gguf", rendered)
        for line in rendered.splitlines():
            self.assertFalse(line.strip().startswith("heretic:"), line)

    def test_an_id_this_checkout_does_not_configure_is_said_so_not_hidden(self):
        replies = self._models_reply(("not-a-preset", "unloaded"))
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            result = status.collect_status()
        self.assertEqual("not-a-preset (not a configured preset)",
                         result["models"][0]["description"])

    def test_unreadable_local_configuration_still_reports_the_router(self):
        """A status read describes a live service; it is not a config gate."""
        with patch.object(status, "local_registry", return_value=({}, {})):
            replies = self._models_reply(("heretic", "loaded"))
            with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
                result = status.collect_status()
        self.assertEqual(["heretic"], result["loaded_models"])
        self.assertEqual("heretic (not a configured preset)",
                         result["models"][0]["description"])


class HostSharingTests(unittest.TestCase):
    """The host-sharing report must never read "I cannot tell" as "go ahead"."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name) / "mission-state.json"

    def write_state(self, document):
        self.state.write_text(json.dumps(document), encoding="utf-8")
        return self.state

    def quiet_supervisor(self, conflicts=(), available=64 << 30):
        supervisor = SimpleNamespace(
            MISSION_BLOCKING_REASONS=frozenset({"kernel-build", "inference"}),
            MIN_AVAILABLE=32 << 30,
            conflicts=lambda: list(conflicts),
            mem_available=lambda: available,
        )
        return patch.object(status, "load_supervisor", return_value=supervisor)

    def collect(self, state_path=None, **kwargs):
        with self.quiet_supervisor(**kwargs), patch.object(
                status, "collect_status", side_effect=status.StatusError("/health: refused")):
            return status.collect_host_sharing(state_path=state_path or self.state)

    # -- mission state ----------------------------------------------------

    def test_absent_state_is_not_reported_as_an_idle_mission(self):
        report = self.collect(state_path=Path(self.directory.name) / "nothing.json")
        self.assertEqual(status.VERDICT_MAY_TAKE_GPU, report["verdict"])
        self.assertEqual("unknown", report["mission"]["disposition"])
        self.assertFalse(report["mission"]["readable"])
        self.assertIn("FileNotFoundError", report["mission"]["problem"])

    def test_corrupt_state_is_not_reported_as_an_idle_mission(self):
        self.state.write_text("{ this is not json", encoding="utf-8")
        report = self.collect()
        self.assertEqual(status.VERDICT_MAY_TAKE_GPU, report["verdict"])
        self.assertEqual("unknown", report["mission"]["disposition"])

    def test_state_read_is_bounded_before_parsing(self):
        with patch.object(status, "MAX_STATE_BYTES", 16), patch.object(
                Path, "open") as opened:
            handle = opened.return_value.__enter__.return_value
            handle.read.return_value = b"x" * 17
            report = status.read_mission_state(self.state)
        handle.read.assert_called_once_with(17)
        self.assertEqual("unknown", report["disposition"])
        self.assertIn("exceeds", report["problem"])

    def test_a_state_from_another_schema_is_not_classified(self):
        self.write_state({"schema": "something-else/9", "status": "functional-foundation-complete"})
        report = self.collect()
        self.assertEqual(status.VERDICT_MAY_TAKE_GPU, report["verdict"])
        self.assertIn("unknown schema", report["mission"]["problem"])

    def test_a_null_current_step_during_the_quiet_wait_is_not_read_as_between_steps(self):
        """The correction this report exists for.

        ``run_step`` sets ``current_step`` only after ``wait_for_quiet`` has
        already returned, and a freshly initialised state has no such key at all.
        So the supervisor counting quiet polls before its very first step -- the
        moment it is closest to claiming the GPU -- looks exactly like "no step is
        running" to a reader of ``current_step``.
        """
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "waiting-safe-host"})
        report = self.collect()
        self.assertIsNone(report["mission"]["current_step"])
        self.assertEqual("pending", report["mission"]["disposition"])
        self.assertEqual(status.VERDICT_MAY_TAKE_GPU, report["verdict"])

    def test_a_stale_current_step_between_steps_does_not_invent_a_running_step(self):
        """The same field is wrong in the other direction too.

        ``current_step`` is cleared only at the three whole-mission exits, never
        when an individual step ends, so it goes on naming a finished step. The
        verdict is taken from ``status`` alone, which is why a terminal mission
        that still records a step name is classified as terminal.
        """
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "blocked-policy", "current_step": "candidate-policy"})
        report = self.collect()
        self.assertEqual("candidate-policy", report["mission"]["current_step"])
        self.assertEqual(status.VERDICT_IDLE_LAST_WRITE, report["verdict"])
        self.assertTrue(report["mission"]["current_step_is_not_a_liveness_signal"])

    def test_a_running_status_reports_a_live_step(self):
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "running", "current_step": "embeddings"})
        report = self.collect()
        self.assertEqual(status.VERDICT_STEP_RUNNING, report["verdict"])

    def test_an_unknown_status_fails_closed_like_the_router_vocabulary(self):
        self.write_state({"schema": "frankenstein-functional-mission/1", "status": "napping"})
        report = self.collect()
        self.assertEqual(status.VERDICT_MAY_TAKE_GPU, report["verdict"])
        self.assertIn("unknown status", report["mission"]["problem"])

    def test_every_supervisor_status_is_classified_exactly_once(self):
        """The three sets must partition what the supervisor can write.

        A status that belongs to two sets would be classified by whichever test
        ran first; one that belongs to none silently becomes "unknown", which is
        safe but hides a real drift from the supervisor.
        """
        sets = (status.RUNNING_MISSION_STATUSES, status.PENDING_MISSION_STATUSES,
                status.TERMINAL_MISSION_STATUSES)
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                self.assertEqual(frozenset(), sets[left] & sets[right])
        self.assertEqual(set(), mission_statuses() - set().union(*sets))

    # -- host scan --------------------------------------------------------

    def test_an_unreadable_process_table_is_not_reported_as_a_quiet_host(self):
        broken = SimpleNamespace(
            MISSION_BLOCKING_REASONS=frozenset({"inference"}), MIN_AVAILABLE=32 << 30,
            conflicts=lambda: (_ for _ in ()).throw(RuntimeError("process table /proc is not readable")),
            mem_available=lambda: 64 << 30)
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "functional-foundation-complete"})
        with patch.object(status, "load_supervisor", return_value=broken), patch.object(
                status, "collect_status", side_effect=status.StatusError("down")):
            report = status.collect_host_sharing(state_path=self.state)
        self.assertEqual(status.VERDICT_MAY_TAKE_GPU, report["verdict"])
        self.assertEqual("unavailable", report["host"]["scan"])

    def test_a_supervisor_that_cannot_be_imported_downgrades_the_verdict(self):
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "functional-foundation-complete"})
        with patch.object(status, "load_supervisor",
                          side_effect=ValueError("HERMES_MISSION_QUIET_TIMEOUT must be positive")), \
                patch.object(status, "collect_status", side_effect=status.StatusError("down")):
            report = status.collect_host_sharing(state_path=self.state)
        self.assertEqual(status.VERDICT_MAY_TAKE_GPU, report["verdict"])
        self.assertIn("ValueError", report["host"]["problem"])

    def test_blocking_and_advisory_conflicts_are_reported_separately(self):
        """The supervisor blocks on two reasons; the rest are recorded, not gating."""
        found = [
            {"pid": 1, "reason": "kernel-build", "command": "cc1", "cwd": "", "cgroup": ""},
            {"pid": 2, "reason": "transfer", "command": "aria2c", "cwd": "", "cgroup": ""},
            {"pid": 3, "reason": "build", "command": "cargo", "cwd": "", "cgroup": ""},
        ]
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "functional-foundation-complete"})
        report = self.collect(conflicts=found)
        self.assertEqual([1], [item["pid"] for item in report["host"]["blocking_conflicts"]])
        self.assertEqual([2, 3], [item["pid"] for item in report["host"]["advisory_conflicts"]])
        self.assertEqual(["inference", "kernel-build"], report["host"]["blocking_reasons"])

    def test_memory_below_the_mission_floor_is_reported(self):
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "functional-foundation-complete"})
        report = self.collect(available=8 << 30)
        self.assertFalse(report["host"]["meets_mission_memory_floor"])
        self.assertTrue(any("below the mission" in reason for reason in report["reasons"]))

    def test_conflict_samples_are_bounded_without_losing_totals_or_reasons(self):
        found = [{"pid": pid, "reason": reason, "command": "fixture"}
                 for reason in ("kernel-build", "inference", "transfer")
                 for pid in range(status.MAX_CONFLICT_SAMPLES + 1)]
        report = self.collect(conflicts=found)
        host = report["host"]
        self.assertEqual(status.MAX_CONFLICT_SAMPLES, len(host["blocking_conflicts"]))
        self.assertEqual(status.MAX_CONFLICT_SAMPLES, len(host["advisory_conflicts"]))
        self.assertEqual(2 * (status.MAX_CONFLICT_SAMPLES + 1), host["blocking_count"])
        self.assertEqual(status.MAX_CONFLICT_SAMPLES + 1, host["advisory_count"])
        self.assertEqual({"kernel-build", "inference"}, set(host["blocking_counts_by_reason"]))
        self.assertTrue(any("inference, kernel-build" in r for r in report["reasons"]))
        self.assertIn(f"blocking={host['blocking_count']}", status.render_host_sharing(report))

    # -- contract ---------------------------------------------------------

    def test_the_report_never_claims_to_be_an_admission_guarantee(self):
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "functional-foundation-complete"})
        report = self.collect()
        self.assertEqual(status.VERDICT_IDLE_LAST_WRITE, report["verdict"])
        self.assertTrue(report["advisory"])
        self.assertFalse(report["atomic_admission"])
        self.assertFalse(report["loads_models"])
        # No verdict spells safety. The idle one is qualified by "per-last-write"
        # precisely because the unit can be started in the next second.
        self.assertNotIn("safe", " ".join(
            [report["verdict"], *report["reasons"]]).lower())

    def test_an_unreachable_router_is_a_fact_about_sharing_not_a_failure(self):
        self.write_state({"schema": "frankenstein-functional-mission/1",
                          "status": "functional-foundation-complete"})
        report = self.collect()
        self.assertFalse(report["router"]["reachable"])
        self.assertEqual(status.VERDICT_IDLE_LAST_WRITE, report["verdict"])
        self.assertIn("/health: refused", report["router"]["problem"])

    def test_the_cli_exits_zero_for_every_verdict(self):
        """Exit status reports production of the report, never admission."""
        for state in ("running", "waiting-safe-host", "functional-foundation-complete"):
            with self.subTest(state=state):
                self.write_state({"schema": "frankenstein-functional-mission/1", "status": state})
                with self.quiet_supervisor(), patch.object(
                        status, "MISSION_STATE", self.state), patch.object(
                        status, "collect_status", side_effect=status.StatusError("down")), patch(
                        "sys.stdout", new_callable=io.StringIO) as out:
                    self.assertEqual(0, status.main(["--host-sharing"]))
                self.assertIn("advisory; not an admission guarantee", out.getvalue())

    def test_the_json_form_carries_the_same_verdict(self):
        self.write_state({"schema": "frankenstein-functional-mission/1", "status": "running"})
        with self.quiet_supervisor(), patch.object(
                status, "MISSION_STATE", self.state), patch.object(
                status, "collect_status", side_effect=status.StatusError("down")), patch(
                "sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(0, status.main(["--host-sharing", "--json"]))
        self.assertEqual(status.VERDICT_STEP_RUNNING, json.loads(out.getvalue())["verdict"])

    def test_the_default_invocation_is_unchanged_by_the_new_flag(self):
        replies = iter([
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [{"id": "heretic", "status": {"value": "unloaded"},
                                    "architecture": {}}]}),
        ])
        with patch.object(status.urllib.request, "urlopen",
                          side_effect=lambda *_a, **_k: next(replies)), patch(
                "sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(0, status.main([]))
        self.assertIn("Router: ok", out.getvalue())
        self.assertNotIn("advisory", out.getvalue())


class SupervisorReuseTests(unittest.TestCase):
    """The report must consume the supervisor's decision, not a second copy of it."""

    def test_the_real_supervisor_module_supplies_the_predicates(self):
        supervisor = status.load_supervisor()
        self.assertEqual(frozenset({"kernel-build", "inference", "download-queue"}),
                         supervisor.MISSION_BLOCKING_REASONS)
        self.assertTrue(callable(supervisor.conflicts))
        self.assertTrue(callable(supervisor.mem_available))
        self.assertIsInstance(supervisor.MIN_AVAILABLE, int)

    def test_importing_the_supervisor_installs_no_signal_handlers(self):
        """Importing it must not arm the handlers a running mission installs."""
        before = {number: signal.getsignal(number)
                  for number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
        status.load_supervisor()
        self.assertEqual(before, {number: signal.getsignal(number) for number in before})

    def test_the_classifier_reads_a_fixture_proc_root_without_touching_the_host(self):
        """`conflicts()` is injectable, so this exercises it against fixtures."""
        supervisor = status.load_supervisor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, comm, cwd, cgroup in (
                    ("101", "cc1", "/linux-tkg", ""),
                    ("102", "llama-server", "/tmp", "0::/user.slice/llama-router.service"),
                    ("103", "llama-server", "/tmp", "0::/user.slice/llama-sidecar@ocr.service"),
            ):
                entry = root / pid
                entry.mkdir()
                (entry / "comm").write_text(comm + "\n")
                (entry / "cgroup").write_text(cgroup + "\n")
                (entry / "target").mkdir()
                (entry / "cwd").symlink_to(cwd)
            found = supervisor.conflicts(proc_root=root)
        # The managed router is excused by cgroup; a sidecar with the same binary
        # name is not, which is why a sidecar left resident blocks the mission.
        self.assertEqual([("101", "kernel-build"), ("103", "inference")],
                         [(str(item["pid"]), item["reason"]) for item in found])

    def test_an_unreadable_proc_root_raises_rather_than_reporting_quiet(self):
        supervisor = status.load_supervisor()
        with self.assertRaisesRegex(RuntimeError, "refusing to read an unreadable host"):
            supervisor.conflicts(proc_root=Path("/nonexistent-proc-root"))


if __name__ == "__main__":
    unittest.main()
