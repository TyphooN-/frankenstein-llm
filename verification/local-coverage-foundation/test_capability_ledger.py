#!/usr/bin/env python3
"""Offline tests for the capability-ledger rebuild. No model, no GPU, no network.

The ledger's only job is to refuse to overstate the stack, so every test here is
a way it could overstate: evidence that is absent, unreadable, interrupted,
failing, or older than the weights it claims to have judged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_capability_ledger as ledger  # noqa: E402

RECENT = "2026-09-03T12:00:00-0400"
OLDER = "2026-09-01T12:00:00-0400"
RECENT_TS = ledger.parse_recorded_at(RECENT)
OLDER_TS = ledger.parse_recorded_at(OLDER)


def artifact(key="a", complete=True, newest=OLDER_TS, missing=(), wrong_size=()):
    return {"key": key, "capability": "synthetic", "complete": complete,
            "missing": list(missing), "wrong_size": list(wrong_size),
            "newest_mtime": newest, "files": []}


def evidence(passed=True, recorded=RECENT, interrupted=False, present=True,
             throughput=False, path="/tmp/gate-synthetic.json"):
    return {"path": path, "present": present, "pass": passed,
            "interrupted": interrupted, "recorded_at": recorded, "error": None,
            "sections_missing": None, "throughput_measured": throughput}


class QueueReadingTests(unittest.TestCase):
    def test_groups_artifacts_by_capability_across_queues(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, capability in enumerate(("asr", "asr", "fim")):
                path = Path(directory) / f"queue{index}.json"
                path.write_text(json.dumps({"artifacts": [
                    {"capability": capability, "key": f"k{index}", "files": []}]}))
                paths.append(path)
            grouped = ledger.read_queues(paths)
        self.assertEqual(["asr", "fim"], sorted(grouped))
        self.assertEqual(2, len(grouped["asr"]))
        self.assertEqual("queue0.json", grouped["asr"][0]["queue"])

    def test_inspect_reports_missing_and_wrong_size_files(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.bin"
            good.write_bytes(b"1234")
            bad = Path(directory) / "bad.bin"
            bad.write_bytes(b"12")
            inspected = ledger.inspect_artifact({
                "capability": "synthetic", "key": "k", "queue": "q.json",
                "files": [
                    {"repo_path": "good.bin", "destination": str(good), "size": 4},
                    {"repo_path": "bad.bin", "destination": str(bad), "size": 4},
                    {"repo_path": "gone.bin", "destination": str(Path(directory) / "gone.bin"),
                     "size": 4},
                ]})
        self.assertFalse(inspected["complete"])
        self.assertEqual(["gone.bin"], inspected["missing"])
        self.assertEqual(["bad.bin"], inspected["wrong_size"])
        self.assertFalse(inspected["sha256_reverified"])


class EvidenceReadingTests(unittest.TestCase):
    def test_absent_artifact_is_not_a_pass(self):
        record = ledger.read_evidence(Path("/nonexistent/gate.json"))
        self.assertFalse(record["present"])
        self.assertFalse(record["pass"])
        self.assertEqual("evidence artifact absent", record["error"])

    def test_unreadable_artifact_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text("{ this is not json")
            record = ledger.read_evidence(path)
        self.assertTrue(record["present"])
        self.assertFalse(record["pass"])
        self.assertIn("JSONDecodeError", record["error"])

    def test_reads_pass_interruption_and_throughput_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps({
                "gate": "synthetic", "pass": False, "interrupted": True,
                "recorded_at": OLDER, "error": "interrupted by SIGTERM",
                "sections_missing": ["grounding"], "throughput_measured": True}))
            record = ledger.read_evidence(path)
        self.assertFalse(record["pass"])
        self.assertTrue(record["interrupted"])
        self.assertTrue(record["throughput_measured"])
        self.assertEqual(["grounding"], record["sections_missing"])

    def test_truthy_pass_must_be_exactly_true(self):
        """A gate that wrote a message where a verdict belongs has not passed."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps({"pass": "yes", "recorded_at": RECENT}))
            self.assertFalse(ledger.read_evidence(path)["pass"])

    def test_timestamp_parsing_accepts_the_gate_format_and_rejects_junk(self):
        self.assertIsNotNone(ledger.parse_recorded_at("2026-09-01T12:57:07-0400"))
        for junk in (None, "", "yesterday", 17):
            with self.subTest(value=junk):
                self.assertIsNone(ledger.parse_recorded_at(junk))


class ClassificationTests(unittest.TestCase):
    def test_incomplete_download_is_never_qualified(self):
        state, reasons = ledger.classify(
            [artifact(complete=False, missing=["shard-2"])], [evidence()])
        self.assertEqual(ledger.STATE_INCOMPLETE, state)
        self.assertIn("1 files missing", reasons[0])

    def test_no_declared_evidence_stays_downloaded(self):
        state, reasons = ledger.classify([artifact()], [])
        self.assertEqual(ledger.STATE_DOWNLOADED, state)
        self.assertIn("no evidence artifact is declared", reasons[0])

    def test_capability_with_neither_artifacts_nor_evidence_is_only_researched(self):
        state, _ = ledger.classify([], [])
        self.assertEqual(ledger.STATE_RESEARCHED, state)

    def test_absent_evidence_is_not_a_pass(self):
        state, _ = ledger.classify(
            [artifact()], [evidence(present=False, passed=False)])
        self.assertEqual(ledger.STATE_DOWNLOADED, state)

    def test_interruption_outranks_everything_else(self):
        """An unfinished run is not a verdict, however far it got."""
        state, reasons = ledger.classify(
            [artifact()], [evidence(passed=True, interrupted=True)])
        self.assertEqual(ledger.STATE_INTERRUPTED, state)
        self.assertIn("not a verdict", reasons[0])

    def test_failing_evidence_is_reported_as_failing(self):
        state, _ = ledger.classify([artifact()], [evidence(passed=False)])
        self.assertEqual(ledger.STATE_FAILED, state)

    def test_evidence_older_than_the_weights_is_stale_not_passing(self):
        """The rule the previous ledger did not have.

        Phase three and four landed the day after the 2026-09-02 snapshot, which
        went on reporting a qualified stack for weights its gates never saw.
        """
        state, reasons = ledger.classify(
            [artifact(newest=RECENT_TS)], [evidence(passed=True, recorded=OLDER)])
        self.assertEqual(ledger.STATE_STALE, state)
        self.assertIn("recorded before the weights", reasons[0])

    def test_evidence_newer_than_the_weights_qualifies(self):
        state, reasons = ledger.classify(
            [artifact(newest=OLDER_TS)], [evidence(passed=True, recorded=RECENT)])
        self.assertEqual(ledger.STATE_QUALIFIED, state)
        self.assertEqual([], reasons)

    def test_unreadable_timestamp_is_stale_rather_than_assumed_fresh(self):
        state, _ = ledger.classify(
            [artifact(newest=OLDER_TS)], [evidence(passed=True, recorded="whenever")])
        self.assertEqual(ledger.STATE_STALE, state)

    def test_a_composed_capability_needs_every_declared_artifact(self):
        state, _ = ledger.classify([], [evidence(passed=True), evidence(passed=False)])
        self.assertEqual(ledger.STATE_FAILED, state)


class LedgerTests(unittest.TestCase):
    def synthetic(self, capability="asr", **evidence_fields):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        weight = root / "weight.bin"
        weight.write_bytes(b"1234")
        # Backdate the weight so the synthetic gate is newer than what it judged;
        # a file written during the test run is always newer than any fixed
        # timestamp and would make every synthetic capability read as stale.
        os.utime(weight, (OLDER_TS, OLDER_TS))
        queue = root / "queue.json"
        queue.write_text(json.dumps({"artifacts": [{
            "capability": capability, "key": "synthetic-artifact",
            "repository": "example/model", "revision": "deadbeef",
            "files": [{"repo_path": "weight.bin", "destination": str(weight), "size": 4}]}]}))
        gate = root / "gate.json"
        gate.write_text(json.dumps({"gate": capability, "pass": True,
                                    "recorded_at": RECENT, **evidence_fields}))
        return (queue,), {capability: (gate,)}

    def test_qualified_capability_is_listed_once_and_pass_is_about_the_ledger(self):
        queues, evidence_map = self.synthetic()
        report = ledger.build_ledger(queues, evidence_map)
        self.assertEqual(ledger.SCHEMA, report["schema"])
        self.assertEqual(["asr"], report["functionally_qualified"])
        self.assertTrue(report["pass"])
        self.assertFalse(report["throughput_measured"])
        self.assertFalse(report["model_inference_performed"])

    def test_measured_throughput_in_evidence_is_a_ledger_problem(self):
        """This workspace forbids throughput measurement; do not copy it forward."""
        queues, evidence_map = self.synthetic(throughput_measured=True)
        report = ledger.build_ledger(queues, evidence_map)
        self.assertFalse(report["pass"])
        self.assertIn("asr: an evidence artifact reports measured throughput",
                      report["problems"])

    def test_evidence_for_an_unknown_capability_is_a_ledger_problem(self):
        queues, evidence_map = self.synthetic()
        evidence_map = {**evidence_map, "invented": (Path("/nonexistent/gate.json"),)}
        report = ledger.build_ledger(queues, evidence_map)
        self.assertFalse(report["pass"])
        self.assertIn("invented: evidence is declared for a capability no queue owns",
                      report["problems"])

    def test_composed_capabilities_stay_visible_without_artifacts(self):
        queues, evidence_map = self.synthetic()
        report = ledger.build_ledger(queues, evidence_map)
        for name in ledger.COMPOSED_CAPABILITIES:
            with self.subTest(capability=name):
                self.assertIn(name, report["capabilities"])
                self.assertEqual([], report["capabilities"][name]["artifacts"])


class TrackedMappingTests(unittest.TestCase):
    """The declared mapping has to keep matching the tracked queues.

    Reads only files that are in Git, so it stays meaningful on a checkout with
    no weights on it.
    """

    def test_every_declared_capability_exists_in_a_queue_or_is_composed(self):
        known = set(ledger.read_queues()) | set(ledger.COMPOSED_CAPABILITIES)
        self.assertEqual(set(), set(ledger.CAPABILITY_EVIDENCE) - known)

    def test_the_four_pinned_queues_are_the_source_of_truth(self):
        self.assertEqual(4, len(ledger.QUEUES))
        for path in ledger.QUEUES:
            with self.subTest(queue=path.name):
                self.assertTrue(path.is_file())

    def test_flux2_and_gemma_heretic_declare_no_evidence_yet(self):
        """Neither has a gate that writes a durable artifact, so neither may pass.

        The functional media gate submits Z-Image, ACE-Step and Qwen-Edit; the
        pinned FLUX.2 graph is not among them, and the 2026-09-03 Gemma-4 checks
        were run against the router by hand.
        """
        for capability in ("image-generation-editing", "uncensored-multimodal"):
            with self.subTest(capability=capability):
                self.assertNotIn(capability, ledger.CAPABILITY_EVIDENCE)
                state, _ = ledger.classify([artifact()], [])
                self.assertNotEqual(ledger.STATE_QUALIFIED, state)


if __name__ == "__main__":
    unittest.main()
