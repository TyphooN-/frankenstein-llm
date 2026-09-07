"""Cross-file consistency for the download queues.

The expected byte totals are repeated in four places: each queue JSON, the
later-phase runners that refuse to start until the previous stamp matches, and
the mission supervisor that refuses to qualify anything until all four do. A
queue edited without updating those copies produces a stamp mismatch
that fails a unit *after* the transfer, or -- worse -- a supervisor that waits
forever for a number nothing will ever write. Cheap to check here; expensive to
discover at 70 GB.

The later slot-comparison queue is not a supervisor stamp, but it still must
not collide on destinations or artifact keys with the four transfer phases.

Reads JSON and source text only. No network, no models directory.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
MODELS_ROOT = Path("/home/typhoon/git/frankenstein-llm/models")
QUEUES = {
    "phase1": HERE / "download-queue.json",
    "phase2": HERE / "download-queue-phase2.json",
    "phase3": HERE / "download-queue-phase3.json",
    "phase4": HERE / "download-queue-phase4.json",
}
ALL_QUEUES = {
    **QUEUES,
    "slot-uncensored-27b": HERE / "download-queue-slot-uncensored-27b.json",
}
STAMP_BYTES = {name: json.loads(path.read_text())["total_bytes"]
               for name, path in QUEUES.items()}


def literals(path: Path) -> list[str]:
    """Every string literal in a module, without importing it."""
    tree = ast.parse(path.read_text())
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


class QueueShapeTests(unittest.TestCase):
    def test_total_bytes_equals_the_sum_of_the_files(self):
        for name, path in ALL_QUEUES.items():
            queue = json.loads(path.read_text())
            declared = queue["total_bytes"]
            computed = sum(int(f["size"]) for a in queue["artifacts"] for f in a["files"])
            with self.subTest(queue=name):
                self.assertEqual(declared, computed)

    def test_every_destination_lands_inside_the_ignored_models_tree(self):
        # models/ is gitignored wholesale. A destination outside it would write
        # weights into tracked space.
        for name, path in ALL_QUEUES.items():
            queue = json.loads(path.read_text())
            for artifact in queue["artifacts"]:
                for entry in artifact["files"]:
                    destination = Path(entry["destination"])
                    with self.subTest(queue=name, key=artifact["key"]):
                        self.assertTrue(destination.is_absolute(), destination)
                        self.assertIn(MODELS_ROOT, destination.parents, str(destination))

    def test_artifact_keys_are_unique_within_and_across_queues(self):
        seen: dict[str, str] = {}
        for name, path in ALL_QUEUES.items():
            for artifact in json.loads(path.read_text())["artifacts"]:
                key = artifact["key"]
                with self.subTest(queue=name, key=key):
                    self.assertNotIn(key, seen, f"also defined in {seen.get(key)}")
                seen[key] = name

    def test_destinations_are_unique_across_all_concurrently_running_queues(self):
        # Each phase has its own process lock and phases 1/2 can run together.
        # A duplicate across manifests would therefore bypass the in-process
        # destination guard and give two curl processes the same partial file.
        seen: dict[str, str] = {}
        for name, path in ALL_QUEUES.items():
            for artifact in json.loads(path.read_text())["artifacts"]:
                for entry in artifact["files"]:
                    destination = str(Path(entry["destination"]).resolve(strict=False))
                    with self.subTest(queue=name, destination=destination):
                        self.assertNotIn(
                            destination, seen,
                            f"destination also belongs to {seen.get(destination)}")
                    seen[destination] = name

    def test_every_artifact_declares_a_pinned_revision_and_files(self):
        for name, path in ALL_QUEUES.items():
            for artifact in json.loads(path.read_text())["artifacts"]:
                with self.subTest(queue=name, key=artifact["key"]):
                    self.assertTrue(artifact["repository"])
                    self.assertTrue(artifact["revision"])
                    self.assertTrue(artifact["files"])
                    for entry in artifact["files"]:
                        self.assertGreater(int(entry["size"]), 0)


class StampConstantTests(unittest.TestCase):
    """The hardcoded byte totals in the gating code must match the queues."""

    def test_phase_two_runner_waits_for_the_real_phase_one_total(self):
        source = (HERE / "run_download_phase2.py").read_text()
        match = re.search(r"EXPECTED_PHASE1_BYTES\s*=\s*'(\d+)'", source)
        self.assertIsNotNone(match, "phase-two runner no longer pins a total")
        self.assertEqual(str(STAMP_BYTES["phase1"]), match.group(1))

    def test_phase_three_runner_waits_for_both_upstream_totals(self):
        found = literals(HERE / "run_download_phase3.py")
        for phase in ("phase1", "phase2"):
            with self.subTest(phase=phase):
                self.assertIn(str(STAMP_BYTES[phase]), found)

    def test_phase_four_runner_waits_for_all_prior_totals(self):
        found = literals(HERE / "run_download_phase4.py")
        for phase in ("phase1", "phase2", "phase3"):
            with self.subTest(phase=phase):
                self.assertIn(str(STAMP_BYTES[phase]), found)

    def test_mission_supervisor_waits_for_all_four_totals(self):
        supervisor = HERE.parent / "mission-supervisor" / "run_functional_mission.py"
        found = literals(supervisor)
        for phase, total in STAMP_BYTES.items():
            with self.subTest(phase=phase):
                self.assertIn(str(total), found,
                              f"{supervisor.name} does not gate on the {phase} total")


if __name__ == "__main__":
    unittest.main()
