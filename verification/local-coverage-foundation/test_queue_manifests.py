"""Cross-file consistency for the download queues.

Each queue's expected byte total is written twice: in the queue JSON and in the
qualification supervisor, which refuses to qualify anything until every
completion stamp matches it. A queue edited without updating the supervisor
produces a supervisor that waits forever for a number nothing will ever write.
The download service must also run exactly the queues the supervisor waits for,
under the state and stamp names it reads. Cheap to check here; expensive to
discover at 70 GB.

The later slot-comparison queue is not a supervisor stamp, but it still must
not collide on destinations or artifact keys with the four maintained queues.

Reads JSON, source text and the download service's file plan. No network, no
models directory.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import download_service

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
        # Every queue file has its own process lock. The download service holds
        # the four maintained ones, but two queues started by hand, or the slot
        # queue beside the service, still run at the same time. A duplicate
        # across manifests would therefore bypass the in-process destination
        # guard and give two curl processes the same partial file.
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

    def test_qualification_supervisor_waits_for_all_four_totals(self):
        supervisor = HERE.parent / "qualification-supervisor" / "run_qualification.py"
        found = literals(supervisor)
        for phase, total in STAMP_BYTES.items():
            with self.subTest(phase=phase):
                self.assertIn(str(total), found,
                              f"{supervisor.name} does not gate on the {phase} total")

    def test_download_service_runs_exactly_the_queues_the_supervisor_waits_for(self):
        supervisor = literals(HERE.parent / "qualification-supervisor" / "run_qualification.py")
        plan = download_service.queue_plan(HERE)
        self.assertEqual(list(QUEUES.values()), [row["queue"] for row in plan])
        for row in plan:
            with self.subTest(queue=row["queue"].name):
                self.assertIn(row["state"].name, supervisor)
                self.assertIn(row["stamp"].name, supervisor)


if __name__ == "__main__":
    unittest.main()
