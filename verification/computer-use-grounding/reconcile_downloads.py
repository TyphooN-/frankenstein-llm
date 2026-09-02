#!/usr/bin/env python3
"""Read-only reconciliation of the download queues, state files and stamps.

Answers one question with evidence: does every artifact the queues claim to have
finished actually exist on disk at the declared size? Uses ``stat`` only -- no
file contents are read -- so it is safe to run while the host is busy.

This reports on capabilities outside the computer-use slice but never rewrites
another slice's state file. A discrepancy is surfaced, not silently repaired.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from culib import record  # noqa: E402

FOUNDATION = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation")
PHASES = [
    {"phase": "phase1", "queue": FOUNDATION / "download-queue.json",
     "state": FOUNDATION / "download-state.json",
     "stamp": FOUNDATION / "downloads-complete.ok"},
    {"phase": "phase2", "queue": FOUNDATION / "download-queue-phase2.json",
     "state": FOUNDATION / "download-state-phase2.json",
     "stamp": FOUNDATION / "downloads-phase2-complete.ok"},
]


def main() -> int:
    summary = {"gate": "download-reconciliation", "phases": [], "problems": []}
    problems = summary["problems"]

    for spec in PHASES:
        queue = json.loads(spec["queue"].read_text())
        state = json.loads(spec["state"].read_text())
        stamp_raw = spec["stamp"].read_text().strip() if spec["stamp"].exists() else None

        phase = {
            "phase": spec["phase"],
            "queue": str(spec["queue"]),
            "queue_built_at": queue.get("built_at"),
            "queue_total_bytes": queue.get("total_bytes"),
            "state_status": state.get("status"),
            "state_completed_at": state.get("completed_at"),
            "stamp_present": stamp_raw is not None,
            "stamp_value": int(stamp_raw) if stamp_raw and stamp_raw.isdigit() else stamp_raw,
            "artifacts": [],
        }

        queue_bytes_sum = 0
        for artifact in queue["artifacts"]:
            key = artifact["key"]
            declared = state.get("artifacts", {}).get(key, {})
            files_present = 0
            bytes_present = 0
            missing, wrong_size = [], []
            for entry in artifact["files"]:
                path = Path(entry["destination"])
                queue_bytes_sum += entry["size"]
                if not path.exists():
                    missing.append(entry["repo_path"])
                    continue
                actual = path.stat().st_size
                bytes_present += actual
                if actual != entry["size"]:
                    wrong_size.append(
                        {"file": entry["repo_path"], "expected": entry["size"], "actual": actual})
                else:
                    files_present += 1

            entry_summary = {
                "key": key,
                "capability": artifact["capability"],
                "repository": artifact["repository"],
                "revision": artifact["revision"],
                "state_status": declared.get("status"),
                "state_files_complete": declared.get("files_complete"),
                "state_files_total": declared.get("files_total"),
                "files_declared": len(artifact["files"]),
                "files_verified_on_disk": files_present,
                "bytes_on_disk": bytes_present,
                "declared_bytes": artifact.get("total_bytes"),
                "missing_files": missing,
                "wrong_size_files": wrong_size,
                "reconciled": not missing and not wrong_size,
            }
            if missing:
                problems.append(f"{spec['phase']}/{key}: {len(missing)} declared files missing")
            if wrong_size:
                problems.append(f"{spec['phase']}/{key}: {len(wrong_size)} files wrong size")
            if declared.get("status") != "complete":
                problems.append(
                    f"{spec['phase']}/{key}: state status is {declared.get('status')!r}")
            if declared.get("files_complete") != len(artifact["files"]):
                problems.append(
                    f"{spec['phase']}/{key}: state claims {declared.get('files_complete')} of "
                    f"{len(artifact['files'])} files")
            phase["artifacts"].append(entry_summary)

        phase["queue_file_bytes_sum"] = queue_bytes_sum
        if phase["stamp_value"] != queue.get("total_bytes"):
            problems.append(
                f"{spec['phase']}: stamp {phase['stamp_value']} != queue total "
                f"{queue.get('total_bytes')}")
        if queue_bytes_sum != queue.get("total_bytes"):
            problems.append(
                f"{spec['phase']}: file sizes sum to {queue_bytes_sum}, queue declares "
                f"{queue.get('total_bytes')}")
        if state.get("status") != "complete":
            problems.append(f"{spec['phase']}: state status {state.get('status')!r}")
        summary["phases"].append(phase)

    summary["capabilities_downloaded"] = sorted(
        a["capability"] for p in summary["phases"] for a in p["artifacts"])
    summary["total_bytes_declared"] = sum(p["queue_total_bytes"] for p in summary["phases"])
    summary["pass"] = not problems
    return record("download-reconciliation", summary)


if __name__ == "__main__":
    raise SystemExit(main())
