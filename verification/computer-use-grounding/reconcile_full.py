#!/usr/bin/env python3
"""Full reconciliation of every download queue, stamp, manifest, hash and provenance.

Extends reconcile_downloads.py (stat-only) with:
  * SHA-256 re-verification of every queued file that declares a digest
  * enumeration of quarantined ``.bad-*`` / stale ``.partial`` files with byte accounting
  * provenance capture (repository, revision, licence, gated flag) per artifact
  * a cross-check that on-disk model trees contain no unaccounted large files

Read-only with respect to model trees: nothing is downloaded, moved or deleted.
Quarantined files are reported as reclaimable, never removed -- that is an
operator decision, not this gate's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from culib import record  # noqa: E402

FOUNDATION = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation")
MODELS = Path("/home/typhoon/git/frankenstein-llm/models")
PHASES = [
    {"phase": "phase1", "queue": FOUNDATION / "download-queue.json",
     "state": FOUNDATION / "download-state.json",
     "stamp": FOUNDATION / "downloads-complete.ok"},
    {"phase": "phase2", "queue": FOUNDATION / "download-queue-phase2.json",
     "state": FOUNDATION / "download-state-phase2.json",
     "stamp": FOUNDATION / "downloads-phase2-complete.ok"},
]
CHUNK = 8 << 20


def sha256_file(path: Path) -> tuple[str, float, int]:
    """Stream a file through SHA-256. Returns (hexdigest, seconds, bytes)."""
    h = hashlib.sha256()
    n = 0
    start = time.monotonic()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
            n += len(block)
    return h.hexdigest(), time.monotonic() - start, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true",
                    help="re-verify SHA-256 for every file that declares one")
    ap.add_argument("--out", default=None, help="evidence basename (default: download-reconciliation-full)")
    args = ap.parse_args()

    summary = {
        "gate": "download-reconciliation-full",
        "hash_verification_requested": args.hash,
        "phases": [],
        "problems": [],
        "quarantined": [],
        "unaccounted_large_files": [],
    }
    problems = summary["problems"]
    accounted: set[str] = set()

    for spec in PHASES:
        queue = json.loads(spec["queue"].read_text())
        state = json.loads(spec["state"].read_text())
        stamp_raw = spec["stamp"].read_text().strip() if spec["stamp"].exists() else None
        stamp_val = int(stamp_raw) if stamp_raw and stamp_raw.isdigit() else stamp_raw

        phase = {
            "phase": spec["phase"],
            "queue": str(spec["queue"]),
            "queue_built_at": queue.get("built_at"),
            "queue_total_bytes": queue.get("total_bytes"),
            "source_of_truth": queue.get("source_of_truth"),
            "policy": queue.get("policy"),
            "state_status": state.get("status"),
            "stamp_present": stamp_raw is not None,
            "stamp_value": stamp_val,
            "artifacts": [],
        }
        queue_bytes = 0
        hashed_bytes = 0
        hashed_files = 0

        for artifact in queue["artifacts"]:
            key = artifact["key"]
            declared = state.get("artifacts", {}).get(key, {})
            rec = {
                "key": key,
                "capability": artifact.get("capability"),
                "provenance": {
                    "repository": artifact.get("repository"),
                    "revision": artifact.get("revision"),
                    "license": artifact.get("license"),
                    "gated": artifact.get("gated"),
                    "metadata_collected_at": artifact.get("metadata_collected_at"),
                },
                "state_status": declared.get("status"),
                "state_files_complete": declared.get("files_complete"),
                "state_files_total": declared.get("files_total"),
                "files_total": len(artifact["files"]),
                "files_present": 0,
                "bytes_present": 0,
                "missing": [],
                "wrong_size": [],
                "sha256_declared": 0,
                "sha256_verified": 0,
                "sha256_mismatch": [],
            }
            for entry in artifact["files"]:
                path = Path(entry["destination"])
                queue_bytes += entry["size"]
                accounted.add(str(path))
                if not path.exists():
                    rec["missing"].append(str(path))
                    continue
                size = path.stat().st_size
                rec["files_present"] += 1
                rec["bytes_present"] += size
                if size != entry["size"]:
                    rec["wrong_size"].append(
                        {"path": str(path), "expected": entry["size"], "actual": size})
                digest = entry.get("sha256")
                if not digest:
                    continue
                rec["sha256_declared"] += 1
                if not args.hash:
                    continue
                actual, secs, nbytes = sha256_file(path)
                hashed_bytes += nbytes
                hashed_files += 1
                if actual == digest:
                    rec["sha256_verified"] += 1
                else:
                    rec["sha256_mismatch"].append(
                        {"path": str(path), "expected": digest, "actual": actual})
                print(f"  [{'ok ' if actual == digest else 'BAD'}] "
                      f"{path.name} {nbytes/2**30:.2f} GiB {secs:.1f}s", flush=True)

            if rec["missing"]:
                problems.append(f"{key}: {len(rec['missing'])} missing file(s)")
            if rec["wrong_size"]:
                problems.append(f"{key}: {len(rec['wrong_size'])} wrong-size file(s)")
            if rec["sha256_mismatch"]:
                problems.append(f"{key}: {len(rec['sha256_mismatch'])} SHA-256 mismatch(es)")
            if declared.get("status") != "complete":
                problems.append(f"{key}: state status is {declared.get('status')!r}")
            phase["artifacts"].append(rec)

        phase["queue_file_bytes_sum"] = queue_bytes
        phase["hashed_files"] = hashed_files
        phase["hashed_bytes"] = hashed_bytes
        if stamp_val != queue_bytes:
            problems.append(
                f"{spec['phase']}: stamp {stamp_val} != queued bytes {queue_bytes}")
        summary["phases"].append(phase)

    # Quarantined and stale staging files: reported for operator review, never deleted.
    reclaimable = 0
    for path in sorted(MODELS.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if ".bad-" in name or name.endswith(".partial"):
            size = path.stat().st_size
            reclaimable += size
            summary["quarantined"].append({
                "path": str(path),
                "bytes": size,
                "kind": "quarantined-bad" if ".bad-" in name else "stale-partial",
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                       time.localtime(path.stat().st_mtime)),
            })
        elif str(path) not in accounted and path.stat().st_size > (256 << 20):
            summary["unaccounted_large_files"].append(
                {"path": str(path), "bytes": path.stat().st_size})

    summary["reclaimable_bytes"] = reclaimable
    summary["total_bytes_declared"] = sum(p["queue_total_bytes"] for p in summary["phases"])
    summary["capabilities"] = sorted(
        {a["capability"] for p in summary["phases"] for a in p["artifacts"]})
    summary["pass"] = not problems
    name = args.out or "download-reconciliation-full"
    return record(name, summary)


if __name__ == "__main__":
    raise SystemExit(main())
