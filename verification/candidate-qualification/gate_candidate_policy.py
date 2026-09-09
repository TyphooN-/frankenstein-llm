#!/usr/bin/env python3
"""Fail-closed, non-inference preflight for phase-four candidates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import candidate_policy as policy
import wemm_remote_code_review as wemm_review

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
ARTIFACT = EVIDENCE / "candidate-policy.json"


def evaluate() -> dict:
    inventory = policy.installed_inventory()
    asr_conflicts = policy.already_satisfied_conflicts()
    index_conflicts = policy.index_conflicts()
    remote_code = wemm_review.scan()
    policy_problems: list[str] = []

    if policy.tool_grant_allowed("gemma4-heretic"):
        policy_problems.append("Gemma-4 Heretic was granted executable tools")
    if policy.abliteration_allowed("ui-mate-9b"):
        policy_problems.append("abliteration was permitted for the GUI actor")
    if policy.control_allowed("ui-mate-9b", None):
        policy_problems.append("UI-Mate control was permitted without a grounding verdict")
    if not policy.tool_grant_allowed("qwen3-coder-next"):
        policy_problems.append("Qwen3-Coder-Next repository-agent tools were not admitted")

    problems = (
        list(inventory["problems"])
        + asr_conflicts
        + index_conflicts
        + list(remote_code["problems"])
        + policy_problems
    )
    return {
        "gate": "phase-four-candidate-policy",
        "candidate_count": len(policy.CANDIDATES),
        "inventory": inventory,
        "already_satisfied_conflicts": asr_conflicts,
        "index_conflicts": index_conflicts,
        "remote_code_review": remote_code,
        "policy_problems": policy_problems,
        "problems": problems,
        "benchmarking_performed": False,
        "benchmark_performed": False,
        "model_inference_performed": False,
        "pass": not problems,
    }


def write_atomic(report: dict, path: Path = ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)


def main() -> int:
    report = evaluate()
    report["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_atomic(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
