#!/usr/bin/env python3
"""Read-only generative-media preflight. Does not import torch or load a model."""
from __future__ import annotations

import json
from pathlib import Path
import time

import media_policy as policy


def inspect_node_contract(name: str, path: Path, node_ids: tuple[str, ...]) -> dict:
    text = path.read_text() if path.is_file() else ""
    missing = [node_id for node_id in node_ids if node_id not in text]
    return {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "required_node_ids": list(node_ids),
        "missing_node_ids": missing,
        "pass": path.is_file() and not missing,
    }


def probe_environment() -> dict:
    """Every fact build_report reads from the host, gathered in one place.

    Keeping the filesystem and process reads here is what lets the report be
    built from a synthetic environment in tests. The gate still calls this for
    real; injection is for describing a host, not for inventing a verdict.
    """
    return {
        "inventory": policy.artifact_inventory(),
        "nodes": [
            inspect_node_contract(name, path, policy.REQUIRED_NODE_IDS[name])
            for name, path in policy.REQUIRED_NODE_FILES.items()
        ],
        "blockers": policy.host_exclusive_blockers(),
        "comfy_python_present": policy.COMFY_PYTHON.is_file(),
        "comfy_main_present": (policy.COMFY_ROOT / "main.py").is_file(),
        "extra_paths_present": policy.EXTRA_PATHS.is_file(),
    }


def build_report(probe: dict | None = None) -> dict:
    probe = probe_environment() if probe is None else probe
    inventory = probe["inventory"]
    nodes = probe["nodes"]
    blockers = probe["blockers"]
    argv = policy.comfy_argv()
    env = policy.comfy_env()
    problems = list(inventory["problems"])
    if not probe["comfy_python_present"]:
        problems.append("ComfyUI venv interpreter missing")
    if not probe["comfy_main_present"]:
        problems.append("ComfyUI main.py missing")
    if not probe["extra_paths_present"]:
        problems.append("extra model paths config missing")
    for entry in nodes:
        if not entry["pass"]:
            problems.append(f"{entry['name']} node contract missing")
    if "2" in env["HIP_VISIBLE_DEVICES"].split(","):
        problems.append("display GPU2 is visible to ComfyUI")
    if argv[argv.index("--listen") + 1] != "127.0.0.1":
        problems.append("ComfyUI is not loopback-only")
    return {
        "gate": "generative-media-preflight",
        "mode": "read-only; no torch import; no GPU allocation",
        "throughput_measured": False,
        "artifacts": inventory,
        "node_contracts": nodes,
        "gpu_roles": policy.GPU_ROLES,
        "visible_compute_devices": policy.VISIBLE_COMPUTE,
        "display_gpu_excluded": "2" not in env["HIP_VISIBLE_DEVICES"].split(","),
        "primary_physical_gpu": policy.PRIMARY_DEVICE,
        "comfy_argv": argv,
        "kernel_build_blockers": blockers,
        "workflow_claims": policy.WORKFLOW_CLAIMS,
        "workflows_functionally_proven": sorted(
            name for name, claim in policy.WORKFLOW_CLAIMS.items()
            if claim["functionally_proven"]),
        "static_readiness_is_not_functional_qualification": True,
        "functional_gate_ready_now": not blockers and not problems,
        "problems": problems,
        "pass": not problems,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main() -> int:
    report = build_report()
    policy.EVIDENCE.mkdir(parents=True, exist_ok=True)
    output = policy.EVIDENCE / "media-preflight.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
