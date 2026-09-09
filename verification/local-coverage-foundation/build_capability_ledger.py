#!/usr/bin/env python3
"""Rebuild the machine-readable local-capability ledger from durable files.

The 2026-09-02 ledger under ``evidence/`` was written by a script that is not in
this repository, so it cannot be regenerated: it still describes phases one and
two only, and its recorded paths predate the workspace move. A status file that
nobody can rebuild stops being a source of truth and becomes a memory.

This rebuilds it from things that survive a reboot, an outage and a path change:

* the pinned download queues say which artifacts belong to which capability, at
  which revision, and how many bytes each file must be;
* the gate artifacts under the various ``evidence/`` directories say what was
  actually observed, including interruptions and missing sections;
* nothing else. No model is loaded, no service is queried, no GPU is touched and
  no throughput is measured or read.

Three rules make the result fail closed:

* a capability with no declared evidence artifact is never qualified, and
  "the file is not there" is treated the same as "the gate failed";
* an artifact recording ``interrupted`` is not a verdict, however far it got;
* evidence older than the weights it was supposed to judge is **stale**, not
  passing. That rule is why this exists: the 2026-09-02 ledger reported a
  qualified stack while phase-three and phase-four weights landed the next day.

Byte sizes are checked from filesystem metadata. SHA-256 is not recomputed here;
the download queue verified digests before promoting each file, and re-hashing
165 GB to print a status line would be a heavy I/O job pretending to be a report.

Staleness is judged on modification time, which also moves when a verified file
is re-promoted rather than re-fetched. That is reported as "this evidence did not
judge these bytes", which is what is actually known; it is not a finding about
the model, and re-running the gate or re-verifying the digests resolves it.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path("/home/typhoon/git/frankenstein-llm")
FOUNDATION = ROOT / "verification" / "local-coverage-foundation"
EVIDENCE = FOUNDATION / "evidence"
LEDGER = EVIDENCE / "capability-ledger.json"
SCHEMA = "hermes-local-capability-ledger/2"

QUEUES = (
    FOUNDATION / "download-queue.json",
    FOUNDATION / "download-queue-phase2.json",
    FOUNDATION / "download-queue-phase3.json",
    FOUNDATION / "download-queue-phase4.json",
)

# Which artifact proves which capability. This is the mapping the prose status
# documents state in a table; tracked here so the ledger can be rebuilt instead
# of transcribed. A capability absent from this map can never be qualified, and
# that is the intended default for anything newly downloaded.
CAPABILITY_EVIDENCE: dict[str, tuple[Path, ...]] = {
    "asr": (EVIDENCE / "gate-asr.json",),
    "embeddings": (EVIDENCE / "gate-embeddings.json",),
    "fim": (EVIDENCE / "gate-fim.json",),
    "ocr": (EVIDENCE / "gate-ocr.json",),
    "reranking": (EVIDENCE / "gate-reranker.json",),
    # Composed from admitted services and presets; no artifact of its own.
    "rag": (EVIDENCE / "gate-rag-structural.json",
            EVIDENCE / "gate-rag-behavioural.json",
            EVIDENCE / "gate-rag-live.json"),
    "native-tool-use": (EVIDENCE / "gate-native-tool-use.json",),
    "vision-grounding": (EVIDENCE / "gate-vision-grounding.json",),
    "computer-use-grounding": (
        ROOT / "verification/computer-use-grounding/evidence/computer-use-grounding.json",),
    "multimodal-embeddings": (
        ROOT / "verification/candidate-qualification/evidence/wemm-functional.json",),
    "repository-agent": (
        ROOT / "verification/repository-agent/evidence/gate-repo-agent-heretic.json",
        ROOT / "verification/repository-agent/evidence/gate-repo-agent-qwen3-coder-next.json"),
    "tts": (ROOT / "verification/tts-local/evidence/gate-tts.json",),
    # One live ComfyUI run submits the Z-Image, ACE-Step and Qwen-Edit graphs and
    # records all three in the same artifact.
    "image": (ROOT / "verification/generative-media/evidence/media-functional.json",),
    "music": (ROOT / "verification/generative-media/evidence/media-functional.json",),
    "image-editing": (
        ROOT / "verification/generative-media/evidence/media-functional.json",),
    # Deliberately empty:
    #   image-generation-editing -- the FLUX.2 Klein graph is pinned but the
    #     functional gate does not submit it, so that artifact's pass would not
    #     be about FLUX.2;
    #   uncensored-multimodal -- the 2026-09-03 Gemma-4 Heretic checks were run
    #     against the router by hand and no gate writes a durable artifact for
    #     them yet.
    # Both therefore report "no evidence artifact is declared", which is true.
}

# A pass bit is meaningful only when the artifact identifies the gate that wrote
# it. Without this binding, copying any passing JSON file onto a declared path
# could qualify an unrelated capability.
EXPECTED_EVIDENCE_GATES: dict[Path, str] = {
    EVIDENCE / "gate-asr.json": "asr",
    EVIDENCE / "gate-embeddings.json": "embeddings",
    EVIDENCE / "gate-fim.json": "fim",
    EVIDENCE / "gate-ocr.json": "ocr",
    EVIDENCE / "gate-reranker.json": "reranker",
    EVIDENCE / "gate-rag-structural.json": "rag-structural",
    EVIDENCE / "gate-rag-behavioural.json": "rag-behavioural",
    EVIDENCE / "gate-rag-live.json": "rag-live",
    EVIDENCE / "gate-native-tool-use.json": "native-tool-use",
    EVIDENCE / "gate-vision-grounding.json": "vision-grounding",
    ROOT / "verification/computer-use-grounding/evidence/computer-use-grounding.json":
        "computer-use-grounding",
    ROOT / "verification/candidate-qualification/evidence/wemm-functional.json":
        "wemm-functional",
    ROOT / "verification/repository-agent/evidence/gate-repo-agent-heretic.json":
        "local-repository-agent",
    ROOT / "verification/repository-agent/evidence/gate-repo-agent-qwen3-coder-next.json":
        "local-repository-agent",
    ROOT / "verification/tts-local/evidence/gate-tts.json": "tts-local",
    ROOT / "verification/generative-media/evidence/media-functional.json":
        "generative-media-functional",
}

# Capabilities that own no downloaded artifact of their own. Listing them keeps
# them visible in the ledger rather than silently absent.
COMPOSED_CAPABILITIES = ("rag", "native-tool-use", "vision-grounding")

# The state vocabulary docs/CANDIDATE-STATUS-2026-09-03.md defines. A later state
# never implies an earlier one was re-checked, so they are reported, not ranked.
STATE_RESEARCHED = "researched"
STATE_INCOMPLETE = "download-incomplete"
STATE_DOWNLOADED = "downloaded"
STATE_STALE = "evidence-stale"
STATE_INTERRUPTED = "evidence-interrupted"
STATE_FAILED = "functionally-failed"
STATE_QUALIFIED = "functionally-qualified"


def read_queues(paths=QUEUES) -> dict[str, list[dict]]:
    """Group every pinned artifact by the capability its queue assigns it."""
    capabilities: dict[str, list[dict]] = {}
    for path in paths:
        document = json.loads(Path(path).read_text())
        for artifact in document.get("artifacts", []):
            capabilities.setdefault(artifact["capability"], []).append(
                {**artifact, "queue": Path(path).name})
    return capabilities


def inspect_artifact(artifact: dict) -> dict:
    """Presence and exact size of one artifact's files, plus its newest mtime."""
    files, missing, wrong_size = [], [], []
    newest = 0.0
    for entry in artifact.get("files", []):
        path = Path(entry["destination"])
        present = path.is_file()
        size = path.stat().st_size if present else 0
        mtime = path.stat().st_mtime if present else 0.0
        newest = max(newest, mtime)
        if not present:
            missing.append(entry["repo_path"])
        elif size != entry["size"]:
            wrong_size.append(entry["repo_path"])
        files.append({"repo_path": entry["repo_path"], "present": present,
                      "bytes": size, "expected_bytes": entry["size"]})
    return {
        "key": artifact.get("key"),
        "capability": artifact["capability"],
        "repository": artifact.get("repository"),
        "revision": artifact.get("revision"),
        "queue": artifact["queue"],
        "files": files,
        "missing": missing,
        "wrong_size": wrong_size,
        "newest_mtime": newest,
        "complete": not missing and not wrong_size,
        "sha256_reverified": False,
    }


def parse_recorded_at(value: str | None) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def read_evidence(path: Path, expected_gate: str | None = None) -> dict:
    """Read one gate artifact. Unreadable and absent are both 'no verdict'."""
    record = {"path": str(path), "present": path.is_file(), "pass": False,
              "interrupted": False, "recorded_at": None, "error": None,
              "sections_missing": None, "benchmark_performed": False}
    if not record["present"]:
        record["error"] = "evidence artifact absent"
        return record
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    record["pass"] = document.get("pass") is True
    record["interrupted"] = bool(document.get("interrupted"))
    record["recorded_at"] = document.get("recorded_at") or document.get("finished_at")
    record["sections_missing"] = document.get("sections_missing")
    record["gate"] = document.get("gate")
    if expected_gate is not None and record["gate"] != expected_gate:
        record["pass"] = False
        record["error"] = (
            f"gate identity mismatch: expected {expected_gate!r}, got {record['gate']!r}")
    elif record["sections_missing"]:
        record["pass"] = False
        record["error"] = "evidence reports missing required sections"
    if document.get("error"):
        record["error"] = str(document["error"])[:400]
    # Dedicated benchmark runs remain separate from functional acceptance.
    # Historical artifacts used throughput_measured for that policy marker;
    # passive per-attempt observations are now kept in a separate sidecar.
    record["benchmark_performed"] = bool(document.get("benchmark_performed", document.get("throughput_measured"))
                                         or document.get("benchmarking_performed"))
    return record


def classify(artifacts: list[dict], evidence: list[dict]) -> tuple[str, list[str]]:
    """Resolve one capability's state, fail-closed, with the reasons for it."""
    reasons: list[str] = []
    if artifacts and not all(item["complete"] for item in artifacts):
        for item in artifacts:
            if item["missing"]:
                reasons.append(f"{item['key']}: {len(item['missing'])} files missing")
            if item["wrong_size"]:
                reasons.append(f"{item['key']}: {len(item['wrong_size'])} files wrong size")
        return STATE_INCOMPLETE, reasons

    if not evidence:
        reasons.append("no evidence artifact is declared for this capability")
        return (STATE_DOWNLOADED if artifacts else STATE_RESEARCHED), reasons

    for item in evidence:
        if item["error"] and not item["present"]:
            reasons.append(f"{Path(item['path']).name}: {item['error']}")
    if any(item["interrupted"] for item in evidence):
        reasons.append("a gate run was interrupted; an unfinished run is not a verdict")
        return STATE_INTERRUPTED, reasons
    if not all(item["pass"] for item in evidence):
        return (STATE_DOWNLOADED if not any(item["present"] for item in evidence)
                else STATE_FAILED), reasons or ["a declared gate artifact does not record a pass"]

    newest_artifact = max((item["newest_mtime"] for item in artifacts), default=0.0)
    for item in evidence:
        recorded = parse_recorded_at(item["recorded_at"])
        if recorded is None:
            reasons.append(f"{Path(item['path']).name}: no readable timestamp")
            return STATE_STALE, reasons
        if newest_artifact and recorded < newest_artifact:
            reasons.append(
                f"{Path(item['path']).name} was recorded before the weights it judges "
                "were installed")
            return STATE_STALE, reasons
    return STATE_QUALIFIED, reasons


def build_ledger(queues=QUEUES, evidence_map=None) -> dict:
    evidence_map = CAPABILITY_EVIDENCE if evidence_map is None else evidence_map
    by_capability = read_queues(queues)
    names = sorted(set(by_capability) | set(COMPOSED_CAPABILITIES))

    capabilities = {}
    problems = []
    for name in names:
        artifacts = [inspect_artifact(artifact) for artifact in by_capability.get(name, [])]
        evidence = [
            read_evidence(path, EXPECTED_EVIDENCE_GATES.get(path))
            for path in evidence_map.get(name, ())
        ]
        state, reasons = classify(artifacts, evidence)
        if any(item["benchmark_performed"] for item in evidence):
            problems.append(f"{name}: an evidence artifact reports measured throughput")
        capabilities[name] = {
            "state": state,
            "reasons": reasons,
            "artifacts": artifacts,
            "evidence": evidence,
            "qualified": state == STATE_QUALIFIED,
        }

    for name in sorted(set(evidence_map) - set(names)):
        problems.append(f"{name}: evidence is declared for a capability no queue owns")

    return {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_of_truth": [str(path) for path in queues],
        "model_inference_performed": False,
        "benchmark_performed": False,
        "sha256_reverified": False,
        "capabilities": capabilities,
        "functionally_qualified": sorted(
            name for name, row in capabilities.items() if row["qualified"]),
        "not_qualified": sorted(
            name for name, row in capabilities.items() if not row["qualified"]),
        "problems": problems,
        # The ledger being internally consistent says nothing about the stack
        # being capable. Those are different questions and only one is answered.
        "pass": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(LEDGER),
                        help="where to publish the ledger (default: ignored evidence dir)")
    parser.add_argument("--print-only", action="store_true",
                        help="render the ledger without writing it")
    args = parser.parse_args(argv)

    ledger = build_ledger()
    print(json.dumps(ledger, indent=2, sort_keys=True))
    if not args.print_only:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(f".tmp.{os.getpid()}")
        temp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        temp.replace(output)
        print(f"\n[ledger] wrote {output}", file=sys.stderr)
    return 0 if ledger["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
