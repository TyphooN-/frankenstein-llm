#!/usr/bin/env python3
"""Execution boundary for WeMM-Embedding-2B's pinned remote code.

``tencent/WeMM-Embedding-2B`` is loaded through ``trust_remote_code``: its
``config.json`` carries an ``auto_map`` that makes ``AutoModel.from_pretrained``
import and execute ``modeling_wemm_embedding.py`` from the model directory, and
``modules.json`` does the same for ``modeling_st_wemm.py`` under Sentence
Transformers. That is arbitrary local code execution triggered by loading a
model, so it gets the same treatment as any other third-party code entering this
host: read it first, pin the exact bytes that were read, and refuse to run
anything else.

This module never imports the reviewed files. It reads them as bytes, digests
them, and compares against the digests recorded by
``WEMM-REMOTE-CODE-REVIEW.md``. A file that changed, appeared, or vanished
revokes approval rather than warning about it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

MODEL_DIR = Path("/home/typhoon/git/frankenstein-llm/models/embedding/WeMM-Embedding-2B")
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
REVIEW_DOCUMENT = HERE / "WEMM-REMOTE-CODE-REVIEW.md"
REVIEWED_REVISION = "bbd6cd4bf52cfc6716f752a2df80b2706720bd95"
REVIEWED_AT = "2026-09-03"

# Every executable file in the pinned snapshot, with the verdict the review
# reached for it. "approved" means the file may be imported by the embedding
# gate; "never-execute" means it must not run on this host at all, and its
# presence is not a reason to refuse the two files that were approved.
APPROVED = "approved"
NEVER_EXECUTE = "never-execute"

REVIEWED_FILES: dict[str, dict] = {
    "modeling_wemm_embedding.py": {
        "sha256": "ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f",
        "verdict": APPROVED,
        "imported_by": "config.json auto_map -> AutoModel / AutoModelForCausalLM",
        "finding": "Subclasses Qwen3_5ForConditionalGeneration and adds embedding(); "
                   "pure tensor arithmetic, no filesystem, network, subprocess, "
                   "eval/exec, or import side effects.",
    },
    "modeling_st_wemm.py": {
        "sha256": "521d02c1c60ae727cc9dc6500cdb0b28c53b259e0ce3d37197920a33ba4dd333",
        "verdict": APPROVED,
        "imported_by": "modules.json -> modeling_st_wemm.WeMMTransformer",
        "finding": "Sentence Transformers wrapper. No filesystem, network, "
                   "subprocess or eval/exec; introduces one undeclared runtime "
                   "dependency, qwen_vl_utils, imported inside the call.",
    },
    "patch_sglang_video.py": {
        "sha256": "c20c73e803a634e5dc54c39bc1c20cec0e7929c0320121769268a3dfb2e58a2d",
        "verdict": NEVER_EXECUTE,
        "imported_by": "nothing; standalone __main__ utility",
        "finding": "Rewrites an installed site-packages file "
                   "(sglang/srt/multimodal/processors/qwen_vl.py) in place. Host "
                   "mutation disguised as model configuration. SGLang is not part "
                   "of this stack and this file has no reason to run here.",
    },
}

# Files whose presence would mean the snapshot gained executable code that no
# review covers. Anything matching this and not named above revokes approval.
EXECUTABLE_SUFFIXES = (".py", ".pyc", ".so", ".sh")


def digest(path: Path) -> str:
    """SHA-256 of one small source file. Model weights are never hashed here."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan(model_dir: Path = MODEL_DIR) -> dict:
    """Compare the pinned snapshot's executable files against the review."""
    files: dict[str, dict] = {}
    problems: list[str] = []

    present = sorted(
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix in EXECUTABLE_SUFFIXES
    )
    for name, expected in REVIEWED_FILES.items():
        path = model_dir / name
        if not path.is_file():
            problems.append(f"{name}: reviewed file is missing from the snapshot")
            files[name] = {"present": False, "verdict": expected["verdict"], "match": False}
            continue
        observed = digest(path)
        match = observed == expected["sha256"]
        files[name] = {
            "present": True,
            "sha256": observed,
            "reviewed_sha256": expected["sha256"],
            "match": match,
            "verdict": expected["verdict"],
            "imported_by": expected["imported_by"],
            "finding": expected["finding"],
        }
        if not match:
            problems.append(f"{name}: bytes changed since review; approval is revoked")

    unreviewed = [name for name in present if name not in REVIEWED_FILES]
    for name in unreviewed:
        problems.append(f"{name}: executable file present that no review covers")

    if not REVIEW_DOCUMENT.is_file():
        problems.append(f"{REVIEW_DOCUMENT.name}: review record is missing")

    approved = sorted(name for name, item in files.items()
                      if item.get("match") and item["verdict"] == APPROVED)
    refused = sorted(name for name, item in REVIEWED_FILES.items()
                     if item["verdict"] == NEVER_EXECUTE)
    return {
        "gate": "wemm-remote-code-review",
        "model_dir": str(model_dir),
        "reviewed_revision": REVIEWED_REVISION,
        "reviewed_at": REVIEWED_AT,
        "review_document": str(REVIEW_DOCUMENT),
        "executable_files_present": present,
        "unreviewed_executable_files": unreviewed,
        "files": files,
        "approved_for_import": approved,
        "never_execute": refused,
        "problems": problems,
        "benchmarking_performed": False,
        "throughput_measured": False,
        "pass": not problems,
    }


def execution_permitted(report: dict | None = None) -> bool:
    """Whether the embedding gate may import WeMM's model code.

    Fail closed on every uncertainty: a missing report, a failing scan, or a
    scan that approved nothing all answer False.
    """
    report = scan() if report is None else report
    if not report.get("pass"):
        return False
    return bool(report.get("approved_for_import"))


def write_atomic(report: dict) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    artifact = EVIDENCE / "wemm-remote-code-review.json"
    temp = artifact.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp, artifact)
    return artifact


def main() -> int:
    report = scan()
    report["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["execution_permitted"] = execution_permitted(report)
    write_atomic(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
