#!/usr/bin/env python3
"""Functional admission gate for the compact FIM completion sidecar.

Uses llama.cpp's /infill endpoint, which consumes the GGUF's prefix/suffix/middle
special tokens directly. The gate asserts the model produces the one middle that
makes each snippet correct, and that it stops instead of running on past the
suffix — the failure mode that makes an editor integration unusable.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/validators")
from gatelib import GateFailure, check, managed_sidecar, post_json, record, unload_gate, vram_used  # noqa: E402

BASE = "http://127.0.0.1:8084"
UNIT = "llama-sidecar@fim.service"

CASES = [
    {
        "name": "python-return",
        "prefix": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    ",
        "suffix": "\n\n\ndef mul(a, b):\n    return a * b\n",
        "must_contain": ["a + b"],
    },
    {
        "name": "python-loop-body",
        "prefix": "total = 0\nfor value in values:\n    total ",
        "suffix": "\nprint(total)\n",
        "must_contain": ["+= value", "= total + value"],
    },
    {
        "name": "shell-guard",
        "prefix": "#!/bin/bash\nset -euo pipefail\nif [ ! -f \"$1\" ]; then\n    echo \"missing file\" >&2\n    ",
        "suffix": "\nfi\ncat \"$1\"\n",
        "must_contain": ["exit 1", "exit 2"],
    },
]


def infill(case: dict) -> str:
    result = post_json(f"{BASE}/infill", {
        "input_prefix": case["prefix"],
        "input_suffix": case["suffix"],
        "n_predict": 48,
        "temperature": 0,
        "top_k": 1,
    })
    return result.get("content", "")


def main() -> int:
    summary: dict = {"gate": "fim", "base_url": BASE, "cases": []}
    try:
        baseline, waited = managed_sidecar(UNIT, BASE)
        summary["vram_baseline"] = baseline
        summary["load_seconds"] = round(waited, 1)
        summary["vram_loaded"] = vram_used()

        for case in CASES:
            completion = infill(case)
            matched = [needle for needle in case["must_contain"] if needle in completion]
            # Repeating the suffix means the FIM stop tokens were not honoured.
            leaked = case["suffix"].strip().splitlines()[-1].strip() if case["suffix"].strip() else ""
            overrun = bool(leaked) and completion.count(leaked) > 0
            record_case = {
                "name": case["name"],
                "completion": completion,
                "matched": matched,
                "suffix_overrun": overrun,
            }
            summary["cases"].append(record_case)
            check(bool(matched), f"{case['name']}: no acceptable middle in {completion!r}")
            check(not overrun, f"{case['name']}: completion re-emitted the suffix, FIM stop tokens not honoured")

        summary["unload"] = unload_gate(UNIT, baseline)
        summary["pass"] = True
    except (GateFailure, Exception) as error:  # noqa: BLE001
        summary["pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    return record("gate-fim", summary)


if __name__ == "__main__":
    raise SystemExit(main())
