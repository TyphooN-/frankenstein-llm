#!/usr/bin/env python3
"""Functional admission gate for the Qwen3-Reranker-8B sidecar.

The specific failure this guards against: a GGUF converted without the reranker
classifier head loads cleanly, answers /v1/rerank, and returns near-identical
scores for every document. So the gate never checks "did it respond" — it checks
that the relevant document wins, that the score spread is real, and that the
ordering inverts when the query changes.
"""
from __future__ import annotations

import statistics
import sys

sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/validators")
from gatelib import GateFailure, check, managed_sidecar, post_json, record, unload_gate, vram_used  # noqa: E402

BASE = "http://127.0.0.1:8082"
UNIT = "llama-sidecar@reranker.service"

DOCUMENTS = [
    "The llama.cpp router exposes five aliases and keeps one model resident at a time.",
    "ZFS reported no known data errors after the recovery scrub completed.",
    "Qwen3-Reranker assigns a relevance score to each query and document pair.",
    "Sourdough needs a long cold retard before the loaf is shaped and baked.",
]
QUERY_A = "How does the local model router manage GPU memory between models?"
QUERY_A_WINNER = 0
QUERY_B = "What did the storage integrity check conclude?"
QUERY_B_WINNER = 1


def rerank(query: str) -> list[dict]:
    result = post_json(f"{BASE}/v1/rerank", {
        "model": "qwen3-reranker-8b",
        "query": query,
        "documents": DOCUMENTS,
        "top_n": len(DOCUMENTS),
    })
    return sorted(result["results"], key=lambda r: r["relevance_score"], reverse=True)


def main() -> int:
    summary: dict = {"gate": "reranker", "base_url": BASE}
    try:
        baseline, waited = managed_sidecar(UNIT, BASE)
        summary["vram_baseline"] = baseline
        summary["load_seconds"] = round(waited, 1)
        summary["vram_loaded"] = vram_used()

        for label, query, winner in (("query_a", QUERY_A, QUERY_A_WINNER), ("query_b", QUERY_B, QUERY_B_WINNER)):
            ranked = rerank(query)
            scores = [r["relevance_score"] for r in ranked]
            summary[label] = {
                "query": query,
                "order": [r["index"] for r in ranked],
                "scores": [round(s, 6) for s in scores],
            }
            check(
                ranked[0]["index"] == winner,
                f"{label}: expected document {winner} first, got {ranked[0]['index']}",
            )
            spread = max(scores) - min(scores)
            summary[label]["spread"] = round(spread, 6)
            # A head-less conversion collapses to a constant; require real separation.
            check(spread > 0.05, f"{label}: score spread {spread} is too flat to be a real ranking")
            check(
                statistics.pstdev(scores) > 1e-3,
                f"{label}: scores are effectively constant, classifier head is likely missing",
            )

        check(
            summary["query_a"]["order"] != summary["query_b"]["order"],
            "ranking did not change between two different queries; scores are query-independent",
        )

        summary["unload"] = unload_gate(UNIT, baseline)
        summary["pass"] = True
    except (GateFailure, Exception) as error:  # noqa: BLE001
        summary["pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    return record("gate-reranker", summary)


if __name__ == "__main__":
    raise SystemExit(main())
