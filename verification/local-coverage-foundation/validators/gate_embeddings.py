#!/usr/bin/env python3
"""Functional admission gate for the Qwen3-Embedding-8B sidecar.

Proves the embedding space is actually useful, not merely that /v1/embeddings
returned an array of floats:
  1. dimensionality matches the published hidden size;
  2. vectors are L2-normalised as configured;
  3. a paraphrase outranks a topically-related distractor, which outranks noise;
  4. the same input embeds identically twice (retrieval indexes must be stable);
  5. batching does not change a vector (ingest batches must match query calls).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/validators")
from gatelib import GateFailure, check, ensure_unloaded, cosine, managed_sidecar, post_json, record, unload_gate, vram_used  # noqa: E402

BASE = "http://127.0.0.1:8081"
UNIT = "llama-sidecar@embeddings.service"
EXPECTED_DIM = 4096

ANCHOR = "The ZFS scrub repaired zero bytes and reported no known data errors."
PARAPHRASE = "A scrub of the ZFS pool finished with nothing repaired and no data errors found."
RELATED = "The router keeps at most one large language model resident in VRAM."
NOISE = "Sourdough needs a long cold retard before the loaf is shaped and baked."


def embed(texts: list[str]) -> list[list[float]]:
    result = post_json(f"{BASE}/v1/embeddings", {"input": texts, "model": "qwen3-embedding-8b"})
    ordered = sorted(result["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in ordered]


def main() -> int:
    summary: dict = {"gate": "embeddings", "base_url": BASE}
    baseline: dict[str, int] | None = None
    try:
        baseline, waited = managed_sidecar(UNIT, BASE)
        summary["vram_baseline"] = baseline
        summary["load_seconds"] = round(waited, 1)
        summary["vram_loaded"] = vram_used()

        anchor, paraphrase, related, noise = embed([ANCHOR, PARAPHRASE, RELATED, NOISE])
        summary["dimension"] = len(anchor)
        check(len(anchor) == EXPECTED_DIM, f"dimension {len(anchor)} != expected {EXPECTED_DIM}")

        norm = sum(x * x for x in anchor) ** 0.5
        summary["l2_norm"] = round(norm, 6)
        check(abs(norm - 1.0) < 1e-3, f"embeddings are not L2-normalised (norm {norm})")

        scores = {
            "paraphrase": cosine(anchor, paraphrase),
            "related": cosine(anchor, related),
            "noise": cosine(anchor, noise),
        }
        summary["similarity"] = {k: round(v, 6) for k, v in scores.items()}
        check(
            scores["paraphrase"] > scores["related"] > scores["noise"],
            f"semantic ordering violated: {scores}",
        )
        check(
            scores["paraphrase"] - scores["noise"] > 0.15,
            f"paraphrase/noise separation too small to support retrieval: {scores}",
        )

        again = embed([ANCHOR])[0]
        drift = max(abs(x - y) for x, y in zip(anchor, again))
        summary["repeat_max_abs_drift"] = drift
        check(drift < 1e-4, f"repeated embedding of identical text drifted by {drift}")

        alone = embed([RELATED])[0]
        batch_drift = max(abs(x - y) for x, y in zip(related, alone))
        summary["batch_vs_single_max_abs_drift"] = batch_drift
        check(batch_drift < 1e-4, f"batched and single embeddings disagree by {batch_drift}")

        summary["unload"] = unload_gate(UNIT, baseline)
        summary["pass"] = True
    except (GateFailure, Exception) as error:  # noqa: BLE001 - gates report, never raise
        summary["pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        ensure_unloaded(summary, UNIT, baseline)
    return record("gate-embeddings", summary)


if __name__ == "__main__":
    raise SystemExit(main())
