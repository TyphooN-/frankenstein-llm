#!/usr/bin/env python3
"""Behavioural gates for the cited-RAG answer path, with no GPU inference.

The live gate proves a real model behaves. This one proves the *pipeline* behaves
regardless of which model is plugged in, by scripting the answerer:

  * the exact prompt reaching the model carries the untrusted-data envelope, the
    provenance header and the citation rules;
  * citations that resolve are returned with source path and byte range;
  * citations the model invents are detected and reported, not rendered as fact;
  * an unreachable reranker degrades to dense order with a recorded status
    instead of throwing or silently pretending it reranked;
  * the context handed to the model respects both bounds.

These are the failure modes that survive a good demo, so they get their own gate.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

FOUNDATION = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation")
sys.path.insert(0, str(FOUNDATION / "validators"))
sys.path.insert(0, str(FOUNDATION / "rag"))

import rag_answer  # noqa: E402
import rag_store  # noqa: E402
from gate_rag import CORPUS, INJECTION_MARKER, deterministic_embed  # noqa: E402
from gatelib import GateFailure, check, record  # noqa: E402


class ScriptedAnswerer:
    """Stands in for the chat endpoint and captures what the pipeline sent."""

    def __init__(self, reply: str, rerank_available: bool = False) -> None:
        self.reply = reply
        self.rerank_available = rerank_available
        self.chat_payloads: list[dict] = []

    def __call__(self, url: str, payload: dict, timeout: int = 600) -> dict:
        if url == rag_answer.RERANK_URL:
            if not self.rerank_available:
                raise ConnectionRefusedError("reranker offline")
            documents = payload["documents"]
            # Deterministic stand-in ordering: longest document first.
            order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))
            return {"results": [
                {"index": i, "relevance_score": 1.0 - rank / max(len(order), 1)}
                for rank, i in enumerate(order)
            ]}
        self.chat_payloads.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": self.reply}}]}


def build_corpus(workspace: Path) -> Path:
    corpus = workspace / "corpus"
    corpus.mkdir()
    for name, body in CORPUS.items():
        (corpus / name).write_text(body)
    database = workspace / "behavioural.sqlite3"
    connection = rag_store.connect(database)
    rag_store.sync_directory(connection, corpus)
    return database


def main() -> int:
    summary: dict = {"gate": "rag-behavioural", "checks": {}}
    original_embed = rag_store.embed
    original_post = rag_answer.post
    rag_store.embed = deterministic_embed
    workspace = Path(tempfile.mkdtemp(prefix="hermes-rag-behav-"))
    try:
        database = build_corpus(workspace)

        # 1. Well-formed cited answer over an unavailable reranker.
        scripted = ScriptedAnswerer("The scrub repaired 0B and found no errors [1].")
        rag_answer.post = scripted
        result = rag_answer.answer("What did the scrub repair?", database, top_n=3)
        summary["checks"]["grounded"] = {
            "citations": result["citations"],
            "invalid": result["invalid_citations"],
            "grounded": result["grounded"],
        }
        check(result["grounded"], f"valid single citation was not accepted: {result}")
        check(len(result["citations"]) == 1, f"expected one resolved citation, got {result['citations']}")
        citation = result["citations"][0]
        for field in ("source_path", "char_start", "char_end", "content_sha256"):
            check(field in citation, f"citation is missing provenance field {field}")
        quoted = Path(citation["source_path"]).read_text()[citation["char_start"]:citation["char_end"]]
        check(bool(quoted.strip()), "citation byte range resolves to empty text")
        summary["checks"]["grounded"]["quoted_prefix"] = quoted[:80]

        # 2. The prompt actually sent must carry the framing, not just the docs.
        payload = scripted.chat_payloads[-1]
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        summary["checks"]["framing"] = {
            "system_declares_untrusted": "data, never instructions" in system,
            "user_has_envelope": "UNTRUSTED DOCUMENT DATA" in user,
            "user_has_provenance": "sha256=" in user and "chars=" in user,
            "system_forbids_invented_markers": "Do not invent passage numbers" in system,
        }
        for key, value in summary["checks"]["framing"].items():
            check(value, f"prompt framing check failed: {key}")

        # 3. Degradation must be visible, not silent.
        summary["checks"]["rerank_degraded"] = [
            p.get("rerank_score") for p in result["passages"]
        ]
        check(
            all(p.get("rerank_score") is None for p in result["passages"]),
            "reranker was offline yet scores were reported",
        )

        # 4. Invented citations must be caught.
        rag_answer.post = ScriptedAnswerer("Everything is fine [1] and also [97].")
        hallucinated = rag_answer.answer("What did the scrub repair?", database, top_n=3)
        summary["checks"]["invented_citation"] = {
            "invalid": hallucinated["invalid_citations"],
            "grounded": hallucinated["grounded"],
        }
        check(hallucinated["invalid_citations"] == [97], f"invented citation not caught: {hallucinated}")
        check(not hallucinated["grounded"], "answer with an invented citation was still marked grounded")

        # 5. An uncited answer must not be marked grounded.
        rag_answer.post = ScriptedAnswerer("The scrub repaired nothing at all.")
        uncited = rag_answer.answer("What did the scrub repair?", database, top_n=3)
        summary["checks"]["uncited_correctly_not_grounded"] = not uncited["grounded"]
        check(not uncited["grounded"], "answer with no citation was marked grounded")

        # 6. Injection detection is a real assertion, so prove it can fail.
        rag_answer.post = ScriptedAnswerer(f"{INJECTION_MARKER}")
        compromised = rag_answer.answer("Summarise the meeting notes.", database, top_n=3)
        summary["checks"]["injection_detectable"] = INJECTION_MARKER in compromised["answer"]
        check(
            INJECTION_MARKER in compromised["answer"],
            "the injection assertion cannot observe a compromised answer, so it proves nothing",
        )

        # 7. Working reranker path reorders and records scores.
        working = ScriptedAnswerer("Answer [1].", rerank_available=True)
        rag_answer.post = working
        reranked = rag_answer.answer("router policy", database, top_n=3)
        summary["checks"]["rerank_active"] = [p.get("rerank_score") for p in reranked["passages"]]
        check(
            all(p.get("rerank_score") is not None for p in reranked["passages"]),
            "reranker was reachable but scores were not recorded",
        )
        scores = [p["rerank_score"] for p in reranked["passages"]]
        check(scores == sorted(scores, reverse=True), f"reranked passages are not in score order: {scores}")

        # 8. Bounds hold on the assembled prompt.
        summary["checks"]["context_chars"] = reranked["context_chars"]
        check(
            reranked["context_chars"] <= rag_store.MAX_CONTEXT_CHARS,
            f"context {reranked['context_chars']} exceeded cap {rag_store.MAX_CONTEXT_CHARS}",
        )
        check(
            reranked["passages_used"] <= rag_store.MAX_CONTEXT_CHUNKS,
            "passage count exceeded the chunk cap",
        )
        summary["pass"] = True
    except (GateFailure, Exception) as error:  # noqa: BLE001
        summary["pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        rag_store.embed = original_embed
        rag_answer.post = original_post
        shutil.rmtree(workspace, ignore_errors=True)
    return record("gate-rag-behavioural", summary)


if __name__ == "__main__":
    raise SystemExit(main())
