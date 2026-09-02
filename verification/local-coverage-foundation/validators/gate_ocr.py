#!/usr/bin/env python3
"""Functional admission gate for the HunyuanOCR sidecar.

Transcribes a locally generated document whose ground truth is exact, then scores
character-level accuracy on the prose lines and exact-cell recovery on the table.
A general VLM that paraphrases instead of transcribing fails this even though it
"understood" the image, which is the distinction the OCR lane exists to enforce.
"""
from __future__ import annotations

import base64
import difflib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/validators")
from gatelib import GateFailure, check, managed_sidecar, post_json, record, unload_gate, vram_used  # noqa: E402

BASE = "http://127.0.0.1:8083"
UNIT = "llama-sidecar@ocr.service"
FIXTURES = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/fixtures")
MIN_CHAR_ACCURACY = 0.95


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def transcribe(image_path: Path, instruction: str) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    result = post_json(f"{BASE}/v1/chat/completions", {
        "model": "hunyuan-ocr",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                {"type": "text", "text": instruction},
            ],
        }],
        "max_tokens": 1024,
        "temperature": 0,
    })
    return result["choices"][0]["message"]["content"]


def main() -> int:
    summary: dict = {"gate": "ocr", "base_url": BASE}
    try:
        truth = json.loads((FIXTURES / "ground-truth.json").read_text())["document"]
        baseline, waited = managed_sidecar(UNIT, BASE)
        summary["vram_baseline"] = baseline
        summary["load_seconds"] = round(waited, 1)
        summary["vram_loaded"] = vram_used()

        text = transcribe(Path(truth["path"]), "Transcribe every character in this document exactly, including the table.")
        summary["transcript"] = text

        observed = normalise(text)
        expected_prose = normalise(" ".join(truth["lines"]))
        accuracy = difflib.SequenceMatcher(None, expected_prose, observed).ratio()
        # The transcript legitimately contains the table too, so compare the prose
        # against its best-matching window rather than the whole output.
        window = len(expected_prose)
        if len(observed) > window:
            accuracy = max(
                difflib.SequenceMatcher(None, expected_prose, observed[i:i + window]).ratio()
                for i in range(0, len(observed) - window + 1, max(1, window // 8))
            )
        summary["prose_char_accuracy"] = round(accuracy, 4)
        check(
            accuracy >= MIN_CHAR_ACCURACY,
            f"prose character accuracy {accuracy:.4f} below required {MIN_CHAR_ACCURACY}",
        )

        cells = [c for row in truth["table_rows"] for c in row] + truth["table_header"]
        missing = [c for c in cells if normalise(c) not in observed]
        summary["table_cells_total"] = len(cells)
        summary["table_cells_missing"] = missing
        check(not missing, f"table cells absent from transcript: {missing}")

        summary["unload"] = unload_gate(UNIT, baseline)
        summary["pass"] = True
    except (GateFailure, Exception) as error:  # noqa: BLE001
        summary["pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    return record("gate-ocr", summary)


if __name__ == "__main__":
    raise SystemExit(main())
