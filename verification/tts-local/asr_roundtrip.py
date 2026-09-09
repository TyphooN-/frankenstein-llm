#!/usr/bin/env python3
"""ASR round-trip verifier for the local Qwen3-TTS admission gate.

The parent gate passes ``--pairs`` as a JSON list containing ``path``, ``text``
and ``name``. This helper emits exactly one compact JSON object as its final
stdout line and exits non-zero unless every generated utterance is intelligible.
Passive timing and actual output-token counts are recorded when requested by the supervisor.
"""
from __future__ import annotations

import argparse
import difflib
import gc
import json
import sys
from pathlib import Path
import time
import unicodedata
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import qualification_performance as performance

MODEL_DIR = Path("/home/typhoon/git/frankenstein-llm/models/asr/Qwen3-ASR-1.7B-hf")
MAX_MEMORY = {0: "2GiB", 1: "4GiB", 2: "2GiB", "cpu": "8GiB"}
SIMILARITY_THRESHOLD = 0.80
MAX_PAIRS = 16
MAX_AUDIO_BYTES = 256 * 1024 * 1024


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(
        "".join(c if c.isalnum() or c.isspace() else " " for c in text).split()
    )


def text_similarity(expected: str, actual: str) -> float:
    return difflib.SequenceMatcher(None, normalize(expected), normalize(actual)).ratio()


def parse_pairs(raw: str) -> list[dict[str, str]]:
    value = json.loads(raw)
    if not isinstance(value, list) or not value or len(value) > MAX_PAIRS:
        raise ValueError(f"pairs must be a non-empty list of at most {MAX_PAIRS} entries")
    pairs: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"pair {index} is not an object")
        name, path, text = item.get("name"), item.get("path"), item.get("text")
        if not all(isinstance(v, str) and v.strip() for v in (name, path, text)):
            raise ValueError(f"pair {index} requires non-empty string name, path, and text")
        audio = Path(path)
        if not audio.is_file():
            raise ValueError(f"pair {index} audio file does not exist: {audio}")
        size = audio.stat().st_size
        if size <= 44 or size > MAX_AUDIO_BYTES:
            raise ValueError(f"pair {index} audio size is invalid: {size} bytes")
        pairs.append({"name": name.strip(), "path": str(audio), "text": text.strip()})
    return pairs


def evaluate_pairs(
    pairs: list[dict[str, str]], transcribe: Callable[[Path], tuple[str, str]]
) -> dict:
    cases = []
    for pair in pairs:
        transcript, language = transcribe(Path(pair["path"]))
        score = text_similarity(pair["text"], transcript)
        passed = bool(normalize(transcript)) and score >= SIMILARITY_THRESHOLD
        cases.append({
            "name": pair["name"],
            "path": pair["path"],
            "expected": pair["text"],
            "transcription": transcript,
            "language": language,
            "similarity": round(score, 6),
            "threshold": SIMILARITY_THRESHOLD,
            "pass": passed,
        })
    return {
        "gate": "tts-asr-roundtrip",
        "model": str(MODEL_DIR),
        "benchmark_performed": False,
        "cases": cases,
        "pass": bool(cases) and all(case["pass"] for case in cases),
    }


def build_local_transcriber():
    """Load the admitted local ASR model and return (callable, cleanup)."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=MAX_MEMORY,
        low_cpu_mem_usage=True,
    )
    model.eval()
    first_device = next(model.parameters()).device

    def transcribe(path: Path) -> tuple[str, str]:
        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size == 0 or not np.all(np.isfinite(samples)):
            raise ValueError(f"invalid waveform: {path}")
        if rate != 16000:
            samples = resample_poly(samples, 16000, rate).astype(np.float32, copy=False)
        inputs = processor.apply_transcription_request(audio=samples, sampling_rate=16000)
        inputs = inputs.to(first_device, model.dtype)
        with torch.inference_mode():
            output_ids = performance.call(model.generate, operation="asr-generate", model_id="Qwen3-ASR-1.7B", mode="tokens", audio_duration=len(samples) / 16000, **inputs, max_new_tokens=256, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = processor.decode(generated, return_format="parsed")[0]
        return parsed["transcription"].strip(), parsed["language"].strip()

    def cleanup() -> None:
        nonlocal model, processor
        model = None
        processor = None
        gc.collect()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        gc.collect()

    return transcribe, cleanup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="JSON list of generated WAV/text pairs")
    args = parser.parse_args(argv)
    cleanup: Callable[[], None] = lambda: None
    try:
        pairs = parse_pairs(args.pairs)
        transcribe, cleanup = build_local_transcriber()
        result = evaluate_pairs(pairs, transcribe)
    except Exception as error:  # fail closed, including model/audio/parser errors
        result = {
            "gate": "tts-asr-roundtrip",
            "model": str(MODEL_DIR),
            "benchmark_performed": False,
            "pass": False,
            "error": f"{type(error).__name__}: {error}"[:1000],
        }
    finally:
        try:
            cleanup()
        except Exception as error:  # unload failure invalidates admission
            result = locals().get("result", {})
            result["pass"] = False
            result["cleanup_error"] = f"{type(error).__name__}: {error}"[:1000]
    result["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
