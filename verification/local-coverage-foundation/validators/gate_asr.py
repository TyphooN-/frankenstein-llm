#!/usr/bin/env python3
"""Functional gate for local Qwen3-ASR-1.7B on the three ROCm GPUs.

This is a correctness/load/unload gate with passive performance observations.
"""
from __future__ import annotations

import difflib
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gatelib import performance, exit_after_verdict, unload_verdict, vram_used  # noqa: E402

MODEL = Path("/home/typhoon/git/frankenstein-llm/models/asr/Qwen3-ASR-1.7B-hf")
AUDIO = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/fixtures/asr/librispeech-mr-quilter.wav")
EVIDENCE = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/evidence/gate-asr.json")
EXPECTED = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
AUDIO_SHA256 = "799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46"
# Memory caps express the host policy GPU0:GPU1:GPU2 = 1:2:1.
MAX_MEMORY = {0: "2GiB", 1: "4GiB", 2: "2GiB", "cpu": "8GiB"}
VRAM_RESIDUE_TOLERANCE = 768 * 1024 * 1024


def load_audio_16k(path: Path) -> np.ndarray:
    """Load WAV without librosa/numba, which break on this Python 3.14 llvmlite."""
    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        samples = resample_poly(samples, 16000, rate).astype(np.float32, copy=False)
    return samples


def gpu_ids(device_map: dict) -> set[int]:
    ids: set[int] = set()
    for value in device_map.values():
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, str) and value.startswith("cuda:"):
            ids.add(int(value.split(":", 1)[1]))
    return ids


def main() -> int:
    summary: dict = {
        "gate": "asr",
        "model": str(MODEL),
        "audio": str(AUDIO),
        "audio_sha256": AUDIO_SHA256,
        "device_memory_policy": {"gpu0": "2GiB", "gpu1_v620": "4GiB", "gpu2": "2GiB"},
        "benchmark_performed": False,
    }
    baseline = vram_used()
    model = processor = inputs = output_ids = None
    try:
        assert torch.cuda.is_available(), "ROCm torch reports no GPUs"
        assert torch.cuda.device_count() >= 3, f"expected 3 GPUs, saw {torch.cuda.device_count()}"
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="balanced",
            max_memory=MAX_MEMORY,
            low_cpu_mem_usage=True,
        )
        mapping = {name: str(device) for name, device in model.hf_device_map.items()}
        used_gpus = gpu_ids(model.hf_device_map)
        summary["device_map"] = mapping
        summary["gpus_used"] = sorted(used_gpus)
        assert used_gpus == {0, 1, 2}, f"model was not distributed over all three GPUs: {mapping}"
        summary["vram_baseline"] = baseline
        summary["vram_loaded"] = vram_used()

        first_device = next(model.parameters()).device
        waveform = load_audio_16k(AUDIO)
        summary["audio_samples"] = int(waveform.shape[0])
        summary["audio_seconds"] = round(float(waveform.shape[0]) / 16000.0, 3)
        inputs = processor.apply_transcription_request(audio=waveform, sampling_rate=16000)
        inputs = inputs.to(first_device, model.dtype)
        with torch.inference_mode():
            output_ids = performance.call(model.generate, operation="asr-generate", model_id="Qwen3-ASR-1.7B", audio_duration=len(waveform) / 16000, **inputs, max_new_tokens=128, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = processor.decode(generated, return_format="parsed")[0]
        transcript = parsed["transcription"].strip()
        language = parsed["language"].strip()
        similarity = difflib.SequenceMatcher(None, transcript.lower(), EXPECTED.lower()).ratio()
        summary["language"] = language
        summary["transcription"] = transcript
        summary["expected"] = EXPECTED
        summary["character_similarity"] = round(similarity, 6)
        assert "english" in language.lower(), f"wrong language: {language!r}"
        assert similarity >= 0.90, f"transcription similarity {similarity:.3f} below 0.90: {transcript!r}"
        for word in ("quilter", "apostle", "middle classes", "gospel"):
            assert word in transcript.lower(), f"missing expected phrase {word!r}: {transcript!r}"
        summary["pass"] = True
    except Exception as error:  # noqa: BLE001
        summary["pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        del output_ids, inputs, model, processor
        gc.collect()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        # Scored through the shared helper so a card whose sysfs node stopped
        # answering cannot be scored as a release: -1 minus a real baseline is a
        # hugely negative residue that clears any tolerance. Reading sysfs also
        # goes through gatelib, which reports -1 instead of raising -- an
        # exception here would escape the finally and leave no evidence file.
        settled = None
        verdict = unload_verdict(baseline, {}, VRAM_RESIDUE_TOLERANCE)
        for _ in range(30):
            time.sleep(2)
            settled = vram_used()
            verdict = unload_verdict(baseline, settled, VRAM_RESIDUE_TOLERANCE)
            if verdict["pass"]:
                break
        summary["vram_after_unload"] = settled
        summary["vram_residue_bytes"] = verdict["vram_residue_bytes"]
        summary["unload"] = verdict
        if not verdict["pass"]:
            summary["pass"] = False
            # Keep whatever failed first; the unload problems are in summary["unload"].
            summary.setdefault("error", f"clean unload not proven: {verdict['problems']}")
        summary["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    # exit_after_verdict, not SystemExit: the ROCm teardown segfault documented
    # there lands after main() has already written the evidence file, and it
    # replaced this gate's pass with exit -11.
    exit_after_verdict(main())
