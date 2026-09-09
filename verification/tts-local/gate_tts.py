#!/usr/bin/env python3
"""Local Qwen3-TTS admission gate: a real, intelligible audio artifact or nothing.

The 1.7B-Base checkpoint is a *voice-clone* model, so the gate clones the voice
from the LibriSpeech clip the ASR gate already validated and then synthesises new
sentences the reference never contained. That matters: if the gate reused the
reference text it could not distinguish synthesis from playback.

"Intelligible" is not asserted by listening. The generated waveform is fed back
through the already-admitted Qwen3-ASR model and the transcript is compared to
the text that was requested, so the claim is checked by a second model rather
than by the gate's own author. Silence, noise, or a truncated tail all fail that
round trip, which is exactly the failure mode a duration/RMS check misses.

Passive performance observations accompany existing inference; no extra benchmark runs.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

# Resolve the shared validator helpers from this gate's own checkout. Naming
# the primary path literally made a linked worktree import the *other*
# tree's gatelib, so a change under test was never the code that ran.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                        / "verification/local-coverage-foundation/validators"))
from gatelib import allocator_report, release_torch_memory, unload_verdict, vram_used  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import qualification_performance as performance

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = Path("/home/typhoon/git/frankenstein-llm/models/tts/Qwen3-TTS-12Hz-1.7B-Base")
ASR_DIR = Path("/home/typhoon/git/frankenstein-llm/models/asr/Qwen3-ASR-1.7B-hf")
REF_AUDIO = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation"
                 "/fixtures/asr/librispeech-mr-quilter.wav")
REF_TEXT = ("Mr. Quilter is the apostle of the middle classes, "
            "and we are glad to welcome his gospel.")

# asr_roundtrip.py loads Qwen3-ASR, which needs a transformers new enough to
# register model_type qwen3_asr. This gate's own TTS stack pins an older one,
# so sys.executable -- the tts venv -- cannot import AutoModelForMultimodalLM
# and the round trip fails on import before it transcribes anything. The two
# venvs are kept apart on purpose; the round trip is spawned in the ASR one.
ASR_PYTHON = ROOT.parents[1] / "venvs" / "asr" / "bin" / "python"

# TTS runs on GPU0: the 32 GB GPU1 is reserved for the large gates and GPU2 drives
# the display, so the small model takes the idle mid-size card.
TTS_DEVICE = 0
VRAM_RESIDUE_TOLERANCE = 512 << 20
SWAP_GROWTH_TOLERANCE = 512 << 20

# Sentences the reference clip does not contain, so a pass cannot be echo.
UTTERANCES = [
    ("short", "The quick brown fox jumps over the lazy dog."),
    ("numeric", "Please call me back at four one five, five five five, "
                "zero one nine eight."),
    ("multi_sentence", "Local verification finished successfully. "
                       "Three graphics cards are now idle. "
                       "No performance figures were recorded."),
]

MALFORMED = [
    ("empty", ""),
    ("whitespace_only", "   \t\n  "),
    ("control_chars", "abc\x00\x07\x1b[31mdef"),
    ("zero_width", "​​​"),
    ("lone_surrogate_stripped", "hello \udcff world"),
    ("only_punctuation", "!!!???...,,,;;;"),
    ("rtl_mixed", "hello ‮evil‬ world"),
]
LONG_INPUT_WORDS = 400
# Hard deadline for the separate ASR process. Generous, because it loads a model
# on a possibly busy host, but finite: an unbounded wait here would hold the
# serialized qualification open indefinitely.
ASR_ROUNDTRIP_TIMEOUT = 3600


SECTION_KEYS = ("gpu_use", "long_input", "malformed_inputs", "memory_safety",
                "unload", "intelligibility")


def note(problems: list, message: str) -> None:
    """Record a problem once. The same failure reported twice is not two failures."""
    if message not in problems:
        problems.append(message)


def finalize(summary: dict) -> dict:
    """Apply the admission rule: every section must be present *and* passing.

    A missing section is treated exactly like a failing one. Admission requires
    positive evidence, so "we did not check" and "the check failed" have to reach
    the same verdict -- otherwise skipping a check would be the cheapest way to
    pass the gate.
    """
    problems = summary.setdefault("problems", [])
    verdicts = {}
    for key in SECTION_KEYS:
        section = summary.get(key)
        recorded = isinstance(section, dict)
        passed = bool(recorded and section.get("pass"))
        verdicts[key] = passed
        if not passed:
            note(problems, f"section {key} did not pass" if recorded
                 else f"section {key} was never recorded")
    summary["section_verdicts"] = verdicts
    summary["pass"] = not problems
    return summary


def memory_sample(label: str) -> dict:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        meminfo[key] = int(rest.strip().split()[0]) * 1024
    return {
        "label": label,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mem_available_bytes": meminfo.get("MemAvailable", -1),
        "swap_used_bytes": meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0),
        "psi_memory": Path("/proc/pressure/memory").read_text().splitlines()[0],
        "vram_used": vram_used(),
    }


def audio_stats(wav, sr: int) -> dict:
    import numpy as np

    a = np.asarray(wav, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"samples": 0, "seconds": 0.0, "rms": 0.0, "peak": 0.0,
                "silent": True, "clipped_fraction": 0.0, "nonfinite": 0}
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a ** 2)))
    # Fraction of 20 ms frames carrying energy: a real utterance is mostly active,
    # a dropout or a dead tail is not.
    frame = max(1, int(0.02 * sr))
    frames = a[: (a.size // frame) * frame].reshape(-1, frame)
    active = float(np.mean(np.sqrt(np.mean(frames ** 2, axis=1)) > max(1e-4, rms * 0.1))) \
        if frames.size else 0.0
    return {
        "samples": int(a.size),
        "sample_rate": sr,
        "seconds": round(a.size / sr, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silent": bool(rms < 1e-4),
        "clipped_fraction": round(float(np.mean(np.abs(a) >= 0.999)), 6),
        "active_frame_fraction": round(active, 4),
        "nonfinite": int(np.sum(~np.isfinite(a))),
    }


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def similarity(a: str, b: str) -> float:
    import difflib
    return round(difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio(), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-asr", action="store_true",
                    help="record artifacts without the intelligibility round trip; "
                         "the gate cannot pass in this mode")
    args = ap.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import soundfile as sf
        import torch
    except Exception as error:                                  # noqa: BLE001
        # Straight after a reboot the likeliest failure is a runtime that no
        # longer imports. That is an admission failure with evidence attached,
        # not an untraceable traceback.
        broken = finalize({
            "gate": "tts-local",
            "model": "Qwen3-TTS-12Hz-1.7B-Base",
            "benchmark_performed": False,
            "problems": [f"runtime unavailable: {type(error).__name__}: {error}"[:300]],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        (EVIDENCE / "gate-tts.json").write_text(
            json.dumps(broken, indent=2, sort_keys=True) + "\n")
        print(json.dumps(broken, indent=2))
        return 1

    summary = {
        "gate": "tts-local",
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
        "model_dir": str(MODEL_DIR),
        "mode": "voice clone from the ASR gate's validated LibriSpeech reference",
        "reference_audio": str(REF_AUDIO),
        "reference_text": REF_TEXT,
        "benchmark_performed": False,
        "problems": [],
    }
    problems = summary["problems"]
    baseline = memory_sample("baseline")
    summary["memory_baseline"] = baseline

    model = None
    try:
        from qwen_tts import Qwen3TTSModel

        started = time.monotonic()
        model = Qwen3TTSModel.from_pretrained(
            str(MODEL_DIR), dtype=torch.bfloat16, device_map=f"cuda:{TTS_DEVICE}")
        summary["load"] = {
            "load_seconds": round(time.monotonic() - started, 1),
            "device": f"cuda:{TTS_DEVICE}",
            "dtype": "bfloat16",
        }
        summary["memory_loaded"] = memory_sample("loaded")

        # GPU residency: the weights must actually be on the target card.
        delta = (summary["memory_loaded"]["vram_used"].get(f"card{TTS_DEVICE}", 0)
                 - baseline["vram_used"].get(f"card{TTS_DEVICE}", 0))
        summary["gpu_use"] = {
            "device": TTS_DEVICE,
            "vram_growth_bytes": delta,
            "pass": delta > (256 << 20),
        }
        if not summary["gpu_use"]["pass"]:
            note(problems, f"model did not occupy GPU{TTS_DEVICE} (growth {delta} bytes)")

        # --- normal synthesis -------------------------------------------------
        generated = []
        for name, text in UTTERANCES:
            wavs, sr = performance.call(model.generate_voice_clone, operation="tts-generate", model_id="Qwen3-TTS-12Hz-1.7B-Base", mode="audio",
                text=text, language="English",
                ref_audio=str(REF_AUDIO), ref_text=REF_TEXT)
            wav = np.asarray(wavs[0] if isinstance(wavs, list) else wavs).reshape(-1)
            path = ARTIFACTS / f"tts-{name}.wav"
            sf.write(path, wav, sr, subtype="PCM_16")
            stats = audio_stats(wav, sr)
            stats.update({"name": name, "text": text, "path": str(path),
                          "bytes": path.stat().st_size})
            generated.append(stats)
            if stats["silent"]:
                note(problems, f"{name}: generated audio is silent")
            if stats["seconds"] < 0.5:
                note(problems, f"{name}: generated audio is implausibly short "
                                f"({stats['seconds']}s)")
            if stats["nonfinite"]:
                note(problems, f"{name}: waveform contains non-finite samples")
        summary["generated"] = generated
        summary["audio_artifacts"] = [g["path"] for g in generated]

        # --- long input -------------------------------------------------------
        long_text = " ".join(["Local verification of the text to speech pipeline "
                              "continues without measuring speed."] * 12)
        long_text = " ".join(long_text.split()[:LONG_INPUT_WORDS])
        try:
            wavs, sr = performance.call(model.generate_voice_clone, operation="tts-generate", model_id="Qwen3-TTS-12Hz-1.7B-Base", mode="audio",
                text=long_text, language="English",
                ref_audio=str(REF_AUDIO), ref_text=REF_TEXT, max_new_tokens=4096)
            wav = np.asarray(wavs[0] if isinstance(wavs, list) else wavs).reshape(-1)
            path = ARTIFACTS / "tts-long.wav"
            sf.write(path, wav, sr, subtype="PCM_16")
            st = audio_stats(wav, sr)
            st.update({"words": len(long_text.split()), "path": str(path),
                       "handled": True})
            summary["long_input"] = st
            summary["long_input"]["pass"] = (not st["silent"]) and st["seconds"] > 5.0
            if not summary["long_input"]["pass"]:
                note(problems, f"long input produced unusable audio: {st['seconds']}s")
        except Exception as exc:                                # noqa: BLE001
            summary["long_input"] = {"handled": False, "pass": False,
                                     "error": f"{type(exc).__name__}: {exc}"[:300]}
            note(problems, f"long input raised {type(exc).__name__}")

        # --- malformed input --------------------------------------------------
        # Requirement: degrade predictably. Either refuse, or return finite audio.
        # A crash, a hang, or NaNs is a failure.
        mal = []
        for name, text in MALFORMED:
            entry = {"name": name, "repr": repr(text)[:80]}
            try:
                wavs, sr = performance.call(model.generate_voice_clone, operation="tts-generate", model_id="Qwen3-TTS-12Hz-1.7B-Base", mode="audio",
                    text=text, language="English",
                    ref_audio=str(REF_AUDIO), ref_text=REF_TEXT, max_new_tokens=256)
                wav = np.asarray(wavs[0] if isinstance(wavs, list) else wavs).reshape(-1)
                st = audio_stats(wav, sr)
                entry.update({"outcome": "returned_audio", "seconds": st["seconds"],
                              "nonfinite": st["nonfinite"], "silent": st["silent"]})
                entry["acceptable"] = st["nonfinite"] == 0
            except Exception as exc:                            # noqa: BLE001
                entry.update({"outcome": "refused",
                              "error": f"{type(exc).__name__}: {exc}"[:200],
                              "acceptable": True})
            mal.append(entry)
            if not entry["acceptable"]:
                note(problems, f"malformed input {name!r} produced non-finite audio")
        summary["malformed_inputs"] = {
            "cases": mal,
            "pass": all(c["acceptable"] for c in mal),
            "note": "acceptable == clean refusal or finite audio; crash/NaN is a failure",
        }

        summary["memory_peak"] = memory_sample("peak")
        swap_growth = (summary["memory_peak"]["swap_used_bytes"]
                       - baseline["swap_used_bytes"])
        summary["memory_safety"] = {
            "swap_growth_bytes": swap_growth,
            "tolerance_bytes": SWAP_GROWTH_TOLERANCE,
            "pass": swap_growth <= SWAP_GROWTH_TOLERANCE,
        }
        if not summary["memory_safety"]["pass"]:
            note(problems, f"swap grew by {swap_growth} bytes")

    except Exception as error:                                  # noqa: BLE001
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback_tail"] = __import__("traceback").format_exc()[-2000:]
        note(problems, summary["error"])
    finally:
        model = None
        gc.collect()
        for problem in release_torch_memory(torch):
            note(problems, f"unload cleanup: {problem}")
        gc.collect()
        time.sleep(8)
        allocator = allocator_report()
        after = memory_sample("after-unload")
        summary["memory_after_unload"] = after
        # Cards that could not be read are not cards that released: an empty
        # residue set used to make ``all(...)`` vacuously true, so a host whose
        # sysfs nodes stopped answering passed the clean-unload gate outright.
        # The allocator reading is what decides whether the *model* came back;
        # this process keeps its HIP context until it exits, and that context is
        # what the card-level residue was being blamed on the model for.
        summary["unload"] = unload_verdict(
            baseline["vram_used"], after["vram_used"], VRAM_RESIDUE_TOLERANCE,
            allocator=allocator)
        for problem in summary["unload"]["problems"]:
            note(problems, f"unload: {problem}")

    # --- intelligibility round trip (separate process so TTS is fully unloaded)
    # This is the gate's central claim, so it is never silently absent: if it did
    # not run, that is recorded as a failed section rather than as no section.
    if args.skip_asr:
        summary["intelligibility"] = {
            "pass": False,
            "skipped": True,
            "reason": "--skip-asr was passed: the round trip did not run, so the "
                      "artifacts were recorded but admission is not granted",
        }
    elif not summary.get("generated"):
        summary["intelligibility"] = {
            "pass": False,
            "skipped": True,
            "reason": "no audio was generated, so there was nothing to transcribe",
        }
    else:
        # The round trip loads a second model, so it is bounded. A blown deadline
        # or a runtime that will not start must still leave gate-tts.json behind:
        # letting TimeoutExpired escape here would abandon the run with no
        # artifact at all, which is the one outcome this gate may not produce.
        try:
            rt = subprocess.run(
                [str(ASR_PYTHON), str(ROOT / "asr_roundtrip.py"),
                 "--pairs", json.dumps([{"path": g["path"], "text": g["text"],
                                         "name": g["name"]} for g in summary["generated"]])],
                capture_output=True, text=True, timeout=ASR_ROUNDTRIP_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            note(problems, f"intelligibility round trip exceeded "
                           f"{ASR_ROUNDTRIP_TIMEOUT}s and was killed")
            summary["intelligibility"] = {
                "pass": False, "timed_out": True,
                "timeout_seconds": ASR_ROUNDTRIP_TIMEOUT,
                "stdout_tail": (error.stdout or b"")[-600:].decode(errors="replace")
                if isinstance(error.stdout, bytes) else (error.stdout or "")[-600:],
            }
        except OSError as error:
            note(problems, f"intelligibility round trip could not start: "
                           f"{type(error).__name__}")
            summary["intelligibility"] = {
                "pass": False,
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        else:
            summary["intelligibility_stdout_tail"] = rt.stdout[-600:]
            try:
                summary["intelligibility"] = json.loads(rt.stdout.strip().splitlines()[-1])
            except Exception:                                   # noqa: BLE001
                summary["intelligibility"] = {"pass": False, "rc": rt.returncode,
                                              "stderr_tail": rt.stderr[-800:]}

    finalize(summary)
    summary["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out = EVIDENCE / "gate-tts.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("generated", "malformed_inputs")}, indent=2)[:4000])
    print(f"\n[tts] wrote {out}", file=sys.stderr)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
