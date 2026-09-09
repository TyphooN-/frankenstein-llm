#!/usr/bin/env python3
"""Live ComfyUI schema/model-discovery gate; does not submit generation."""
from __future__ import annotations

import json
from pathlib import Path
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
EVIDENCE = Path(__file__).resolve().parent / "evidence"
REQUIRED_NODES = (
    "TextEncodeZImageOmni",
    "TextEncodeAceStepAudio1.5",
    "EmptyAceStep1.5LatentAudio",
    "CheckpointLoaderSimple",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "SaveImage",
    "SaveAudio",
    "TextEncodeQwenImageEditPlus",
    "FluxKontextMultiReferenceLatentMethod",
    "FluxKontextImageScale",
    "CFGNorm",
    "LoraLoaderModelOnly",
)
REQUIRED_MODELS = {
    "checkpoints": ("ace_step_1.5_turbo_aio.safetensors",),
    "diffusion_models": (
        "z_image_turbo_bf16.safetensors",
        "qwen_image_edit_2511_int8_convrot.safetensors",
    ),
    "text_encoders": (
        "qwen_3_4b.safetensors",
        "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    ),
    "vae": ("ae.safetensors", "qwen_image_vae.safetensors"),
    "loras": ("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",),
}


def get_json(path: str, timeout: int = 30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return json.load(response)


def build_report() -> dict:
    object_info = get_json("/object_info", 120)
    missing_nodes = [name for name in REQUIRED_NODES if name not in object_info]
    models = {}
    problems = []
    for category, required in REQUIRED_MODELS.items():
        available = get_json(f"/models/{category}")
        missing = [filename for filename in required if filename not in available]
        models[category] = {
            "required": list(required),
            "available_count": len(available),
            "present": not missing,
            "missing": missing,
        }
        if missing:
            problems.append(f"{category}: missing {missing}")
    if missing_nodes:
        problems.append(f"missing nodes: {missing_nodes}")
    return {
        "gate": "generative-media-live-schema",
        "server": BASE,
        "benchmark_performed": False,
        "generation_submitted": False,
        "required_nodes": list(REQUIRED_NODES),
        "missing_nodes": missing_nodes,
        "models": models,
        "problems": problems,
        "pass": not problems,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main() -> int:
    try:
        report = build_report()
    except Exception as error:
        report = {
            "gate": "generative-media-live-schema",
            "server": BASE,
            "benchmark_performed": False,
            "generation_submitted": False,
            "pass": False,
            "error": f"{type(error).__name__}: {error}"[:1000],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    output = EVIDENCE / "media-live-schema.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
