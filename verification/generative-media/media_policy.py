#!/usr/bin/env python3
"""GPU/host policy for local ComfyUI image and music gates.

GPU2 is the display adapter and is never exposed to ComfyUI. GPU0 and GPU1 are
the headless ROCm compute cards. Z-Image Turbo BF16 is ~20.6 GiB on disk and
cannot live on a 16 GiB card, so GPU1 (V620 32 GiB) is the primary device and
GPU0 is overflow-only. Weights are never duplicated to fake utilization.
"""
from __future__ import annotations

from pathlib import Path
import os

ROOT = Path("/home/typhoon/git/frankenstein-llm")
COMFY_ROOT = ROOT / "tools" / "ComfyUI"
COMFY_PYTHON = ROOT / "venvs" / "comfy" / "bin" / "python"
EXTRA_PATHS = Path(__file__).resolve().parent / "extra_model_paths.yaml"
EVIDENCE = Path(__file__).resolve().parent / "evidence"

ARTIFACTS = {
    "z-image-unet": ROOT / "models/comfy/diffusion_models/z_image_turbo_bf16.safetensors",
    "z-image-clip": ROOT / "models/comfy/text_encoders/qwen_3_4b.safetensors",
    "z-image-vae": ROOT / "models/comfy/vae/ae.safetensors",
    "ace-step-1.5-aio": ROOT / "models/comfy/checkpoints/ace_step_1.5_turbo_aio.safetensors",
    "qwen-image-edit-unet": ROOT / "models/comfy/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors",
    "qwen-image-edit-clip": ROOT / "models/comfy/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen-image-edit-vae": ROOT / "models/comfy/vae/qwen_image_vae.safetensors",
    "qwen-image-edit-lora": ROOT / "models/comfy/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "flux2-klein-unet": ROOT / "models/comfy/flux2-klein-4b/flux-2-klein-4b.safetensors",
}

REQUIRED_BYTES = {
    "z-image-unet": 12309866400,
    "z-image-clip": 8044982048,
    "z-image-vae": 335304388,
    "ace-step-1.5-aio": 10025478736,
    "qwen-image-edit-unet": 20499083824,
    "qwen-image-edit-clip": 9384670680,
    "qwen-image-edit-vae": 253806246,
    "qwen-image-edit-lora": 849608296,
    "flux2-klein-unet": 7751105712,
}

# Workflow coverage claims, stated as data so a passing preflight cannot be read
# as covering more than it does.
#
# Z-Image Turbo is a text-to-image generation checkpoint. Instruction-following
# editors are inventoried separately below; they are on disk and still unproven.
#
# artifacts_present answers "is the weight on disk"; functionally_proven answers
# "did a gate observe it working". Only the second is admission.
WORKFLOW_CLAIMS = {
    "image-generation": {
        "artifacts_present": True,
        "artifacts": ["z-image-unet", "z-image-clip", "z-image-vae"],
        "functionally_proven": False,
        "note": "Z-Image Turbo BF16 text-to-image; static preflight only",
    },
    "image-editing": {
        "artifacts_present": True,
        "artifacts": [
            "qwen-image-edit-unet",
            "qwen-image-edit-clip",
            "qwen-image-edit-vae",
            "qwen-image-edit-lora",
            "flux2-klein-unet",
        ],
        "functionally_proven": False,
        "note": "Qwen Image Edit 2511 INT8 + Lightning LoRA and FLUX.2 Klein 4B "
                "are on disk; no live ComfyUI run has been admitted",
    },
    "music-generation": {
        "artifacts_present": True,
        "artifacts": ["ace-step-1.5-aio"],
        "functionally_proven": False,
        "note": "ACE-Step 1.5 turbo all-in-one; static preflight only",
    },
}

REQUIRED_NODE_FILES = {
    "z-image": COMFY_ROOT / "comfy_extras" / "nodes_zimage.py",
    "ace-step": COMFY_ROOT / "comfy_extras" / "nodes_ace.py",
}

REQUIRED_NODE_IDS = {
    "z-image": ("TextEncodeZImageOmni",),
    "ace-step": ("TextEncodeAceStepAudio1.5", "EmptyAceStep1.5LatentAudio"),
}

# Physical roles. ComfyUI is launched with HIP_VISIBLE_DEVICES=0,1 so its
# cuda:0 is GPU0 and cuda:1 is GPU1. GPU2 never appears.
GPU_ROLES = {
    0: {"name": "RX 6900 XT", "role": "headless-compute", "nominal_gib": 16},
    1: {"name": "Radeon Pro V620", "role": "headless-compute-primary", "nominal_gib": 32},
    2: {"name": "RX 6900 XT", "role": "display", "nominal_gib": 16},
}

VISIBLE_COMPUTE = "0,1"
PRIMARY_DEVICE = 1
LISTEN = "127.0.0.1"
PORT = 8188

KERNEL_BUILD_NEEDLES = (
    "make -j45 LLVM=1",
    "link-vmlinux",
    "ld.lld -m elf_x86_64",
)


def host_exclusive_blockers() -> list[str]:
    """Return process fingerprints that must not overlap a GPU gate."""
    blockers = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if any(needle in cmd for needle in KERNEL_BUILD_NEEDLES):
            blockers.append(f"pid={entry.name} kernel-build")
            break
    return blockers


def comfy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = VISIBLE_COMPUTE
    env["ROCR_VISIBLE_DEVICES"] = VISIBLE_COMPUTE
    env["CUDA_VISIBLE_DEVICES"] = VISIBLE_COMPUTE
    env.pop("DISPLAY", None)
    return env


def comfy_argv() -> list[str]:
    return [
        str(COMFY_PYTHON),
        str(COMFY_ROOT / "main.py"),
        "--listen", LISTEN,
        "--port", str(PORT),
        "--disable-auto-launch",
        "--extra-model-paths-config", str(EXTRA_PATHS),
        "--cuda-device", str(PRIMARY_DEVICE),
    ]


def artifact_inventory() -> dict:
    files = {}
    problems = []
    for key, path in ARTIFACTS.items():
        expected = REQUIRED_BYTES[key]
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        ok = exists and size == expected
        files[key] = {
            "path": str(path),
            "exists": exists,
            "bytes": size,
            "expected_bytes": expected,
            "ok": ok,
        }
        if not ok:
            problems.append(f"{key}: missing or wrong size")
    return {"files": files, "problems": problems, "pass": not problems}
