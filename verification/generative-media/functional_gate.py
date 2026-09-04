#!/usr/bin/env python3
"""Execute local image, music, and image-edit workflows; no performance metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageStat

BASE = "http://127.0.0.1:8188"
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
INPUTS = EVIDENCE / "inputs"
OUTPUTS = EVIDENCE / "outputs"
ARTIFACT = EVIDENCE / "media-functional.json"
CLIENT_ID = str(uuid.uuid4())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def request_json(path: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def free_models() -> None:
    data = json.dumps({"unload_models": True, "free_memory": True}).encode()
    request = urllib.request.Request(
        BASE + "/free",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120):
        pass
    time.sleep(5)


def submit(prompt: dict, timeout: int = 1800) -> dict:
    response = request_json("/prompt", {"prompt": prompt, "client_id": CLIENT_ID})
    if response.get("node_errors"):
        raise RuntimeError(f"workflow rejected: {response['node_errors']}")
    prompt_id = response["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = request_json(f"/history/{prompt_id}", timeout=30)
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error" or not status.get("completed", False):
                raise RuntimeError(f"workflow failed: {status}")
            return record
        time.sleep(2)
    raise TimeoutError(f"workflow {prompt_id} did not finish")


def output_files(record: dict) -> list[Path]:
    files = []
    for node in record.get("outputs", {}).values():
        for value in node.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                subfolder = item.get("subfolder", "")
                path = (OUTPUTS / subfolder / item["filename"]).resolve()
                if OUTPUTS.resolve() not in path.parents:
                    raise RuntimeError(f"output escaped evidence directory: {path}")
                files.append(path)
    return files


def image_report(path: Path, source: Path | None = None) -> dict:
    with Image.open(path) as image:
        image = image.convert("RGB")
        stat = ImageStat.Stat(image)
        report = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "format": path.suffix.lower(),
            "width": image.width,
            "height": image.height,
            "channel_stddev": [round(value, 3) for value in stat.stddev],
        }
        varied = max(stat.stddev) > 12 and path.stat().st_size > 10_000
        report["nontrivial"] = varied
        if source is not None:
            with Image.open(source) as original:
                original = original.convert("RGB").resize(image.size)
                center = (image.width // 4, image.height // 4, 3 * image.width // 4, 3 * image.height // 4)
                edited_center = ImageStat.Stat(image.crop(center)).mean
                source_center = ImageStat.Stat(original.crop(center)).mean
                report["center_rgb"] = [round(value, 2) for value in edited_center]
                report["source_center_rgb"] = [round(value, 2) for value in source_center]
                report["instruction_blue_over_red"] = edited_center[2] > edited_center[0] * 1.15
                report["changed"] = sum(abs(a - b) for a, b in zip(edited_center, source_center)) > 35
        return report


def audio_report(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    duration = None
    if result.returncode == 0:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "duration_seconds": duration,
        "nontrivial": path.stat().st_size > 100_000 and duration is not None and duration >= 8,
        "ffprobe_exit_code": result.returncode,
    }


def z_image_workflow() -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "A blue ceramic robot beside a yellow sunflower, clean studio lighting, detailed photograph"}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": 424242, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["8", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "functional/z-image"}},
    }


def flux2_klein_workflow() -> dict:
    """API graph for the installed FLUX.2 Klein 4B native ComfyUI checkpoint.

    Not executed by main() until a live ComfyUI run is admitted. The VAE file is
    the publisher Diffusers name under extra_model_paths vae/.
    """
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "A blue ceramic robot beside a yellow sunflower, clean studio lighting, detailed photograph"}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0], "seed": 4242, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": "diffusion_pytorch_model.safetensors"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["7", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "functional/flux2-klein"}},
    }


def music_workflow() -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ace_step_1.5_turbo_aio.safetensors"}},
        "2": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["1", 1], "tags": "instrumental ambient synth, warm pads, gentle pulse", "lyrics": "", "seed": 8181, "bpm": 90, "duration": 10.0, "timesignature": "4", "language": "en", "keyscale": "C major", "generate_audio_codes": True, "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "3": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}},
        "4": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 10.0, "batch_size": 1}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": 8181, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveAudio", "inputs": {"audio": ["7", 0], "filename_prefix": "functional/ace-step"}},
    }


def make_edit_source() -> Path:
    INPUTS.mkdir(parents=True, exist_ok=True)
    path = INPUTS / "qwen-edit-source.png"
    image = Image.new("RGB", (512, 512), (230, 230, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((128, 128, 384, 384), fill=(210, 35, 35), outline=(40, 40, 40), width=8)
    draw.line((0, 64, 512, 64), fill=(30, 30, 30), width=6)
    draw.line((64, 0, 64, 512), fill=(30, 30, 30), width=6)
    image.save(path)
    return path


def upload_image(path: Path) -> str:
    boundary = f"----frankenstein-{uuid.uuid4().hex}"
    chunks = []
    fields = (("type", "input"), ("overwrite", "true"))
    for name, value in fields:
        chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\nContent-Type: image/png\r\n\r\n".encode())
    chunks.append(path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(BASE + "/upload/image", data=b"".join(chunks), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["name"]


def edit_workflow(image_name: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_int8_convrot.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["3", 0], "shift": 3.1}},
        "5": {"class_type": "CFGNorm", "inputs": {"model": ["4", 0], "strength": 1.0}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["5", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["7", 0], "vae": ["8", 0], "image1": ["2", 0], "prompt": "Change only the large red square to a saturated blue square. Preserve the beige background, dark border, and guide lines."}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["7", 0], "vae": ["8", 0], "image1": ["2", 0], "prompt": ""}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["9", 0], "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["10", 0], "reference_latents_method": "index_timestep_zero"}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["8", 0]}},
        "14": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["11", 0], "negative": ["12", 0], "latent_image": ["13", 0], "seed": 2511, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["8", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "functional/qwen-edit"}},
    }


def atomic_write(report: dict) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    temp = ARTIFACT.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp, ARTIFACT)


def main() -> int:
    report = {
        "gate": "generative-media-functional",
        "server": BASE,
        "benchmarking_performed": False,
        "throughput_measured": False,
        "started_at": now(),
        "workflows": {},
        "problems": [],
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        record = submit(z_image_workflow())
        files = [path for path in output_files(record) if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if not files:
            raise RuntimeError("Z-Image produced no image file")
        check = image_report(files[0])
        report["workflows"]["image-generation"] = check
        if not check["nontrivial"]:
            report["problems"].append("Z-Image output was trivial")
        atomic_write(report)
        free_models()

        record = submit(music_workflow())
        files = output_files(record)
        if not files:
            raise RuntimeError("ACE-Step produced no audio file")
        check = audio_report(files[0])
        report["workflows"]["music-generation"] = check
        if not check["nontrivial"]:
            report["problems"].append("ACE-Step output failed duration/content checks")
        atomic_write(report)
        free_models()

        source = make_edit_source()
        uploaded = upload_image(source)
        record = submit(edit_workflow(uploaded))
        files = [path for path in output_files(record) if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if not files:
            raise RuntimeError("Qwen Image Edit produced no image file")
        check = image_report(files[0], source)
        report["workflows"]["image-editing"] = check
        if not check["nontrivial"] or not check.get("changed") or not check.get("instruction_blue_over_red"):
            report["problems"].append("Qwen Image Edit output did not prove the requested red-to-blue edit")
        free_models()
    except Exception as error:  # noqa: BLE001 - durable evidence on every failure
        report["problems"].append(f"{type(error).__name__}: {error}"[:2000])
    report["finished_at"] = now()
    report["pass"] = not report["problems"] and set(report["workflows"]) == {"image-generation", "music-generation", "image-editing"}
    atomic_write(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
