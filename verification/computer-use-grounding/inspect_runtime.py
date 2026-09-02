#!/usr/bin/env python3
"""Static runtime-compatibility probe for UI-TARS-1.5-7B on this host.

Establishes everything that can be known *without* allocating GPU memory or
reading weights: interpreter/torch/ROCm versions, device inventory, whether
transformers can construct the processor and chat template from the local
checkout, and the residency arithmetic that decides which GPU the gate targets.

It also cross-checks ``culib.smart_resize`` against the real Qwen2-VL image
processor. The gate's grounding verdict depends on mapping model coordinates
back to fixture pixels; if that mapping is wrong the verdict is meaningless, so
it is verified against the library rather than assumed.

No GPU allocation is performed here by design -- this runs safely on a busy host.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from culib import (  # noqa: E402
    EVIDENCE, FIXTURES, MODEL_DIR, image_token_count, load_processor,
    memory_sample, record, smart_resize,
)
from gate_computer_use import (  # noqa: E402
    ARTIFACT_NAME, RUN_STATE_PATH, boot_id, record_atomic,
)
from gpu_telemetry import summarize_records  # noqa: E402

BYTES_PER_F32 = 4
BYTES_PER_BF16 = 2

UNIT = "local-ai-computer-use-gate.service"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _unit_state(unit: str = UNIT) -> dict:
    fields = ("LoadState", "ActiveState", "SubState", "Result", "ExecMainPID",
              "ExecMainStatus", "ExecMainCode", "NRestarts", "InvocationID",
              "ActiveEnterTimestamp", "InactiveEnterTimestamp")
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=" + ",".join(fields)],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        return {"unit": unit, "error": f"{type(error).__name__}: {error}"[:200]}
    state = {"unit": unit}
    for line in out.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            state[key] = value
    return state


def classify_last_run(run_state: dict | None, current_boot: str | None,
                      artifact: dict | None) -> dict:
    """Explain what happened to the previous run without guessing.

    The distinction that matters is between "the gate ended and said so" and
    "the gate stopped existing". A breadcrumb left at a mid-run phase carrying a
    boot id that is no longer current is the signature of the host going down
    under the run -- no signal was delivered, no exit path ran, and no artifact
    could have been written. That is not the same failure as a killed process,
    and conflating them sends the fix to the wrong place.
    """
    if run_state is None:
        return {"state": "no-run-state",
                "detail": "no run has recorded a breadcrumb since this file was last removed"}
    phase = run_state.get("phase")
    same_boot = bool(current_boot and run_state.get("boot_id") == current_boot)
    pid = run_state.get("pid")
    alive = False
    if same_boot and isinstance(pid, int):
        alive = Path(f"/proc/{pid}").exists()
    if run_state.get("finished"):
        return {"state": "finished", "phase": phase,
                "outcome": run_state.get("outcome"),
                "exit_code": run_state.get("exit_code"),
                "artifact_pass": bool(artifact and artifact.get("pass"))}
    if not same_boot:
        return {"state": "vanished-across-reboot", "phase": phase,
                "detail": f"breadcrumb stopped at phase={phase} under boot "
                          f"{run_state.get('boot_id')}, which is no longer the running "
                          f"kernel; the host went down mid-run and no exit path ran",
                "artifact_pass": bool(artifact and artifact.get("pass"))}
    if alive:
        return {"state": "running", "phase": phase, "pid": pid}
    return {"state": "vanished-same-boot", "phase": phase, "pid": pid,
            "detail": "process is gone with the breadcrumb unfinished; consistent with "
                      "SIGKILL or an OOM kill, neither of which can be handled"}


def lifecycle_report() -> dict:
    """Everything the control plane needs, without loading or allocating anything."""
    artifact_path = EVIDENCE / f"{ARTIFACT_NAME}.json"
    telemetry_path = EVIDENCE / "computer-use-gpu-telemetry.jsonl"
    artifact = _read_json(artifact_path)
    run_state = _read_json(RUN_STATE_PATH)
    runner_result = _read_json(EVIDENCE / "computer-use-runner-result.json")
    current_boot = boot_id()

    summary = {
        "gate": "computer-use-lifecycle",
        "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "current_boot_id": current_boot,
        "gpu_allocation_performed": False,
        "model_loaded": False,
        "unit": _unit_state(),
        "run_state": run_state,
        "runner_result": runner_result,
        "artifact": {
            "path": str(artifact_path),
            "exists": artifact_path.exists(),
            "readable": artifact is not None,
            "recorded_at": (artifact or {}).get("recorded_at"),
            "outcome": (artifact or {}).get("outcome"),
            "pass": bool(artifact and artifact.get("pass")),
            "section_verdicts": (artifact or {}).get("section_verdicts"),
            "interrupt": (artifact or {}).get("interrupt"),
            "error": (artifact or {}).get("error"),
        },
    }
    summary["last_run"] = classify_last_run(run_state, current_boot, artifact)
    if telemetry_path.exists():
        rebuilt = summarize_records(telemetry_path)
        summary["telemetry"] = {
            "path": str(telemetry_path),
            "records": rebuilt["sample_count"],
            "duration_seconds": rebuilt["duration_seconds"],
            "cards_reporting": rebuilt["cards_reporting"],
            "busy_max_percent": {card: values["busy_max_percent"]
                                 for card, values in rebuilt["cards"].items()},
            "busy_samples_ge_20": {card: values["busy_samples_ge_20"]
                                   for card, values in rebuilt["cards"].items()},
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                   time.localtime(os.stat(telemetry_path).st_mtime)),
            "note": "rebuilt from the on-disk JSONL; the baseline is the first record, "
                    "so deltas here are weaker than the gate's own aggregate",
        }
    # The single rule the control plane must obey.
    summary["control_plane_success"] = bool(artifact and artifact.get("pass"))
    summary["pass"] = summary["control_plane_success"]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-only", action="store_true",
                        help="report gate lifecycle state only: unit, run-state "
                             "breadcrumb, artifact verdict and telemetry. Imports no "
                             "torch, loads no model, touches no GPU")
    args = parser.parse_args()
    if args.lifecycle_only:
        return record_atomic("computer-use-lifecycle", lifecycle_report())

    summary = {"gate": "runtime-compatibility", "problems": []}
    problems = summary["problems"]

    import torch
    import transformers

    summary["interpreter"] = {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "venv": "/home/typhoon/git/frankenstein-llm/venvs/computer-use",
        "wiring": "venv created with --system-site-packages (system ROCm torch) plus "
                  "symlinks to the pre-existing transformers/tokenizers in venvs/convert; "
                  "no packages were downloaded or compiled",
    }
    summary["torch"] = {
        "version": torch.__version__,
        "hip": torch.version.hip,
        "path": torch.__file__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
    }
    summary["transformers"] = {
        "version": transformers.__version__,
        "path": transformers.__file__,
    }
    for optional in ("accelerate", "qwen_vl_utils", "flash_attn", "bitsandbytes"):
        summary.setdefault("optional_packages", {})[optional] = bool(
            importlib.util.find_spec(optional))

    if not summary["torch"]["cuda_available"]:
        problems.append("torch reports no ROCm device")

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "arch": props.gcnArchName,
            "total_bytes": props.total_memory,
            "total_gib": round(props.total_memory / 2**30, 2),
            "multi_processor_count": props.multi_processor_count,
        })
    summary["devices"] = devices

    # --- residency arithmetic ------------------------------------------------
    index_json = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    on_disk_bytes = index_json["metadata"]["total_size"]
    parameters = on_disk_bytes // BYTES_PER_F32
    bf16_bytes = parameters * BYTES_PER_BF16
    summary["residency"] = {
        "parameters": parameters,
        "on_disk_bytes_f32": on_disk_bytes,
        "weights_bytes_bf16": bf16_bytes,
        "weights_gib_bf16": round(bf16_bytes / 2**30, 2),
        "note": "shards are F32 on disk; the loader casts to bf16 so resident VRAM is "
                "about half the file size",
    }
    summary["residency"]["placement_policy"] = (
        "all three GPUs; maximize headless GPU0/GPU1 residency, GPU1 > GPU0, "
        "display-connected GPU2 smallest")
    summary["residency"]["expected_unique_gib"] = {
        "GPU0": 5.31, "GPU1": 6.51, "GPU2": 2.60}
    summary["residency"]["gpu2_execution"] = (
        "decoder layers 22-27; lm_head is tied to GPU0 embed_tokens and is not "
        "counted as independent residency")

    # --- processor / chat template ------------------------------------------
    # AutoProcessor is unusable here: it constructs an AutoVideoProcessor which
    # hard-requires torchvision. torchvision is absent and this phase may not
    # download anything, so the gate drives the image processor and tokenizer
    # directly. That is a recorded runtime constraint, not a model defect.
    summary["torchvision_present"] = bool(importlib.util.find_spec("torchvision"))
    summary["processor_strategy"] = (
        "Qwen2VLImageProcessor + AutoTokenizer + chat_template.json, with manual "
        "<|image_pad|> expansion (culib.build_inputs); AutoProcessor avoided because "
        "AutoVideoProcessor requires absent torchvision"
    )
    image_processor = None
    try:
        image_processor, tokenizer, chat_template = load_processor()
        summary["processor"] = {
            "loaded": True,
            "image_processor": type(image_processor).__name__,
            "tokenizer": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "patch_size": image_processor.patch_size,
            "merge_size": image_processor.merge_size,
            "min_pixels": getattr(image_processor, "min_pixels", None),
            "max_pixels": getattr(image_processor, "max_pixels", None),
        }
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": "SYS"},
             {"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": "PROBE"}]}],
            tokenize=False, add_generation_prompt=True, chat_template=chat_template)
        summary["processor"]["chat_template_renders"] = bool(rendered)
        summary["processor"]["chat_template_sample"] = rendered[:260]
        for token in ("<|vision_start|>", "<|image_pad|>", "<|im_start|>assistant"):
            if token not in rendered:
                problems.append(f"chat template omitted {token}")
        # The tokenizer_config template is the wrong one for image turns; prove the
        # gate would notice if it were ever selected by accident.
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": "X"}]}],
                tokenize=False, add_generation_prompt=True)
            summary["tokenizer_config_template_handles_list_content"] = True
        except Exception as error:  # noqa: BLE001
            summary["tokenizer_config_template_handles_list_content"] = False
            summary["tokenizer_config_template_error"] = f"{type(error).__name__}: {error}"[:200]
    except Exception as error:  # noqa: BLE001
        summary["processor"] = {"loaded": False,
                                "error": f"{type(error).__name__}: {error}"}
        problems.append(f"processor construction failed: {type(error).__name__}: {error}")

    # --- coordinate-frame cross-check ---------------------------------------
    checks = []
    try:
        from PIL import Image
        for name in ("screen-app.png", "screen-distractor.png",
                     "screen-injection.png", "ocr-document.png"):
            path = FIXTURES / name
            if not path.exists():
                continue
            with Image.open(path) as handle:
                width, height = handle.size
                predicted = smart_resize(width, height)
                entry = {"fixture": name, "original_size": [width, height],
                         "culib_smart_resize": list(predicted)}
                if image_processor is not None:
                    vision = image_processor(images=handle.convert("RGB"), return_tensors="pt")
                    grid = vision["image_grid_thw"][0].tolist()
                    actual = (grid[2] * image_processor.patch_size,
                              grid[1] * image_processor.patch_size)
                    entry["image_grid_thw"] = grid
                    entry["processor_resize"] = list(actual)
                    entry["agrees"] = actual == predicted
                    entry["vision_tokens"] = image_token_count(grid, image_processor.merge_size)
                    if not entry["agrees"]:
                        problems.append(
                            f"{name}: culib.smart_resize {predicted} != processor {actual}")
                checks.append(entry)
    except Exception as error:  # noqa: BLE001
        problems.append(f"resize cross-check failed: {type(error).__name__}: {error}")
    summary["coordinate_frame_checks"] = checks

    # --- load plan -----------------------------------------------------------
    summary["load_plan"] = {
        "class": "Qwen2_5_VLForConditionalGeneration",
        "dtype": "torch.bfloat16",
        "device_map": "explicit GPU0/GPU1-heavy, GPU2-minimal map in gate_computer_use.py",
        "accelerate_required": True,
        "expected_host_peak_bytes": 5 * 1024**3,
        "expected_host_peak_gib": 5.0,
        "host_available_bytes": memory_sample("load-plan")["mem_available_bytes"],
        "attn_implementation": "eager",
        "reason_eager": "flash_attn is absent and gfx1030 has no upstream FA2 build",
    }
    if summary["load_plan"]["expected_host_peak_bytes"] > \
            summary["load_plan"]["host_available_bytes"]:
        summary["load_plan"]["host_headroom_warning"] = (
            "projected CPU-staging peak exceeds currently available RAM; run the gate "
            "on an idle host or install accelerate for direct-to-GPU streaming")

    summary["memory"] = memory_sample("runtime-inspection")
    summary["gpu_allocation_performed"] = False
    summary["pass"] = not problems
    return record("runtime-compatibility", summary)


if __name__ == "__main__":
    raise SystemExit(main())
