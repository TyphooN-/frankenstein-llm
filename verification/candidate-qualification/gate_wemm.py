#!/usr/bin/env python3
"""Functional WeMM embedding gate. Passive item-rate observations. Separate 2048-D space."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import qualification_performance as performance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_policy as policy  # noqa: E402
import wemm_remote_code_review as review  # noqa: E402

MODEL_DIR = review.MODEL_DIR
EVIDENCE = HERE / "evidence"
ARTIFACT = EVIDENCE / "wemm-functional.json"
EXPECTED_DIM = policy.MULTIMODAL_INDEX["dimension"]
ANCHOR = "The ZFS scrub repaired zero bytes and reported no known data errors."
PARAPHRASE = "A scrub of the ZFS pool finished with nothing repaired and no data errors found."
RELATED = "The router keeps at most one large language model resident in VRAM."
NOISE = "Sourdough needs a long cold retard before the loaf is shaped and baked."


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def l2(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def vram_used() -> dict[str, int]:
    values: dict[str, int] = {}
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        path = card / "device" / "mem_info_vram_used"
        try:
            values[card.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return values


def to_list(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().float().cpu().reshape(-1).tolist()]


def write_atomic(payload: dict) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    temp = ARTIFACT.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, ARTIFACT)


def run() -> dict:
    summary: dict = {
        "gate": "wemm-functional",
        "benchmarking_performed": False,
        "benchmark_performed": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "problems": [],
        "checks": {},
        "index": {
            "alias": policy.MULTIMODAL_INDEX["alias"],
            "database": str(policy.MULTIMODAL_INDEX["database"]),
            "dimension": EXPECTED_DIM,
            "text_index_database": str(policy.TEXT_INDEX["database"]),
        },
    }
    conflicts = policy.index_conflicts()
    if conflicts:
        summary["problems"].extend(conflicts)
        summary["pass"] = False
        return summary

    remote = review.scan()
    summary["remote_code"] = {
        "pass": remote["pass"],
        "execution_permitted": review.execution_permitted(remote),
        "approved_for_import": remote.get("approved_for_import"),
        "problems": remote.get("problems"),
    }
    if not review.execution_permitted(remote):
        summary["problems"].append("remote-code review refused execution")
        summary["pass"] = False
        return summary

    import torch
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        device = torch.device("cuda:1")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    summary["device"] = str(device)
    summary["torch_hip"] = bool(getattr(torch.version, "hip", None))
    summary["before_vram"] = vram_used()
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        model = AutoModel.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).to(device)
        model.eval()

        def embed_texts(texts: list[str]):
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                return performance.call(model.embedding, operation="embedding", model_id="WeMM-Embedding-2B", mode="items", **encoded)

        vectors = embed_texts([ANCHOR, PARAPHRASE, RELATED, NOISE])
        anchor, paraphrase, related, noise = (to_list(row) for row in vectors)
        summary["text_dimension"] = len(anchor)
        summary["text_l2_norm"] = round(l2(anchor), 6)
        if len(anchor) != EXPECTED_DIM:
            summary["problems"].append(f"text dim {len(anchor)} != {EXPECTED_DIM}")
        if abs(l2(anchor) - 1.0) > 5e-3:
            summary["problems"].append(f"text embedding is not L2-normalised ({l2(anchor)})")
        scores = {
            "paraphrase": cosine(anchor, paraphrase),
            "related": cosine(anchor, related),
            "noise": cosine(anchor, noise),
        }
        summary["similarity"] = {key: round(value, 6) for key, value in scores.items()}
        if not (scores["paraphrase"] > scores["related"] > scores["noise"]):
            summary["problems"].append(f"semantic ordering violated: {scores}")
        again = to_list(embed_texts([ANCHOR])[0])
        drift = max(abs(x - y) for x, y in zip(anchor, again))
        summary["repeat_max_abs_drift"] = drift
        if drift > 1e-3:
            summary["problems"].append(f"repeated text embedding drifted by {drift}")
        summary["checks"]["text"] = (
            len(anchor) == EXPECTED_DIM
            and abs(l2(anchor) - 1.0) <= 5e-3
            and scores["paraphrase"] > scores["related"] > scores["noise"]
            and drift <= 1e-3
        )

        summary["checks"]["image"] = False
        summary["image_status"] = (
            "not claimed: AutoProcessor requires torchvision, which is not "
            "installed in the ROCm candidate runtime"
        )
        summary["loaded_vram"] = vram_used()
    except Exception as error:  # noqa: BLE001
        summary["problems"].append(f"{type(error).__name__}: {error}"[:800])
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary["after_vram"] = vram_used()
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["pass"] = not summary["problems"] and summary.get("checks", {}).get("text") is True
    return summary


def main() -> int:
    summary = run()
    write_atomic(summary)
    print(json.dumps({
        "pass": summary["pass"],
        "problems": summary["problems"],
        "checks": summary.get("checks"),
        "text_dimension": summary.get("text_dimension"),
        "image_dimension": summary.get("image_dimension"),
        "similarity": summary.get("similarity"),
        "device": summary.get("device"),
    }, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
