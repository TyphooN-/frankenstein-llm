#!/usr/bin/env python3
"""Build phase four from the reviewed candidate metadata.

The metadata inputs come directly from the Hugging Face tree API through
collect_hf_metadata.py. Selections intentionally avoid redundant quant families
and duplicate packaging while retaining every file needed by the chosen runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "research" / "hf"
MODELS = Path("/home/typhoon/git/frankenstein-llm/models")
OUT = ROOT / "download-queue-phase4.json"
EXPECTED_REVISIONS = {
    "Qwen/Qwen3-Coder-Next-GGUF": "b82fb7382639d97b38fa7672e526c760c2fb358e",
    "SC117/Gemma-4-12B-it-heretic-GGUF": "efa14611b0b04ab1ab1e38356596ac8d673a619a",
    "tencent/UI-Mate-9B": "05dd5f2975195a5bb03d4363e8767f12158c8421",
    "tencent/WeMM-Embedding-2B": "bbd6cd4bf52cfc6716f752a2df80b2706720bd95",
    "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
}


def metadata(repository: str) -> dict:
    document = json.loads((HF / f"{repository.replace('/', '__')}.json").read_text())
    expected = EXPECTED_REVISIONS[repository]
    if document.get("revision") != expected:
        raise SystemExit(
            f"{repository}: expected reviewed revision {expected}, got {document.get('revision')}")
    return document


def select_prefix(document: dict, prefix: str) -> list[dict]:
    return [item for item in document["files"] if item["path"].startswith(prefix)]


def select_paths(document: dict, paths: list[str]) -> list[dict]:
    index = {item["path"]: item for item in document["files"]}
    missing = sorted(set(paths) - set(index))
    if missing:
        raise SystemExit(f"{document['repository']}: selected paths absent: {missing}")
    return [index[path] for path in paths]


def runtime_snapshot(document: dict, excluded: set[str]) -> list[dict]:
    return [item for item in document["files"] if item["path"] not in excluded]


def artifact(*, key: str, capability: str, repository: str, destination: Path,
             selected: list[dict], notes: str, keep_tree: bool = True,
             license_override: str | None = None) -> dict:
    document = metadata(repository)
    files = []
    for item in selected:
        relative = Path(item["path"]) if keep_tree else Path(item["path"]).name
        files.append({
            "repo_path": item["path"],
            "destination": str(destination / relative),
            "size": item["size"],
            "sha256": item["sha256"],
        })
    return {
        "key": key,
        "capability": capability,
        "repository": repository,
        "revision": document["revision"],
        "license": license_override or document["license"],
        "gated": document["gated"],
        "metadata_collected_at": document["collected_at"],
        "notes": notes,
        "total_bytes": sum(int(item["size"]) for item in files),
        "files": files,
    }


def main() -> int:
    coder_repo = "Qwen/Qwen3-Coder-Next-GGUF"
    coder = metadata(coder_repo)
    gemma_repo = "SC117/Gemma-4-12B-it-heretic-GGUF"
    gemma = metadata(gemma_repo)
    ui_repo = "tencent/UI-Mate-9B"
    ui = metadata(ui_repo)
    wemm_repo = "tencent/WeMM-Embedding-2B"
    wemm = metadata(wemm_repo)
    flux_repo = "black-forest-labs/FLUX.2-klein-4B"
    flux = metadata(flux_repo)

    flux_paths = [
        "flux-2-klein-4b.safetensors",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]

    artifacts = [
        artifact(
            key="repository-agent-qwen3-coder-next-q4km",
            capability="repository-agent",
            repository=coder_repo,
            destination=MODELS / "repository-agent" / "Qwen3-Coder-Next-Q4_K_M",
            selected=select_prefix(coder, "Qwen3-Coder-Next-Q4_K_M/"),
            notes="Official Q4_K_M GGUF. Highest official quant that can use the aggregate "
                  "GPU pool with bounded host/display spill; Qwen2.5 Coder remains the FIM model.",
            keep_tree=False,
        ),
        artifact(
            key="uncensored-multimodal-gemma4-heretic-q6k",
            capability="uncensored-multimodal",
            repository=gemma_repo,
            destination=MODELS / "gemma4-heretic",
            selected=select_paths(gemma, [
                "Gemma-4-12B-it-heretic-Q6_K.gguf",
                "mmproj-Gemma-4-12B-it-BF16.gguf",
            ]),
            notes="Q6_K text weights plus the publisher's BF16 multimodal projector. "
                  "Low-privilege serving only; modality support must be qualified independently.",
            keep_tree=False,
        ),
        artifact(
            key="computer-use-ui-mate-9b",
            capability="computer-use-grounding",
            repository=ui_repo,
            destination=MODELS / "computer-use" / "UI-Mate-9B",
            selected=runtime_snapshot(ui, {".gitattributes", "README.md"}),
            notes="Complete pinned runtime snapshot for an A/B against UI-TARS after the "
                  "existing grounding baseline. No privileged control before grounding passes.",
        ),
        artifact(
            key="multimodal-embedding-wemm-2b",
            capability="multimodal-embeddings",
            repository=wemm_repo,
            destination=MODELS / "embedding" / "WeMM-Embedding-2B",
            selected=runtime_snapshot(wemm, {".gitattributes", "README.md"}),
            notes="Complete pinned runtime snapshot including remote model code. The current "
                  "LICENSE explicitly grants Apache-2.0 despite the Hub's generic 'other' tag. "
                  "Use a separate multimodal vector index.",
            license_override="apache-2.0 (repository LICENSE)",
        ),
        artifact(
            key="image-edit-flux2-klein-4b",
            capability="image-generation-editing",
            repository=flux_repo,
            destination=MODELS / "comfy" / "flux2-klein-4b",
            selected=select_paths(flux, flux_paths),
            notes="Root ComfyUI checkpoint plus official text encoder, tokenizer and VAE. "
                  "The duplicate Diffusers transformer copy and sample images are excluded.",
        ),
    ]

    queue = {
        "schema": "hermes-hf-artifact-queue/1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_of_truth": str(HF),
        "policy": {
            "single_writer": True,
            "resume": "aria2 bounded multi-range continuation with its piece map; curl fallback",
            "promotion": "fsync content, replace, and fsync parent only after exact size and SHA-256 when published",
            "hash_available": "LFS files use publisher SHA-256; small runtime files are exact-size gated",
            "selection": "one useful runtime artifact per reviewed backlog candidate; no redundant quant families",
        },
        "already_satisfied": [{
            "key": "asr-qwen3-1.7b-hf",
            "repository": "Qwen/Qwen3-ASR-1.7B-hf",
            "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
            "source_queue": "download-queue.json",
        }],
        "total_bytes": sum(item["total_bytes"] for item in artifacts),
        "artifacts": artifacts,
    }
    OUT.write_text(json.dumps(queue, indent=2) + "\n")
    for item in artifacts:
        hashed = sum(file["sha256"] is not None for file in item["files"])
        print(f"{item['key']:43} {item['total_bytes']:>14,} files={len(item['files']):>2} hashed={hashed:>2}")
    print(f"{'TOTAL':43} {queue['total_bytes']:>14,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
