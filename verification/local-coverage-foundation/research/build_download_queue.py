#!/usr/bin/env python3
"""Build the exact artifact download queue from collected Hugging Face metadata.

Every size and SHA-256 is copied from research/hf/*.json, which was populated
straight from the Hugging Face model API. Nothing here is hand-typed, so the
queue cannot drift from the publisher tree that was actually observed.
"""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "research" / "hf"
MODELS = Path("/home/typhoon/git/frankenstein-llm/models")
OUT = ROOT / "download-queue.json"

# key, capability, repository, destination directory, file selector, notes
SELECTION = [
    (
        "embedding-qwen3-8b-q6k",
        "embeddings",
        "Qwen/Qwen3-Embedding-8B-GGUF",
        MODELS / "embedding",
        ["Qwen3-Embedding-8B-Q6_K.gguf"],
        "Official Qwen GGUF. Served by the existing stable llama.cpp with "
        "--embedding --pooling last. Q6_K keeps 4096-dim quality while leaving "
        "room to co-reside with the reranker.",
    ),
    (
        "reranker-qwen3-8b-source",
        "reranking",
        "Qwen/Qwen3-Reranker-8B",
        MODELS / "reranker-src" / "Qwen3-Reranker-8B",
        None,  # complete snapshot minus .gitattributes
        "Publisher safetensors snapshot. Qwen ships no reranker GGUF and community "
        "conversions commonly drop cls.output.weight and score garbage, so the GGUF "
        "is produced locally by llama.cpp convert_hf_to_gguf.py, which detects "
        "Qwen3-Reranker, keeps the classifier head and sets pooling_type=RANK.",
    ),
    (
        "ocr-hunyuanocr-bf16",
        "ocr",
        "ggml-org/HunyuanOCR-GGUF",
        MODELS / "ocr",
        ["HunyuanOCR-bf16.gguf", "mmproj-HunyuanOCR-bf16.gguf"],
        "First-party ggml-org conversion, so llama.cpp mtmd support is authoritative "
        "rather than inferred. 1B class: bf16 costs little and avoids quantisation "
        "damage on dense-layout character recognition.",
    ),
    (
        "asr-qwen3-1.7b-hf",
        "asr",
        "Qwen/Qwen3-ASR-1.7B-hf",
        MODELS / "asr" / "Qwen3-ASR-1.7B-hf",
        None,
        "Transformers-native snapshot. Runs in a dedicated ROCm venv; there is no "
        "llama.cpp path for this architecture.",
    ),
    (
        "tts-qwen3-12hz-1.7b-base",
        "tts",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        MODELS / "tts" / "Qwen3-TTS-12Hz-1.7B-Base",
        None,
        "Complete snapshot including the bundled speech_tokenizer/ codec, so the "
        "separate Qwen3-TTS-Tokenizer-12Hz repository is not required.",
    ),
    (
        "music-acestep-1.5-turbo-aio",
        "music",
        "Comfy-Org/ace_step_1.5_ComfyUI_files",
        MODELS / "comfy" / "checkpoints",
        ["checkpoints/ace_step_1.5_turbo_aio.safetensors"],
        "All-in-one turbo checkpoint: bundles diffusion, text encoder and VAE, so no "
        "split_files dependency is downloaded.",
    ),
    (
        "image-z-image-turbo-bf16",
        "image",
        "Comfy-Org/z_image_turbo",
        MODELS / "comfy",
        [
            "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            "split_files/text_encoders/qwen_3_4b.safetensors",
            "split_files/vae/ae.safetensors",
        ],
        "bf16 weights chosen deliberately: gfx1030 is RDNA2 and has no FP8/FP4/INT8 "
        "tensor acceleration, so the int8_convrot and nvfp4 variants would add "
        "conversion risk without a speed win.",
    ),
    (
        "fim-qwen2.5-coder-7b-q8",
        "fim",
        "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
        MODELS / "fim",
        ["qwen2.5-coder-7b-q8_0.gguf"],
        "Exactly the artifact the installed llama-server blesses via "
        "--fim-qwen-7b-default, so /infill prefix/suffix/middle tokens are known good "
        "rather than assumed.",
    ),
]

SKIP = {".gitattributes"}


def flatten(destination: Path, repo_path: str, whole_snapshot: bool) -> Path:
    """Snapshots keep their tree; cherry-picked files land in a flat category dir."""
    if whole_snapshot:
        return destination / repo_path
    return destination / Path(repo_path).name


def main() -> int:
    artifacts = []
    for key, capability, repository, destination, selector, notes in SELECTION:
        document = json.loads((HF / f"{repository.replace('/', '__')}.json").read_text())
        whole_snapshot = selector is None
        wanted = [f for f in document["files"] if f["path"] not in SKIP]
        if not whole_snapshot:
            index = {f["path"]: f for f in document["files"]}
            missing = [p for p in selector if p not in index]
            if missing:
                raise SystemExit(f"{repository}: selected paths absent from tree: {missing}")
            wanted = [index[p] for p in selector]

        # z_image_turbo keeps ComfyUI's split_files/<kind>/ layout under models/comfy.
        keep_tree = whole_snapshot or key == "image-z-image-turbo-bf16"
        files = []
        for f in wanted:
            relative = f["path"]
            if key == "image-z-image-turbo-bf16":
                relative = relative.removeprefix("split_files/")
            local = (destination / relative) if keep_tree else flatten(destination, f["path"], False)
            files.append({
                "repo_path": f["path"],
                "destination": str(local),
                "size": f["size"],
                "sha256": f["sha256"],
            })

        artifacts.append({
            "key": key,
            "capability": capability,
            "repository": repository,
            "revision": document["revision"],
            "license": document["license"],
            "gated": document["gated"],
            "metadata_collected_at": document["collected_at"],
            "notes": notes,
            "total_bytes": sum(f["size"] for f in files),
            "files": files,
        })

    queue = {
        "schema": "hermes-hf-artifact-queue/1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_of_truth": str(HF),
        "policy": {
            "single_writer": True,
            "resume": "curl --continue-at - against a .partial sibling",
            "promotion": "os.replace only after exact size and, when published, SHA-256 match",
            "hash_available": "LFS files only; small config/tokenizer files are size-gated",
        },
        "total_bytes": sum(a["total_bytes"] for a in artifacts),
        "artifacts": artifacts,
    }
    OUT.write_text(json.dumps(queue, indent=2) + "\n")
    for artifact in artifacts:
        hashed = sum(1 for f in artifact["files"] if f["sha256"])
        print(
            f"{artifact['key']:34} {artifact['total_bytes']:>14,}  "
            f"files={len(artifact['files']):>2} hashed={hashed:>2}  {artifact['repository']}"
        )
    print(f"{'TOTAL':34} {queue['total_bytes']:>14,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
