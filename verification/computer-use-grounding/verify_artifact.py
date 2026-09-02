#!/usr/bin/env python3
"""Bounded integrity and provenance check for the UI-TARS-1.5-7B artifact.

Deliberately does NOT re-hash the 33 GB of shards. The downloader
(`local-coverage-foundation/download_queue.py`) promotes a shard with
``os.replace`` only after an exact size *and* SHA-256 match against the queue
manifest, so a full re-read would re-prove a property that was already proven at
write time -- at the cost of 33 GB of I/O. This host is currently linking a
kernel, so the check is restricted to operations that touch kilobytes:

  * stat every file in the manifest and compare exact sizes
  * parse each shard's safetensors header (a header read, not a weight read)
  * prove every tensor's byte range lies inside its shard
  * cross-check ``model.safetensors.index.json`` against the headers
  * read the last byte of each shard to prove it is fully allocated and readable

A full re-hash remains available for an idle host via ``--full-hash``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from culib import MODEL_DIR, record  # noqa: E402

QUEUE = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/"
             "download-queue-phase2.json")
ARTIFACT_KEY = "computer-use-ui-tars-1.5-7b"

# Anything wider than this is a weight read, not a header read.
MAX_HEADER_BYTES = 64 * 1024 * 1024


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header, header_end_offset) for a safetensors file."""
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path.name}: file shorter than the 8-byte length prefix")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= MAX_HEADER_BYTES:
            raise ValueError(f"{path.name}: implausible header length {length}")
        header = json.loads(handle.read(length))
    return header, 8 + length


def last_byte_readable(path: Path, size: int) -> bool:
    """Prove the tail of a multi-GB shard is present without reading the whole file."""
    if size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(size - 1)
        return len(handle.read(1)) == 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-hash", action="store_true",
                        help="re-hash every shard; idle host only, reads ~33 GB")
    args = parser.parse_args()

    queue = json.loads(QUEUE.read_text())
    artifact = next(a for a in queue["artifacts"] if a["key"] == ARTIFACT_KEY)

    summary = {
        "gate": "computer-use-artifact-integrity",
        "artifact": ARTIFACT_KEY,
        "model_dir": str(MODEL_DIR),
        "provenance": {
            "repository": artifact["repository"],
            "revision": artifact["revision"],
            "license": artifact["license"],
            "gated": artifact["gated"],
            "metadata_collected_at": artifact["metadata_collected_at"],
            "queue_source_of_truth": queue["source_of_truth"],
            "sha256_enforced_at_download": True,
            "sha256_enforcement_site":
                "local-coverage-foundation/download_queue.py:download() "
                "promotes via os.replace only after size and SHA-256 match",
        },
        "full_hash_requested": args.full_hash,
        "files": [],
        "shards": [],
        "problems": [],
    }
    problems = summary["problems"]

    # --- manifest file presence and exact sizes ----------------------------
    total_expected = 0
    for entry in artifact["files"]:
        path = Path(entry["destination"])
        record_entry = {
            "repo_path": entry["repo_path"],
            "expected_bytes": entry["size"],
            "published_sha256": entry["sha256"],
            "exists": path.exists(),
        }
        total_expected += entry["size"]
        if path.exists():
            actual = path.stat().st_size
            record_entry["actual_bytes"] = actual
            record_entry["size_match"] = actual == entry["size"]
            if not record_entry["size_match"]:
                problems.append(f"{entry['repo_path']}: size {actual} != {entry['size']}")
            if args.full_hash and entry["sha256"]:
                digest = sha256(path)
                record_entry["recomputed_sha256"] = digest
                record_entry["sha256_match"] = digest == entry["sha256"]
                if not record_entry["sha256_match"]:
                    problems.append(f"{entry['repo_path']}: SHA-256 mismatch")
        else:
            problems.append(f"{entry['repo_path']}: missing")
        summary["files"].append(record_entry)

    summary["manifest_total_bytes"] = total_expected
    summary["queue_total_bytes"] = artifact["total_bytes"]
    if total_expected != artifact["total_bytes"]:
        problems.append(
            f"manifest file sizes sum to {total_expected}, queue declares {artifact['total_bytes']}")

    # --- unexpected files in the model directory ---------------------------
    expected_names = {Path(e["destination"]).name for e in artifact["files"]}
    present = {p.name for p in MODEL_DIR.iterdir() if p.is_file()}
    extra = sorted(present - expected_names)
    summary["unexpected_files"] = [
        {"name": name, "bytes": (MODEL_DIR / name).stat().st_size,
         "note": "download-time quarantine; reclaimable, left in place by policy"
                 if ".bad-" in name or name.endswith(".partial") else "unexpected"}
        for name in extra
    ]
    # Quarantined partials are evidence the hash gate fired, not corruption of the
    # live artifact. They are reported for reclamation, never deleted here.
    for item in summary["unexpected_files"]:
        if item["note"] == "unexpected":
            problems.append(f"unexpected file in model dir: {item['name']}")
    summary["reclaimable_bytes"] = sum(
        i["bytes"] for i in summary["unexpected_files"] if i["note"] != "unexpected")

    # --- safetensors headers and tensor extents ----------------------------
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    declared_total = index["metadata"]["total_size"]

    header_tensor_bytes = 0
    seen_tensors: set[str] = set()
    dtypes: dict[str, int] = {}
    for shard_name in sorted(set(weight_map.values())):
        path = MODEL_DIR / shard_name
        entry = {"shard": shard_name}
        try:
            header, data_start = read_header(path)
            size = path.stat().st_size
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            max_end = 0
            for name, spec in tensors.items():
                start, end = spec["data_offsets"]
                if start < 0 or end < start:
                    problems.append(f"{shard_name}: tensor {name} has inverted offsets")
                if data_start + end > size:
                    problems.append(
                        f"{shard_name}: tensor {name} ends at {data_start + end} "
                        f"beyond file size {size}")
                max_end = max(max_end, end)
                header_tensor_bytes += end - start
                dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
                seen_tensors.add(name)
            entry.update({
                "file_bytes": size,
                "header_bytes": data_start,
                "tensor_count": len(tensors),
                "data_end_offset": data_start + max_end,
                "trailing_bytes": size - (data_start + max_end),
                "last_byte_readable": last_byte_readable(path, size),
                "safetensors_metadata": header.get("__metadata__"),
            })
            if not entry["last_byte_readable"]:
                problems.append(f"{shard_name}: final byte unreadable")
            if entry["trailing_bytes"] != 0:
                problems.append(
                    f"{shard_name}: {entry['trailing_bytes']} bytes of trailing slack")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            problems.append(f"{shard_name}: {entry['error']}")
        summary["shards"].append(entry)

    summary["dtype_histogram"] = dtypes
    summary["tensors_in_index"] = len(weight_map)
    summary["tensors_in_headers"] = len(seen_tensors)
    summary["index_total_size"] = declared_total
    summary["header_tensor_bytes"] = header_tensor_bytes

    missing_from_headers = sorted(set(weight_map) - seen_tensors)
    extra_in_headers = sorted(seen_tensors - set(weight_map))
    summary["tensors_missing_from_headers"] = missing_from_headers[:20]
    summary["tensors_absent_from_index"] = extra_in_headers[:20]
    if missing_from_headers:
        problems.append(f"{len(missing_from_headers)} indexed tensors absent from shard headers")
    if extra_in_headers:
        problems.append(f"{len(extra_in_headers)} shard tensors absent from the index")
    if header_tensor_bytes != declared_total:
        problems.append(
            f"tensor bytes {header_tensor_bytes} != index total_size {declared_total}")

    # --- architecture sanity ------------------------------------------------
    config = json.loads((MODEL_DIR / "config.json").read_text())
    summary["architecture"] = {
        "architectures": config["architectures"],
        "model_type": config["model_type"],
        "torch_dtype": config["torch_dtype"],
        "num_hidden_layers": config["num_hidden_layers"],
        "hidden_size": config["hidden_size"],
        "vision_depth": config["vision_config"]["depth"],
        "max_position_embeddings": config["max_position_embeddings"],
    }
    if config["architectures"] != ["Qwen2_5_VLForConditionalGeneration"]:
        problems.append(f"unexpected architecture {config['architectures']}")

    # F32 on disk for a bf16-declared checkpoint is expected for this upload and
    # is why the load plan casts to bf16 rather than trusting the file dtype.
    summary["on_disk_dtype_note"] = (
        "shards store F32 while config declares bfloat16; loader must pass "
        "dtype=torch.bfloat16 so resident VRAM is ~16.6 GiB, not ~33 GiB")

    summary["bounded_check_only"] = not args.full_hash
    summary["deferred"] = [] if args.full_hash else [
        "full SHA-256 re-hash of 7 shards (~33 GB read) deferred to an idle host: "
        "run `python3 verify_artifact.py --full-hash`"
    ]
    summary["pass"] = not problems
    return record("artifact-integrity", summary)


if __name__ == "__main__":
    raise SystemExit(main())
