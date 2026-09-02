#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.parse import quote

signal.signal(signal.SIGHUP, signal.SIG_IGN)

MANIFEST = Path("/home/typhoon/git/frankenstein-llm/verification/obliterated-mmproj-manifest.json")
DESTINATION = Path("/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf")


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    expected_size = int(manifest["size"])
    expected_hash = manifest["sha256"]
    partial = DESTINATION.with_suffix(DESTINATION.suffix + ".partial")

    if DESTINATION.exists():
        if DESTINATION.stat().st_size != expected_size:
            raise RuntimeError("existing projector has the wrong size")
        actual_hash = sha256(DESTINATION)
        if actual_hash != expected_hash:
            raise RuntimeError(f"existing projector hash mismatch: {actual_hash}")
        log(f"already verified size={expected_size} sha256={actual_hash}")
        return 0

    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError("partial projector is oversized")

    if not partial.exists() or partial.stat().st_size < expected_size:
        repository = manifest["repository"]
        revision = manifest["repository_revision"]
        filename = quote(manifest["filename"])
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{filename}?download=true"
        command = [
            "curl", "--fail", "--location", "--continue-at", "-",
            "--retry", "20", "--retry-delay", "5", "--retry-all-errors",
            "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1048576",
            "--output", str(partial), url,
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            log(f"curl failed exit={result.returncode}; partial preserved")
            return result.returncode

    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"projector size mismatch: {partial.stat().st_size} != {expected_size}")
    actual_hash = sha256(partial)
    if actual_hash != expected_hash:
        raise RuntimeError(f"projector hash mismatch: {actual_hash} != {expected_hash}")
    os.replace(partial, DESTINATION)
    log(f"verified and promoted size={expected_size} sha256={actual_hash}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"fatal: {error}")
        raise
