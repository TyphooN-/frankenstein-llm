#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Transfers must survive supervising-shell/context SIGHUP events. The ignored
# disposition is inherited by curl, while SIGTERM remains available for an
# intentional, clean stop.
signal.signal(signal.SIGHUP, signal.SIG_IGN)
from urllib.parse import quote

REPO = "6block/GLM-5.3-Flash-GGUF"
ROOT = Path("/home/typhoon/git/frankenstein-llm/models/glm53flash-regular-iq3xxs")
MANIFEST = Path("/home/typhoon/git/frankenstein-llm/verification/glm53flash-local/manifest.tsv")
STATE = Path("/home/typhoon/git/frankenstein-llm/verification/glm53flash-local/verified-state.json")
STAMP = Path("/home/typhoon/git/frankenstein-llm/verification/glm53flash-local/force-reverify.ok")
FORCE_REVERIFY = os.environ.get("GLM_FORCE_REVERIFY") == "1"


def sha256(path: Path) -> str:
    last_error = None
    for attempt in range(1, 6):
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(16 * 1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as error:
            last_error = error
            if error.errno != 5 or attempt == 5:
                raise
            log(f"retrying sha256 after EIO attempt {attempt}/5 for {path.name}")
            time.sleep(2 * attempt)
    raise last_error


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def load_state() -> dict:
    if not STATE.exists():
        return {}
    try:
        state = json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(state: dict) -> None:
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATE)


def state_matches(state: dict, path: Path, expected_size: int, expected_hash: str) -> bool:
    entry = state.get(path.name)
    if not isinstance(entry, dict):
        return False
    stat = path.stat()
    return (
        stat.st_size == expected_size
        and entry.get("size") == expected_size
        and entry.get("mtime_ns") == stat.st_mtime_ns
        and entry.get("sha256") == expected_hash
    )


def record_verified(state: dict, path: Path, expected_hash: str) -> None:
    stat = path.stat()
    state[path.name] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": expected_hash,
    }
    save_state(state)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    rows = []
    for line in MANIFEST.read_text().splitlines():
        expected_hash, expected_size, name = line.split("\t")
        rows.append((expected_hash, int(expected_size), name))
    if len(rows) != 15:
        raise RuntimeError(f"expected 15 manifest rows, found {len(rows)}")
    expected_total = sum(size for _, size, _ in rows)
    if STAMP.exists() and STAMP.read_text().strip() == str(expected_total):
        log(f"force reverify already complete bytes={expected_total}")
        return 0

    for index, (expected_hash, expected_size, name) in enumerate(rows, 1):
        final = ROOT / name
        partial = ROOT / f"{name}.partial"
        if final.exists():
            size = final.stat().st_size
            if size == expected_size:
                if not FORCE_REVERIFY and state_matches(state, final, expected_size, expected_hash):
                    log(f"cached verification {index}/15 {name} size={size} sha256={expected_hash}")
                    continue
                try:
                    actual_hash = sha256(final)
                except OSError as error:
                    if error.errno != 5:
                        raise
                    quarantine = final.with_name(f"{name}.bad-eio-{int(time.time())}")
                    final.rename(quarantine)
                    state.pop(name, None)
                    save_state(state)
                    log(f"quarantined unreadable EIO file {name} as {quarantine.name}")
                    actual_hash = None
                if actual_hash == expected_hash:
                    record_verified(state, final, actual_hash)
                    log(f"verified {index}/15 {name} size={size} sha256={actual_hash}")
                    continue
                if actual_hash is not None:
                    quarantine = final.with_name(f"{name}.bad-{actual_hash[:12]}")
                    final.rename(quarantine)
                    state.pop(name, None)
                    save_state(state)
                    log(f"quarantined hash mismatch {name} as {quarantine.name}")
            else:
                if partial.exists():
                    raise RuntimeError(f"both wrong-size final and partial exist for {name}")
                final.rename(partial)
                log(f"converted incomplete final to resumable partial: {name} size={size}")

        if partial.exists() and partial.stat().st_size > expected_size:
            raise RuntimeError(f"oversized partial for {name}: {partial.stat().st_size} > {expected_size}")
        if partial.exists() and partial.stat().st_size == expected_size:
            actual_hash = sha256(partial)
            if actual_hash != expected_hash:
                raise RuntimeError(f"sha256 mismatch for complete partial {name}: {actual_hash} != {expected_hash}")
            os.replace(partial, final)
            record_verified(state, final, actual_hash)
            log(f"verified and promoted complete partial {index}/15 {name} size={expected_size} sha256={actual_hash}")
            continue

        url = f"https://huggingface.co/{REPO}/resolve/main/{quote(name)}?download=true"
        log(f"downloading {index}/15 {name} from byte {partial.stat().st_size if partial.exists() else 0}")
        command = [
            "curl", "--fail", "--location", "--continue-at", "-",
            "--retry", "20", "--retry-delay", "5", "--retry-all-errors",
            "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1048576",
            "--output", str(partial), url,
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            log(f"curl failed for {name} exit={result.returncode}; partial preserved")
            return result.returncode
        size = partial.stat().st_size
        if size != expected_size:
            raise RuntimeError(f"size mismatch for {name}: {size} != {expected_size}")
        actual_hash = sha256(partial)
        if actual_hash != expected_hash:
            raise RuntimeError(f"sha256 mismatch for {name}: {actual_hash} != {expected_hash}")
        os.replace(partial, final)
        record_verified(state, final, actual_hash)
        log(f"verified and promoted {index}/15 {name} size={size} sha256={actual_hash}")

    total = sum((ROOT / name).stat().st_size for _, _, name in rows)
    log(f"complete shards=15 bytes={total}")
    STAMP.write_text(f"{total}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"fatal: {error}")
        raise
