#!/usr/bin/env python3
"""Wait for both recovery queues, then run the image-editing artifact queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation")
PHASES = (
    (ROOT / "download-state.json", ROOT / "downloads-complete.ok", "72134030730"),
    (ROOT / "download-state-phase2.json", ROOT / "downloads-phase2-complete.ok", "33184695056"),
)


def phase_ready(state_path: Path, stamp_path: Path, expected_bytes: str) -> bool:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if state.get("status") == "failed":
        raise SystemExit(f"upstream queue failed: {state_path}: {state.get('error', 'unknown error')}")
    if state.get("status") != "complete":
        return False
    try:
        return stamp_path.read_text(encoding="utf-8").strip() == expected_bytes
    except OSError:
        return False


while not all(phase_ready(*phase) for phase in PHASES):
    time.sleep(30)

environment = os.environ.copy()
environment.update({
    "HERMES_DOWNLOAD_QUEUE": str(ROOT / "download-queue-phase3.json"),
    "HERMES_DOWNLOAD_STATE": str(ROOT / "download-state-phase3.json"),
    "HERMES_DOWNLOAD_LOCK": str(ROOT / "download-queue-phase3.lock"),
    "HERMES_DOWNLOAD_STAMP": str(ROOT / "downloads-phase3-complete.ok"),
    "HERMES_DOWNLOAD_LOG": str(ROOT / "downloads-phase3.log"),
})
os.execve(sys.executable, [sys.executable, str(ROOT / "download_queue.py")], environment)
