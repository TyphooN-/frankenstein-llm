#!/usr/bin/env python3
"""Wait for phase-one completion, then exec the independently locked phase-two queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation')
PHASE1_STAMP = ROOT / 'downloads-complete.ok'
PHASE1_STATE = ROOT / 'download-state.json'
EXPECTED_PHASE1_BYTES = '72134030730'

while True:
    if PHASE1_STAMP.exists():
        value = PHASE1_STAMP.read_text(encoding='utf-8').strip()
        if value != EXPECTED_PHASE1_BYTES:
            raise SystemExit(f'phase-one stamp mismatch: expected {EXPECTED_PHASE1_BYTES}, got {value!r}')
        break
    if PHASE1_STATE.exists():
        try:
            state = json.loads(PHASE1_STATE.read_text(encoding='utf-8'))
            if state.get('status') == 'failed':
                raise SystemExit(f"phase one failed: {state.get('error', 'unknown error')}")
        except json.JSONDecodeError:
            pass
    time.sleep(30)

environment = os.environ.copy()
environment.update({
    'HERMES_DOWNLOAD_QUEUE': str(ROOT / 'download-queue-phase2.json'),
    'HERMES_DOWNLOAD_STATE': str(ROOT / 'download-state-phase2.json'),
    'HERMES_DOWNLOAD_LOCK': str(ROOT / 'download-queue-phase2.lock'),
    'HERMES_DOWNLOAD_STAMP': str(ROOT / 'downloads-phase2-complete.ok'),
    'HERMES_DOWNLOAD_LOG': str(ROOT / 'downloads-phase2.log'),
})
os.execve(sys.executable, [sys.executable, str(ROOT / 'download_queue.py')], environment)
