#!/usr/bin/env python3
import json
import time
from pathlib import Path

PID = 26193
PGID = 26193
ROOT = Path('/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation')
TRANSCRIPT = ROOT / 'transcript-resume.jsonl'
HANDOFF = ROOT / 'handoff.json'

def members():
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / 'stat').read_text()
            close = stat.rfind(')')
            fields = stat[close + 2:].split()
            state = fields[0]
            pgrp = int(fields[2])
            if pgrp != PGID:
                continue
            # Only membership and liveness are used below. cmdline would add
            # nothing and is served through access_remote_vm, where a target
            # holding its mmap write lock can wedge this poll.
            rows.append((int(p.name), state))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            continue
    return rows

last_size = -1
while True:
    rows = members()
    live = [r for r in rows if r[1] != 'Z']
    size = TRANSCRIPT.stat().st_size if TRANSCRIPT.exists() else 0
    if size != last_size:
        print(f'transcript_bytes={size} live_members={len(live)}', flush=True)
        last_size = size
    if not live:
        final = False
        if TRANSCRIPT.exists():
            for line in reversed(TRANSCRIPT.read_text(errors='replace').splitlines()[-50:]):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get('type') == 'result':
                    final = True
                    break
        print(f'CLAUDE_GROUP_EXITED final_result={final} handoff={HANDOFF.exists()}', flush=True)
        raise SystemExit(0)
    time.sleep(20)
