#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation')
READY = ROOT / 'machine-ready.ok'

def active_release_max():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / 'comm').read_text().strip()
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if comm in {'cargo', 'rustc', 'bash', 'sh', 'makepkg'} and ('release-max' in cmd or 'makepkg -si' in cmd):
            found.append((proc.name, comm, cmd[:300]))
    return found

def scrub_active():
    p = subprocess.run(['zpool', 'status', 'zroot'], text=True, capture_output=True)
    text = p.stdout + p.stderr
    return 'scrub in progress' in text, text

last = None
while True:
    builds = active_release_max()
    scrub, status = scrub_active()
    state = (bool(builds), scrub)
    if state != last:
        print(f'release_max_active={bool(builds)} scrub_active={scrub}', flush=True)
        last = state
    if not builds and not scrub:
        READY.write_text('ready\n')
        print('MACHINE_READY_FOR_LOCAL_AI_EXECUTION', flush=True)
        raise SystemExit(0)
    time.sleep(30)
