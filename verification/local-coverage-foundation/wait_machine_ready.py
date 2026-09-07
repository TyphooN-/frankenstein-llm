#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation')
READY = ROOT / 'machine-ready.ok'

# Task names as /proc/<pid>/comm reports them. cmdline is not read: Linux serves
# it through access_remote_vm, so a poll every 30 seconds against a running
# rustc can block on that task's own mmap write lock. Losing the argument vector
# loses the "release-max" qualifier with it, so this now waits out any Rust or
# makepkg build rather than one named profile -- broader, and the right
# direction for a wait whose whole purpose is an uncontended host.
BUILD_COMMANDS = {'cargo', 'rustc', 'makepkg', 'make', 'gmake', 'ninja', 'cmake',
                  'cc', 'gcc', 'g++', 'clang', 'clang++', 'cc1', 'cc1plus',
                  'ld', 'ld.lld', 'lld', 'mold'}


def active_builds():
    found = []
    pids = sorted(int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit())
    for proc in (Path('/proc') / str(pid) for pid in pids):
        try:
            comm = (proc / 'comm').read_text().strip()
        except OSError:
            continue
        if comm in BUILD_COMMANDS:
            found.append((proc.name, comm))
    return found

def scrub_active():
    p = subprocess.run(['zpool', 'status', 'zroot'], text=True, capture_output=True)
    text = p.stdout + p.stderr
    return 'scrub in progress' in text, text

last = None
while True:
    builds = active_builds()
    scrub, status = scrub_active()
    state = (bool(builds), scrub)
    if state != last:
        print(f'build_active={bool(builds)} scrub_active={scrub}', flush=True)
        last = state
    if not builds and not scrub:
        READY.write_text('ready\n')
        print('MACHINE_READY_FOR_LOCAL_AI_EXECUTION', flush=True)
        raise SystemExit(0)
    time.sleep(30)
