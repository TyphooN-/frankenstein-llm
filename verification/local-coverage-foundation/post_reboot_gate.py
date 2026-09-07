#!/usr/bin/env python3
"""Fail-closed gate for optional post-reboot benchmarking only.

Functional builds, model loads, correctness checks, memory checks, and clean-unload
checks are explicitly allowed before this gate passes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path('/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation')
BASELINE = json.loads((ROOT / 'pre-reboot-kernel-baseline.json').read_text(encoding='utf-8'))
current = {
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'uname_release': platform.release(),
    'proc_version': Path('/proc/version').read_text().strip(),
}
reasons: list[str] = []
if current['boot_id'] == BASELINE['boot_id']:
    reasons.append('host has not rebooted since the flawed-kernel baseline')
if current['proc_version'] == BASELINE['proc_version']:
    reasons.append('booted kernel build signature is unchanged from the flawed baseline')

# Kernel metadata only. Linux serves /proc/<pid>/cmdline through
# access_remote_vm, so reading it can block on the mmap write lock held by the
# very wide parallel compile this gate is looking for. comm and the cwd symlink
# cannot. Dropping the old "is 'linux' in the argument vector" qualifier makes
# this block on any build rather than only a kernel one, which is the safe
# direction for a gate that exists to refuse a contended benchmark; cwd still
# says which kind it was, and this script's own name is not in the set.
BUILD_COMMANDS = {'makepkg', 'make', 'gmake', 'ninja', 'cmake', 'cc', 'gcc', 'g++',
                  'clang', 'clang++', 'cc1', 'cc1plus', 'ld', 'ld.lld', 'lld', 'mold'}
KERNEL_TREE = '/linux-tkg'
DELETED_SUFFIX = ' (deleted)'
self_pid = os.getpid()
for pid in sorted(int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()):
    if pid == self_pid:
        continue
    proc = Path('/proc') / str(pid)
    try:
        comm = (proc / 'comm').read_text().strip()
    except OSError:
        continue
    if comm not in BUILD_COMMANDS:
        continue
    try:
        # Refused for another user's process as a matter of course; that costs
        # the label, not the detection, because comm already matched.
        cwd = os.readlink(proc / 'cwd').removesuffix(DELETED_SUFFIX)
    except OSError:
        cwd = ''
    kind = 'kernel build' if KERNEL_TREE in cwd else 'build'
    reasons.append(f'{kind} process remains active: pid={pid} comm={comm} cwd={cwd or "unreadable"}')
    break

services = {}
for unit in ('local-ai-model-downloads.service', 'local-ai-model-downloads-phase2.service', 'llama-router.service'):
    result = subprocess.run(
        ['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'ExecMainPID'],
        text=True, capture_output=True, check=False,
    )
    services[unit] = result.stdout.strip()

report = {
    'gate': 'post-reboot-benchmark-ready',
    'pass': not reasons,
    'baseline': BASELINE,
    'current': current,
    'reasons': reasons,
    'services': services,
    'next_when_passed': [
        'benchmarking may be considered separately if the user requests it',
        'functional builds and model tests never depend on this gate',
        'tokens-per-second measurement remains prohibited by the active task',
    ],
}
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['pass'] else 3)
