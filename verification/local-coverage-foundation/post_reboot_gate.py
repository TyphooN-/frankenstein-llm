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

# Exact process-name check; inspect cmdlines so this script cannot self-match.
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc / 'comm').read_text().strip()
        cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except OSError:
        continue
    if comm in {'makepkg', 'make', 'clang', 'ld.lld'} and ('linux' in cmd or 'Makefile.build' in cmd):
        reasons.append(f'kernel build process remains active: pid={proc.name} comm={comm}')
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
