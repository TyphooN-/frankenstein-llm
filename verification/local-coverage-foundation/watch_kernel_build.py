#!/usr/bin/env python3
"""Record completion of the current optimized-kernel build without controlling it."""
from __future__ import annotations

import json
from pathlib import Path
import time

ROOT = Path('/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation')
PGID = 14562
STAMP = ROOT / 'kernel-build-finished.json'

def members() -> list[dict]:
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            if int(fields[2]) != PGID or fields[0] == 'Z':
                continue
            rows.append({
                'pid': int(proc.name),
                'state': fields[0],
                # comm, not cmdline. Linux serves cmdline through
                # access_remote_vm, so reading it can block on the target's mmap
                # write lock -- and every task this watcher polls is a compiler
                # in a wide parallel kernel build, which is exactly the task
                # that holds it. stat and comm are kernel metadata.
                'command': (proc / 'comm').read_bytes().decode(errors='replace').strip(),
            })
        except (OSError, ValueError, IndexError):
            continue
    return rows

while members():
    time.sleep(30)
images = []
for path in sorted(Path('/boot').glob('vmlinuz-*')):
    try:
        stat = path.stat()
        images.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    except OSError:
        pass
record = {
    'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'tracked_pgid': PGID,
    'non_zombie_members': [],
    'boot_images': images,
    'next': 'Functional work continues now. Only optional benchmarking waits for reboot and post_reboot_gate.py.',
}
temporary = STAMP.with_suffix('.json.tmp')
temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
temporary.replace(STAMP)
print(json.dumps(record, indent=2), flush=True)
