#!/usr/bin/env python3
"""Explicit, offline migration of pre-rebrand state; never promotes receipts."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LEGACY = 'mission-supervisor'
CURRENT = 'qualification-supervisor'


def renamed(value):
    if isinstance(value, str):
        return (value.replace('local-ai-functional-mission', 'local-ai-qualification')
                .replace('run_functional_mission', 'run_qualification')
                .replace('mission-supervisor', CURRENT)
                .replace('mission-state.json', 'qualification-state.json')
                .replace('mission.log', 'qualification.log'))
    if isinstance(value, list):
        return [renamed(item) for item in value]
    if isinstance(value, dict):
        return {key: renamed(item) for key, item in value.items()}
    return value


def migration_plan(root):
    """Return bounded known runtime targets, retaining every source as backup."""
    old = root / 'verification' / LEGACY
    new = root / 'verification' / CURRENT
    rows = []
    if not old.exists():
        return rows
    if old.is_symlink() or new.is_symlink():
        raise ValueError('state directories must not be symlinks')
    candidates = list(old.glob('*.log'))
    state = old / 'mission-state.json'
    if state.exists():
        candidates.append(state)
    cache = old / 'qualification-cache'
    if cache.is_symlink():
        raise ValueError('cache directory must not be a symlink')
    if cache.exists():
        candidates.extend(cache.glob('*.json'))
    if len(candidates) > 10000:
        raise ValueError('runtime inventory exceeds migration bound')
    total = 0
    for source in sorted(candidates):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'not a regular runtime file: {source}')
        relative = source.relative_to(old)
        target = new / relative
        if source == state:
            target = new / 'qualification-state.json'
        elif source.name == 'mission.log':
            target = new / 'qualification.log'
        if target.is_symlink() or target.parent.is_symlink():
            raise ValueError(f'target must not be a symlink: {target}')
        size = source.stat().st_size
        total += size
        if size > 64 << 20 or total > 128 << 20:
            raise ValueError('runtime evidence exceeds migration memory bound; archive logs first')
        with source.open('rb') as handle:
            raw = handle.read((64 << 20) + 1)
        if len(raw) != size:
            raise ValueError('runtime source changed during migration preflight')
        if source == state:
            if len(raw) > 4 << 20:
                raise ValueError('state exceeds 4 MiB')
            value = json.loads(raw)
            if not isinstance(value, dict) or not isinstance(value.get('steps'), dict):
                raise ValueError('invalid state mapping')
            raw = (json.dumps(renamed(value), indent=2) + '\n').encode()
        # Receipt bytes, keys, statuses and timestamps are deliberately unchanged.
        # New criteria still invalidate them through the normal exact-key check.
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f'refusing to overwrite divergent target: {target}')
        rows.append((source, target, raw))
    return rows


def assert_no_supervisor(root):
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for arg in args:
            if arg in (str(root / 'verification' / LEGACY / 'run_functional_mission.py').encode(),
                       str(root / 'verification' / CURRENT / 'run_qualification.py').encode()):
                raise RuntimeError('a supervisor is alive; stop it before migration')


def migrate(root, apply=False):
    root = root.resolve()
    if not apply:
        return migration_plan(root)
    assert_no_supervisor(root)
    old = root / 'verification' / LEGACY
    new = root / 'verification' / CURRENT
    if not old.exists():
        return []
    if old.is_symlink() or new.is_symlink():
        raise ValueError('state directories must not be symlinks')
    new.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (old / 'mission.lock', new / 'qualification.lock'):
            if path.is_symlink():
                raise ValueError('lock must not be a symlink')
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = migration_plan(root)  # Preflight every target before any copy.
        for _, target, raw in rows:
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, staging = tempfile.mkstemp(prefix='.migration-', dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Hard-link publication refuses an unexpectedly appeared target.
                os.link(staging, target)
                directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                Path(staging).unlink(missing_ok=True)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--apply', action='store_true', help='copy state while both supervisors are stopped')
    args = parser.parse_args()
    rows = migrate(args.root, args.apply)
    print(json.dumps({'applied': args.apply, 'files': [
        {'source': str(source), 'target': str(target)} for source, target, _ in rows]}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
