#!/usr/bin/env python3
"""Verify locked upstream plus exact declared patches without modifying source/index.

The temporary index may write ordinary Git objects. It never resets, stages into
 the real index, or applies a patch to the working tree. Unknown tracked edits
are rejected before their contents are read. Ignored build outputs are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def verify(root: Path, source: Path) -> dict:
    lock = json.loads((root / 'upstream/llama-cpp.lock.json').read_text())
    base = lock['commit']
    if not isinstance(base, str) or not re.fullmatch(r'[0-9a-f]{40}', base):
        raise ValueError('invalid locked commit')
    env = dict(os.environ)
    for key in ('GIT_INDEX_FILE', 'GIT_DIR', 'GIT_WORK_TREE'):
        env.pop(key, None)

    def git(*args, index=None):
        child_env = dict(env)
        if index is not None:
            child_env['GIT_INDEX_FILE'] = str(index)
        result = subprocess.run(['git', '-C', str(source), *args], env=child_env,
                                capture_output=True, timeout=30)
        if result.returncode:
            raise ValueError(f'git {args[0]} failed ({result.returncode}): '
                             + result.stderr.decode(errors='replace').strip())
        return result.stdout

    if git('rev-parse', 'HEAD').decode().strip() != base:
        raise ValueError('llama.cpp HEAD does not match locked commit')
    # Git status deliberately trusts these index flags. Refuse them before
    # either manifest branch, rather than certifying hidden compiler inputs.
    flagged = [entry[2:] for entry in git('ls-files', '-v', '-z').split(b'\0')
               if entry and (entry[:1] == b'S' or entry[:1].islower())]
    if flagged:
        raise ValueError('visibility-suppressing index flags are forbidden: '
                         + ', '.join(os.fsdecode(path) for path in flagged))
    status = git('status', '--porcelain', '-z', '--no-renames', '--untracked-files=no')
    patches = lock.get('local_patches', [])
    if not isinstance(patches, list):
        raise ValueError('local_patches must be a list')
    if not patches:
        if status:
            raise ValueError('llama.cpp tracked worktree is dirty without declared patches')
        return {'status': 'VERIFIED', 'commit': base, 'patch_count': 0}

    paths = []
    for entry in patches:
        relative = entry['path']
        if not isinstance(relative, str):
            raise ValueError('invalid patch path')
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root / 'upstream/patches'):
            raise ValueError('patch path must stay inside upstream/patches')
        digest = entry['sha256']
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid patch SHA256')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('patch SHA256 mismatch: ' + relative)
        paths.append(path)

    with tempfile.TemporaryDirectory(prefix='llama-patch-proof-') as temp:
        index = Path(temp) / 'index'
        git('read-tree', base, index=index)
        for path in paths:
            git('apply', '--cached', '--whitespace=error-all', str(path), index=index)
        expected = git('write-tree', index=index).decode().strip()
        owned = set(git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z',
                        base, expected, index=index).split(b'\0')) - {b''}
        changed = {item[3:] for item in status.split(b'\0') if item}
        if changed - owned:
            raise ValueError('tracked edits outside declared patch: '
                             + ', '.join(os.fsdecode(path) for path in sorted(changed - owned)))
        git('read-tree', base, index=index)
        if owned:
            git('add', '--all', '--', *sorted(owned), index=index)
        actual = git('write-tree', index=index).decode().strip()
        if actual != expected:
            raise ValueError('llama.cpp source does not match declared patch tree')
    return {'status': 'VERIFIED', 'commit': base, 'patch_count': len(paths),
            'patched_tree': expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path, help='isolated checkout to verify instead of the submodule')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        result = verify(root, (args.source or root / 'upstream/llama.cpp').resolve())
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
