"""Opt-in source backport; never change profiles, credentials, services or Git staging."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

BUNDLE = Path(__file__).resolve().parent / 'runtime'
MAX_BYTES = 2 * 1024 * 1024
SOURCE_ROOTS = {'agent', 'hermes_cli', 'scripts', 'tests', 'tui_gateway', 'website'}
GIT_ROUTING_ENV = {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY'}


class InstallError(ValueError):
    """Refuse a package or checkout before writing."""


def _git(root, *args, data=None):
    env = {k: v for k, v in os.environ.items() if k not in GIT_ROUTING_ENV}
    result = subprocess.run(['git', '-C', str(root), *args], input=data, capture_output=True,
                            env=env, timeout=30)
    if result.returncode:
        raise InstallError('Git refused the package or checkout: ' + result.stderr.decode(errors='replace')[:1200])
    return result.stdout


def _read(path, limit=MAX_BYTES):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise InstallError('Expected a regular, non-symlink file: ' + str(path))
    chunks = []
    remaining = limit + 1
    with path.open('rb') as stream:
        while remaining:
            chunk = stream.read(min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    if not remaining:
        raise InstallError('File exceeds package read limit: ' + str(path))
    return b''.join(chunks)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def _path(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_./-]{1,240}', name):
        raise InstallError('Invalid package path')
    parts = name.split('/')
    forbidden = {'.', '..', '.git', '.hermes', '.env', 'auth.json', 'config.yaml'}
    if (not all(parts) or any(p in forbidden for p in parts) or parts[0] not in SOURCE_ROOTS
            or PurePosixPath(name).suffix not in ('.py', '.md')):
        raise InstallError('Package path is outside the source-only boundary: ' + name)
    return name


def _manifest(bundle):
    value = json.loads(_read(bundle / 'manifest.json', 65536))
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
            or value['schema_version'] != 1 or not _hex(value.get('base_commit'), 40)
            or value.get('patch') != 'hermes-local-helper.patch'
            or not _hex(value.get('patch_sha256'), 64)):
        raise InstallError('Invalid package manifest header')
    files = value.get('files')
    if not isinstance(files, list) or not 1 <= len(files) <= 128:
        raise InstallError('Package must declare a bounded nonempty source set')
    seen = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {'path', 'base_sha256', 'sha256', 'base_mode', 'mode'}:
            raise InstallError('Invalid source manifest entry')
        name = _path(entry['path'])
        if name in seen:
            raise InstallError('Duplicate source path: ' + name)
        seen.add(name)
        if not _hex(entry['sha256'], 64) or entry['mode'] not in ('100644', '100755'):
            raise InstallError('Invalid source postimage')
        if entry['base_sha256'] is None:
            if entry['base_mode'] is not None:
                raise InstallError('New source file must have no base mode')
        elif not _hex(entry['base_sha256'], 64) or entry['base_mode'] not in ('100644', '100755'):
            raise InstallError('Invalid source preimage')
    return value


def _target(root, name):
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink():
            raise InstallError('Refusing symlink in target path: ' + name)
    return current


def _state(root, manifest):
    matches_base = matches_post = True
    for entry in manifest['files']:
        path = _target(root, entry['path'])
        if path.exists():
            data = _read(path)
            mode = '100755' if path.stat().st_mode & 0o111 else '100644'
            observed = (_digest(data), mode)
        else:
            observed = (None, None)
        matches_base &= observed == (entry['base_sha256'], entry['base_mode'])
        matches_post &= observed == (entry['sha256'], entry['mode'])
    if matches_post:
        return 'already_applied'
    if matches_base:
        return 'ready'
    raise InstallError('Target source differs from both complete preimage and complete postimage; no files changed')


def _inspect(root, bundle):
    root = Path(root).resolve(strict=True)
    bundle = Path(bundle).resolve(strict=True)
    manifest = _manifest(bundle)
    # Freeze and validate the exact bytes later supplied to Git, not a second path read.
    patch = _read(bundle / manifest['patch'])
    if _digest(patch) != manifest['patch_sha256']:
        raise InstallError('Patch SHA-256 mismatch')
    top = Path(_git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    if top != root or _git(root, 'rev-parse', 'HEAD').decode().strip() != manifest['base_commit']:
        raise InstallError('Target must be the root of the exact pinned Hermes checkout')
    records = _git(root, 'apply', '--numstat', '-z', '-', data=patch).split(b'\0')
    paths = []
    for record in filter(None, records):
        fields = record.decode().split('\t')
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            raise InstallError('Only ordinary text additions/modifications are supported')
        paths.append(_path(fields[2]))
    declared = [entry['path'] for entry in manifest['files']]
    if len(paths) != len(set(paths)) or set(paths) != set(declared):
        raise InstallError('Patch source set differs from manifest')
    status = _state(root, manifest)
    if status == 'ready':
        if _git(root, 'diff', '--cached', '--name-only', '--', *declared).strip():
            raise InstallError('Target paths have staged changes; no files changed')
        _git(root, 'apply', '--check', '--whitespace=error', '-', data=patch)
    report = {'status': status, 'base_commit': manifest['base_commit'], 'files': len(declared)}
    return root, report, manifest, patch


def check(root, bundle=BUNDLE):
    """Read-only validation; recognize the exact already-installed source snapshot."""
    return _inspect(root, bundle)[1]


def apply(root, bundle=BUNDLE):
    """Apply only to the complete preimage; unrelated work and index remain untouched.

    The operator must own the target source paths exclusively for this operation.
    No conflict cleanup or automatic rollback erases a user's checkpoint on I/O failure.
    """
    root, report, manifest, patch = _inspect(root, bundle)
    if report['status'] == 'already_applied':
        return report
    _git(root, 'apply', '--whitespace=error', '-', data=patch)
    if (_state(root, manifest) != 'already_applied'
            or _git(root, 'rev-parse', 'HEAD').decode().strip() != manifest['base_commit']):
        raise InstallError('Post-install verification failed; preserve the checkout for inspection')
    return {**report, 'status': 'already_applied'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-source', type=Path, required=True, help='Existing exact pinned Hermes source root')
    parser.add_argument('--apply', action='store_true', help='Apply the validated backport; default is read-only check')
    args = parser.parse_args()
    try:
        result = (apply if args.apply else check)(args.hermes_source)
    except (InstallError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status': 'refused', 'error': str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
