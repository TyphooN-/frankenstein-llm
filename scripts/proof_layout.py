#!/usr/bin/env python3
"""Prepare canonical proof storage; preserve historical paths with directory links."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (
    'logs',
    'verification/candidate-qualification/evidence',
    'verification/computer-use-grounding/evidence',
    'verification/generative-media/evidence',
    'verification/local-coverage-foundation/evidence',
    'verification/local-coverage-foundation/research/hf',
    'verification/prompt-corpus-admission/evidence',
    'verification/qualification-supervisor/qualification-cache',
    'verification/qualification-supervisor/evidence',
    'verification/glm53flash-local/evidence',
    'verification/manual/evidence',
    'verification/qualitative-characterization/evidence',
    'verification/repository-agent/evidence',
    'verification/router-functional/evidence',
    'verification/security-agents/strix-scaffold/evidence',
    'verification/tts-local/evidence',
    'verification/tts-local/artifacts',
    'verification/qwen38-obliterated-functional',
)


def prepare(root, migrate=False):
    """Never overwrite a destination or move tracked inputs/source files."""
    root = Path(root).resolve()
    operations = []
    for relative in DIRECTORIES:
        source = root / relative
        target = root / 'proofs' / relative
        target.resolve().relative_to(root / 'proofs')
        if source.is_symlink():
            if source.resolve() != target.resolve():
                raise ValueError(f'Unexpected link: {source}')
            continue
        if source.exists() and not source.is_dir():
            raise ValueError(f'Expected a directory: {source}')
        entries = list(source.iterdir()) if source.exists() else []
        if entries and not migrate:
            raise ValueError(f'Existing artifacts require --migrate: {source}')
        for entry in entries:
            if (target / entry.name).exists() or (target / entry.name).is_symlink():
                raise ValueError(f'Destination collision: {target / entry.name}')
        operations.append((source, target, entries))
    # Validate the whole plan before changing any directory.
    occupied = {target / entry.name for _, target, entries in operations for entry in entries}
    flat = []
    specifications = (
        ('verification/qualification-supervisor', ('*.log', '*.performance.jsonl', 'qualification-state.json')),
        ('verification/glm53flash-local', ('gate-glm32.json', 'glm32-server.log')),
        ('verification', ('direct-*.json', 'router-state-after-direct.json', 'hermes-*.png', 'hermes-*-output.txt', 'hermes-*-usage.json', 'obliteratus-hf-inventory-*.json')),
    )
    if migrate:
        for directory, patterns in specifications:
            destination = directory if directory != 'verification' else 'verification/manual'
            for pattern in patterns:
                for source in (root / directory).glob(pattern):
                    target = root / 'proofs' / destination / 'evidence' / source.name
                    target.resolve().relative_to(root / 'proofs')
                    if source.is_symlink() and source.resolve() == target.resolve():
                        continue
                    if source.is_symlink() or not source.is_file() or target.exists() or target in occupied:
                        raise ValueError(f'Unsafe or colliding proof: {source}')
                    flat.append((source, target))
                    occupied.add(target)
    for source, target, entries in operations:
        target.mkdir(parents=True, exist_ok=True)
        (target / '.gitkeep').touch(exist_ok=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            entry.rename(target / entry.name)
        if source.exists():
            source.rmdir()  # Only an empty directory; artifacts were preserved.
        source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=True)
    for source, target in flat:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        source.symlink_to(os.path.relpath(target, source.parent))
    return len(operations) + len(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrate', action='store_true', help='move existing artifacts; stop all writers first')
    args = parser.parse_args()
    if args.migrate:
        result = subprocess.run(['systemctl', '--user', 'list-units', '--no-legend',
                                 '--plain', '--state=active,activating,deactivating',
                                 'local-ai-*.service', 'llama-*.service'],
                                capture_output=True, text=True, check=True)
        if result.stdout.strip():
            parser.error('Stop artifact-writing local-ai/llama services before migration')
    print(f'Prepared {prepare(ROOT, args.migrate)} proof paths')


if __name__ == '__main__':
    main()
