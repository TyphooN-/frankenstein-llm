#!/usr/bin/env python3
"""Standalone serving for every router preset. Preview by default; --execute runs."""
import argparse
import configparser
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'upstream/llama.cpp/build/bin/llama-server'
BOOL_FLAGS = {'jinja': ('--jinja', '--no-jinja'), 'mmap': ('--mmap', '--no-mmap'),
              'embedding': ('--embedding', None), 'reranking': ('--reranking', None)}
VALUE_FLAGS = {'model', 'mmproj', 'mmproj-device', 'ctx-size', 'gpu-layers', 'flash-attn',
               'cache-type-k', 'cache-type-v', 'parallel', 'device', 'tensor-split',
               'reasoning', 'spec-type', 'spec-draft-n-max', 'temp', 'repeat-penalty',
               'min-p', 'batch-size', 'ubatch-size', 'pooling', 'embd-normalize'}


def presets(path):
    ini = configparser.ConfigParser(interpolation=None)
    with Path(path).open() as f:
        ini.read_file(f)
    common = dict(ini['*']) if '*' in ini else {}
    return {name: {**common, **dict(ini[name])} for name in ini.sections() if name != '*'}


def command(alias, values, serving):
    if not isinstance(serving, dict) or set(serving) != {'host', 'port'} or serving['host'] != '127.0.0.1':
        raise ValueError('serving config must contain only host=127.0.0.1 and port')
    if type(serving['port']) is not int or not 1024 <= serving['port'] <= 65535:
        raise ValueError('port must be an integer from 1024 to 65535')
    if set(values) - VALUE_FLAGS - set(BOOL_FLAGS):
        raise ValueError('unknown preset keys: ' + ', '.join(sorted(set(values) - VALUE_FLAGS - set(BOOL_FLAGS))))
    if not values.get('model'):
        raise ValueError('model path is required')
    cmd = [str(BINARY), '--host', serving['host'], '--port', str(serving['port']), '--alias', alias]
    for key, value in values.items():
        if key in BOOL_FLAGS:
            if value not in ('0', '1'):
                raise ValueError(f'{key} must be 0 or 1')
            flag = BOOL_FLAGS[key][0 if value == '1' else 1]
            if flag:
                cmd.append(flag)
        else:
            cmd.extend(['--' + key, value])
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('alias', nargs='?')
    p.add_argument('--list', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--presets', type=Path, default=ROOT / 'llama-models.ini')
    p.add_argument('--config', type=Path, default=ROOT / 'config/serving.json')
    p.add_argument('--port', type=int)
    p.add_argument('--device', help='explicit device override; no automatic fallback')
    p.add_argument('--tensor-split', help='comma-separated llama-server proportions')
    p.add_argument('--ctx-size', type=int)
    a = p.parse_args(argv)
    try:
        registry = presets(a.presets)
        if a.list:
            for alias, preset in sorted(registry.items()):
                print(f"{alias}  ({Path(preset['model']).name})")
                if preset.get('mmproj'):
                    print(f"  projector: {Path(preset['mmproj']).name}")
            return 0
        if a.alias not in registry:
            raise ValueError('unknown alias; use --list')
        serving = json.loads(a.config.read_text())
        if a.port is not None:
            serving['port'] = a.port
        values = registry[a.alias]
        for key in ('device', 'tensor_split', 'ctx_size'):
            value = getattr(a, key)
            if value is not None:
                if key == 'ctx_size' and value <= 0:
                    raise ValueError('ctx-size must be positive')
                values[key.replace('_', '-')] = str(value)
        cmd = command(a.alias, values, serving)
        if not a.execute:
            print(json.dumps({'plan_only': True, 'alias': a.alias, 'command': cmd}, indent=2))
            return 0
        for key in ('model', 'mmproj'):
            if key in values and not Path(values[key]).is_file():
                raise ValueError(f'missing {key}: {values[key]}')
        if not os.access(BINARY, os.X_OK):
            raise ValueError('build the pinned llama.cpp runtime first')
        with socket.socket() as probe:
            probe.bind((serving['host'], serving['port']))
        # Bind is advisory; the server still fails if another process wins the race.
        # Exec preserves direct signal handling and returns the actual server status.
        os.execv(str(BINARY), cmd)
    except (OSError, ValueError, configparser.Error) as e:
        print(f'Serve refused: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
