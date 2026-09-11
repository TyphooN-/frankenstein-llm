"""Passive performance observations; no inference, scoring, or benchmark policy.

The supervisor supplies a unique sidecar path per attempt. No prompts, outputs,
headers or arbitrary runtime fields are persisted. Missing timings remain absent.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time

LIMIT = 256
_written = 0
_default_target = None


def number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or value > 1e100 or not math.isfinite(value) or (positive and value == 0):
        return None
    return value


def observation(response=None, elapsed=None, *, operation='inference', model=None,
                completion_tokens=None, audio_seconds=None, items=None):
    row = {'schema': 1, 'kind': 'qualification-observation',
           'benchmark': False, 'operation': str(operation)[:100],
           'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'pid': os.getpid(), 'comparability': 'uncontrolled; includes current host load'}
    if isinstance(model, str):
        row['model'] = model[:200]
    seconds = number(elapsed, positive=True)
    if seconds is not None:
        row['wall_seconds'] = seconds
    response = response if isinstance(response, dict) else {}
    usage = response.get('usage')
    usage = usage if isinstance(usage, dict) else {}
    timings = response.get('timings')
    timings = timings if isinstance(timings, dict) else {}
    for key in ('prompt_tokens', 'completion_tokens'):
        value = number(usage.get(key))
        if value is not None:
            row[key] = value
    if number(completion_tokens) is not None:
        row['completion_tokens'] = completion_tokens
    from_runtime = False
    for source, target in (('prompt_per_second', 'prompt_tokens_per_second'),
                           ('predicted_per_second', 'decode_tokens_per_second'),
                           ('prompt_ms', 'prompt_milliseconds'),
                           ('predicted_ms', 'decode_milliseconds'),
                           ('prompt_n', 'runtime_prompt_tokens'),
                           ('predicted_n', 'runtime_completion_tokens')):
        value = number(timings.get(source))
        if value is not None:
            row[target] = value
            from_runtime = True
    # Attribute only what the runtime actually supplied. Testing key *names*
    # here credited response.timings with prompt_tokens, which comes from
    # usage: a timings block carrying nothing usable then produced a row that
    # named a source for numbers it had not contributed.
    if from_runtime:
        row['runtime_timing_source'] = 'response.timings'
    tokens = row.get('completion_tokens', row.get('runtime_completion_tokens'))
    if tokens is not None and seconds:
        row['end_to_end_tokens_per_second'] = tokens / seconds
        row['end_to_end_note'] = 'includes prefill and request overhead; not decode-only'
    audio = number(audio_seconds, positive=True)
    if audio is not None and seconds:
        row['audio_seconds'] = audio
        row['real_time_factor'] = seconds / audio
    count = number(items)
    if count is not None and seconds:
        row['items'] = count
        row['items_per_second'] = count / seconds
    row['token_rate_available'] = any(k in row for k in (
        'decode_tokens_per_second', 'prompt_tokens_per_second', 'end_to_end_tokens_per_second'))
    if not row['token_rate_available']:
        row['token_rate_unavailable_reason'] = 'not exposed or not applicable to this operation'
    return row


def emit(row):
    """Best-effort, capped observation output; metrics cannot fail a gate."""
    global _written, _default_target
    target = os.environ.get('QUALIFICATION_PERFORMANCE_PATH')
    if target is None and Path(sys.argv[0]).suffix == '.py':
        if _default_target is None:
            parent = Path(__file__).resolve().parents[1] / 'proofs/verification/qualification-supervisor/evidence'

            _default_target = parent / f'direct-{time.time_ns()}-{os.getpid()}.performance.jsonl'
        target = str(_default_target)
    if not target or _written >= LIMIT:
        return
    _written += 1
    try:
        data = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
        if len(data) > 8192:
            return
        # Each attempt has its own path; each append is one write, including
        # subprocess ASR observations. Never truncate another process's samples.
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except (OSError, ValueError, TypeError):
        return


def guarded(function, *args):
    """Run one telemetry step; a failure costs the sample and nothing else.

    Both callers sit in a ``finally`` beside a real gate result. Left unguarded,
    a device synchronize that raises after a failed generate -- or any accounting
    error -- would replace the gate's own exception, or turn a passing gate into
    a failing one. Losing an observation is the correct price.
    """
    if function is None:
        return
    try:
        function(*args)
    except Exception:                                           # noqa: BLE001
        pass


def record(values, operation, model, elapsed):
    row = observation(values.get('response'), elapsed, operation=operation, model=model,
                      completion_tokens=values.get('completion_tokens'),
                      audio_seconds=values.get('audio_seconds'), items=values.get('items'))
    row['request_failed'] = values.get('failed', False)
    emit(row)


@contextmanager
def measure(operation, *, model=None, synchronize=None):
    """Time existing work. Caller may fill response/count/duration fields."""
    values = {}
    guarded(synchronize)
    start = time.perf_counter()
    try:
        yield values
    except BaseException:
        values['failed'] = True
        raise
    finally:
        # Elapsed is read after the synchronize so queued device work is inside
        # the interval, exactly as before.
        guarded(synchronize)
        guarded(record, values, operation, model, time.perf_counter() - start)


def synchronize_devices():
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def call(function, *args, operation, model_id=None, mode='tokens', audio_duration=None, **kwargs):
    """Observe one existing in-process call without changing its arguments."""
    with measure(operation, model=model_id, synchronize=synchronize_devices) as values:
        values['audio_seconds'] = audio_duration
        result = function(*args, **kwargs)
        try:
            if mode == 'tokens':
                output = getattr(result, 'sequences', result)
                prompt = kwargs.get('input_ids')
                # These callers are decoder-only; input tokens are not output.
                if prompt is not None and len(output.shape) == 2:
                    values['completion_tokens'] = int(output.shape[0] *
                                                       (output.shape[1] - prompt.shape[1]))
            elif mode == 'audio':
                waves, rate = result
                if rate > 0:
                    wave = waves[0] if isinstance(waves, list) else waves
                    values['audio_seconds'] = int(math.prod(wave.shape)) / rate
            elif mode == 'items':
                values['items'] = int(result.shape[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            pass  # Unknown output shape is unavailable telemetry, not gate failure.
        return result
