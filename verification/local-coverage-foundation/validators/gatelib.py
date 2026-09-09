#!/usr/bin/env python3
"""Shared helpers for local-capability admission gates.

Design rule for every gate in this directory: a component is admitted only on
observed behaviour. "The server answered 200" and "the model card says so" are
not evidence. Each gate asserts a property of the *content* returned, and every
component additionally has to release its GPU memory when stopped.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import typing
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import qualification_performance as performance

EVIDENCE = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/evidence")

# Measured 2026-09-01 from sysfs. Index is the DRM card, not the ROCm ordinal;
# gates compare totals rather than trusting the two orderings to agree.
CARDS = {
    "card0": {"pci": "1002:73BF", "total": 17163091968, "name": "RX 6900 XT 16 GiB"},
    "card1": {"pci": "1002:73A1", "total": 32195477504, "name": "Radeon Pro V620 32 GiB"},
    "card2": {"pci": "1002:73BF", "total": 17163091968, "name": "RX 6900 XT 16 GiB"},
}


class GateFailure(AssertionError):
    """Raised when observed behaviour does not satisfy an admission gate."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def vram_used() -> dict[str, int]:
    """Per-card VRAM bytes in use, read from sysfs. No GPU work is submitted."""
    readings = {}
    for card in CARDS:
        path = Path(f"/sys/class/drm/{card}/device/mem_info_vram_used")
        try:
            readings[card] = int(path.read_text().strip())
        except (OSError, ValueError):
            readings[card] = -1
    return readings


def vram_residue(baseline: dict[str, int], released: dict[str, int]) -> tuple[dict[str, int], list[str]]:
    """Per-card VRAM still held after unload, plus the cards that could not be read.

    ``vram_used`` reports -1 for a card whose sysfs node did not answer. Treating
    that as a number makes ``released - baseline`` hugely negative, which sails
    under any residue tolerance and turns "we could not measure" into "it
    released cleanly". Unreadable cards are therefore returned separately so a
    caller can fail on them instead of scoring them.
    """
    residue: dict[str, int] = {}
    unreadable: list[str] = []
    for card, base in baseline.items():
        if base < 0:
            unreadable.append(card)
            continue
        after = released.get(card, -1)
        if after < 0:
            unreadable.append(card)
            continue
        residue[card] = after - base
    return residue, sorted(unreadable)


def release_torch_memory(torch) -> list[str]:
    """Release runtime-owned BLAS workspaces after callers drop model references.

    empty_cache cannot release an allocated BLAS workspace. Synchronize every
    device first, clear those workspace references, then empty the allocator.
    Return cleanup errors so a finally block can retain its original verdict.
    This does not relax allocator or card-level unload checks.
    """
    problems = []

    def attempt(label, action):
        try:
            action()
        except Exception as error:
            problems.append(f"{label}: {type(error).__name__}: {error}"[:300])

    try:
        if not torch.cuda.is_available():
            return problems
        devices = range(torch.cuda.device_count())
    except Exception as error:
        return [f"device discovery: {type(error).__name__}: {error}"[:300]]
    for index in devices:
        attempt(f"synchronize cuda:{index}", lambda index=index: torch.cuda.synchronize(index))
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if callable(clear):
        attempt("BLAS workspace release", clear)
    attempt("allocator cache release", torch.cuda.empty_cache)
    return problems


def allocator_report() -> dict[str, dict[str, int]] | None:
    """Bytes this process's tensor allocator still holds, per device, or None.

    Read-only accounting: it submits no work and allocates nothing. ``None``
    means the question could not be answered -- no torch, no initialised
    runtime, an accounting call that raised -- which is not the same answer as
    zero and never relaxes a verdict.
    """
    try:
        import torch
    except Exception:                                           # noqa: BLE001
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        return {f"cuda:{index}": {
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index))}
                for index in range(torch.cuda.device_count())}
    except Exception:                                           # noqa: BLE001
        return None


def allocator_outstanding(report) -> dict[str, dict[str, int]] | None:
    """Devices whose allocator still holds bytes, or None if it cannot be read."""
    if not isinstance(report, dict) or not report:
        return None
    outstanding = {}
    for device, entry in report.items():
        if not isinstance(entry, dict):
            return None
        try:
            held = {name: int(entry[name])
                    for name in ("allocated_bytes", "reserved_bytes")}
        except (KeyError, TypeError, ValueError):
            return None
        if any(value < 0 for value in held.values()):
            return None
        if any(held.values()):
            outstanding[device] = held
    return outstanding


def unload_verdict(baseline: dict[str, int], released: dict[str, int],
                   tolerance: int = 256 * 1024 * 1024, allocator=None) -> dict:
    """Score a clean-unload check so that missing evidence never reads as a pass.

    Card-level VRAM is a reading about the *host*. For a gate that stops a
    sidecar it is also a reading about the model, because the process holding
    the weights exits. For a gate that loads a model in-process it is not: the
    HIP context, its compiled kernels and its BLAS workspaces are created on
    first device use and live until the process does, so a few hundred megabytes
    per card survive a correct unload and were being reported as "VRAM still
    held after unload". The three in-process gates had each answered that with a
    different tolerance -- 256, 512 and 768 MiB -- which is what tuning a
    constant around the wrong measurement looks like.

    ``allocator`` adds framework accounting. Zero allocator bytes cannot prove
    that arbitrary device residue belongs to the runtime: direct allocations,
    other processes, and driver leaks are outside that allocator. Preserve the
    card threshold until process-exit or independently attributed context proof
    exists; never turn unexplained residue into a pass.
    """
    residue, unreadable = vram_residue(baseline, released)
    problems = []
    if unreadable:
        problems.append(f"VRAM unreadable for {unreadable}; release is unproven")
    if not residue:
        problems.append("no card produced a usable before/after VRAM pair")
    over = {card: value for card, value in residue.items() if value > tolerance}
    verdict = {
        "vram_residue_bytes": residue,
        "unreadable_cards": unreadable,
        "tolerance_bytes": tolerance,
    }
    outstanding = allocator_outstanding(allocator)
    if outstanding is None:
        if over:
            problems.append(f"VRAM still held after unload: {over}")
    else:
        verdict["allocator_bytes"] = allocator
        if outstanding:
            # Stricter than the card-level check it replaces: a live tensor is a
            # failed unload at any size, including one under the tolerance.
            problems.append(f"model memory still allocated after unload: {outstanding}")
        elif over:
            verdict["unattributed_device_bytes"] = over
            verdict["note"] = "zero tensor allocator bytes do not attribute device residue"
            problems.append(f"VRAM release remains unproven: {over}")
    verdict["problems"] = problems
    verdict["pass"] = not problems
    return verdict


def post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with performance.measure("http-inference", model=payload.get("model")) as observed:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        observed["response"] = result
        if "input" in payload:
            observed["items"] = len(payload["input"]) if isinstance(payload["input"], list) else 1
        elif isinstance(payload.get("documents"), list):
            observed["items"] = len(payload["documents"])
        return result


def wait_healthy(base_url: str, timeout: int = 900) -> float:
    """Block until llama-server reports ready. Returns seconds waited."""
    started = time.monotonic()
    deadline = started + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last = error
        time.sleep(2)
    raise GateFailure(f"{base_url} never became healthy within {timeout}s (last: {last!r})")


def systemd(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def cosine(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


def unload_gate(unit: str, baseline: dict[str, int], tolerance: int = 256 * 1024 * 1024) -> dict:
    """Stop a sidecar and prove its VRAM actually came back.

    A model that loads and answers but leaks VRAM on stop is not admissible: the
    one-resident-large-model policy depends on eviction genuinely freeing memory.
    """
    systemd("stop", unit)
    verdict: dict = {"vram_residue_bytes": {}, "unreadable_cards": sorted(baseline),
                     "tolerance_bytes": tolerance, "problems": ["unload never sampled"],
                     "pass": False}
    for _ in range(30):
        time.sleep(2)
        verdict = unload_verdict(baseline, vram_used(), tolerance)
        if verdict["pass"]:
            break
    active = systemd("is-active", unit).stdout.strip()
    check(active != "active", f"{unit} still active after stop")
    check(verdict["pass"], f"{unit} unload not proven: {verdict['problems']}")
    return {"unit": unit, "residue_bytes": verdict["vram_residue_bytes"],
            "unreadable_cards": verdict["unreadable_cards"],
            "tolerance_bytes": tolerance, "pass": True}


def ensure_unloaded(summary: dict, unit: str, baseline: dict[str, int] | None,
                    tolerance: int = 256 * 1024 * 1024) -> None:
    """Stop a sidecar even when the gate failed before its unload check.

    ``unload_gate`` lives on the success path, so any failing assertion used to
    return through the error handler with the model still resident. That breaks
    the one-resident-large-model policy and stalls the serialized qualification, whose
    quiet-host check treats a live unmanaged llama-server as a conflict and waits
    for it. Cleanup is therefore unconditional, and it is recorded as cleanup:
    an unload check that never ran is not a passing unload check.

    Callers invoke this from a ``finally``, so it must not raise. An exception
    escaping a ``finally`` would replace the gate's own failure and skip
    ``record``, leaving the run with no evidence file at all -- the one outcome a
    gate may never produce. Anything that goes wrong here is recorded instead.
    """
    if summary.get("unload"):
        return
    report: dict = {
        "unit": unit,
        "ran_as": "cleanup",
        "reason": "the gate failed before its unload check; the sidecar was stopped anyway",
        "pass": False,
    }
    summary["unload"] = report
    try:
        stop = systemd("stop", unit)
        report["stop_returncode"] = stop.returncode
        settled = None
        for _ in range(30):
            time.sleep(2)
            if systemd("is-active", unit).stdout.strip() == "active":
                continue
            # The unit is gone; give the driver a moment to hand the memory back
            # before deciding what the cleanup actually released.
            settled = vram_used()
            if baseline is None or unload_verdict(baseline, settled, tolerance)["pass"]:
                break
        report["still_active"] = systemd("is-active", unit).stdout.strip() == "active"
        report["vram_release"] = (unload_verdict(baseline, settled, tolerance)
                                  if baseline and settled is not None else None)
    except Exception as error:                                  # noqa: BLE001
        report["cleanup_error"] = f"{type(error).__name__}: {error}"[:300]


def managed_sidecar(unit: str, base_url: str, timeout: int = 900) -> tuple[dict, float]:
    """Stop the unit, sample a true idle VRAM baseline, then start and wait.

    Sampling the baseline while the model is already resident makes the unload
    check compare a loaded state against itself, which can pass without proving
    anything. Gates therefore own the whole lifecycle.
    """
    systemd("stop", unit)
    for _ in range(30):
        if systemd("is-active", unit).stdout.strip() != "active":
            break
        time.sleep(2)
    time.sleep(5)
    baseline = vram_used()
    result = systemd("start", unit)
    check(result.returncode == 0, f"could not start {unit}: {result.stderr.strip()}")
    waited = wait_healthy(base_url, timeout)
    return baseline, waited


def exit_after_verdict(code: int) -> typing.NoReturn:
    """End the process on the gate's own verdict rather than on ROCm's teardown.

    ROCm's HSA runtime segfaults inside its own process-exit teardown on this
    host. Observed three times on boot 1669f3ad-9ef9-4a6e-a5aa-f61a9c4ba276,
    always the same signature -- ``libhsa-runtime64.so.1.18.0``, error 4, at
    interpreter finalization, with no accompanying kernel GPU fault, RCU stall,
    OOM or MCE. The ASR gate hit it one second after it had transcribed the
    fixture exactly, scored a clean unload, and written ``gate-asr.json`` with
    ``"pass": true``; the supervisor then read ``Popen.wait() == -11`` and
    recorded the gate as failed.

    This does not soften any gate. Every assertion has already run and the
    artifact is already on disk by the time this is called -- what is dropped is
    only the interpreter finalization that follows the verdict. A crash *before*
    the verdict still escapes as an exception, is recorded in the artifact, and
    still fails the gate. Callers must therefore invoke this last, after the
    evidence file is written.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def record(name: str, summary: dict) -> int:
    """Persist a gate result and return the process exit code it implies."""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    summary["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    passed = bool(summary.get("pass"))
    summary["pass"] = passed
    (EVIDENCE / f"{name}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1
