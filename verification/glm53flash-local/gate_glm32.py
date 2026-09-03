#!/usr/bin/env python3
"""Guarded GLM-5.3-Flash 32K functional gate. No throughput measurement."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/validators")
from gatelib import unload_verdict, vram_used  # noqa: E402

ROOT = Path("/home/typhoon/git/frankenstein-llm/verification/glm53flash-local")
BINARY = ROOT / "build-pr27752-c9ddd682/bin/llama-server"
MODEL = Path("/home/typhoon/git/frankenstein-llm/models/glm53flash-regular-iq3xxs/GLM-5.3-Flash-IQ3_XXS-00001-of-00015.gguf")
LOG = ROOT / "glm32-server.log"
EVIDENCE = ROOT / "gate-glm32.json"
BASE = "http://127.0.0.1:8093"
MIN_AVAILABLE = 8 * 1024**3
MAX_SWAP_DELTA = 4 * 1024**3
VRAM_TOLERANCE = 768 * 1024**2
CARDS = ("card0", "card1", "card2")
COMMAND = [
    str(BINARY), "--host", "127.0.0.1", "--port", "8093",
    "--model", str(MODEL), "--alias", "glm53flash-iq3xxs-32k",
    "--ctx-size", "32768", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    "--jinja", "--reasoning", "off", "--reasoning-format", "none",
    "--device", "ROCm0,ROCm1,ROCm2", "--n-gpu-layers", "auto",
    "--split-mode", "layer",
    "--fit", "on", "--fit-target", "1536,1024,4096", "--fit-ctx", "32768",
    "--flash-attn", "on", "--load-mode", "mmap", "--parallel", "1", "--no-perf",
]


def memory() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
            values[key] = int(raw.split()[0]) * 1024
    return values


def vram() -> dict[str, int]:
    return vram_used()


def unreadable_cards(readings: dict[str, int]) -> list[str]:
    """Return required cards whose VRAM reading is absent or unreadable."""
    return sorted(card for card in CARDS if readings.get(card, -1) < 0)


def request(path: str, payload: dict | None = None, timeout: int = 600) -> tuple[int, dict | str]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode(errors="replace")
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw


def chat(prompt: str, max_tokens: int, response_format: dict | None = None) -> str:
    payload = {
        "model": "glm53flash-iq3xxs-32k",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    status, body = request("/v1/chat/completions", payload)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"chat failed: HTTP {status}: {body!r}")
    return body["choices"][0]["message"]["content"]


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main() -> int:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    baseline_mem = memory()
    baseline_vram = vram()
    summary: dict = {
        "gate": "glm-32k", "model": str(MODEL), "command": COMMAND,
        "runtime_revision": "c9ddd6821c93871c53741344d35d1e440d60d9ea", "runtime_pr": 27752,
        "started_at": started_at, "throughput_measured": False,
        "baseline_memory": baseline_mem, "baseline_vram": baseline_vram,
        "minimum_mem_available_bytes": baseline_mem["MemAvailable"],
        "peak_vram": baseline_vram.copy(), "samples": 0,
    }
    process = None
    received_signal: int | None = None
    guard_stop = threading.Event()
    guard_errors: list[str] = []
    guard_thread = None
    log_handle = LOG.open("w")

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        guard_errors.append(f"external signal received: {signal.Signals(signum).name}")
        if process is not None:
            terminate(process)

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        missing = unreadable_cards(baseline_vram)
        if missing:
            raise RuntimeError(f"baseline VRAM telemetry unreadable for {missing}")
        process = subprocess.Popen(COMMAND, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)

        def resource_guard() -> None:
            while not guard_stop.wait(2):
                mem = memory()
                gpu = vram()
                missing = unreadable_cards(gpu)
                if missing:
                    guard_errors.append(f"VRAM telemetry unreadable for {missing}")
                    terminate(process)
                    return
                summary["samples"] += 1
                summary["minimum_mem_available_bytes"] = min(summary["minimum_mem_available_bytes"], mem["MemAvailable"])
                summary["peak_vram"] = {card: max(summary["peak_vram"][card], gpu[card]) for card in CARDS}
                if mem["MemAvailable"] < MIN_AVAILABLE:
                    guard_errors.append(f"memory guard tripped: MemAvailable={mem['MemAvailable']}")
                    terminate(process)
                    return
                swap_delta = baseline_mem["SwapFree"] - mem["SwapFree"]
                if swap_delta > MAX_SWAP_DELTA:
                    guard_errors.append(f"swap guard tripped: delta={swap_delta}")
                    terminate(process)
                    return

        guard_thread = threading.Thread(target=resource_guard, name="glm32-resource-guard", daemon=True)
        guard_thread.start()
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if process.poll() is not None:
                reason = guard_errors[-1] if guard_errors else f"server exited during load with status {process.returncode}"
                raise RuntimeError(reason)
            try:
                status, health = request("/health", timeout=5)
                if status == 200:
                    summary["health"] = health
                    break
            except (OSError, urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise RuntimeError("server did not become healthy within 1800 seconds")

        status, models = request("/v1/models")
        if status != 200:
            raise RuntimeError(f"models endpoint failed: HTTP {status}")
        summary["models"] = models

        exact = chat("Reply with exactly the single word pong and nothing else.", 16).strip()
        summary["exact_output"] = exact
        if exact.lower().strip(". `\n") != "pong":
            raise AssertionError(f"exact response failed: {exact!r}")

        coding = chat(
            "Write only a Python function safe_div(a, b) that returns None when b is zero and otherwise returns a / b. No prose or markdown.",
            128,
        ).strip()
        summary["coding_output"] = coding
        if "def safe_div" not in coding or "b == 0" not in coding or "a / b" not in coding:
            raise AssertionError(f"coding response failed: {coding!r}")

        structured = chat(
            "Return a JSON object with exactly these values: status is ok, count is 3, enabled is true.",
            64, {"type": "json_object"},
        ).strip()
        summary["structured_output"] = structured
        parsed = json.loads(structured)
        if parsed != {"status": "ok", "count": 3, "enabled": True}:
            raise AssertionError(f"structured response mismatch: {parsed!r}")

        malformed_status, malformed_body = request("/v1/chat/completions", {"model": "glm53flash-iq3xxs-32k", "messages": "invalid"})
        summary["malformed"] = {"status": malformed_status, "body": malformed_body}
        if malformed_status < 400:
            raise AssertionError(f"malformed request unexpectedly accepted: HTTP {malformed_status}")

        deltas = {card: summary["peak_vram"][card] - baseline_vram[card] for card in CARDS}
        summary["peak_vram_delta"] = deltas
        if not all(deltas[card] > 256 * 1024**2 for card in CARDS):
            raise AssertionError(f"not all GPUs materially used: {deltas}")
        if not (deltas["card1"] > deltas["card0"] > deltas["card2"]):
            raise AssertionError(f"GPU placement policy not met: {deltas}")
        summary["functional_pass"] = True
    except Exception as error:
        summary["functional_pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        guard_stop.set()
        if guard_thread is not None:
            guard_thread.join(timeout=5)
        if process is not None:
            terminate(process)
            summary["server_exit_status"] = process.returncode
        log_handle.close()
        settled: dict[str, int] = {}
        verdict = unload_verdict(baseline_vram, settled, VRAM_TOLERANCE)
        for _ in range(30):
            time.sleep(2)
            settled = vram()
            verdict = unload_verdict(baseline_vram, settled, VRAM_TOLERANCE)
            if verdict["pass"]:
                break
        summary["vram_after_unload"] = settled
        summary["unload"] = verdict
        summary["vram_residue"] = verdict["vram_residue_bytes"]
        summary["memory_after_unload"] = memory()
        summary["clean_unload"] = verdict["pass"]
        if received_signal is not None:
            summary["received_signal"] = signal.Signals(received_signal).name
        summary["pass"] = bool(summary.get("functional_pass") and summary["clean_unload"])
        summary["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        EVIDENCE.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
