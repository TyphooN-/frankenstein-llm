#!/usr/bin/env python3
"""Low-overhead GPU telemetry for the UI-TARS admission gate.

Telemetry is the only continuous record of what the three GPUs actually did, so
it is written to disk as it is collected rather than buffered until the end. A
run that dies without an exit path -- a host crash, an OOM kill, SIGKILL -- still
leaves the JSONL behind, and the aggregate can be rebuilt from it afterwards by
``summarize_records``. That is deliberate: the 2026-09-01T21:06 six-block run
vanished with the host under it, and the JSONL was the only surviving proof that
all three adapters had been loaded and busy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

CARDS = ("card0", "card1", "card2")
DEFAULT_INTERVAL = 0.1
# Line-buffered writes survive a process kill; only a host crash can lose them,
# and this bounds that loss to a few seconds instead of the whole run.
FSYNC_EVERY_SAMPLES = 50


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def sample_gpu_sysfs(root: Path = Path("/sys/class/drm")) -> dict:
    sample = {"monotonic": time.monotonic(), "cards": {}}
    for card in CARDS:
        device = root / card / "device"
        power_paths = sorted((device / "hwmon").glob("hwmon*/power1_average"))
        sample["cards"][card] = {
            "vram_used_bytes": _read_int(device / "mem_info_vram_used"),
            "busy_percent": _read_int(device / "gpu_busy_percent"),
            "power_microwatts": _read_int(power_paths[0]) if power_paths else None,
        }
    return sample


def _span_seconds(samples: list[dict]) -> float | None:
    """Monotonic span actually covered by the samples, not an assumed rate."""
    stamps = [s["monotonic"] for s in samples if isinstance(s.get("monotonic"), (int, float))]
    if len(stamps) < 2:
        return None
    return round(max(stamps) - min(stamps), 3)


def summarize_samples(samples: list[dict], baseline: dict,
                      interval: float = DEFAULT_INTERVAL) -> dict:
    cards = {}
    for card in CARDS:
        rows = [s["cards"][card] for s in samples if card in s.get("cards", {})]
        busy = [r["busy_percent"] for r in rows if r.get("busy_percent") is not None]
        power = [r["power_microwatts"] for r in rows if r.get("power_microwatts") is not None]
        vram = [r["vram_used_bytes"] for r in rows if r.get("vram_used_bytes") is not None]
        base = baseline.get("cards", {}).get(card, {})
        base_power = base.get("power_microwatts")
        base_vram = base.get("vram_used_bytes")
        cards[card] = {
            "samples": len(rows),
            "readings": len(busy),
            "busy_max_percent": max(busy) if busy else None,
            "busy_mean_percent": round(sum(busy) / len(busy), 2) if busy else None,
            "busy_samples_ge_20": sum(value >= 20 for value in busy),
            "power_baseline_microwatts": base_power,
            "power_max_microwatts": max(power) if power else None,
            "power_peak_delta_microwatts": (
                max(power) - base_power if power and base_power is not None else None),
            "vram_baseline_bytes": base_vram,
            "vram_max_bytes": max(vram) if vram else None,
            "vram_peak_delta_bytes": (
                max(vram) - base_vram if vram and base_vram is not None else None),
        }
    return {
        "interval_seconds": interval,
        "sample_count": len(samples),
        "duration_seconds": _span_seconds(samples),
        "cards_reporting": sorted(c for c, v in cards.items() if v["readings"]),
        "cards": cards,
    }


def read_records(path: Path) -> list[dict]:
    """Load whatever the sampler managed to write, tolerating a torn last line.

    A crash mid-write leaves a partial final record. Discarding it is correct;
    refusing to read the file at all would throw away the entire run's evidence.
    """
    records: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue  # torn trailing write from an abrupt death
    return records


def summarize_records(path: Path, baseline: dict | None = None,
                      interval: float = DEFAULT_INTERVAL) -> dict:
    """Rebuild the aggregate from the on-disk JSONL after an abrupt death."""
    records = read_records(path)
    if baseline is None:
        baseline = records[0] if records else {"cards": {}}
    summary = summarize_samples(records, baseline, interval)
    summary["evidence_path"] = str(path)
    summary["reconstructed_from_disk"] = True
    return summary


class GpuTelemetry:
    """Sample all DRM GPUs continuously and retain compact aggregate evidence."""

    def __init__(self, evidence_path: Path, interval: float = DEFAULT_INTERVAL):
        self.evidence_path = evidence_path
        self.interval = interval
        self.baseline = sample_gpu_sysfs()
        self.samples: list[dict] = []
        self.records_written = 0
        self.sampler_errors: list[str] = []
        self.stop_reason: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._summary: dict | None = None

    def start(self) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text("")
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        with self.evidence_path.open("a") as handle:
            while not self._stop.is_set():
                try:
                    sample = sample_gpu_sysfs()
                    self.samples.append(sample)
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                    handle.flush()
                    self.records_written += 1
                    if self.records_written % FSYNC_EVERY_SAMPLES == 0:
                        os.fsync(handle.fileno())
                except Exception as error:  # noqa: BLE001 - telemetry must not kill the gate
                    if len(self.sampler_errors) < 10:
                        self.sampler_errors.append(f"{type(error).__name__}: {error}"[:200])
                self._stop.wait(self.interval)
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    def stop(self, reason: str = "completed") -> dict:
        """Stop sampling and summarize. Idempotent: the signal path may race the
        normal ``finally`` path, and neither may lose or double-count evidence."""
        if self._summary is not None:
            return self._summary
        self.stop_reason = reason
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 40))
            self._thread = None
        summary = summarize_samples(self.samples, self.baseline, self.interval)
        summary["evidence_path"] = str(self.evidence_path)
        summary["records_written"] = self.records_written
        summary["stop_reason"] = reason
        summary["sampler_errors"] = list(self.sampler_errors)
        summary["reconstructed_from_disk"] = False
        self._summary = summary
        return summary
