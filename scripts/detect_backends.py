#!/usr/bin/env python3
"""Detect which llama.cpp backends this host can build, and for which GPUs.

``scripts/build-llama-cpp.sh`` builds one llama.cpp per backend found here
(ADR 0009). Detection reads the devices, not the toolchain: a GPU with no
toolchain installed is reported with ``toolchain: false`` so the build can say
what to install, instead of silently building for fewer devices than the host
has.

* HIP targets come from the KFD topology. ``gfx_target_version`` encodes
  major, minor and stepping as decimal pairs (100300 -> ``gfx1030``); CPU nodes
  report 0 and are skipped.
* CUDA architectures come from ``nvidia-smi``'s ``compute_cap`` (``8.6`` ->
  ``86``). No ``nvidia-smi``, or a driver that answers with an error, is "no
  NVIDIA devices", reported with the reason.

``HIP_GPU_TARGETS`` and ``CUDA_ARCHITECTURES`` (semicolon- or comma-separated)
override detection, for building on a host without the GPU installed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

KFD_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")
BACKENDS = ("hip", "cuda", "vulkan", "cpu")


def gfx_name(version: int) -> str | None:
    """``gfx_target_version`` as the ``gfx`` target name the HIP compiler takes."""
    if version <= 0:
        return None
    major, minor, stepping = version // 10000, (version // 100) % 100, version % 100
    return f"gfx{major}{minor}{stepping:x}"


def hip_targets(kfd_root: Path = KFD_ROOT) -> list[str]:
    targets: set[str] = set()
    for properties in sorted(kfd_root.glob("*/properties")):
        try:
            text = properties.read_text()
        except OSError:
            continue
        match = re.search(r"^gfx_target_version (\d+)$", text, re.MULTILINE)
        name = gfx_name(int(match.group(1))) if match else None
        if name:
            targets.add(name)
    return sorted(targets)


def cuda_architectures(run=subprocess.run) -> tuple[list[str], str | None]:
    """Compute capabilities of the NVIDIA GPUs present, and why none if none."""
    if shutil.which("nvidia-smi") is None and run is subprocess.run:
        return [], "nvidia-smi not found"
    try:
        result = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], f"nvidia-smi failed: {error}"
    if result.returncode != 0:
        return [], f"nvidia-smi exited {result.returncode}: {result.stderr.strip()[:200]}"
    architectures = set()
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", line)
        if match:
            architectures.add(f"{match.group(1)}{match.group(2)}")
    return sorted(architectures, key=int), None if architectures else "no NVIDIA GPU reported"


def override(name: str) -> list[str] | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    items = [item.strip() for item in re.split(r"[;,]", value) if item.strip()]
    if not items:
        raise ValueError(f"{name}: override contains no targets")
    return list(dict.fromkeys(items))


def find_nvcc() -> str | None:
    """PATH first, then common NVIDIA and distribution toolkit locations."""
    found = shutil.which("nvcc")
    if found:
        return found
    for path in ("/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"):
        if Path(path).is_file() and os.access(path, os.X_OK):
            return path
    return None


def detect(kfd_root: Path = KFD_ROOT, run=subprocess.run) -> dict:
    hip = override("HIP_GPU_TARGETS")
    hip_source = "HIP_GPU_TARGETS" if hip is not None else "kfd-topology"
    if hip is None:
        hip = hip_targets(kfd_root)
    cuda = override("CUDA_ARCHITECTURES")
    cuda_reason = None
    cuda_source = "CUDA_ARCHITECTURES" if cuda is not None else "nvidia-smi"
    if cuda is None:
        cuda, cuda_reason = cuda_architectures(run)
    nvcc = find_nvcc()
    return {
        "hip": {"targets": hip, "source": hip_source,
                "toolchain": shutil.which("hipconfig") is not None},
        "cuda": {"architectures": cuda, "source": cuda_source, "reason": cuda_reason,
                 "toolchain": nvcc is not None, "nvcc": nvcc},
        "vulkan": {"toolchain": shutil.which("glslc") is not None},
        "cpu": {"toolchain": True},
    }


def auto(detected: dict) -> list[str]:
    """Every GPU backend with a device; CPU only when there is no GPU at all."""
    chosen = []
    if detected["hip"]["targets"]:
        chosen.append("hip")
    if detected["cuda"]["architectures"]:
        chosen.append("cuda")
    return chosen or ["cpu"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--auto", action="store_true",
                        help="print the backends an automatic build selects, one per line")
    parser.add_argument("--get", metavar="BACKEND.FIELD",
                        help="print one field (lists joined with ';'), e.g. hip.targets")
    args = parser.parse_args(argv)
    try:
        detected = detect()
    except ValueError as error:
        parser.error(str(error))
    if args.auto:
        print("\n".join(auto(detected)))
    elif args.get:
        backend, _, name = args.get.partition(".")
        value = detected[backend][name]
        print(";".join(value) if isinstance(value, list) else ("" if value is None else value))
    else:
        print(json.dumps(detected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
