#!/usr/bin/env python3
"""Report VRAM per GPU against the ROCm index a preset actually names.

``llama-models.ini`` addresses cards as ``ROCm0``/``ROCm1``/``ROCm2`` and
``--tensor-split`` is positional, so this report is only useful if its row
labels are the indices the runtime uses. Counting ``/sys/class/drm/card*`` does
not produce those indices. A sysfs card number is a DRM minor handed out in
device-registration order across *every* DRM driver on the host; a ROCm index is
a position in the HIP device list, which is built from the KFD topology and then
filtered by ``ROCR_VISIBLE_DEVICES`` and ``HIP_VISIBLE_DEVICES``. The two agree
on this host today and diverge the moment a non-KFD display device appears, a
card fails to bind KFD, or either visibility variable is set -- and they diverge
silently, printing one card's bytes under another card's name.

Each ROCm index is therefore derived from the KFD topology and joined back to
sysfs by identity that both sides publish:

* ``drm_render_minor`` in a node's ``properties`` names the render node
  (``/sys/class/drm/renderD<minor>``) that the same physical device exposes.
* ``domain`` and ``location_id`` encode the PCI address, and a DRM card's
  ``device`` symlink resolves to that same address.

Both joins run and are compared. Agreement is the evidence that a row label is
correct; disagreement is reported, not resolved by preference. A device whose
sysfs attributes cannot be read keeps its row and its index, carrying the reason
instead of bytes: dropping it would renumber every device behind it, which is
the failure this module exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

KFD_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")
DRM_ROOT = Path("/sys/class/drm")

# HIP applies ROCR_VISIBLE_DEVICES at the ROCr agent level first, then indexes
# into what survives with HIP_VISIBLE_DEVICES, falling back to the CUDA spelling
# only when the HIP one is unset.
ROCR_VARIABLE = "ROCR_VISIBLE_DEVICES"
HIP_VARIABLES = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

MIB = 1 << 20


def read_properties(path: Path) -> dict[str, int]:
    """One KFD ``properties`` file as ``name -> integer``.

    Non-integer values are skipped rather than raised on: the file is a stable
    kernel interface but not a closed one, and an unparsable future field must
    not cost the caller every field beside it.
    """
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, _, raw = line.strip().partition(" ")
        try:
            values[key] = int(raw)
        except ValueError:
            continue
    return values


def pci_address(domain: int, location_id: int) -> str:
    """``domain:bus:device.function`` from the pair KFD publishes.

    KFD packs bus and devfn into ``location_id`` as ``(bus << 8) | devfn`` and
    reports the PCI domain separately, which is the same address sysfs uses for
    its device directory names.
    """
    bus = (location_id >> 8) & 0xFF
    device = (location_id >> 3) & 0x1F
    function = location_id & 0x07
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


def kfd_gpu_nodes(kfd_root: Path = KFD_ROOT) -> list[dict]:
    """KFD GPU nodes in node order, which is the order ROCr reports agents in.

    ``simd_count`` is the CPU/GPU discriminator: the host shows up as a node
    with no SIMDs and must not consume a ROCm index.
    """
    nodes: list[dict] = []
    for entry in sorted(kfd_root.glob("[0-9]*"), key=lambda p: int(p.name)):
        if not entry.name.isdigit():
            continue
        try:
            properties = read_properties(entry / "properties")
        except OSError:
            continue
        if properties.get("simd_count", 0) <= 0:
            continue
        try:
            name = (entry / "name").read_text().strip()
        except OSError:
            name = ""
        unique_id = properties.get("unique_id")
        nodes.append({
            "kfd_node": int(entry.name),
            "kfd_name": name,
            # rocminfo prints this same value as the agent Uuid, so it is the
            # one identifier a reader can compare against ROCm's own tooling.
            "uuid": None if unique_id is None else f"GPU-{unique_id:016x}",
            "pci": pci_address(properties.get("domain", 0),
                               properties.get("location_id", 0)),
            "drm_render_minor": properties.get("drm_render_minor"),
            "simd_count": properties.get("simd_count"),
            "gfx_target_version": properties.get("gfx_target_version"),
        })
    return nodes


def _select(nodes: list[dict], spec: str | None) -> list[dict]:
    """Apply one CUDA-style visibility list to an ordered device list.

    Indices and ``GPU-<uuid>`` names are both accepted, as HIP accepts them.
    Parsing stops at the first entry that does not resolve, and at a repeat,
    because that is what the runtime does: reporting the whole list would claim
    devices the runtime will not hand out.
    """
    if spec is None:
        return list(nodes)
    by_uuid = {node["uuid"]: index for index, node in enumerate(nodes)
               if node["uuid"] is not None}
    chosen: list[dict] = []
    taken: set[int] = set()
    for token in (part.strip() for part in spec.split(",")):
        if token in by_uuid:
            index = by_uuid[token]
        else:
            try:
                index = int(token)
            except ValueError:
                break
            if not 0 <= index < len(nodes):
                break
        if index in taken:
            break
        taken.add(index)
        chosen.append(nodes[index])
    return chosen


def visible_nodes(nodes: list[dict],
                  environ=None) -> tuple[list[dict], list[tuple[str, str]]]:
    """The nodes a HIP process would see, and the variables that decided it."""
    environ = os.environ if environ is None else environ
    applied: list[tuple[str, str]] = []
    agents = nodes
    if ROCR_VARIABLE in environ:
        agents = _select(agents, environ[ROCR_VARIABLE])
        applied.append((ROCR_VARIABLE, environ[ROCR_VARIABLE]))
    for name in HIP_VARIABLES:
        if name in environ:
            agents = _select(agents, environ[name])
            applied.append((name, environ[name]))
            break
    return agents, applied


def drm_cards_by_pci(drm_root: Path = DRM_ROOT) -> dict[str, Path]:
    """``PCI address -> /sys/class/drm/cardN`` for every primary DRM node."""
    cards: dict[str, Path] = {}
    for card in sorted(drm_root.glob("card[0-9]*")):
        if re.fullmatch(r"card\d+", card.name) is None:
            continue
        try:
            address = (card / "device").resolve(strict=True).name
        except OSError:
            continue
        cards.setdefault(address, card)
    return cards


def drm_card_for_render_minor(minor: int | None,
                              drm_root: Path = DRM_ROOT) -> Path | None:
    """The primary card sharing a PCI device with ``renderD<minor>``."""
    if minor is None:
        return None
    try:
        address = (drm_root / f"renderD{minor}" / "device").resolve(strict=True).name
    except OSError:
        return None
    return drm_cards_by_pci(drm_root).get(address)


def display_connectors(card: Path) -> list[str]:
    """Connected outputs on this card, which is what makes it a display GPU.

    Read rather than configured: a role label in a document can go stale against
    a moved cable, and the placement policy reserves headroom on whichever card
    is actually driving the desktop.
    """
    connected = []
    for status in sorted(card.parent.glob(f"{card.name}-*/status")):
        try:
            if status.read_text().strip() == "connected":
                connected.append(status.parent.name)
        except OSError:
            continue
    return connected


def _memory(card: Path) -> tuple[int | None, int | None, str | None]:
    device = card / "device"
    try:
        used = int((device / "mem_info_vram_used").read_text())
        total = int((device / "mem_info_vram_total").read_text())
    except (OSError, ValueError) as error:
        return None, None, f"VRAM counters unreadable: {type(error).__name__}"
    return used, total, None


def devices(kfd_root: Path = KFD_ROOT, drm_root: Path = DRM_ROOT,
            environ=None) -> tuple[list[dict], list[tuple[str, str]]]:
    """One row per ROCm index, in ROCm index order, complete or not."""
    nodes, applied = visible_nodes(kfd_gpu_nodes(kfd_root), environ)
    by_pci = drm_cards_by_pci(drm_root)
    rows: list[dict] = []
    for index, node in enumerate(nodes):
        row = dict(node, rocm_index=index, card=None, used_bytes=None,
                   total_bytes=None, display_connectors=[], problems=[])
        by_address = by_pci.get(node["pci"])
        by_render = drm_card_for_render_minor(node["drm_render_minor"], drm_root)
        card = by_address or by_render
        if by_address is not None and by_render is not None and by_address != by_render:
            # Two kernel-published joins that disagree mean the identity of this
            # row is unknown; naming one of them anyway is the false label this
            # module exists to avoid.
            row["problems"].append(
                f"PCI join {by_address.name} and render-node join "
                f"{by_render.name} disagree; identity unresolved")
            card = None
        elif card is None:
            row["problems"].append(
                f"no DRM card for PCI {node['pci']} or "
                f"renderD{node['drm_render_minor']}")
        if card is not None:
            row["card"] = card.name
            row["used_bytes"], row["total_bytes"], problem = _memory(card)
            if problem:
                row["problems"].append(problem)
            row["display_connectors"] = display_connectors(card)
        rows.append(row)
    return rows, applied


def format_row(row: dict) -> str:
    label = f"ROCm{row['rocm_index']}"
    where = row["card"] or "no card"
    identity = f"{label} ({where}, {row['pci']}, {row['uuid'] or 'no uuid'})"
    if row["used_bytes"] is None or row["total_bytes"] is None:
        detail = "used=       ? total=       ? free=       ?"
    else:
        free = row["total_bytes"] - row["used_bytes"]
        detail = (f"used={row['used_bytes'] / MIB:8.0f} MiB"
                  f" total={row['total_bytes'] / MIB:8.0f} MiB"
                  f" free={free / MIB:8.0f} MiB")
    role = "display" if row["display_connectors"] else "headless"
    line = f"{identity}: {detail} role={role}"
    if row["display_connectors"]:
        line += f" ({','.join(row['display_connectors'])})"
    for problem in row["problems"]:
        line += f"\n  ! {problem}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="machine-readable rows instead of a table")
    args = parser.parse_args(argv)
    rows, applied = devices()
    if args.json:
        print(json.dumps({"schema": "frankenstein-gpu-vram/1",
                          "visibility_filters": [
                              {"variable": name, "value": value}
                              for name, value in applied],
                          "devices": rows}, indent=2, sort_keys=True))
    else:
        for name, value in applied:
            print(f"# {name}={value} applied; ROCm indices are filtered")
        if not rows:
            print("no ROCm-visible GPUs")
        for row in rows:
            print(format_row(row))
    # A row that could not be identified is a reporting failure, not a note in
    # the margin: a caller sizing a tensor-split off this output must not treat
    # an unresolved device as absent.
    return 1 if any(row["problems"] for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
