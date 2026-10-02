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

The additive ``physical_inventory()`` / ``--physical-inventory`` mode reports
unfiltered KFD and NVIDIA management-tool observations. It does not establish
runtime indices or placement budgets. Legacy APIs and default CLI output keep
their ROCm semantics; no existing caller is migrated by this mode.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from pathlib import Path
import subprocess
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


NVIDIA_FIELDS = ("index", "uuid", "pci.bus_id", "name", "memory.total",
                 "memory.used", "display_active", "display_mode", "mig.mode.current")


def _nvidia_pci(value: str) -> str | None:
    """Validate before converting nvidia-smi's domain width to a sysfs key."""
    match = re.fullmatch(r"([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):"
                         r"([01][0-9a-fA-F])\.([0-7])", value)
    if match is None or int(match[1], 16) > 0xFFFF:
        return None
    return f"{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}"


def physical_inventory(kfd_root: Path = KFD_ROOT, drm_root: Path = DRM_ROOT,
                       environ=None, *, run=None) -> dict:
    """KFD + nvidia-smi physical observations, not placement/runtime authority.

    Visibility variables are recorded verbatim but never applied. All runtime
    indices remain unresolved, including AMD physical ordinals: legacy ROCm
    callers must keep using devices(). NVIDIA memory is physical-parent VRAM,
    never a MIG-instance budget. Only devices exposed by these sources can be
    reported; this is not a probe of inaccessible hardware or CUDA/HIP runtimes.
    """
    environ = os.environ if environ is None else environ
    amd, _ = devices(kfd_root, drm_root, environ={})
    for row in amd:
        row["rocm_physical_index"] = row.pop("rocm_index")
        row.update(vendor="amd", backend="HIP", runtime_index=None,
                   runtime_index_error="HIP runtime index unresolved in physical inventory; use legacy devices()",
                   placement_usable=False, display_mode=None, mig_mode=None,
                   display_active=bool(row["display_connectors"]) if row["card"] is not None else None)
    inventory = _nvidia_inventory(run=run)
    inventory["devices"] = amd + inventory["devices"]
    inventory["runtime_visibility_applied"] = False
    inventory["runtime_environment"] = {
        key: environ[key] for key in (ROCR_VARIABLE, *HIP_VARIABLES,
                                     "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES")
        if key in environ}
    return inventory


def _nvidia_inventory(*, run=None) -> dict:
    """Read-only management-tool observations, with failures distinct from absence."""
    run = subprocess.run if run is None else run
    inventory = {"schema": "frankenstein-gpu-physical-inventory/1",
                 "devices": [], "problems": []}
    try:
        result = run(["nvidia-smi", "--query-gpu=" + ",".join(NVIDIA_FIELDS),
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=10, check=False)
    except subprocess.TimeoutExpired:
        inventory["problems"].append("nvidia-smi timed out after 10 seconds")
        return inventory
    except (OSError, UnicodeError) as error:
        inventory["problems"].append(f"nvidia-smi unavailable: {type(error).__name__}")
        return inventory
    if result.returncode != 0:
        inventory["problems"].append(
            f"nvidia-smi exited {result.returncode}: {result.stderr.strip()[:200]}")
        return inventory
    rows = inventory["devices"]
    try:
        for values in csv.reader(io.StringIO(result.stdout), strict=True):
            if values:
                rows.append(_nvidia_row(values))
    except csv.Error as error:
        inventory["problems"].append(f"nvidia-smi malformed CSV: {error}")
    if not rows and not inventory["problems"]:
        inventory["problems"].append("nvidia-smi returned no physical GPU rows")
    # Detect all collisions before clearing any keys: early invalidation would
    # hide a later transitive collision or a third copy of the same identity.
    contested = set()
    for key in ("nvidia_smi_index", "pci", "uuid"):
        owners = {}
        for index, row in enumerate(rows):
            value = row[key]
            if value is None:
                continue
            if value in owners:
                contested.update((owners[value], index))
            else:
                owners[value] = index
    for index in sorted(contested):
        row = rows[index]
        row["problems"].append("duplicate physical identity; all joins unresolved")
        row["nvidia_smi_index"] = row["pci"] = row["uuid"] = None
    return inventory


def _nvidia_row(values: list[str]) -> dict:
    """Retain even malformed rows; validated keys never replace raw identifiers."""
    raw = dict(zip(NVIDIA_FIELDS, values))
    row = {
        "vendor": "nvidia", "backend": "CUDA", "raw": raw, "raw_fields": values,
        "nvidia_smi_index": None, "pci": None, "uuid": None, "name": None,
        "total_bytes": None, "used_bytes": None,
        "display_active": None, "display_mode": None, "mig_mode": None,
        "runtime_index": None,
        "runtime_index_error": "CUDA runtime index unresolved; independent runtime probe required",
        "placement_usable": False, "problems": [],
    }
    if len(values) != len(NVIDIA_FIELDS):
        row["problems"].append(
            f"malformed CSV row: expected {len(NVIDIA_FIELDS)} fields, found {len(values)}")
        return row
    fields = {key: value.strip() for key, value in raw.items()}
    row["pci"] = _nvidia_pci(fields["pci.bus_id"])
    row["name"] = fields["name"]
    for field in ("display_active", "display_mode", "mig.mode.current"):
        value = fields[field]
        key = "mig_mode" if field == "mig.mode.current" else field
        if value in ("Enabled", "Disabled"):
            row[key] = value.lower() if key == "mig_mode" else value == "Enabled"
        elif value not in ("[N/A]", "[Not Supported]", ""):
            row["problems"].append(f"invalid metadata {field}; state unknown")
    if row["mig_mode"] == "enabled":
        row["runtime_index_error"] += "; MIG physical parent is not a runtime device mapping"
    if re.fullmatch(r"[0-9]{1,10}", fields["index"]):
        row["nvidia_smi_index"] = int(fields["index"])
    if re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                    fields["uuid"]):
        row["uuid"] = "GPU-" + fields["uuid"][4:].lower()
    for key in ("nvidia_smi_index", "pci", "uuid"):
        if row[key] is None:
            row["problems"].append(f"invalid physical identity {key}; join unresolved")
    for field, key in (("memory.total", "total_bytes"), ("memory.used", "used_bytes")):
        value = fields[field]
        if re.fullmatch(r"[0-9]{1,16}", value) and (field != "memory.total" or int(value) > 0):
            row[key] = int(value) * MIB
        else:
            row["problems"].append(f"VRAM counter {field} unreadable or invalid")
    if (row["used_bytes"] is not None and row["total_bytes"] is not None
            and row["used_bytes"] > row["total_bytes"]):
        row["used_bytes"] = row["total_bytes"] = None
        row["problems"].append("VRAM used exceeds total; counters invalid")
    return row


def logical_card_paths(kfd_root: Path = KFD_ROOT, drm_root: Path = DRM_ROOT) -> dict[str, Path]:
    """``cardN`` label -> the DRM card directory that is physical ROCm index N now.

    Gate evidence has always labelled GPUs ``card0``/``card1``/``card2`` meaning
    ROCm0/1/2, which matched the sysfs names until a reinstall renumbered the DRM
    minors (2026-09-29: the GPUs became card1..card3 with no card0, and every
    VRAM read of ``/sys/class/drm/card0`` failed). The labels stay logical; this
    resolves them through the same KFD/PCI join as ``devices``. Visibility
    variables are deliberately ignored (``environ={}``): a gate that narrows
    ``HIP_VISIBLE_DEVICES`` must not renumber the physical cards it measures.
    Rows whose identity is unresolved are omitted, so callers read them as
    unreadable rather than as some other card's bytes.
    """
    rows, _ = devices(kfd_root, drm_root, environ={})
    return {f"card{row['rocm_index']}": drm_root / row["card"]
            for row in rows if row["card"] is not None and not any(
                "identity unresolved" in problem for problem in row["problems"])}


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


def _physical_main(as_json: bool) -> int:
    inventory = physical_inventory()
    if as_json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print("# physical observations only; runtime visibility not applied")
        for row in inventory["devices"]:
            source = (f" nvidia-smi index={row['nvidia_smi_index']}"
                      if row["vendor"] == "nvidia" else "")
            print(f"{row['vendor']}/{row['backend']} physical: "
                  f"pci={row['pci']} uuid={row['uuid']}{source} "
                  f"used_bytes={row['used_bytes']} total_bytes={row['total_bytes']} "
                  f"display_active={row['display_active']} mig_mode={row['mig_mode']} "
                  "runtime=unresolved placement=unusable")
            print(f"  ! {row['runtime_index_error']}")
            for problem in row["problems"]:
                print(f"  ! {problem}")
        for problem in inventory["problems"]:
            print(f"! {problem}")
    # Exit status measures observation integrity, not suitability for placement.
    # Unresolved runtime mappings are expected in this explicitly physical mode.
    return 1 if inventory["problems"] or any(row["problems"] for row in inventory["devices"]) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="machine-readable rows instead of a table")
    parser.add_argument("--physical-inventory", action="store_true",
                        help="unfiltered AMD/NVIDIA observations, never runtime indices or placement budgets")
    args = parser.parse_args(argv)
    if args.physical_inventory:
        return _physical_main(args.json)
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
