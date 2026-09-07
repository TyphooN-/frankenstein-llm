#!/usr/bin/env python3
"""Size every configured preset against the three GPUs and propose a split.

``llama-models.ini`` carries one ``tensor-split`` for almost every preset, and
the numbers in it were chosen to match card capacity: 16 / 32 / 16 GiB became
``3,6,2``. That is a reasonable rule for *fitting* a model and a poor one for
deciding where work should go, because it sends the majority of every model to
the one card the operator least wants loaded. This module separates the two
questions it conflates:

* **What does this preset actually need?** Weight bytes come from the files on
  disk. KV-cache bytes come from the model's own GGUF header -- block count, KV
  head count, key and value head width -- multiplied by the context and cache
  types the preset configures. Neither figure is copied from a document.
* **Where should that go?** A policy in ``config/gpu-placement.json`` names the
  preferred devices, the leftover device, and the reserves that must stay free.

Nothing here is a measurement of speed and nothing here loads a model. The
outputs are byte estimates and the split that follows from them. ``--verify``
compares an estimate against live sysfs residency for whatever is resident now,
which is the only way an estimate in this file earns any trust.

Known bounds of the estimate, none of them hidden in the arithmetic:

* KV is computed for the full context on every block. A model with
  sliding-window attention allocates less than that for its local blocks, so
  its estimate is an over-estimate and is labelled as one.
* Bytes are assumed to fall in proportion to layers. Non-repeating tensors, the
  compute buffers and the CPU-resident input layer do not, which is what
  ``per_device_reserve_bytes`` absorbs and what ``--verify`` exists to check.
* A proportional byte figure is not an allocator result and not an upper bound
  on what a device will actually be asked for. "Fits" here means "inside the
  budget this policy reserved", which is a planning statement; the only proof
  that a preset loads is loading it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf_header  # noqa: E402
import gpu_vram  # noqa: E402
import model_catalog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config/gpu-placement.json"
GIB = 1 << 30

# ggml block layouts: bytes per element for the cache types this repository
# configures. A quantized cache stores a block of 32 values plus its scale
# (and, for the ``_1`` variants, a minimum), so the cost per element is not a
# whole number of bytes.
CACHE_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}
DEFAULT_CACHE_TYPE = "f16"


class PolicyError(ValueError):
    """The placement policy file is missing something the split depends on."""


def load_policy(path=POLICY) -> dict:
    policy = json.loads(Path(path).read_text())
    if policy.get("schema") != 1:
        raise PolicyError("unknown placement policy schema")
    required = ("per_device_reserve_bytes", "display_reserve_bytes",
                "prefer", "fallback", "split_scale")
    missing = [key for key in required if key not in policy]
    if missing:
        raise PolicyError(f"placement policy missing {', '.join(missing)}")
    if not policy["prefer"]:
        raise PolicyError("placement policy names no preferred device")
    overlap = set(policy["prefer"]) & set(policy["fallback"])
    if overlap:
        raise PolicyError(f"device {sorted(overlap)} is both preferred and fallback")
    if not isinstance(policy["split_scale"], int) or policy["split_scale"] < 1:
        raise PolicyError("split_scale must be a positive integer")
    baselines = policy.get("idle_baseline_bytes", {})
    if not isinstance(baselines, dict):
        raise PolicyError("idle_baseline_bytes must be an object keyed by GPU UUID")
    for key, value in baselines.items():
        # These were keyed by ROCm index until a boot renumbered the cards. A
        # stale index-keyed policy would now match no device at all and quietly
        # drop every baseline, so it is rejected rather than half-honoured.
        if not isinstance(key, str) or not key.strip() or key.strip().isdigit():
            raise PolicyError(
                f"idle_baseline_bytes key {key!r} is not a GPU UUID; re-key the "
                "policy off the ROCm index, which is not stable across boots")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PolicyError(f"idle_baseline_bytes[{key!r}] must be a byte count")
    return policy


def shards(model: Path) -> list[Path]:
    """Every file the runtime opens for one ``model =`` value.

    llama.cpp is given the first shard of a split model and finds the rest by
    name, so a size taken from the configured path alone understates a split
    model by however many shards follow it -- for the repository agent, by 30 GiB.
    """
    name = model.name
    marker = "-00001-of-"
    if marker not in name:
        return [model]
    prefix = name.split(marker)[0]
    found = sorted(model.parent.glob(f"{prefix}-*-of-*.gguf"))
    return found or [model]


def cache_bytes_per_element(name: str | None) -> float:
    key = (name or DEFAULT_CACHE_TYPE).strip().lower()
    if key not in CACHE_TYPE_BYTES:
        raise PolicyError(f"unknown cache type {name!r}")
    return CACHE_TYPE_BYTES[key]


def kv_heads_per_block(shape: dict) -> list[int]:
    """KV head count for each block, expanded from whichever form GGUF used."""
    blocks = shape.get("block_count")
    heads = shape.get("head_count_kv")
    if not blocks:
        return []
    if isinstance(heads, list):
        if len(heads) != blocks:
            # A per-layer list that does not match the block count is not a
            # list this arithmetic can trust; fall back to its widest entry so
            # the estimate stays on the conservative side and say so upstream.
            return [max(heads)] * blocks
        return list(heads)
    if isinstance(heads, int):
        return [heads] * blocks
    return []


def kv_bytes(shape: dict, context: int, type_k: str | None,
             type_v: str | None) -> int | None:
    """Bytes the KV cache needs for ``context`` tokens across every block."""
    heads = kv_heads_per_block(shape)
    key_length = shape.get("key_length")
    value_length = shape.get("value_length")
    if not heads or not key_length or not value_length:
        return None
    per_token = sum(
        count * (key_length * cache_bytes_per_element(type_k)
                 + value_length * cache_bytes_per_element(type_v))
        for count in heads)
    return int(per_token * context)


def measure(alias: str, preset: dict) -> dict:
    """Weight bytes, KV bytes and the caveats that apply, for one preset."""
    notes: list[str] = []
    record: dict = {"alias": alias, "notes": notes}
    model = Path(preset["model"]) if preset.get("model") else None
    if model is None:
        notes.append("preset configures no model file")
        return record
    files = shards(model)
    record["files"] = [str(path) for path in files]
    record["shard_count"] = len(files)
    missing = [str(path) for path in files if not path.is_file()]
    weight_bytes = sum(path.stat().st_size for path in files if path.is_file())
    projector = Path(preset["mmproj"]) if preset.get("mmproj") else None
    if projector is not None:
        record["projector"] = str(projector)
        record["projector_device"] = preset.get("mmproj-device")
        if projector.is_file():
            record["projector_bytes"] = projector.stat().st_size
            # The projector is pinned to one device by mmproj-device, so it is
            # not part of what the tensor-split distributes. It still has to fit
            # on that device, and it is reported separately for that reason.
            #
            # Without mmproj-device nothing pins it, and the runtime picks the
            # device. Saying "pinned" there would assert a placement the preset
            # never made, so the two cases are reported as the different facts
            # they are; charge_projector declines to bill the unpinned one.
            notes.append("projector bytes are pinned by mmproj-device, "
                         "not distributed by tensor-split"
                         if preset.get("mmproj-device") else
                         "preset sets no mmproj-device: the projector is not "
                         "pinned and its bytes are charged to no device below")
        else:
            missing.append(str(projector))
    if missing:
        record["missing_files"] = missing
        notes.append("weight files missing; sizes below cover only what is on disk")
    record["weight_bytes"] = weight_bytes

    try:
        header = gguf_header.read_header(files[0])
    except (OSError, gguf_header.GGUFError) as error:
        notes.append(f"GGUF header unreadable: {type(error).__name__}: {error}")
        return record
    shape = gguf_header.shape(header)
    record["shape"] = shape
    context = int(preset.get("ctx-size") or shape.get("context_length") or 0)
    record["context"] = context
    record["cache_type_k"] = preset.get("cache-type-k") or DEFAULT_CACHE_TYPE
    record["cache_type_v"] = preset.get("cache-type-v") or DEFAULT_CACHE_TYPE
    if preset.get("embedding") or preset.get("reranking"):
        # An embedding or reranking server holds no rolling cache: it encodes a
        # batch and returns. Charging it a KV cache would misreport it as the
        # largest small model in the catalog.
        record["kv_bytes"] = 0
        notes.append("embedding/reranking preset: no persistent KV cache")
    else:
        try:
            record["kv_bytes"] = kv_bytes(shape, context, record["cache_type_k"],
                                          record["cache_type_v"])
        except PolicyError as error:
            notes.append(str(error))
            record["kv_bytes"] = None
        if record.get("kv_bytes") is None:
            notes.append("GGUF header lacks the attention fields KV sizing needs")
    if isinstance(shape.get("head_count_kv"), list):
        notes.append("per-block KV head counts vary; KV summed block by block")
    if shape.get("sliding_window"):
        notes.append(f"sliding-window attention ({shape['sliding_window']} tokens): "
                     "KV shown is the full-context upper bound and overstates it")
    if record.get("kv_bytes") is not None:
        record["required_bytes"] = weight_bytes + record["kv_bytes"]
    return record


def budgets(devices: list[dict], policy: dict) -> list[dict]:
    """Usable bytes per ROCm index after the reserves the policy holds back."""
    rows = []
    for device in devices:
        total = device.get("total_bytes")
        reserve = policy["per_device_reserve_bytes"]
        display = bool(device.get("display_connectors"))
        if display:
            reserve += policy["display_reserve_bytes"]
        rows.append({
            "rocm_index": device["rocm_index"],
            "card": device.get("card"),
            "uuid": device.get("uuid"),
            "display": display,
            "total_bytes": total,
            "reserved_bytes": reserve,
            "budget_bytes": None if total is None else max(total - reserve, 0),
            "problems": list(device.get("problems", [])),
        })
    return rows


def _reduce(weights: list[int]) -> list[int]:
    """Divide a split by its common factor; ``4,4,4`` and ``1,1,1`` are one split."""
    from math import gcd
    from functools import reduce
    divisor = reduce(gcd, weights, 0)
    return [value // divisor for value in weights] if divisor > 1 else weights


def _integerise(shares: dict[int, float], order: list[int], scale: int,
                budget: dict[int, int | None], required: float) -> list[int]:
    """Integer weights for a proportional split, repaired to respect budgets.

    ``--tensor-split`` takes proportions, so only the ratio matters -- but a
    proportion is coarse, and rounding a share up can push a device past the
    budget the share was computed to stay inside. Largest-remainder rounding
    fixes the sum; the repair pass after it fixes the fit, moving whole units
    off any device the rounding pushed over onto one with room. Without it a
    plan that fits in bytes is emitted as a split that does not.
    """
    total = sum(shares.values()) or 1.0
    exact = {index: shares.get(index, 0.0) / total * scale for index in order}
    weights = {index: int(value) for index, value in exact.items()}
    shortfall = scale - sum(weights.values())
    by_remainder = sorted(order, key=lambda i: (exact[i] - weights[i], shares.get(i, 0.0)),
                          reverse=True)
    for index in by_remainder[:shortfall]:
        weights[index] += 1
    # A device the policy gave real work to must not round away to zero: a zero
    # entry means "no layers here", which is a different placement than "a small
    # share here" and would silently move those bytes onto another card.
    for index in order:
        if weights[index] == 0 and shares.get(index, 0.0) > 0:
            donor = max(order, key=lambda i: weights[i])
            if weights[donor] > 1:
                weights[donor] -= 1
                weights[index] = 1

    def over(index: int) -> float:
        room = budget.get(index)
        if room is None:
            return 0.0
        return required * weights[index] / scale - room

    for _ in range(scale):
        worst = max(order, key=over)
        if over(worst) <= 0 or weights[worst] == 0:
            break
        receivers = [i for i in order if i != worst and over(i) < 0]
        # Prefer a device the plan already gave work to. Handing the overflow to
        # an idle device would quietly overrule the placement rule that chose to
        # leave it idle -- a rounding unit is not a reason to start using the
        # fallback card.
        engaged = [i for i in receivers if weights[i] > 0]
        candidates = engaged or receivers
        if not candidates:
            break
        target = min(candidates, key=over)
        weights[worst] -= 1
        weights[target] += 1
    return _reduce([weights[index] for index in order])


def _round(shares: dict[int, float], order: list[int], scale: int) -> dict[int, int]:
    """Largest-remainder rounding of proportional shares onto ``scale`` units."""
    total = sum(shares.values()) or 1.0
    exact = {index: shares.get(index, 0.0) / total * scale for index in order}
    weights = {index: int(value) for index, value in exact.items()}
    shortfall = scale - sum(weights.values())
    by_remainder = sorted(order, key=lambda i: (exact[i] - weights[i], shares.get(i, 0.0)),
                          reverse=True)
    for index in by_remainder[:shortfall]:
        weights[index] += 1
    return weights


def _simplest(shares: dict[int, float], order: list[int], scale: int,
              budget: dict[int, int | None], headroom: dict[int, float],
              required: float) -> list[int]:
    """The smallest split that still expresses the plan.

    ``13.98 : 21.30 : 11.48`` rounds exactly onto 96 units and prints as
    ``29,44,23``, which is correct and unreadable -- an operator cannot check it
    against ``llama-models.ini`` at a glance, and a number nobody can check is a
    number nobody will maintain. Coarser scales are tried first and the first one
    that still respects the plan is taken: it must put work on exactly the
    devices the plan chose, and keep every device inside its working headroom.
    Falling back to the fine scale is always available, so simplicity is only
    ever bought where it costs nothing.
    """
    engaged = {index for index in order if shares.get(index, 0.0) > 0}
    for candidate in range(2, scale):
        weights = _round(shares, order, candidate)
        if {index for index in order if weights[index] > 0} != engaged:
            continue
        if any(required * weights[index] / candidate > headroom.get(index, 0)
               for index in engaged):
            continue
        return _reduce([weights[index] for index in order])
    return _integerise(shares, order, scale, budget, required)


def propose(required: int | None, rows: list[dict], policy: dict,
            pinned: dict[int, int] | None = None) -> dict:
    """The split the policy implies for one model size.

    ``pinned`` is bytes a preset places on a named device without going through
    ``--tensor-split`` -- today, a projector held by ``mmproj-device``. They are
    withheld from that device's planning headroom *before* a rule runs, because
    a share chosen against headroom the projector will occupy fits the
    arithmetic and not the card.

    The order of the rules is the whole policy, so it is written out here rather
    than left implicit in the branches:

    1. **One preferred device, if the model fits on one.** Splitting a model
       that fits on a single card buys nothing and costs something: layer mode
       is pipelined, so the split adds device-to-device handoffs, and it holds
       budget on cards that would otherwise be free. Most of this catalog --
       every embedding, reranking and code-completion preset -- is in this case.
    2. **Across the preferred devices, if it fits on those.** This is the
       operator preference expressed directly: the two RX 6900 XTs carry the
       model and the V620 carries none of it.
    3. **Equal across all three, if an equal share fits everywhere.** This is
       the "equal split where sensible" case, and it only becomes sensible once
       the model is too big for the preferred pair -- at which point an equal
       share is both the simplest thing to reason about and the smallest share
       the fallback device can be given.
    4. **Preferred devices filled to budget, remainder on the fallback.** The
       last resort for a model larger than three equal shares can hold.
    """
    order = [row["rocm_index"] for row in rows]
    budget = {row["rocm_index"]: row["budget_bytes"] for row in rows}
    result: dict = {"devices": order, "reasons": []}
    if required is None:
        result["reasons"].append("model size unknown; no split proposed")
        return result
    usable = [index for index in order if budget.get(index)]
    if not usable:
        result["reasons"].append("no device reported a usable budget")
        return result

    # Every rule plans against working headroom, not the whole budget. Filling a
    # card to its last modelled byte produced knife-edge splits -- fable left
    # 0.13 GiB on the display card, and the repository agent 0.30 GiB -- for a
    # figure that ``--verify`` measures as roughly 5% conservative. One margin,
    # applied in one place, is what makes "where sensible" mean something.
    margin = policy.get("min_slack_bytes", policy["per_device_reserve_bytes"])
    pinned = pinned or {}
    headroom = {index: max((budget[index] or 0) - margin - pinned.get(index, 0), 0)
                for index in usable}
    for index, reserved in sorted(pinned.items()):
        if index in headroom:
            result["reasons"].append(
                f"{reserved / GIB:.2f} GiB pinned to ROCm{index} is held back "
                "from its share before the split is chosen")

    prefer = [i for i in policy["prefer"] if i in usable]
    fallback = [i for i in policy["fallback"] if i in usable]
    rest = [i for i in usable if i not in prefer and i not in fallback]
    preferred_capacity = sum(headroom[i] for i in prefer)
    shares: dict[int, float] = {}

    # Only a *preferred* device wins the single-device rule. Considering the
    # fallback here would hand it every model between one preferred card and two
    # of them -- ridge and the vision presets all fit on the 32 GiB card alone --
    # which is the placement this policy exists to stop. It becomes a candidate
    # only when no preferred device is usable at all.
    single_candidates = prefer or (fallback + rest)
    single = next((i for i in single_candidates if required <= headroom[i]), None)
    if single is not None:
        shares = {single: float(required)}
        result["reasons"].append(
            f"fits on ROCm{single} alone ({required / GIB:.2f} GiB within "
            f"{headroom[single] / GIB:.2f} GiB of working headroom); no split needed")
    elif prefer and required <= preferred_capacity:
        for index in prefer:
            shares[index] = required * headroom[index] / preferred_capacity
        result["reasons"].append(
            f"preferred devices {prefer} hold {preferred_capacity / GIB:.2f} GiB "
            f"of working headroom; fallback device carries none of it")
    elif (policy.get("equal_split_when_it_fits", True)
            and len(usable) == len(order)
            and required / len(usable) <= min(headroom[i] for i in usable)):
        shares = {index: required / len(order) for index in order}
        result["reasons"].append(
            f"too large for the preferred pair; equal split puts "
            f"{required / len(order) / GIB:.2f} GiB on each device, within every "
            "device's working headroom")
    else:
        for index in prefer:
            shares[index] = float(headroom[index])
        remainder = required - preferred_capacity
        spill = fallback + rest
        spill_capacity = sum(headroom[i] for i in spill)
        if spill_capacity <= 0:
            result["reasons"].append("no fallback device available for the remainder")
        for index in spill:
            shares[index] = (remainder * headroom[index] / spill_capacity
                             if spill_capacity else 0.0)
        result["reasons"].append(
            f"preferred devices filled to working headroom; "
            f"{max(remainder, 0) / GIB:.2f} GiB remainder placed on {spill}")

    result["split"] = _simplest(shares, order, policy["split_scale"], budget,
                                headroom, float(required))
    total_weight = sum(result["split"]) or 1
    result["projected_bytes"] = {
        index: required * weight / total_weight
        for index, weight in zip(order, result["split"])}
    result["slack_bytes"] = {
        index: (budget.get(index) or 0) - result["projected_bytes"][index]
        for index in order}
    result["fits"] = all(value >= 0 for value in result["slack_bytes"].values())
    if not result["fits"]:
        shortfall = sum(-value for value in result["slack_bytes"].values() if value < 0)
        result["reasons"].append(
            f"does not fit: {shortfall / GIB:.2f} GiB beyond the reserved budgets; "
            "reduce context, use a smaller cache type, or accept CPU spill")
    else:
        tight = [index for index in order
                 if result["split"][order.index(index)]
                 and result["slack_bytes"][index] < margin]
        if tight:
            result["reasons"].append(
                "marginal: " + ", ".join(
                    f"ROCm{i} keeps only {result['slack_bytes'][i] / GIB:.2f} GiB "
                    "above its reserve" for i in tight))
    return result


MODELS = ROOT / "models"
# Partial downloads and quarantined replacements live beside the weights they
# would replace. Sizing them would double-count a model and, for a truncated
# file, report a header error as if the catalog were broken.
SCAN_SKIP_PARTS = frozenset({".staging", ".cache", ".incomplete"})


def unconfigured_models(presets: dict, models_root: Path = MODELS) -> list[Path]:
    """GGUF files on disk that no preset loads, first shard only.

    ``--scan`` exists because "every model this host holds" and "every model the
    router can serve" are different sets, and only the second one appears in
    ``llama-models.ini``. A weight file nobody has written a preset for still
    occupies the disk and still has to be placeable before a preset is worth
    writing.
    """
    configured: set[Path] = set()
    for preset in presets.values():
        for key in ("model", "mmproj"):
            if preset.get(key):
                configured.update(shards(Path(preset[key])))
    found = []
    for path in sorted(models_root.rglob("*.gguf")):
        if SCAN_SKIP_PARTS.intersection(path.parts):
            continue
        if path in configured:
            continue
        name = path.name
        if "-of-" in name and "-00001-of-" not in name:
            continue
        if any(path in shards(other) for other in configured):
            continue
        found.append(path)
    return found


def configured_split(preset: dict, count: int) -> list[int] | None:
    raw = preset.get("tensor-split")
    if not raw:
        return None
    try:
        values = [int(part) for part in raw.replace("/", ",").split(",") if part.strip()]
    except ValueError:
        return None
    return values if len(values) == count else None


def projector_index(record: dict) -> int | None:
    """The ROCm index ``mmproj-device`` names, or ``None`` if it names none."""
    device = record.get("projector_device")
    if not device:
        return None
    try:
        return int(str(device).lower().removeprefix("rocm"))
    except ValueError:
        return None


def pinned_bytes(record: dict) -> dict[int, int]:
    """Bytes this preset places on a named device without a tensor-split.

    Handed to ``propose`` so the rules plan around them. Charging a projector
    only after a split has been chosen leaves the split free to hand its device
    a share that fits right up until the projector lands on top of it -- which
    is how ``obliterated-vision`` came to be reported as fitting with 0.21 GiB
    left on a 16 GiB card.
    """
    index = projector_index(record)
    size = record.get("projector_bytes")
    if index is None or not size:
        return {}
    return {index: size}


def charge_projector(record: dict, proposal: dict, margin: int = 0) -> None:
    """Bill a pinned projector to the device ``mmproj-device`` pins it to.

    ``--tensor-split`` does not distribute the projector, so the split-based
    slack is blind to it: a vision preset can be reported as fitting while the
    device holding its projector is over budget by most of a gigabyte. Charging
    it here is the difference between a plan and a plan that loads.

    ``margin`` is the policy's working headroom. It is only used to say when a
    charge leaves a device technically inside its budget and practically on the
    edge of it; the reserve itself was already applied when the budget was cut.
    """
    device = record.get("projector_device")
    size = record.get("projector_bytes")
    if not size or "slack_bytes" not in proposal:
        return
    if not device:
        # An unpinned projector still occupies a card; which card is the
        # runtime's choice, not this policy's. Guessing one would put real bytes
        # against a device the preset never named, so the gap is reported
        # instead -- an unaccounted gigabyte must not read as a clean fit.
        proposal["reasons"].append(
            f"projector of {size / GIB:.2f} GiB is not pinned by mmproj-device; "
            "the runtime chooses its device and these bytes are in no slack "
            "figure below")
        return
    index = projector_index(record)
    if index is None:
        proposal["reasons"].append(
            f"projector device {device!r} is not a ROCm index; not charged to any card")
        return
    if index not in proposal["slack_bytes"]:
        proposal["reasons"].append(
            f"projector pinned to ROCm{index}, which is not a visible device")
        return
    proposal["slack_bytes"][index] -= size
    proposal["projector_charged_to"] = index
    proposal["fits"] = all(value >= 0 for value in proposal["slack_bytes"].values())
    if proposal["slack_bytes"][index] < 0:
        proposal["reasons"].append(
            f"projector of {size / GIB:.2f} GiB puts ROCm{index} "
            f"{-proposal['slack_bytes'][index] / GIB:.2f} GiB over budget")
    else:
        proposal["reasons"].append(
            f"projector of {size / GIB:.2f} GiB charged to ROCm{index}, leaving "
            f"{proposal['slack_bytes'][index] / GIB:.2f} GiB there")
        if margin and proposal["slack_bytes"][index] < margin:
            proposal["reasons"].append(
                f"marginal: ROCm{index} keeps only "
                f"{proposal['slack_bytes'][index] / GIB:.2f} GiB above its "
                "reserve once the projector is charged")


def evaluate_preset(alias: str, preset: dict, rows: list[dict], policy: dict) -> dict:
    """Size one preset and place it, projector included.

    This is the single path from a preset to a proposal. The comparison test
    calls it for the same reason ``evaluate`` does: a placement checked by a
    shorter route than the one that produced it is not really checked.
    """
    margin = policy.get("min_slack_bytes", policy["per_device_reserve_bytes"])
    record = measure(alias, preset)
    record["configured_split"] = configured_split(preset, len(rows))
    record["configured_devices"] = preset.get("device")
    record["proposal"] = propose(record.get("required_bytes"), rows, policy,
                                 pinned_bytes(record))
    charge_projector(record, record["proposal"], margin)
    return record


def evaluate(presets: dict, rows: list[dict], policy: dict) -> list[dict]:
    catalog = model_catalog.local_registry()[1]
    report = []
    for alias, preset in presets.items():
        record = evaluate_preset(alias, preset, rows, policy)
        record["purpose"] = model_catalog.purpose(alias, catalog)
        report.append(record)
    return report


def format_devices(rows: list[dict]) -> str:
    lines = ["Devices (ROCm index order, budgets after reserves):"]
    for row in rows:
        total = row["total_bytes"]
        lines.append(
            f"  ROCm{row['rocm_index']} {row['card'] or 'no card':<7}"
            f" total={(total or 0) / GIB:6.2f} GiB"
            f" reserved={row['reserved_bytes'] / GIB:5.2f} GiB"
            f" budget={(row['budget_bytes'] or 0) / GIB:6.2f} GiB"
            f" {'display' if row['display'] else 'headless'}")
        for problem in row["problems"]:
            lines.append(f"    ! {problem}")
    return "\n".join(lines)


def format_models(report: list[dict]) -> str:
    lines = []
    for record in sorted(report, key=lambda r: -(r.get("required_bytes") or 0)):
        required = record.get("required_bytes")
        proposal = record["proposal"]
        head = (f"  {record['alias']:<22}"
                f" weights={(record.get('weight_bytes') or 0) / GIB:6.2f} GiB")
        if record.get("kv_bytes") is not None:
            head += f" kv={record['kv_bytes'] / GIB:6.2f} GiB @ ctx {record.get('context')}"
        if required is not None:
            head += f" total={required / GIB:6.2f} GiB"
        lines.append(head)
        # "Not sized" and "does not fit" are different answers. A projector or a
        # header this reader cannot parse has no size, and calling that a fit
        # failure would send an operator looking for VRAM that is not the problem.
        if required is None:
            verdict = "not sized"
        elif proposal.get("fits"):
            verdict = "fits"
        else:
            verdict = "DOES NOT FIT"
        current = record.get("configured_split")
        proposed = proposal.get("split")
        lines.append(f"    configured {current}  proposed {proposed}  [{verdict}]")
        projected = proposal.get("projected_bytes") or {}
        if projected:
            lines.append("    projected " + "  ".join(
                f"ROCm{index}={projected[index] / GIB:.2f} GiB"
                for index in sorted(projected)))
        for reason in proposal.get("reasons", []):
            lines.append(f"    - {reason}")
        for note in record.get("notes", []):
            lines.append(f"    note: {note}")
    return "\n".join(lines)


def device_baselines(rows: list[dict], policy: dict) -> tuple[dict[int, int], list[int]]:
    """Idle baselines matched to the cards actually present, keyed by GPU UUID.

    These were keyed by ROCm index until a boot came up without the V620 and
    renumbered what was left. An index is a position in the runtime's device
    list, not an identity, so on that topology index 2's baseline would have
    been subtracted from a different physical card. A UUID survives it.

    A device whose UUID is absent from the policy gets no baseline rather than a
    borrowed one. That understates the desktop's share and so reports the
    estimate as *less* conservative than it is -- the direction that asks for
    attention instead of granting it. Which devices those were is returned, not
    swallowed.
    """
    configured = policy.get("idle_baseline_bytes") or {}
    matched: dict[int, int] = {}
    unmatched: list[int] = []
    for row in rows:
        uuid = row.get("uuid")
        if uuid is not None and uuid in configured:
            matched[row["rocm_index"]] = configured[uuid]
        else:
            unmatched.append(row["rocm_index"])
    return matched, unmatched


def verify(rows: list[dict], devices: list[dict], report: list[dict],
           policy: dict) -> dict:
    """Check the estimate against live residency for whatever is resident now.

    An estimate nobody checks is a number with a confident format. This reads
    what the router currently holds -- without loading anything, ``GET /models``
    is an observer -- and subtracts the configured idle baseline so the desktop's
    own allocation is not charged to the model. The residual it prints is the
    error of the arithmetic above, in the direction it errs. A positive residual
    means the estimate is conservative, which is the safe direction for a budget
    but still worth knowing the size of.
    """
    used = {device["rocm_index"]: device.get("used_bytes") for device in devices}
    baseline, without_baseline = device_baselines(rows, policy)
    result: dict = {
        "measured_used_bytes": used,
        "idle_baseline_bytes": baseline,
        "devices_without_baseline": without_baseline,
        "devices": [
            {"rocm_index": row["rocm_index"], "card": row["card"],
             "used_bytes": used.get(row["rocm_index"]),
             "idle_baseline_bytes": baseline.get(row["rocm_index"]),
             "budget_bytes": row["budget_bytes"],
             "total_bytes": row["total_bytes"]}
            for row in rows],
    }
    try:
        import local_model_status
        summary = local_model_status.collect_status()
    except Exception as error:  # noqa: BLE001 - an absent router is not a failure here
        result["router"] = f"unavailable: {type(error).__name__}: {error}"
        return result
    resident = summary.get("loaded_models") or []
    result["router"] = {"endpoint": summary.get("endpoint"), "resident": resident}
    if len(resident) != 1:
        result["comparison"] = (
            "no comparison: exactly one resident model is needed and the router "
            f"reports {len(resident)}")
        return result
    alias = resident[0]
    record = next((item for item in report if item["alias"] == alias), None)
    if record is None or record.get("required_bytes") is None:
        result["comparison"] = f"no comparison: no size estimate for resident {alias!r}"
        return result
    # A device whose residency could not be read contributes nothing to the
    # measurement, so its idle baseline must not be subtracted either: doing
    # both would charge the desktop's allocation against a card that reported no
    # bytes and inflate the residual by that much. Which cards were skipped is
    # reported, because a comparison over two of three cards is a different
    # claim from one over all three.
    contributing = [row["rocm_index"] for row in rows
                    if used.get(row["rocm_index"]) is not None]
    unread = [row["rocm_index"] for row in rows
              if used.get(row["rocm_index"]) is None]
    measured = sum(used[index] for index in contributing)
    attributed = measured - sum(baseline.get(index, 0) for index in contributing)
    estimate = record["required_bytes"]
    result["comparison"] = {
        "resident_model": alias,
        "configured_split": record.get("configured_split"),
        "estimated_bytes": estimate,
        "measured_bytes": measured,
        "attributed_bytes": attributed,
        "residual_bytes": estimate - attributed,
        "residual_fraction": (estimate - attributed) / estimate if estimate else None,
        "compared_devices": contributing,
        "unread_devices": unread,
        "note": ("Baseline subtraction uses configured idle figures, not a "
                 "reading taken with the model unloaded, so the residual "
                 "carries whatever those figures are stale by."),
    }
    if without_baseline:
        result["comparison"]["note"] += (
            f" Devices {without_baseline} have no idle baseline under their GPU "
            "UUID, so their whole residency is charged to the model and the "
            "residual is smaller than the arithmetic alone would make it.")
    if unread:
        result["comparison"]["note"] += (
            f" Devices {unread} reported no residency and are excluded from "
            "both sides of this comparison, so it covers only part of the model.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--policy", default=str(POLICY))
    parser.add_argument("--presets", default=str(model_catalog.PRESETS))
    parser.add_argument("--preset", action="append",
                        help="limit the report to these aliases")
    parser.add_argument("--verify", action="store_true",
                        help="also report live per-device residency from sysfs")
    parser.add_argument("--scan", action="store_true",
                        help="also size GGUF weights on disk that no preset loads")
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    devices, filters = gpu_vram.devices()
    rows = budgets(devices, policy)
    presets = model_catalog.presets(args.presets)
    if args.preset:
        unknown = [name for name in args.preset if name not in presets]
        if unknown:
            parser.error(f"unknown preset(s): {', '.join(unknown)}")
        presets = {name: presets[name] for name in args.preset}
    report = evaluate(presets, rows, policy)

    payload = {
        "schema": "frankenstein-gpu-placement/1",
        "benchmarking_performed": False,
        "throughput_measured": False,
        "visibility_filters": [{"variable": n, "value": v} for n, v in filters],
        "devices": rows,
        "presets": report,
    }
    unconfigured: list[dict] = []
    if args.scan:
        all_presets = model_catalog.presets(args.presets)
        for path in unconfigured_models(all_presets):
            # A file with no preset has no configured context or cache type, so
            # it is sized at the context its own header declares with the f16
            # cache llama.cpp defaults to. That is the ceiling, and the number a
            # preset would have to be written down from.
            record = measure(path.stem, {"model": str(path)})
            record["note_unconfigured"] = (
                "no preset loads this file; sized at the model's own maximum "
                "context with the default f16 cache")
            record["proposal"] = propose(record.get("required_bytes"), rows, policy)
            unconfigured.append(record)
        payload["unconfigured_models"] = unconfigured
    if args.verify:
        payload["verification"] = verify(rows, devices, report, policy)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(format_devices(rows))
        print("\nPresets (weights + KV upper bound at the configured context):")
        print(format_models(report))
        if unconfigured:
            print("\nOn disk, no preset (sized at the model's own maximum context):")
            print(format_models(unconfigured))
        if args.verify:
            print("\nLive residency (all models currently resident, desktop included):")
            for row in payload["verification"]["devices"]:
                used = row["used_bytes"]
                print(f"  ROCm{row['rocm_index']} {row['card']}: "
                      f"used={(used or 0) / GIB:6.2f} GiB of "
                      f"{(row['total_bytes'] or 0) / GIB:6.2f} GiB")
    # A preset the policy cannot place is a configuration problem, not a note.
    return 1 if any(not r["proposal"].get("fits", False) for r in report) else 0


if __name__ == "__main__":
    sys.exit(main())
