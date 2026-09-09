# ADR 0006: Placement policy prefers the RX 6900 XTs and is computed, not hand-written

- Status: Accepted
- Date: 2026-09-06
- Amends: [ADR 0003](0003-hardware-allocation-and-memory-policy.md)

## Context

Every router preset inherited `tensor-split = 3,6,2` from the `[*]` section of
`llama-models.ini`. Those proportions came from card capacity — 16 / 32 / 16 GiB
became 3 / 6 / 2 — which is a sound rule for *fitting* a model and a poor one for
deciding where work should go. Three consequences followed:

- The Radeon Pro V620 received 54.5% of the layers of every preset in the
  catalog, including presets small enough to fit on a single 16 GiB card. The
  operator reports that the V620 thermally throttles while both RX 6900 XTs run
  undervolted at 2,710 MHz with excellent cooling. Those are operator reports,
  not measurements taken here, and no measurement is claimed by this decision.
- Presets that fit comfortably on one card were split across three anyway.
  `--split-mode` is `layer`, which llama.cpp's own help describes as
  *pipelined*; splitting a model that fits on one card adds device-to-device
  handoffs and holds budget on cards that would otherwise be free.
- `gemma4-heretic` and `gemma4-heretic-vision` ran `0,1,0` — entirely on the
  V620 — and both vision presets pinned their projector there with
  `mmproj-device = ROCm1`, which was the only reason either preset touched that
  card at all.

Separately, `scripts/gpu_vram.py` reported per-device VRAM by counting
`/sys/class/drm/card*` entries and labelling the Nth one `ROCmN`. A sysfs card
number is a DRM minor assigned across every DRM driver on the host; a ROCm index
is a position in the HIP device list, built from the KFD topology and then
filtered by `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`. The two coincide
on this host and diverge silently the moment a non-KFD display device appears, a
card fails to bind KFD, or either variable is set — printing one card's bytes
under another card's name, to a reader who is sizing a positional
`--tensor-split` from it.

The user's own attempt at this was `6,6,6` in `config/model-runs.json`. That is
an equal split expressed in weights rather than gigabytes, and it was the right
instinct; it disagreed with `llama-models.ini`, which still read `3,6,2`, and
broke a test asserting the literal `3/6/2`.

## Decision

**Placement is computed from measured facts, not written by hand.**
`scripts/gpu_placement.py` sizes each preset from the weight files on disk
(every shard included) and the KV-cache dimensions in the model's own GGUF
header, at the context and cache types the preset configures. It reads the
policy from `config/gpu-placement.json` and emits a split.
`llama-models.ini` records the result, and
`verification/gpu-placement/test_gpu_placement.py` compares the two so a preset
cannot drift into a placement nobody evaluated.

**ROCm indices are derived, not assumed.** `scripts/gpu_vram.py` builds each
index from the KFD topology and joins it back to sysfs by both the render minor
and the PCI address that the node and the card each publish. It applies
`ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` in the order HIP applies them.
Disagreement between the two joins is reported rather than resolved by
preference, and a device whose attributes cannot be read keeps its row and its
index rather than renumbering everything behind it.

**Preference order is ROCm0 and ROCm2 (the RX 6900 XTs), then ROCm1 (the
V620).** Budgets are the card total less a 1 GiB per-device reserve, less a
further 2.5 GiB on whichever card has connected display outputs. Every rule
plans against *working headroom*, the budget less a 1 GiB margin. The rules, in
order:

1. one preferred card, when the model fits on one;
2. across the two RX 6900 XTs, when it fits on those;
3. equal across all three, once it does not;
4. the 6900 XTs filled to their working headroom with the remainder on the
   V620.

The `[*]` default becomes `1,1,1`: an equal three-way split is the safe
placement for a preset added without its own evaluation.

**The display role is read, not configured.** It comes from connected DRM
connectors, so a moved cable moves the reserve.

## Consequences

- Seven of twelve presets no longer place any layer on the V620.
  `qwen3-coder-next` is the only preset that requires it — for capacity, at
  46.77 GiB across four shards — and the remaining four (`heretic`,
  `obliterated`, `phr00ty`, `fable`) use it through the equal split because they
  exceed what the 6900 XT pair can hold.
- Both vision presets pin their projector to `ROCm0`, a card their weights
  already occupy. `--tensor-split` does not distribute a pinned projector, so
  its bytes are withheld from that card's headroom before the split is chosen
  and charged to it again when fit is reported. Planning around the projector
  rather than only accounting for it afterwards is what moved
  `obliterated-vision` from `3,0,2` — which left 0.21 GiB on a 16 GiB card — to
  `5,0,4`, which leaves 1.24 GiB.
- ADR 0003's consequence "Router presets split tensors across ROCm0/ROCm1/ROCm2
  as `3,6,2` unless a gate overrides that for a specific model" is superseded.
  Its decisions on HugeTLB, zram, kernel `_version` and the single resident
  model are unchanged. Its "minimize GPU2 compute" line is narrowed: GPU2 now
  carries compute, and what is preserved is its *memory* headroom, through the
  display reserve.
- The estimate is a planning figure, not an upper bound. It distributes bytes in
  proportion to layers, which is not what an allocator does: non-repeating
  tensors, compute buffers and fragmentation do not follow that proportion, and
  nothing here bounds how far they can depart from it. In the one comparison
  taken against live residency — `heretic` resident, before the 2026-09-06
  reboot — it was about 5% high. That is one observation in the direction a
  budget wants, not a guarantee, and it is why the reserves exist rather than a
  reason to trust the arithmetic without them. For models with sliding-window
  attention the over-estimate is larger and is labelled per preset rather than
  smoothed away.
- **These placements are evaluated, not yet exercised.** No preset has been
  loaded under its new split. That is the functional qualification's job
  (`local-ai-qualification.service`), which is gated on an uncontended host
  and was waiting on an active kernel build when this was written.
- No throughput was measured for these placements. The operator subsequently
  authorized targeted larger-model upgrade benchmarks, subject to the
  [evaluation strategy](../reference/MODEL-UPGRADE-EVALUATION.md); this does not
  blanket-enable throughput qualifications. The preference for the
  6900 XTs rests on operator reports about clocks and cooling, and on the
  structural argument about pipelined layer mode — not on a timing run. Nothing
  here may be cited as a performance result.

## Rollback

Restore the previous `[*]` default and per-preset splits in `llama-models.ini`
(`tensor-split = 3,6,2`, `0,1,0` for the two gemma4 presets, `13,26,6` for
`qwen3-coder-next`, `mmproj-device = ROCm1` for both vision presets), or set
`prefer` to `[1]` in `config/gpu-placement.json` and regenerate. The comparison
test reports the mismatch either way, so a partial rollback cannot pass quietly.
