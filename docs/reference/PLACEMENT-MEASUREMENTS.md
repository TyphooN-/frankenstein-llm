# Placement, measured

[ADR 0006](../decisions/0006-placement-policy-prefers-the-rx-6900-xts.md) chose a
placement policy on capacity arithmetic and an operator's thermal preference, and
[GPU-EXECUTION-AND-MODEL-LOADING.md](GPU-EXECUTION-AND-MODEL-LOADING.md) said so
plainly: *"not a measured performance claim and this document makes none."*
[CANDIDATE-STATUS-2026-09-06.md](../CANDIDATE-STATUS-2026-09-06.md) carried the
same gap as an open item, "live three-GPU placement qualification (ADR 0006:
computed, not exercised)".

This file exercises it. Every number below is native `llama-bench` output from a
run directory under `logs/model-runs/`, taken on 2026-09-07 on kernel
`7.2.3-273-tkg-eevdf-llvm`, boot `9b83466d-2582-4fad-ba3c-00af2cedbdd8`, with the
router stopped and no other GPU owner. It measures **placement**, not weights.

## Method

One workload throughout: `-p 512 -n 128 -r 3`, devices `ROCm0/ROCm1/ROCm2`, and
only `-ts` varied. Holding the device list fixed and moving the proportions is
what makes the arms comparable; a card with a zero share carries no layers.
Arms were run back to back per model. Each run had
`scripts/bench_report.py sample` polling VRAM and RAM beside it.

`llama-bench` sizes context to the workload, so **these runs are not carrying the
preset's serving context**. That is what makes them a clean placement A/B, and
also why they are not the rates the router delivers.

## 1. The V620 in or out of a three-way split

`heretic`, `obliterated`, `fable` and `phr00ty` take the `1,1,1` default because
their weights *plus their configured context* exceed the 25.47 GiB of working
headroom on the RX 6900 XT pair. At the benchmark's own small context the weights
alone do fit on the pair, which is what let both arms run.

| Preset | pp `1,1,1` | pp `1,0,1` | pp | tg `1,1,1` | tg `1,0,1` | tg |
|---|---|---|---|---|---|---|
| `heretic` | 387.49 | 408.93 | +5.5% | 13.83 | 14.44 | +4.4% |
| `obliterated` | 385.72 | 408.55 | +5.9% | 13.55 | 14.41 | +6.3% |
| `fable` | 385.85 | 408.41 | +5.8% | 13.26 | 13.92 | +5.0% |
| `phr00ty` | 333.62 | 354.54 | +6.3% | 12.58 | 13.32 | +5.9% |
| `ridge` (control, serves `1,0,1`) | 489.67 | 519.01 | +6.0% | 17.14 | 17.74 | +3.5% |

Rates are tok/s. Standard deviations were between 0.02 and 0.26 on generation and
5.4 to 17.3 on prompt processing, so the direction is not noise.

Including the V620 costs about **5–6% of prompt processing and 4–6% of
generation**. The `ridge` row is the control: it serves `1,0,1` already, and
adding the V620 to it moves the same way and by the same margin.

This is a measured penalty for a card that is *carrying layers*, and it is small.
It is not an argument for stranding capacity: `qwen3-coder-next` needs the V620
and pays this to exist at all.

## 2. Splitting a model that fits on one card

This is the large effect, and it is a different mechanism. Layer split is
pipelined, so a decoded token crosses every device in the split, one after
another. Prompt processing batches and parallelises; single-token decode cannot.

Same weight file, same workload, only `-ts` changed:

| Weights | pp `1,0,0` | pp `1,0,1` | pp | tg `1,0,0` | tg `1,0,1` | tg |
|---|---|---|---|---|---|---|
| `Gemma-4-12B-it-heretic-Q6_K.gguf` | 935.8 | 928.3 | +0.8% | 40.30 | 23.60 | **+70.8%** |
| `Qwen3-Embedding-8B-Q6_K.gguf` | 1425 | 1367.7 | +4.2% | 65.42 | 29.32 | **+123.1%** |

The gemma4 prefill figure is the mean of three `1,0,0` runs (934.93, 938.10,
934.29). A fourth returned 755.48 with a standard deviation of 338.26 against
about 25 on every other run; it is excluded as the outlier its own variance says
it is, and is named here rather than dropped silently.

Prefill is a wash. Generation nearly doubles. **A model that fits on one card
should be on one card**, which is rule 1 of the placement policy, and this is the
first measurement of what breaking it costs.

## 3. The split the sidecars were still serving

`services/sidecar-*.env` kept `--tensor-split 3,6,2` after `f2576ac` moved
`llama-models.ini` off it. Each of these models fits on a single card, so the
split bought no capacity and cost two device hops per token. The gates start
these units, so this was live, not latent.

| Sidecar | pp `3,6,2` | pp `1,0,0` | pp | tg `3,6,2` | tg `1,0,0` | tg |
|---|---|---|---|---|---|---|
| `embeddings` | 1234.53 | 1430.10 | +15.8% | 34.63 | 65.32 | +88.7% |
| `reranker` | 1233.73 | 1425.14 | +15.5% | 34.72 | 65.31 | +88.1% |
| `fim` | 2074.37 | 2467.06 | +18.9% | 42.13 | 58.17 | +38.1% |

**Read the right column for the right service.** `embeddings` and `reranker`
encode and return; they never decode, so the honest figure for them is the
prefill one, **+15.8% and +15.5%**. The generation column is what `llama-bench`
measured, not a rate those two serve. `fim` and `ocr` do generate, so both
columns describe their work.

## What changed

- All four sidecars now serve `1,0,0`, matching the preset that loads the same
  file. `verification/gpu-placement/test_gpu_placement.py` fails if they drift
  apart again, which is how the previous drift survived a policy change.
- `gemma4-heretic` moved from `1,0,1` to `1,0,0`. That one needed a residency
  measurement rather than an A/B, because the question was whether it fits:
  loaded alone on ROCm0 at 32,768 context it holds **9.88 GiB** against 13.98 GiB
  of working headroom, and unloads to its baseline. `scripts/gpu_placement.py`
  still proposes `1,0,1` because it charges every block the full context and
  Gemma 4's window is 1,024 tokens.

## What did not change

`heretic`, `obliterated`, `fable` and `phr00ty` stay on `1,1,1`. Their context is
what pushes them off the pair, not their weights, so buying the 4–6% means buying
it with context:

| Preset | Configured context | Largest context that fits the pair |
|---|---|---|
| `obliterated` | 131,072 | 65,667 |
| `heretic` | 131,072 | 64,294 |
| `fable` | 65,536 | 44,264 |
| `phr00ty` | 65,536 | 6,312 |

Only `obliterated` has a clean option — 65,536 fits inside 65,667 — and it costs
half the context for about 6%. `heretic` misses a 64K context by roughly 1,200
tokens, which is too close to plan against. `fable` and `phr00ty` are not close.
None of these were changed; the arithmetic is recorded so the trade can be made
deliberately rather than rediscovered.

## What these numbers are not

- Not a ranking of weights. Every rate here was measured **without** the router's
  `spec-type = draft-mtp` preset, so an MTP-served preset and a non-MTP one may
  not be ordered against each other by these figures.
- Not the serving context. `llama-bench` sizes context to the workload and uses
  f16 KV, while these presets serve `q4_0`. Residency and rates both move.
- Not a quality measurement. Placement does not change what a model answers.
- Not thermal. ADR 0006's preference is thermal and these runs are minutes long;
  nothing here measures sustained clocks.
