# Capability matrix

> Operator-interface update: [Model runs](../MODEL-RUNS.md) is the current reference
> for standalone serving, qualification and native benchmark scripts. The former
> standalone Ridge environment overrides are replaced by shared presets/configs.
> Any older statement below about absent benchmark tooling predates this interface;
> native throughput tooling exists, but comparative agent-quality evaluation remains absent.

What each local capability is backed by, what would prove it, and how to find out
what is true right now. This file is the *map*; the last observed values live in
the dated snapshots it links, and the machine reading is regenerated on demand.

Related: [architecture](ARCHITECTURE.md) · [operations](OPERATIONS.md) ·
[user guide](../USER-GUIDE.md)

## Contents

- [State vocabulary](#state-vocabulary)
- [How to get the current reading](#how-to-get-the-current-reading)
- [The matrix](#the-matrix)
- [Capabilities with no artifact of their own](#capabilities-with-no-artifact-of-their-own)
- [What is genuinely absent](#what-is-genuinely-absent)
- [No numeric model-characterization harness](#no-numeric-model-characterization-harness)
- [Dated observation](#dated-observation)

## State vocabulary

Four human states are defined by
[`docs/CANDIDATE-STATUS-2026-09-06.md`](../CANDIDATE-STATUS-2026-09-06.md), and a
later state never rewrites an earlier document:

| State | Means |
|---|---|
| **researched** | named and pinned in a review |
| **downloaded** | queued exact size and any recorded SHA-256 verified, and bytes promoted; ancillary files without a digest are not SHA-256-verified |
| **policy-admitted** | the non-inference candidate policy gate passed |
| **functionally qualified** | a live gate observed load, behaviour and unload |

The ledger refines those into seven machine states. The three that exist to stop
overstatement are worth knowing by name:

- **`evidence-interrupted`** — a gate run was cut short. However far it got, an
  unfinished run is not a verdict.
- **`evidence-stale`** — the evidence predates the weights it claims to judge.
  This is not a finding about the model; it says "this evidence did not judge
  these bytes". Re-running the gate resolves it.
- **`functionally-failed`** — a declared artifact exists and does not record a
  pass.

Passing the policy gate is explicitly **not** a functional verdict. Being
downloaded is explicitly not readiness. "The model card says so" is not evidence
about this host.

## How to get the current reading

```bash
python3 verification/local-coverage-foundation/build_capability_ledger.py --print-only
```

It loads no model, starts no service, queries nothing and touches no GPU: it
reads the pinned queues, stats the files they declare, and reads whatever gate
artifacts exist under the ignored `evidence/` directories. Drop `--print-only` to
publish `evidence/capability-ledger.json`.

Because evidence is untracked runtime state, a fresh clone has none and every
capability reads `downloaded` or `researched`. That is correct, not a bug.

## The matrix

`Queue` names the artifact key(s) whose files must be present and exactly sized
before evidence is even considered. `Decided by` is the artifact the ledger reads
— absent means no verdict.

| Capability | Queue artifact(s) | Served by | Decided by |
|---|---|---|---|
| `embeddings` | `embedding-qwen3-8b-q6k` | sidecar `:8081`, preset `qwen3-embedding-8b` | `evidence/gate-embeddings.json` |
| `reranking` | `reranker-qwen3-8b-source` | sidecar `:8082`, preset `qwen3-reranker-8b` | `evidence/gate-reranker.json` |
| `ocr` | `ocr-hunyuanocr-bf16` | sidecar `:8083` (alias `hunyuan-ocr`) | `evidence/gate-ocr.json` |
| `fim` | `fim-qwen2.5-coder-7b-q8` | sidecar `:8084`, preset `qwen25-coder-7b-fim` | `evidence/gate-fim.json` |
| `asr` | `asr-qwen3-1.7b-hf` | `venvs/asr` Transformers | `evidence/gate-asr.json` |
| `tts` | `tts-qwen3-12hz-1.7b-base` | `venvs/tts` Transformers | `tts-local/evidence/gate-tts.json` |
| `image` | `image-z-image-turbo-bf16` | ComfyUI `:8188` | `generative-media/evidence/media-functional.json` |
| `music` | `music-acestep-1.5-turbo-aio` | ComfyUI `:8188` | same artifact |
| `image-editing` | 4 phase-three Qwen-Image-Edit artifacts | ComfyUI `:8188` | same artifact |
| `image-generation-editing` | `image-edit-flux2-klein-4b` | ComfyUI `:8188` | **nothing declared** — see below |
| `computer-use-grounding` | `computer-use-ui-tars-1.5-7b`, `computer-use-ui-mate-9b` | `venvs/computer-use` Transformers | `computer-use-grounding/evidence/computer-use-grounding.json` |
| `multimodal-embeddings` | `multimodal-embedding-wemm-2b` | `venvs/candidates` Transformers | `candidate-qualification/evidence/wemm-functional.json` |
| `repository-agent` | `repository-agent-qwen3-coder-next-q4km` | router presets `heretic`, `qwen3-coder-next` | `repository-agent/evidence/gate-repo-agent-{heretic,qwen3-coder-next}.json` (both) |
| `uncensored-multimodal` | `uncensored-multimodal-gemma4-heretic-q6k` | presets `gemma4-heretic`, `gemma4-heretic-vision` | **nothing declared** — see below |
| `rag` | none | store + sidecars + router | `gate-rag-structural.json`, `gate-rag-behavioural.json`, `gate-rag-live.json` (all three) |
| `native-tool-use` | none | router presets | `evidence/gate-native-tool-use.json` |
| `vision-grounding` | none | preset `obliterated-vision` | `evidence/gate-vision-grounding.json` |

Two capabilities have **no declared evidence on purpose**, and the ledger
therefore reports "no evidence artifact is declared", which is true rather than
convenient:

- **`image-generation-editing`** (FLUX.2 Klein) — the graph is pinned and the
  preflight resolves it, but the functional gate does not submit it, so that
  artifact's pass would not be about FLUX.2.
- **`uncensored-multimodal`** (Gemma-4 Heretic) — the 2026-09-03 checks were run
  against the router by hand and no gate writes a durable artifact for them yet.

Declaring an artifact for either without a gate that writes it would convert an
honest gap into a false pass.

## Capabilities with no artifact of their own

`rag`, `native-tool-use` and `vision-grounding` own no downloaded weights; they
are composed from services and presets that are admitted separately. They are
listed explicitly (`COMPOSED_CAPABILITIES`) so they stay visible in the ledger
rather than silently absent.

`rag` is the strictest: all three of its artifacts must pass. The structural and
behavioural halves need no GPU and can run on a busy host, which is why the
failure modes they cover — a stale chunk surviving an edit, a deleted file still
citable, a citation pointing at the wrong byte range, an invented citation
rendered as fact, a silently-not-reranked result — get their own gates instead of
relying on a live demo query.

## What is genuinely absent

Distinguish "the weights are missing" from "the workflow is unproven" from "the
service does not exist". They need different work.

| Gap | Kind | What would close it |
|---|---|---|
| End-to-end computer control | **service** | A bounded screenshot → action → state-readback scaffold in the shape of the repository-agent and Strix scaffolds. Not a model download; the grounding model is already on disk. Blocked behind a grounding verdict, because without one there is nothing to build on. |
| UI-Mate vs UI-TARS A/B | **baseline** | The UI-TARS grounding gate finished for the first time on 2026-09-07 and did not pass, so there is a baseline to compare against but not an admitted one. Scored: grounding 4/6 against a required 5; OCR 3 of 4 headers and 1 of 2 table rows, misreading `Cksum` and `nvme0n1`; the distractor case clicked outside screen bounds; unload left residue over tolerance. Decoding is greedy (`do_sample=False`), so this is the model, not a sampler or a chat template. The shared contract in `groundlib.py` is ready and UI-Mate can now be run against a real number. |
| FLUX.2 Klein workflow | **gate** | A functional gate that actually submits the pinned FLUX.2 graph, plus an evidence declaration. |
| Gemma-4 Heretic durable evidence | **gate** | A gate that writes an artifact for the text and vision presets, replacing the by-hand 2026-09-03 checks. |
| WeMM image/video retrieval | **runtime** | `AutoProcessor` needs torchvision, which is not in the ROCm candidate runtime. Text-only is claimed; image/video is not. |
| WeMM multimodal index | **build step** | The 2048-D index file has not been built. The separation policy exists; the store does not. |
| PDF and scanned-document ingest | **dependency** | PDF needs an optional parser; scanned documents need the OCR sidecar, which is not qualified. Both are reported as explicit skip reasons, never silent omissions. |
| GLM-5.3-Flash | **upstream support** | Pinned v0.4.0 has no `glm5next` architecture. Requires a reviewed experimental worktree below the submodule, and a clean idle-host 32K retry. No production alias is permitted first. |
| Security-agent pilot | **by design** | The Strix scaffold builds and validates specifications and executes nothing. There is no functional run to have, and adding one is a separate authorized decision. |
| Prompt-corpus evaluation | **data + evaluator** | Catalog and admission exist; no rows have been admitted and no model score is claimed. |
| Model characterization | **harness** | See below. |

## No numeric model-characterization harness

There is no local harness that *scores* a model. Qualitative characterization
exists (`verification/qualitative-characterization`): preview by default,
`--execute` for inference, saved prompts and outputs. That is inspectable
admission evidence, not tokens/sec, perplexity, or a comparable ranking.

This remains a deliberate gap for numeric claims, so it should not be
discovered by surprise.

What does not exist here:

- no tokens/sec, latency, time-to-first-token or throughput measurement;
- no perplexity, KL-divergence or quantisation-damage measurement;
- no refusal-rate, over-refusal or jailbreak-success scoring;
- no MMLU-style or task-benchmark runner;
- no scoring harness that produces a comparable number for two models on the same
  workload.

What exists instead is **pass/fail admission**. Gates assert properties of
returned content — this paraphrase outranks this distractor, this coordinate is
inside this box, this transcript matches this text, this suite went green in a
sandbox — and record a boolean. The "A/B" lanes (repository-agent
`heretic` vs `qwen3-coder-next`, UI-TARS vs UI-Mate) run the same fixtures
through both models and produce two verdicts; they do not produce a score, a
ranking or a margin.

Consequences to keep in mind when reading any document here:

1. **Every numeric model claim in this repository is a publisher claim.** KL
   figures, refusal counts, MMLU deltas, perplexity, vendor benchmark tables —
   all of them are cited as what a publisher or vendor reported, and none is
   independently reproduced on this host. The strategy documents say so
   repeatedly and the interpretation warning at the end of
   [`docs/LOCAL-AI-MODEL-STRATEGY.md`](../LOCAL-AI-MODEL-STRATEGY.md) is not
   boilerplate.
2. **Promotion rules that say "materially beats"** — install a specialist only if
   it beats the incumbent on a named workflow — currently have no instrument. A
   promotion decision of that shape cannot be executed today without first
   building the missing harness.
3. **This is a constraint, not an oversight.**
   [ADR 0002](../decisions/0002-serialized-functional-qualification.md) blocks
   benchmarking until the user explicitly authorizes performance testing after confirming the intended kernel; reboot alone is insufficient,
   and `verification/local-coverage-foundation/post_reboot_gate.py` is the
   fail-closed gate for that: it compares the current `boot_id` and
   `/proc/version` against a recorded pre-reboot baseline and exits 3 while they
   are unchanged or a kernel build is still running. Functional builds, model
   loads, correctness checks, memory checks and clean-unload checks never depend
   on it.

Building the harness later means: an offline scorer with pinned inputs, per-model
evidence artifacts carrying explicit measurement fields, a ledger extension that
does not confuse a measurement with an admission, and removal of the
`benchmark_performed: false` invariant that every gate currently asserts. None of
that exists yet.

## Dated observation

> **Snapshot — 2026-09-06.** A reading, not a claim about any later state.
> Regenerate with the command in
> [how to get the current reading](#how-to-get-the-current-reading).

Rebuilding the ledger on 2026-09-06 reported no `problems`, and these states:

| State | Capabilities |
|---|---|
| `functionally-qualified` | `asr`, `multimodal-embeddings`, `native-tool-use`, `rag`, `vision-grounding` |
| `evidence-stale` | `embeddings`, `fim`, `ocr`, `reranking` |
| `evidence-interrupted` | `computer-use-grounding` |
| `downloaded` | `image`, `image-editing`, `image-generation-editing`, `music`, `repository-agent`, `tts`, `uncensored-multimodal` |

The four `evidence-stale` rows are the ledger doing its job, not a regression:
those gates passed on 2026-09-01, and phase-three and phase-four weights landed
afterwards, so their artifacts are older than the artifact set they are being
read against. The 2026-09-02 snapshot at
[`verification/local-coverage-foundation/docs/CAPABILITY-STATUS-2026-09-02.md`](../../verification/local-coverage-foundation/docs/CAPABILITY-STATUS-2026-09-02.md)
listed eight qualified capabilities and remains correct *for 2026-09-02*; the
difference is the staleness rule.

For the download, policy, runtime and functional narrative behind these states,
read [`docs/CANDIDATE-STATUS-2026-09-06.md`](../CANDIDATE-STATUS-2026-09-06.md).
