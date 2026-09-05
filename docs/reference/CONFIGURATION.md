# Configuration reference

Undated reference for every tracked configuration file, schema, port, environment
variable and data flow. Where a value is also stated elsewhere, this file names
the single owner of that value.

Related: [architecture](ARCHITECTURE.md) · [operations](OPERATIONS.md) ·
[coverage map](COVERAGE-MAP.md)

## Contents

- [Ownership](#ownership)
- [Router presets](#router-presets)
- [Upstream dependency lock](#upstream-dependency-lock)
- [Sidecar environment files](#sidecar-environment-files)
- [systemd units](#systemd-units)
- [Ports](#ports)
- [Environment variables](#environment-variables)
- [Download queue schema](#download-queue-schema)
- [Download state and stamps](#download-state-and-stamps)
- [Mission state schema](#mission-state-schema)
- [Capability ledger schema](#capability-ledger-schema)
- [Prompt corpus schemas](#prompt-corpus-schemas)
- [Other tracked manifests](#other-tracked-manifests)
- [ComfyUI model paths](#comfyui-model-paths)
- [Hermes provider configuration](#hermes-provider-configuration)
- [What Git tracks](#what-git-tracks)
- [Data flows](#data-flows)

## Ownership

One writer per fact. If two files state the same number, one of them is a copy
and something checks them against each other.

| Fact | Owner | Copies, and what reconciles them |
|---|---|---|
| llama.cpp revision | `upstream/llama-cpp.lock.json` | outer gitlink and ADR 0005 prose; reconciled by `verification/upstream-pin/test_llama_cpp_pin.py` |
| Build backend, targets, parallelism | `upstream/llama-cpp.lock.json` | `scripts/build-llama-cpp.sh`; reconciled by the same suite |
| Router aliases and runtime flags | `llama-models.ini` | none — services pass the file, they do not restate it |
| Which model file backs an alias | `llama-models.ini` | prose in `docs/local-hermes-models.md`; prose is descriptive |
| Sidecar model, port, flags | `services/sidecar-*.env` | none — the unit template substitutes them |
| Pinned artifact, revision, size, digest | `verification/local-coverage-foundation/download-queue*.json` | generated from `research/hf/`; consumed by policy and ledger without restatement |
| Phase byte totals | the queue's `total_bytes` | the phase stamp, the next phase runner, the mission supervisor; reconciled by `test_queue_manifests.py` |
| Candidate privilege and prerequisites | `verification/candidate-qualification/candidate_policy.py` | none — the router gate derives its check list from it |
| Approved remote-code digests | `wemm_remote_code_review.py` (`REVIEWED_FILES`) | `WEMM-REMOTE-CODE-REVIEW.md` table |
| Which artifact proves which capability | `build_capability_ledger.py` (`CAPABILITY_EVIDENCE`) | [capability matrix](CAPABILITY-MATRIX.md) prose |
| Media artifact inventory and sizes | `verification/generative-media/media_policy.py` | none |
| Prompt source disposition | `verification/prompt-corpus-admission/source-catalog.json` | ADR 0004 prose |
| Hardware allocation policy | [ADR 0003](../decisions/0003-hardware-allocation-and-memory-policy.md) | `llama-models.ini` `tensor-split`, `media_policy.GPU_ROLES` |

## Router presets

`llama-models.ini`, loaded with `llama-server --models-preset`. Keys are
llama-server long options with the leading `--` removed. **An unrecognised key is
fatal**: llama.cpp raises during `load_from_ini` and the router never finishes
starting, so a key dropped by a future release breaks every alias rather than
degrading one. `PresetCompatibilityTests` checks every key against
`llama-server --help` from the installed build; four preset-only keys
(`load-on-startup`, `stop-timeout`, `dedup-cache-models`, `version`) are exempt
because `--help` never lists them.

### Shared defaults — `[*]`

| Key | Value | Note |
|---|---|---|
| `ctx-size` | `131072` | overridden per preset where the model or memory requires |
| `gpu-layers` | `all` | |
| `flash-attn` | `on` | turned off for the embedding and reranker presets |
| `cache-type-k` / `cache-type-v` | `q4_0` | |
| `parallel` | `1` | one slot; qualification is serialized |
| `device` | `ROCm0,ROCm1,ROCm2` | |
| `tensor-split` | `3,6,2` | matches 16 / 32 / 16 GiB; per ADR 0003 |
| `jinja` | `1` | use the GGUF's own chat template |
| `mmap` | `0` | |
| `reasoning` | `off` | |

Do not add `split-mode = tensor`: Qwen3.8 MTP backend sampling is incompatible
with that path.

### Aliases

| Alias | Model | Overrides | Privilege |
|---|---|---|---|
| `ridge` | `models/Qwen3.8-27B-Ridge-3.7bpw.gguf` | `spec-type=draft-mtp`, `spec-draft-n-max=2` | tool-using |
| `heretic` | `models/RVN-Q6_K-multilingual-mtp.gguf` | same MTP pair | tool-using |
| `obliterated` | `models/Qwen3.8-27B-OBLITERATED-Q6_K.gguf` | MTP pair, `temp=0.2`, `repeat-penalty=1.15` | tool-using |
| `obliterated-vision` | same GGUF | adds `mmproj=…-mmproj-bf16.gguf`, `mmproj-device=ROCm1`, `ctx-size=32768` | tool-using |
| `fable` | `models/Qwen3.6-27B-Fable-Fus-711-…-Q6_K.gguf` | MTP pair, `ctx-size=65536` | tool-using |
| `phr00ty` | `models/Phr00tyMix-v4-32B-imat-Q6_K.gguf` | `ctx-size=65536`, `temp=1.5`, `min-p=0.1`; **no MTP** | tool-using |
| `qwen3-embedding-8b` | `models/embedding/Qwen3-Embedding-8B-Q6_K.gguf` | `ctx=4096`, `batch/ubatch=4096`, `embedding=1`, `pooling=last`, `embd-normalize=2`, `flash-attn=off`, `cache-type-v=f16` | read-only |
| `qwen3-reranker-8b` | `models/reranker/Qwen3-Reranker-8B-Q6_K.gguf` | `ctx=4096`, `batch/ubatch=1024`, `reranking=1`, `pooling=rank`, `flash-attn=off`, `cache-type-v=f16` | read-only |
| `qwen25-coder-7b-fim` | `models/fim/qwen2.5-coder-7b-q8_0.gguf` | `ctx=32768`, `batch=1024`, `ubatch=512`, `cache-type-k/v=q8_0` | read-only |
| `qwen3-coder-next` | `models/repository-agent/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf` | `ctx=65536`, `tensor-split=13,26,6`, `temp=0.2`, `repeat-penalty=1.05` | tool-using |
| `gemma4-heretic` | `models/gemma4-heretic/Gemma-4-12B-it-heretic-Q6_K.gguf` | `ctx=32768`, `tensor-split=0,1,0`, `temp=0.7` | low |
| `gemma4-heretic-vision` | same GGUF | adds `mmproj=mmproj-Gemma-4-12B-it-BF16.gguf`, `mmproj-device=ROCm1`, `ctx=16384`, `tensor-split=0,1,0`, `temp=0.7` | low |

`phr00ty` omits draft-MTP intentionally: its GGUF declares a 131,072-token native
training context but carries no MTP tensors. `qwen3-coder-next` keeps a bounded
GPU2 share because Q4_K_M weights are ~48 GB and pretending `0,1,0` would hold KV
is worse than spending some display-GPU VRAM.

**Adding an alias requires a privilege decision.** `candidate_policy` maps every
alias to a tier; an alias it does not know returns `None` and is refused tools.
`uncovered_router_presets()` is the check, and the router gate asserts it is
empty. See [developer guide](DEVELOPER-GUIDE.md#adding-a-router-preset).

## Upstream dependency lock

`upstream/llama-cpp.lock.json`, schema `frankenstein-upstream-dependency/1`:

| Field | Value | Enforced by |
|---|---|---|
| `repository` | `https://github.com/ggml-org/llama.cpp.git` | build script, `.gitmodules`, pin suite |
| `submodule_path` | `upstream/llama.cpp` | `.gitmodules`, pin suite |
| `tag` | `v0.4.0` | pin suite |
| `commit` | 40-hex | build script (worktree `HEAD`), pin suite (staged gitlink) |
| `build.backend` | `ROCm/HIP` | pin suite |
| `build.gpu_targets` | `["gfx1030"]` | build script, pin suite |
| `build.build_type` | `Release` | pin suite |
| `build.generator` | `Ninja` | pin suite |
| `build.parallelism` | `"nproc"` | pin suite — a fixed job count outlives its machine |
| `build.targets` | `llama-server`, `llama-cli`, `llama-quantize`, `llama-gguf` | build script, pin suite |

## Sidecar environment files

`services/sidecar-<instance>.env`, consumed by `llama-sidecar@<instance>.service`.

| Variable | Meaning |
|---|---|
| `SIDECAR_PORT` | loopback port |
| `SIDECAR_ALIAS` | `--alias` reported by `/v1/models` |
| `SIDECAR_MODEL` | absolute GGUF path |
| `SIDECAR_DEVICE` | `--device` list |
| `SIDECAR_NGL` | `--n-gpu-layers` |
| `SIDECAR_CTX` / `SIDECAR_BATCH` / `SIDECAR_UBATCH` | context and batch sizes |
| `SIDECAR_EXTRA_ARGS` | additional flags, word-split by systemd |

`SIDECAR_EXTRA_ARGS` is referenced in the unit as `$SIDECAR_EXTRA_ARGS`, not
`${…}`: systemd only word-splits the unbraced form, and the braced form would
pass the whole string as one argument that llama-server rejects.

Per-instance choices worth keeping:

- **embeddings** — `--pooling last` because Qwen3-Embedding is a causal LM whose
  sentence vector is the final-token hidden state; mean/cls silently degrades
  recall. `UBATCH` must be ≥ the longest single chunk, because a pooled sequence
  cannot be split across micro-batches.
- **reranker** — `--pooling rank` needs the classifier head the official
  converter preserves. A GGUF without it loads fine and returns meaningless
  scores, which is why the gate checks score *order*.
- **ocr** — bf16 on a 1B model costs about 2 GiB and avoids quantisation damage
  on dense glyphs; `--temp 0` because OCR is transcription, not generation.
- **fim** — exactly the artifact `llama-server --fim-qwen-7b-default` resolves
  to, so the prefix/suffix/middle tokens `/infill` needs are known present.

## systemd units

Tracked copies of user units. They are **not** installed by cloning; see
[operations → installing the units](OPERATIONS.md#installing-the-user-units).

| Unit | Type | `[Install]` | Purpose |
|---|---|---|---|
| `llama-router.service` | simple, restart on-failure | yes | the router on `:8080` |
| `llama-sidecar@.service` | simple template, restart on-failure | yes | one sidecar per env file |
| `llama-ridge.service` | simple | yes | single-model rollback; conflicts with the router |
| `local-ai-model-downloads.service` | simple, restart on-failure | yes | phase-one queue |
| `local-ai-model-downloads-phase2.service` | simple | yes | UI-TARS |
| `local-ai-model-downloads-phase3.service` | simple | yes | Qwen Image Edit |
| `local-ai-model-downloads-phase4.service` | simple | yes | researched candidates |
| `obliterated-mmproj-download.service` | oneshot | yes | one projector download |
| `glm53flash-reverify.service` | oneshot | yes | force publisher-hash reverify of GLM shards |
| `local-ai-functional-mission.service` | oneshot | yes | the serialized mission |
| `local-ai-computer-use-gate.service` | exec | **no** | grounding gate; started explicitly |
| `local-ai-tts-gate.service` | oneshot | no | TTS→ASR round trip |
| `local-ai-repo-agent-gate.service` | oneshot | no | repository-agent A/B |
| `local-ai-media-schema-gate.service` | oneshot | no | live ComfyUI schema discovery |

Notable hardening: the download phases run `ProtectSystem=strict`,
`ProtectHome=read-only`, `NoNewPrivileges=true` with `ReadWritePaths` limited to
`models/` and their own foundation directory. The mission unit adds
`/run/user/%U` so it can reach the user bus. The computer-use unit deliberately
omits `[Install]` because it places ~15.5 GiB across all three GPUs and must
never start at boot; it bounds restarts (`StartLimitBurst=3`), excludes its own
verdict exit codes from restart (`RestartPreventExitStatus=1 3 5`), stops rather
than restarts on OOM, and runs an `ExecStopPost` hook so a result line is written
even when the payload never got to describe its own ending.

## Ports

All loopback-only. Nothing here may be exposed to a LAN or the Internet.

| Port | Service | Started by |
|---|---|---|
| 8080 | llama.cpp router (`/v1`, `/models`, `/health`, `/infill`) | `llama-router.service` |
| 8081 | embeddings sidecar | `llama-sidecar@embeddings.service` |
| 8082 | reranker sidecar | `llama-sidecar@reranker.service` |
| 8083 | OCR sidecar | `llama-sidecar@ocr.service` |
| 8084 | FIM sidecar | `llama-sidecar@fim.service` |
| 8188 | ComfyUI | a media gate runner, for the duration of the gate |

## Environment variables

### Downloader — `download_queue.py`

| Variable | Default | Bounds |
|---|---|---|
| `HERMES_DOWNLOAD_QUEUE` | `download-queue.json` | path |
| `HERMES_DOWNLOAD_STATE` | `download-state.json` | path |
| `HERMES_DOWNLOAD_LOCK` | `download-queue.lock` | path |
| `HERMES_DOWNLOAD_STAMP` | `downloads-complete.ok` | path |
| `HERMES_DOWNLOAD_LOG` | `downloads.log` | path |
| `HERMES_DOWNLOAD_WORKERS` | `16` | 1–64; non-integer or out-of-range raises |
| `HERMES_DOWNLOAD_CONNECTION_BUDGET` | `32` | 1–256; divided across active files, capped at 16 per file |

The queue takes **no command-line arguments** — every path comes from the
environment so a unit and an operator shell cannot disagree about them. `-h` and
`--help` print usage and exit 0 before any lock, log or state write; any other
argument exits 2 the same way. Phase runners set these variables and `execve`
into the queue.

Service policy: phase one and two use budget 32; phases three and four use 64.
All four use 16 workers.

### Mission supervisor

| Variable | Default | Behaviour |
|---|---|---|
| `HERMES_MISSION_QUIET_TIMEOUT` | `21600` (6 h) | positive integer seconds; empty or unset uses the default, anything else raises rather than silently defaulting |

### GLM isolated runtime

| Variable | Default | Behaviour |
|---|---|---|
| `GLM_SRC` | `upstream/llama.cpp/.worktrees/glm53flash-local` | experimental worktree root |
| `GLM_BINARY` | derived from `GLM_SRC` | direct binary override; the gate refuses to start if absent |
| `GLM_FORCE_REVERIFY` | unset | set to `1` by `glm53flash-reverify.service` to re-hash all 15 shards |

### Legacy rollback server — `scripts/serve-ridge.sh`

`RIDGE_MODEL`, `LLAMA_HOST`, `LLAMA_PORT`, `LLAMA_CTX`, `SPEC_DRAFT_N_MAX`,
`LLAMA_SERVER_BIN`, `LLAMA_LOG`. Defaults reproduce the historical Ridge-only
configuration, including `--tensor-split 1,2,1` — which is the *rollback*
proportion, not the router's `3,6,2`.

### Download scripts

`scripts/download-ridge.sh` reads `LLM_MODELS_DIR`, `RIDGE_FILE` and
`RIDGE_MMPROJ` (`1` to also fetch the projector).

## Download queue schema

`hermes-hf-artifact-queue/1`.

```jsonc
{
  "schema": "hermes-hf-artifact-queue/1",
  "built_at": "2026-09-03T18:01:01-0400",
  "source_of_truth": ".../local-coverage-foundation/research/hf",
  "policy": { "single_writer": true, "resume": "...", "promotion": "...", ... },
  "total_bytes": 98653778996,
  "already_satisfied": [                 // phase four only
    { "key": "asr-qwen3-1.7b-hf", "repository": "...", "revision": "...",
      "source_queue": "download-queue.json" }
  ],
  "artifacts": [
    {
      "key": "repository-agent-qwen3-coder-next-q4km",
      "capability": "repository-agent",
      "repository": "Qwen/Qwen3-Coder-Next-GGUF",
      "revision": "b82fb738...",
      "files": [
        { "repo_path": "...gguf", "destination": "/abs/path.gguf",
          "size": 12345678, "sha256": "…" }      // sha256 for LFS files only
      ]
    }
  ]
}
```

`capability` is the join key: the ledger groups artifacts by it, and
`candidate_policy` looks artifacts up by `key`. `sha256` is optional because
small config/tokenizer files are size-gated by Hugging Face rather than LFS.

The four queues:

| Queue | Built | Artifacts | Files | `total_bytes` | Capabilities |
|---|---|---|---|---|---|
| `download-queue.json` | 2026-09-01 | 8 | 46 | 72,134,030,730 | embeddings, reranking, ocr, asr, tts, music, image, fim |
| `download-queue-phase2.json` | 2026-09-01 | 1 | 18 | 33,184,695,056 | computer-use-grounding |
| `download-queue-phase3.json` | 2026-09-02 | 4 | 4 | 30,987,169,046 | image-editing |
| `download-queue-phase4.json` | 2026-09-03 | 5 | 50 | 98,653,778,996 | repository-agent, uncensored-multimodal, computer-use-grounding, multimodal-embeddings, image-generation-editing |
| **total** | | **18** | **118** | **234,959,673,828** (≈218.8 GiB) | |

## Download state and stamps

`download-state[-phaseN].json`, schema `hermes-hf-download-state/1` (ignored):

| Field | Meaning |
|---|---|
| `status` | `running` / `complete` / `failed` |
| `queue_built_at` | copied from the queue |
| `first_started_at` / `started_at` / `completed_at` / `failed_at` | timestamps |
| `error` | `repr()` of the failure |
| `artifacts.<key>` | `capability`, `repository`, `revision`, `status`, `files_complete`, `files_total`, optional `errors[]`, `completed_at` |

`downloads[-phaseN]-complete.ok` holds the queue's `total_bytes` as decimal text
plus a newline. It is written with the same atomic-plus-fsync path as the state,
because a torn stamp is not a retryable state — it reads as a permanent mismatch
against a queue that is in fact complete.

## Mission state schema

`verification/mission-supervisor/mission-state.json`,
`frankenstein-functional-mission/1` (ignored):

| Field | Meaning |
|---|---|
| `status` | `starting`, `waiting-artifacts`, `waiting-safe-host`, `running`, `running-with-failures`, `blocked-policy`, `functional-foundation-incomplete`, `functional-foundation-complete`, `interrupted`, `failed` |
| `boot_id`, `kernel_release`, `kernel_build_signature` | which boot produced this run |
| `input_fingerprint` | SHA-256 over tracked sources, promotion records, queues and queued-file metadata; a step is skipped only when its recorded fingerprint still matches |
| `steps.<name>` | `command`, `input_fingerprint`, `started_at`, `finished_at`, `exit_code`, `status` (`running`/`passed`/`failed`/`interrupted`/`blocked-policy`), `benchmarking_performed: false`, `throughput_measured: false` |
| `failed_steps[]`, `remaining[]`, `exit_code` | terminal summary |
| `signal`, `interrupted_step`, `step_exit_code`, `step_status` | operator-stop detail |

Per-step logs land beside it as `<step>.log`, with `mission.log` as the lifecycle
log.

## Capability ledger schema

`verification/local-coverage-foundation/evidence/capability-ledger.json`,
`hermes-local-capability-ledger/2` (ignored):

| Field | Meaning |
|---|---|
| `built_at`, `source_of_truth[]` | when, and from which queues |
| `model_inference_performed`, `throughput_measured`, `sha256_reverified` | all `false` by construction |
| `capabilities.<name>.state` | one of the seven states in [architecture → layer 6](ARCHITECTURE.md#layer-6--reporting) |
| `capabilities.<name>.reasons[]` | why that state, in words |
| `capabilities.<name>.artifacts[]` | per-file presence, bytes vs expected, `newest_mtime`, `complete` |
| `capabilities.<name>.evidence[]` | per-artifact `present`, `pass`, `interrupted`, `recorded_at`, `gate`, `error`, `sections_missing` |
| `functionally_qualified[]` / `not_qualified[]` | sorted names |
| `problems[]` | throughput claims, evidence declared for a capability no queue owns |
| `pass` | `problems` is empty — a statement about the ledger, not the stack |

## Prompt corpus schemas

`source-catalog.json` — `hermes-prompt-source-catalog/1`:

| Field | Meaning |
|---|---|
| `id` | stable source key referenced by manifests |
| `url`, `revision` | immutable pin (40-hex commit or dataset revision) |
| `license_spdx`, `license_status` | `verified` / `unverified` / `missing` / `mixed-or-incomplete` |
| `disposition` | `candidate` (may supply artifacts), `reference-only`, `rejected` |
| `suites[]` | `harmful-refusal`, `false-refusal`, `benign-utility`, `tool-integrity`, `mcp-tool-integrity`, `computer-use-safety`, `response-judge`, `multimodal-safety`, `prompt-research`, `reference-catalog` |
| `execution_policy` | `inert-text-only` / `never-execute` / `never-ingest` |
| `reason` | why this disposition |

Artifact manifest — `hermes-prompt-corpus-artifact/1`: catalog source ID, exact
40-character revision, one approved suite, a relative path beneath the ignored
`corpora/`, `jsonl` or `csv`, exact bytes, SHA-256, row count, field schema, and
`inert_text_only: true`.

Score families are kept separate and a pass in one never qualifies another:
harmful-request refusal and harmful compliance; false refusal and benign utility;
indirect prompt injection and tool integrity; future MCP and computer-use safety
in disposable synthetic environments.

## Other tracked manifests

| File | Schema | Purpose |
|---|---|---|
| `verification/obliterated-mmproj-manifest.json` | ad hoc | repository, revision, filename, size, SHA-256, local path and verification time for the Qwen3.8 OBLITERATED projector |
| `verification/glm53flash-local/manifest.tsv` | `sha256<TAB>bytes<TAB>filename` | 15 GLM-5.3-Flash IQ3_XXS shards, 120,994,791,264 bytes total |
| `verification/local-coverage-foundation/research/hf-lfs-index.tsv` | flat index | every LFS file observed during metadata collection |
| `verification/computer-use-grounding/fixtures/ground-truth.json` | ad hoc | control rectangles and OCR text, emitted by `make_fixtures.py` from the same `LAYOUT` that drew the pixels |
| `verification/local-coverage-foundation/fixtures/ground-truth.json` | ad hoc | OCR and screen-grounding ground truth for the validator fixtures |
| `verification/candidate-qualification/requirements.in` / `.lock` | uv | the isolated `venvs/candidates` runtime; `.lock` is hash-locked and is the one `*.lock` file Git tracks |

## ComfyUI model paths

`verification/generative-media/extra_model_paths.yaml` is passed with
`--extra-model-paths-config` and is deliberately **not** installed into the
ComfyUI checkout. `base_path` is `models/comfy`, and each category may list
several newline-separated directories, which ComfyUI searches in turn.

FLUX.2-klein-4B is installed in the publisher's own layout — a native ComfyUI
transformer at the repository root plus Diffusers subfolders — so rather than
copying files into the flat tree, `flux2-klein-4b`, `flux2-klein-4b/text_encoder`,
`flux2-klein-4b/vae` and `flux2-klein-4b/tokenizer` are added to the
`diffusion_models`, `text_encoders`, `vae` and `configs` categories.

The preflight resolves every bare filename in the pinned graphs through these
categories, so a reference that resolves to nothing, to two files, or to a file
no workflow claim declares is a failure rather than a surprise at submit time.

## Hermes provider configuration

`hermes-llamacpp-ridge.yaml` is a historical single-model example, superseded by
the router. It is kept because it documents the shape of a Hermes provider entry:
a `providers.<name>` block with `api`, `transport: chat_completions`,
`default_model`, `discover_models` and `context_length`, plus `model_aliases`
entries binding an alias to a provider and base URL.

All Hermes front ends — CLI, TUI and Desktop — share `~/.hermes/config.yaml`.
There is no separate Desktop model registry. That file is **not** tracked here;
this repository configures the server side only.

## What Git tracks

Per [ADR 0001](../decisions/0001-git-tracked-workspace-without-weights.md):
source, configuration, scripts, systemd templates, verification code, small named
fixtures and documentation. Not tracked:

| Ignored | Why |
|---|---|
| `/models/` | weights and shards |
| `/venvs/`, `/tools/ComfyUI/`, `/tools/qwen38-mtp/` | local runtimes and nested upstream checkouts |
| `/logs/`, `router-state.json`, `*.log`, `*.lock`, `*.tmp`, `*.tmp.*`, `*.ok` | runtime state, locks, stamps; `*.tmp.*` alone would miss the plain-suffix crash residue |
| `*.partial`, `*.recovered` | transfer debris |
| `verification/**/evidence/`, `**/download-state*.json`, `**/mission-state.json`, `**/transcript-*.jsonl`, `**/lane.json`, `**/build-*/` | gate evidence and per-run state |
| `/verify-*`, several named probe dumps | ad-hoc probe output |
| `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `*.egg-info/` | Python debris |

One negation: `!/verification/candidate-qualification/requirements.lock`, because
that is a dependency hash lock rather than a transient process lock.

## Data flows

**Acquisition.** Hugging Face API → `research/hf/*.json` + `hf-lfs-index.tsv` →
`build_download_queue.py` / `build_candidate_queue.py` → `download-queue*.json`
→ `download_queue.py` → `models/**` + `download-state*.json` +
`downloads*-complete.ok`.

**Admission.** `download-queue-phase4.json` + `models/**` + `llama-models.ini` +
`WEMM-REMOTE-CODE-REVIEW.md` → `gate_candidate_policy.py` →
`candidate-qualification/evidence/candidate-policy.json` → mission step one.

**Qualification.** stamps + quiet host → mission supervisor → each gate →
`verification/**/evidence/*.json` + `mission-state.json`.

**Reporting.** `download-queue*.json` + `models/**` metadata + gate artifacts →
`build_capability_ledger.py` → `evidence/capability-ledger.json` → the prose
[capability matrix](CAPABILITY-MATRIX.md).

**Serving.** `llama-models.ini` → `llama-router.service` → `:8080` → Hermes,
gates, RAG. `services/sidecar-*.env` → `llama-sidecar@*.service` → `:8081`–`:8084`
→ RAG ingest/answer, FIM clients, OCR fallback.

**Retrieval.** files → `rag_store.ingest_file` → embeddings sidecar →
`rag/hermes-rag.sqlite3` (4096-D) → `rag_answer` → reranker sidecar → bounded
context → router. WeMM's 2048-D multimodal vectors go to a separate database and
alias; `index_conflicts` refuses a configuration that merges them.
