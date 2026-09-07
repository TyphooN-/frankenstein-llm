# Architecture and implementation reference

> Operator-interface update: [Model runs](../MODEL-RUNS.md) is the current reference
> for standalone serving, qualification and native benchmark scripts. The former
> standalone Ridge environment overrides are replaced by shared presets/configs.
> Any older statement below about absent benchmark tooling predates this interface;
> native throughput tooling exists, but comparative agent-quality evaluation remains absent.

Undated reference. It describes how the tracked code is built, not what the host
happened to be running when it was written. Dated readings live in
[the capability matrix](CAPABILITY-MATRIX.md) and the snapshots it links.

**Backend: AMD ROCm/HIP, exclusively.** `upstream/llama-cpp.lock.json` pins
`"backend": "ROCm/HIP"` with `"gpu_targets": ["gfx1030"]`, and
`scripts/build-llama-cpp.sh` refuses a build whose `GPU_TARGETS` is anything else.
Every layer below inherits that: device selectors are `ROCm<N>`, VRAM is read from
`amdgpu` sysfs nodes, and the media stack is scoped with `HIP_VISIBLE_DEVICES`.
There is no CUDA code path to fall back to. How placement works across the three
cards is in [GPU execution and model loading](GPU-EXECUTION-AND-MODEL-LOADING.md).

Related: [configuration reference](CONFIGURATION.md) ·
[operations runbook](OPERATIONS.md) · [developer guide](DEVELOPER-GUIDE.md)

## Contents

- [What the stack is](#what-the-stack-is)
- [Layer map](#layer-map)
- [Layer 0 — pinned upstream](#layer-0--pinned-upstream)
- [Layer 1 — serving](#layer-1--serving)
- [Layer 2 — acquisition](#layer-2--acquisition)
- [Layer 3 — admission policy](#layer-3--admission-policy)
- [Layer 4 — functional gates](#layer-4--functional-gates)
- [Layer 5 — supervision](#layer-5--supervision)
- [Layer 6 — reporting](#layer-6--reporting)
- [Application lanes](#application-lanes)
- [Cross-cutting rules](#cross-cutting-rules)
- [Evidence artifacts](#evidence-artifacts)

## What the stack is

A single-host, loopback-only local AI workspace. Three things are kept separate
on purpose, and most of the design follows from that separation:

1. **Source, configuration and verification code** — tracked in Git.
2. **Model weights and runtime state** — on local disk, never tracked.
3. **Capability claims** — produced only by gates that observed behaviour, never
   by a document asserting it.

The repository tracks (1). It contains the pins, policy and gates that decide
whether (2) may be used, and it refuses to let (3) be written by hand.

Two rules constrain nearly every module:

- **Fail closed.** Absent evidence is never a pass. A missing artifact, an
  interrupted run, an unreadable timestamp, an unknown model status and an
  unknown router alias all resolve to "not allowed" rather than "probably fine".
- **No throughput.** Functional load, coherence, memory fit and clean unload are
  in scope. Tokens/sec, latency and comparative benchmarks are not, per
  [ADR 0002](../decisions/0002-serialized-functional-qualification.md). Gate
  artifacts carry an explicit `throughput_measured: false`, and the capability
  ledger reports an artifact that claims otherwise as a problem.

## Layer map

```
                       ┌──────────────────────────────────────────────┐
  Hermes CLI/TUI/      │ layer 1  SERVING                             │
  Desktop, gates,      │   llama-router.service  :8080  1 resident    │
  editors      ───────▶│   llama-sidecar@{embeddings,reranker,        │
                       │      ocr,fim}          :8081-:8084           │
                       │   ComfyUI (gate-launched only)  :8188        │
                       └───────▲──────────────────────────┬───────────┘
                               │ execs binaries           │ observed by
  ┌────────────────────────────┴───────────┐  ┌───────────▼───────────┐
  │ layer 0  PINNED UPSTREAM               │  │ layer 4  GATES        │
  │   upstream/llama.cpp  (submodule)      │  │   router-functional   │
  │   upstream/llama-cpp.lock.json         │  │   validators/*        │
  │   scripts/build-llama-cpp.sh           │  │   computer-use        │
  └────────────────────────────────────────┘  │   tts-local           │
                                              │   repository-agent    │
  ┌────────────────────────────────────────┐  │   generative-media    │
  │ layer 2  ACQUISITION                   │  │   candidate-qual      │
  │   download-queue*.json  (pins)         │  └───────────┬───────────┘
  │   download_queue.py     (transfers)    │              │ writes
  │   run_download_phase{2,3,4}.py         │              ▼
  └───────────────┬────────────────────────┘   verification/*/evidence/
                  │ promotes verified bytes                │  (ignored)
                  ▼                                        │
        models/ (ignored)  ──────────────┐                 │
                                         ▼                 ▼
  ┌────────────────────────────────────────┐  ┌────────────────────────┐
  │ layer 3  ADMISSION POLICY              │  │ layer 6  REPORTING     │
  │   candidate_policy.py                  │  │   build_capability_    │
  │   wemm_remote_code_review.py           │  │     ledger.py          │
  │   prompt-corpus-admission/             │  └────────────────────────┘
  └───────────────┬────────────────────────┘
                  │ blocking prerequisite
                  ▼
  ┌────────────────────────────────────────┐
  │ layer 5  SUPERVISION                   │
  │   mission-supervisor/run_functional_   │
  │     mission.py  (serialized, resumable)│
  └────────────────────────────────────────┘
```

## Layer 0 — pinned upstream

`upstream/llama.cpp` is a Git submodule of `https://github.com/ggml-org/llama.cpp.git`.
The revision this stack runs is written down in three places, and the three are
deliberately checked against each other because only one of them is what a fresh
clone actually checks out:

| Record | Where | Checked by |
|---|---|---|
| Outer gitlink | `git ls-files -s upstream/llama.cpp` | `verification/upstream-pin/test_llama_cpp_pin.py` |
| Dependency lock | `upstream/llama-cpp.lock.json` | the build script, before CMake runs |
| Prose | [ADR 0005](../decisions/0005-track-llama-cpp-submodule.md) | review |

`scripts/build-llama-cpp.sh` is the only supported build entry point. It reads
the lock, refuses to continue when the submodule `HEAD` is not the locked commit,
refuses a dirty submodule worktree, refuses an unexpected repository URL or GPU
target, then configures HIP/ROCm + Ninja + Release and builds exactly the four
locked targets. Job count comes from `nproc`, so a job number cannot outlive the
machine it was measured on. See
[operations → upgrading llama.cpp](OPERATIONS.md#upgrading-llamacpp).

The gitlink is checked separately from the worktree because the build script
reads the worktree and cannot see a stale index: staging the submodule while its
worktree sat on some other revision pins that revision for every later clone
while the lock, the documents and every local binary still say v0.4.0.

Binaries are produced into the submodule's ignored `build/bin/` and executed from
there directly. There is no supported runtime symlink under `~/.local/bin` and no
second checkout under `/home/typhoon/src`; the pin test greps every tracked file
for both, exempting prose so documents stay free to name a path in order to
forbid it.

## Layer 1 — serving

### Router

One `llama-server` process, started by `llama-router.service`, reads
`llama-models.ini` through `--models-preset` and serves an OpenAI-compatible API
on `127.0.0.1:8080`. `--models-max 1` keeps a single resident model: selecting a
different alias evicts the previous one, so the first reply after a switch waits
for a load and later replies do not.

Every alias in the INI is a *preset*, not a running process. Preset keys are
llama-server command-line options with the leading dashes removed, and llama.cpp
treats an unknown key as fatal at load time — a key dropped by a future release
does not degrade one alias, it stops the router from starting at all. That is why
`PresetCompatibilityTests` in the pin suite checks every key in the INI against
the option list `llama-server --help` prints from the build that is actually
installed. The full preset table is in
[configuration → router presets](CONFIGURATION.md#router-presets).

### Sidecars

`llama-sidecar@.service` is a template unit. Each instance reads
`services/sidecar-<instance>.env` and starts an independent `llama-server` bound
to its own loopback port with `--load-mode none`. Sidecars are deliberately *not*
bound to the router unit: they are small, single-purpose and pinned to their own
GPU share, so a router restart must not cycle them.

They run at `Nice=5` with `IOSchedulingClass=best-effort` and
`IOSchedulingPriority=6`, which the unit comments as keeping them from winning a
resource fight against the interactive router. Read that as intent, not as a
mechanism that prevents contention:

- `Nice=` weights **CPU** scheduling only. The contention that matters between a
  sidecar and the router is VRAM residency, and a resident sidecar holds its
  share of a card whatever its nice value. GPU command submission is not ordered
  by a CPU nice value either.
- `Nice=5` is also what `local-ai-functional-mission.service` runs at, so it does
  not order the sidecars against the mission.
- The I/O settings are one notch below the kernel default — best-effort priority
  4 (`systemd.exec(5)`) — and take effect at all only "in conjunction with an I/O
  scheduler that supports I/O priorities" (`ioprio_set(2)`). Which scheduler a
  device uses is readable per device:
  `cat /sys/block/<device>/queue/scheduler`.

The real serialization is elsewhere: `--models-max 1` on the router, the
supervisor's quiet-host wait, and the fact that a resident sidecar is classified
as an `inference` conflict and therefore blocks a mission step outright.

| Instance | Port | Alias | Purpose |
|---|---|---|---|
| `embeddings` | 8081 | `qwen3-embedding-8b` | 4096-D text embeddings, last-token pooling, L2-normalised |
| `reranker` | 8082 | `qwen3-reranker-8b` | cross-encoder rerank, `rank` pooling |
| `ocr` | 8083 | `hunyuan-ocr` | dense-document transcription with a bf16 projector |
| `fim` | 8084 | `qwen25-coder-7b-fim` | `/infill` fill-in-the-middle completion |

### ComfyUI

ComfyUI is not a managed service. It is launched by the media gate runners for
the duration of a gate, on `127.0.0.1:8188`, with `HIP_VISIBLE_DEVICES=0,1` so
the display GPU is never exposed, `--cuda-device 1` to select the 32 GiB V620,
and `--extra-model-paths-config` pointing at
`verification/generative-media/extra_model_paths.yaml`. The runners stop the
router first and restart it afterwards, then assert the V620 released its
residency.

### Rollback path

`llama-ridge.service` runs `scripts/serve-ridge.sh`, a single-model server for
one alias. It exists as a rollback and must never run alongside the router: both
bind port 8080 and own the same GPUs.

## Layer 2 — acquisition

Weights arrive through four pinned queues, each a `hermes-hf-artifact-queue/1`
document listing artifacts, their Hugging Face repository and exact revision, and
every file's `repo_path`, `destination`, `size` and (for LFS files) `sha256`.
Queues are generated from collected metadata rather than hand-typed —
`research/collect_hf_metadata.py` reads the Hugging Face API,
`research/build_download_queue.py` and `research/build_candidate_queue.py` emit
the queues — so a queue cannot drift from the publisher tree that was observed.

`download_queue.py` executes one queue. It takes an exclusive `flock`, so only
one writer per queue exists, and rejects a queue whose destinations collide after
`realpath` resolution, so only one worker per file exists. Independent files then
run concurrently in a thread pool.

The transfer path is deliberately state-machine-shaped, because the unit is
restarted on failure and survives reboots:

| Wake-up state | Action |
|---|---|
| Destination present, size and digest match | Verified; nothing transferred |
| Destination present, wrong | Quarantined to `.bad-<epoch>`, transfer restarts |
| `.partial` larger than expected | Quarantined, transfer restarts |
| `.partial` exactly full size | Verified and promoted, or quarantined to `.bad-<digest>` — never resumed, because a resume at EOF returns HTTP 416 and the unit would restart forever without moving a byte |
| `.partial` short | Resumed |

Resume uses `aria2c` with bounded per-file range connections when aria2 is
installed *and* the partial is either fresh or already aria2-managed; a partial
created by curl cannot have its valid ranges inferred, so it stays on curl.
Concurrency is `HERMES_DOWNLOAD_WORKERS` files with a shared
`HERMES_DOWNLOAD_CONNECTION_BUDGET` divided across active files and capped at 16
per file.

Promotion is `os.replace` **after** an exact size match and, when the publisher
publishes one, an exact SHA-256 match — and after the partial's own bytes are
fsynced. The parent directory is fsynced too, because downstream phases trust the
completion stamp and a rename that evaporates on power loss would leave a stamp
describing bytes that are not there.

On success the queue writes `downloads[-phaseN]-complete.ok` containing the exact
byte total. Phases 2–4 refuse to start until every earlier phase's state says
`complete` *and* its stamp matches the byte total compiled into the runner; the
mission supervisor makes the same check across all four. Those repeated totals
are cross-checked offline by `test_queue_manifests.py`, because a queue edited
without updating its copies produces either a post-transfer stamp mismatch or a
supervisor waiting forever for a number nothing will write.

## Layer 3 — admission policy

Downloading proves that pinned bytes arrived intact. It says nothing about what
the model is allowed to do. Three policy modules answer that, and none of them
loads a model, starts a service or touches a GPU.

### Candidate policy

`verification/candidate-qualification/candidate_policy.py` is the privilege
model. Every fact about *what is installed* is read from the phase-four queue —
the artifact the transfer actually verified — so no size or revision is
re-declared anywhere. What the module adds is the part a download manifest cannot
express.

Three privilege tiers, most restrictive first:

| Tier | May generate text | May receive tools | Holders |
|---|---|---|---|
| `read-only` | no | no | `qwen3-embedding-8b`, `qwen3-reranker-8b`, `qwen25-coder-7b-fim`, UI-Mate-9B, WeMM-Embedding-2B, FLUX.2-klein-4B, Qwen3-ASR-1.7B |
| `low` | yes | never | `gemma4-heretic`, `gemma4-heretic-vision` |
| `tool-using` | yes | yes | `ridge`, `heretic`, `obliterated`, `obliterated-vision`, `fable`, `phr00ty`, `qwen3-coder-next` |

Four structural rules, each a function that returns `False` when the evidence it
needs is missing:

- `tool_grant_allowed` / `tool_grant_allowed_for_preset` — only the tool-using
  tier receives executable tools. An **unknown alias returns `None` and is
  refused**, so a preset added to the INI without a privilege decision does not
  inherit tools by falling through. `uncovered_router_presets()` is the
  fail-closed check for that, and the router gate asserts it is empty.
- `abliteration_allowed` — ablated weights are refused for anything in
  `ACTING_CAPABILITIES` (`computer-use-grounding`, `computer-use-end-to-end`,
  `repository-agent`). Removing refusals from an actor removes a safety boundary,
  so this is structural rather than reviewer discipline.
- `control_allowed` — a GUI candidate gets desktop-control authority only after
  its own grounding verdict exists and passed. `None` (no verdict) is refused
  exactly as a failure is.
- `remote_code_execution_allowed` — a candidate that ships executable model code
  may not be imported until the tracked review approves those exact bytes.

`index_conflicts` refuses any configuration that would merge the 4096-D Qwen3
text index and the 2048-D WeMM multimodal index into one database or alias, and
refuses to let dimension difference alone be the justification.

`gate_candidate_policy.py` runs all of it, adds an exact-size inventory of every
phase-four file, proves the phase-four "already satisfied" ASR claim against the
phase-one queue rather than trusting it, and publishes
`evidence/candidate-policy.json`. It is step one of the mission and blocks every
model-backed step behind it.

### Remote-code review

`wemm_remote_code_review.py` is the execution boundary for
`tencent/WeMM-Embedding-2B`, whose `config.json` `auto_map` makes
`AutoModel.from_pretrained` import publisher-authored Python from the model
directory. The module never imports those files: it reads them as bytes, digests
them, and compares against the digests recorded in
[`WEMM-REMOTE-CODE-REVIEW.md`](../../verification/candidate-qualification/WEMM-REMOTE-CODE-REVIEW.md).
Two files are approved for import; `patch_sglang_video.py`, which rewrites an
installed site-packages file in place, is recorded `never-execute`. A file that
changed, appeared or vanished revokes approval rather than warning.

### Prompt-corpus admission

`verification/prompt-corpus-admission/` decides which external prompt/safety
sources may become local evaluation input. `source-catalog.json` is policy, not
proof of download: each source carries an immutable revision, SPDX licence and
licence status, a disposition (`candidate` / `reference-only` / `rejected`), a
suite assignment and an execution policy. Only `candidate` + verified licence +
`inert-text-only` may produce an artifact, and only via a separate
`hermes-prompt-corpus-artifact/1` manifest binding source, revision, suite,
relative path, format, exact bytes, SHA-256, row count and field schema.

`admit_corpus.py` validates that manifest against the bytes. It never downloads,
never invokes a model, never executes corpus content, and rejects symlinks,
parent traversal, executable permissions, NUL/binary content, invalid UTF-8,
credential markers, schema drift, unknown sources and reference-only/rejected
sources. Existing passing evidence is not overwritten by a later failing
candidate. Corpus bytes stay ignored by Git; admission evidence is runtime state.
Rationale is [ADR 0004](../decisions/0004-prompt-corpus-admission.md).

## Layer 4 — functional gates

A gate is admitted only on observed behaviour of the *content* returned. "The
server answered 200" and "the model card says so" are explicitly not evidence,
and every component additionally has to give its GPU memory back when stopped.
`validators/gatelib.py` holds the shared primitives: VRAM sampling per DRM card,
residue arithmetic, a clean-unload verdict that **fails when it could not
measure**, a managed-sidecar context manager, and cleanup that runs whether or
not the gate reached its own unload step and never raises out of `finally`.

| Gate | Entry point | What it proves | Evidence artifact |
|---|---|---|---|
| Candidate policy | `candidate-qualification/gate_candidate_policy.py` | inventory, dedup claim, index separation, privileges, reviewed digests | `candidate-qualification/evidence/candidate-policy.json` |
| WeMM embeddings | `candidate-qualification/gate_wemm.py` via `run_wemm.sh` | 2048-D vectors, L2 norm, paraphrase > related > noise | `candidate-qualification/evidence/wemm-functional.json` |
| Router models | `router-functional/gate_router_models.py` | per-preset coherence, structured output, tool call, vision, memory fit, clean release | `router-functional/evidence/router-functional.json` |
| Embeddings | `validators/gate_embeddings.py` | dimension, normalisation, semantic ordering, determinism, batch invariance | `evidence/gate-embeddings.json` |
| Reranker | `validators/gate_reranker.py` | relevant document wins, real score spread, ordering inverts with the query | `evidence/gate-reranker.json` |
| OCR | `validators/gate_ocr.py` | character accuracy on prose plus exact table-cell recovery | `evidence/gate-ocr.json` |
| FIM | `validators/gate_fim.py` | correct middle via `/infill`, and stopping instead of running past the suffix | `evidence/gate-fim.json` |
| ASR | `validators/gate_asr.py` | transcription of a pinned waveform, load and clean unload | `evidence/gate-asr.json` |
| Native tool use | `validators/gate_native_tool_use.py` | `message.tool_calls[]`, not XML in `content`; and a final answer after the tool result | `evidence/gate-native-tool-use.json` |
| Vision grounding | `validators/gate_vision_grounding.py` | returned point lands inside the real bounding box; injection resistance | `evidence/gate-vision-grounding.json` |
| RAG structural | `validators/gate_rag.py --structural` | chunk spans, update replacement, delete propagation, context bounds, citation fidelity — offline embedder, no GPU | `evidence/gate-rag-structural.json` |
| RAG behavioural | `validators/gate_rag_behavioral.py` | prompt envelope, resolvable citations, invented-citation detection, reranker degradation | `evidence/gate-rag-behavioural.json` |
| RAG live | `validators/gate_rag.py --live` | semantic retrieval quality and end-to-end injection resistance | `evidence/gate-rag-live.json` |
| Computer use | `computer-use-grounding/gate_computer_use.py` | placement, GPU execution, grounding, distractors, typed action selection, OCR, injection, bounds, malformed inputs, memory safety, unload | `computer-use-grounding/evidence/computer-use-grounding.json` |
| TTS → ASR | `tts-local/gate_tts.py` via `run_serialized.sh` | synthesised speech transcribed back by the already-admitted ASR model | `tts-local/evidence/gate-tts.json` |
| Repository agent | `repository-agent/gate_repo_agent.py` via `run_serialized.sh` | sandboxed repair of a defective fixture with an untamperable oracle | `repository-agent/evidence/gate-repo-agent-<model>.json` |
| Generative media | `generative-media/functional_gate.py` via `run_functional_serialized.sh` | real image, music and edit outputs from pinned graphs | `generative-media/evidence/media-functional.json` |
| Media schema | `generative-media/live_schema_gate.py` via `run_schema_serialized.sh` | live ComfyUI node/model discovery without submitting generation | `generative-media/evidence/` |
| GLM 32K | `glm53flash-local/gate_glm32.py` | isolated-runtime 32K load, generation and unload | `glm53flash-local/` (ignored) |

Three gates deserve their own note.

**Repository agent.** The agent gets exactly three tools over the router —
`read_file`, `write_file`, `run_tests` — confined to a disposable copy of
`verification/repository-agent/fixture/`. `run_tests` executes candidate-written
Python, and import-time code in a "repair" runs before any assertion does, so the
oracle runs under Bubblewrap: read-only `/usr`, private PID and network
namespaces (the router on host loopback is unreachable), a size-capped private
`/tmp`, all capabilities dropped, a cleared environment, `--disable-userns`, and
one writable bind. There is **no unisolated fallback** — a missing `bwrap`
refuses the run — and a sandbox that cannot start, cannot be executed or does not
finish aborts the qualification instead of returning a retryable tool error,
because a host-side failure reported as a red suite is a false verdict about the
candidate. The test suite is the oracle and is not writable; `run_tests` re-checks
its SHA-256 after every run, and a completed `write_file` invalidates an earlier
green result because a pass describes one exact generation of the workspace.

**Computer-use grounding.** Its scoring is only as trustworthy as its geometry.
`groundlib.py` exists because UI-TARS-1.5-7B and UI-Mate-9B disagree before
either sees a pixel — patch 14 vs 16, a `min_pixels`/`max_pixels` budget vs
`size.shortest_edge`/`longest_edge`, `Thought:`/`Action:` text vs XML tool calls
with a leading `<think>` block. Each geometry is read from that checkpoint's own
`preprocessor_config.json`, a config that does not state its budget is refused,
and both action spaces are normalised so one scorer grades both on the same
fixtures. Injection compliance is judged through the action space the model was
actually given. Actions are applied to `sandbox.py`, a pure-Python model of the
same fixture pixels — nothing synthesises input or touches a display server.
`finalize_verdict` treats a missing section as a failing section.

**Generative media.** `media_policy.py` states coverage as data
(`WORKFLOW_CLAIMS`), separating `artifacts_present` from `functionally_proven`,
so a passing preflight cannot be read as covering more than it does. Because the
pinned API graphs name weights by bare filename, `preflight.py` resolves every
graph reference through the categories declared in `extra_model_paths.yaml`; a
reference that does not resolve, resolves to two files, resolves to an
uninventoried file, or is not declared by its workflow claim fails. That is what
catches the FLUX.2 Klein graph loading the Qwen3-4B text encoder the Z-Image lane
owns.

## Layer 5 — supervision

`verification/mission-supervisor/run_functional_mission.py` runs the serialized
qualification mission. It is a `oneshot` unit, holds an exclusive `flock`
(exit 75 if another supervisor owns it), and is designed to be resumed across
reboots and operator stops.

**Ordering.** `candidate-policy` is step one and is a blocking prerequisite: if
it fails, every later step is recorded `blocked-policy` and nothing model-backed
runs. After it passes, the remaining eleven steps continue past a failure so one
capability cannot hide the status of every later one; the aggregate mission still
fails closed.

The twelve steps, in order: `candidate-policy`, `wemm-embeddings`,
`router-reload-presets`, `router-models`, `embeddings`, `reranker`, `fim`, `asr`,
`computer-use-grounding`, `tts-asr-roundtrip`, `repository-agent`,
`generative-media-functional`.

**Waiting for inputs.** `wait_for_inputs` is deliberately unbounded: it waits on
roughly 219 GiB of downloads that legitimately take days. A completion stamp that
matches the expected byte total is readiness even if a downloader has marked
state `running` while it re-hashes existing files after reboot. A `failed` state
aborts; a stamp that does not match its expected byte total aborts.

**Waiting for a quiet host.** `wait_for_quiet` requires two consecutive clean
polls with `MemAvailable ≥ 32 GiB` and no competing kernel-tree build or
unmanaged llama.cpp/inference process. Unrelated cargo/rustc, download re-hash,
and desktop load average are not blockers: they are not a missing kernel and
must not hold the mission after a new-kernel boot. Unlike the input wait it is
bounded — default six hours, overridable with `HERMES_MISSION_QUIET_TIMEOUT`,
which rejects a non-positive or unparseable value rather than silently
substituting a default. On timeout it records the blockers it actually observed
into durable state *before* raising.

**How conflicts are found.** Classification reads only `/proc/<pid>/comm`, the
`cwd` symlink and `cgroup`. It never opens `/proc/<pid>/cmdline`: Linux serves
that file through `access_remote_vm`, so a task holding its own mmap write lock
can wedge a supervisor that merely tries to inspect it — which is how a
command-line sweep of a host gets stuck behind a browser. All three reads
tolerate the task exiting underneath them, a `cwd` whose directory was deleted
has its ` (deleted)` suffix stripped so a build in a removed tree still counts,
undecodable task names are forced to valid UTF-8, and the proc root is injectable
so classification is tested against fixtures. Classification is coarse and errs
towards reporting: builds, transfers, second model servers and workspace Python
are named on the strength of the task name alone. The single exclusion is the
managed `llama-router.service`, excused by its **cgroup** rather than its name,
so `llama-server` started by hand is still a conflict. Blocker reporting is
bounded — counts by reason plus at most eight samples — so a 44-thread compiler
fan-out cannot flood the log or an error string.

**Resume.** State is published atomically and durably (`.tmp.<pid>`, fsync, rename,
directory fsync) as `frankenstein-functional-mission/1`. A step is skipped on a
later run only when it is recorded `passed` *and* its `input_fingerprint` still
matches. That fingerprint covers `git ls-files -s` and `git diff --binary HEAD`
over `verification/`, `llama-models.ini` and `scripts/`, the four promotion
records and stamps, every queue file, the size and mtime of every queued
destination, and the step list itself — so editing a gate, replacing a weight or
reordering the mission re-runs the affected steps instead of trusting a stale
pass.

**Signals.** `SIGHUP`/`SIGINT`/`SIGTERM` are forwarded to the running step's
process group. Signal state is tested *before* the return code, because a
forwarded `SIGTERM` makes the step exit non-zero and testing the code first
recorded every operator stop as "step X failed". A step that still exited 0 under
a stop really did pass and is kept, so a completed gate is not repeated.

## Layer 6 — reporting

`verification/local-coverage-foundation/build_capability_ledger.py` rebuilds
`evidence/capability-ledger.json` (`hermes-local-capability-ledger/2`, ignored)
from the pinned queues and the gate artifacts. Nothing else: no model is loaded,
no service queried, no GPU touched, no digest recomputed — the download queue
already verified digests at write time, and re-hashing 219 GiB to print a status
line would be a heavy I/O job pretending to be a report.

The generator is tracked precisely because its predecessor was not: the previous
ledger was written by a script outside this repository, could not be rebuilt, and
went on reporting a 2026-09-02 reading after phases three and four landed.

States it can assign, and the fail-closed rules behind them:

| State | Meaning |
|---|---|
| `researched` | named in a review; no artifact owned |
| `download-incomplete` | a declared file is missing or the wrong size |
| `downloaded` | bytes present; no declared evidence, or no artifact on disk |
| `evidence-interrupted` | a gate run was interrupted — an unfinished run is not a verdict |
| `functionally-failed` | a declared artifact exists and does not record a pass |
| `evidence-stale` | evidence was recorded before the weights it judges were installed |
| `functionally-qualified` | every declared artifact records a pass, recorded after the weights |

Two further bindings: an artifact only counts when its `gate` field matches the
expected gate identity, so copying any passing JSON onto a declared path cannot
qualify an unrelated capability; and an artifact reporting measured throughput is
surfaced as a `problem` rather than copied forward. The ledger's own `pass` field
is a statement about the ledger, not about the stack.

## Application lanes

Two lanes are built on the serving layer rather than being gates themselves.

### RAG

`verification/local-coverage-foundation/rag/rag_store.py` is a bounded,
provenance-carrying document store: SQLite for metadata plus NumPy brute-force
cosine, no vector-database daemon. Every chunk keeps source path, content hash,
ordinal and character span, so any retrieved sentence points back at exact bytes.
Update and delete are first-class — re-ingesting a changed file atomically
replaces its chunks, and a removed file's chunks stop being retrievable in the
same transaction that marks the document deleted. Extraction covers plain text,
Markdown, HTML (visible text only) and OOXML, with bounded expansion, an archive
member cap and a replacement-ratio check; PDF needs an optional parser and
scanned documents need the OCR sidecar, and both are reported as explicit skip
reasons rather than silent omissions.

`rag_answer.py` is dense retrieve → optional cross-encoder rerank → bounded
context → answer → citation validation. Two properties matter more than answer
fluency: every `[n]` the model emits must resolve to a passage that was actually
in the context, with unresolvable citations reported rather than rendered; and
retrieved text is framed as untrusted data, so instructions inside a document are
content to summarise, never commands to follow. An unreachable reranker degrades
to dense order with a recorded status instead of pretending it reranked.

### Security-agent scaffold

`verification/security-agents/strix-scaffold/` builds and validates a hardened
run specification for an authorized-testing pilot and **executes nothing** — no
container is started and no packet is sent. `policy.py` encodes the required
profile as data so a configuration can be mechanically refused; `runspec.py`
validates any spec including hostile ones; `scope.py` is a deny-by-default
destination allowlist enforced outside the agent, on the network namespace,
because an agent that can run `exec_command` can curl past rules that live inside
it — and a hostname is in scope only if it resolves entirely into admitted IP
space, which is what stops a DNS rebind. `preflight.py` emits the docker argv a
human would review, a refusal matrix pairing each control with the mutation that
trips it, and a scope-denial matrix over an injected offline resolver. The design
rule: a control counts as present only when something fails without it, so every
constant has a negative test.

## Cross-cutting rules

**Durable publication.** Anything a later run or a reboot reads is written
`.tmp`/`.staged` → fsync → `os.replace` → parent-directory fsync. Where a
filesystem refuses a directory fsync the code tolerates it for state and is
stricter for model promotion, because downstream execution trusts the stamp.
`.tmp` and `.tmp.*` are both gitignored, since a crash between write and rename
leaves the plain suffix behind.

**Single writer.** Every long-running writer takes an exclusive `flock` and exits
75 rather than racing: the download queue per phase, the mission supervisor, the
computer-use gate. The kernel releases the lock on any death, including SIGKILL.

**Untrusted data.** Screen text, retrieved documents and corpus rows are data,
never instructions. The grounding gate scores injection compliance explicitly,
the RAG prompt carries an untrusted-data envelope, and ADR 0004 states that
corpus content has no authority.

**Isolation by construction.** Candidate code runs in Bubblewrap; ComfyUI sees
only the two headless GPUs; the router and every sidecar bind loopback only; the
Strix scaffold never executes; remote model code is digest-pinned before import.

**Environment separation.** Python runtimes are per-lane and ignored:
`venvs/candidates` (hash-locked, Python 3.14, host ROCm Torch via
`--system-site-packages`, refuses a non-ROCm build), `venvs/comfy`, `venvs/tts`,
`venvs/computer-use`. Resolvers are kept away from Torch so an NVIDIA CUDA wheel
cannot be installed beside the AMD runtime.

## Evidence artifacts

Every gate artifact is JSON with, at minimum: a `gate` identity string, a
boolean `pass`, `throughput_measured: false`, and a timestamp
(`recorded_at`/`finished_at`) that `datetime.fromisoformat` can parse. Gates that
can be interrupted also carry `interrupted` and, where sections exist,
`sections_missing`. The ledger reads exactly those fields; see
[developer guide → writing an evidence artifact](DEVELOPER-GUIDE.md#writing-an-evidence-artifact).

All `verification/**/evidence/` directories are gitignored. Evidence is runtime
state, not source: it describes one host at one moment, and a fresh clone is
supposed to have none.
