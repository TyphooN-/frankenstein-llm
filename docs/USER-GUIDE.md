# User guide

For the unified user/agent command interface and per-model serve scripts, see
[Model runs](MODEL-RUNS.md). Benchmark tooling is now available for explicit user-run
measurements; no performance result is implied by its existence.

How to set this workspace up, use it day to day, and tell the difference between
a capability that is downloaded and one that actually works.

**This is an AMD ROCm workspace.** All three GPUs are AMD, the pinned llama.cpp is
built with `-DGGML_HIP=ON` for `gfx1030`, presets address devices as `ROCm0`,
`ROCm1` and `ROCm2`, and ComfyUI runs under `HIP_VISIBLE_DEVICES`. Instructions
written for CUDA/NVIDIA do not apply, and no CUDA build exists here.

Undated. Where something depends on what the host is running right now, this
guide gives you the command rather than an answer.

**Start here**, then follow the references:
[architecture](reference/ARCHITECTURE.md) ·
[configuration](reference/CONFIGURATION.md) ·
[operations](reference/OPERATIONS.md) ·
[troubleshooting](reference/TROUBLESHOOTING.md) ·
[capability matrix](reference/CAPABILITY-MATRIX.md) ·
[developer guide](reference/DEVELOPER-GUIDE.md)

## Contents

- [What this is](#what-this-is)
- [Before you start](#before-you-start)
- [Set up the runtime](#set-up-the-runtime)
- [Get the weights](#get-the-weights)
- [Everyday chat](#everyday-chat)
- [Picking a model](#picking-a-model)
- [Tools and structured output](#tools-and-structured-output)
- [Sharing the host with the qualification mission](#sharing-the-host-with-the-qualification-mission)
- [Sharing one model between several agents](#sharing-one-model-between-several-agents)
- [Vision](#vision)
- [Embeddings, reranking and RAG](#embeddings-reranking-and-rag)
- [Code completion](#code-completion)
- [Images, editing and music](#images-editing-and-music)
- [Speech](#speech)
- [Screen grounding and computer use](#screen-grounding-and-computer-use)
- [Knowing what actually works](#knowing-what-actually-works)
- [Qualifying the stack](#qualifying-the-stack)
- [Checking the workspace](#checking-the-workspace)
- [Safety boundaries](#safety-boundaries)
- [Where things live](#where-things-live)

## What this is

A private, single-host, loopback-only local AI workspace for the `frankenstein`
X99 workstation: a llama.cpp router with on-demand model switching, four
capability sidecars, a ComfyUI media lane, a bounded RAG store, and a set of
gates whose whole job is to refuse to claim a capability that has not been
observed working.

What it is **not**: a hosted service, a multi-user deployment, or a place where
model weights live in Git. It also does not measure models — there is no
authorized comparative-performance result here. Functional gates do score specific correctness criteria; those scores are not throughput benchmarks. See
[no numeric model-characterization harness](reference/CAPABILITY-MATRIX.md#no-numeric-model-characterization-harness).

## Before you start

The host this is written for: Xeon E5-2696 v4 (22c/44t), ~94 GiB RAM, and three
AMD GPUs —

| Device | Card | VRAM | Role |
|---|---|---|---|
| `ROCm0` | RX 6900 XT | 16 GiB | headless, preferred compute |
| `ROCm1` | Radeon Pro V620 | 32 GiB | headless, preferred large-model and media |
| `ROCm2` | RX 6900 XT | 16 GiB | display; minimise compute |

Aggregate VRAM is about 61–64 GiB, but frameworks do not automatically aggregate
it. "It fits on disk" is never the test: KV cache and runtime overhead come out
of the same pool. Three 1 GiB HugeTLB pages are reserved for XMRig and must be
preserved. Compressed swap is `/dev/zram0` zstd 16G — zram, not zswap. Full
rationale: [ADR 0003](decisions/0003-hardware-allocation-and-memory-policy.md).

Software you need: a ROCm toolchain with `hipconfig`, CMake, Ninja, Python 3,
`curl`, and `bubblewrap` for the repository-agent gate. `aria2c` is optional and
only speeds downloads up.

## Set up the runtime

```bash
git clone git@github.com:TyphooN-/frankenstein-llm.git /home/typhoon/git/frankenstein-llm
cd /home/typhoon/git/frankenstein-llm
git submodule update --init --recursive upstream/llama.cpp
scripts/build-llama-cpp.sh
```

The submodule is pinned to llama.cpp v0.4.0 at commit `5266f24`, and the build
script refuses to run if the worktree is on any other revision, is dirty, or if
the lock names an unexpected repository or GPU target. It builds HIP/ROCm for
`gfx1030` with Ninja in Release, uses every logical CPU via `nproc`, and prints
`llama-server --version` when it finishes. Do not run it alongside another
optimized build.

Then install the user services — a deliberate step, not something cloning does:

```bash
install -Dm644 services/systemd/llama-router.service   ~/.config/systemd/user/llama-router.service
install -Dm644 services/systemd/llama-sidecar@.service ~/.config/systemd/user/llama-sidecar@.service
systemctl --user daemon-reload
systemctl --user start llama-router.service
```

The router listens on `127.0.0.1:8080` and nowhere else. Confirm it:

```bash
systemctl --user is-active llama-router.service
scripts/local-model-status.sh
```

If the unit fails with `status=203/EXEC`, it is executing a path that no longer
exists — see [troubleshooting](reference/TROUBLESHOOTING.md#router-will-not-start).

Full procedure, including which units have no `[Install]` section on purpose:
[operations → first-time setup](reference/OPERATIONS.md#first-time-setup).

## Get the weights

Nothing in `models/` is tracked. Two paths:

**One-off scripts** for the chat models, each with a publisher SHA-256 gate and
atomic promotion:

```bash
scripts/download-ridge.sh
scripts/download-uncensored-models.sh          # obliterated + heretic
scripts/download-writing-models.sh all         # fable + phr00ty
```

**The chained download queues** for everything else — 18 artifacts, 118 files,
about 219 GiB in four queues, pinned to repository revisions and exact sizes.
LFS weight files carry SHA-256 values; some ancillary files have no SHA-256 in the
queue. Do not describe those ancillary files as independently SHA-256 verified.

All four have already completed on this host. Run them on a fresh checkout, or to
re-verify: each revalidates finished files rather than refetching them.

```bash
systemctl --user start local-ai-model-downloads.service         # core capabilities: embeddings, reranker, OCR, ASR, TTS, music, image, FIM
systemctl --user start local-ai-model-downloads-phase2.service  # computer-use grounding: UI-TARS
systemctl --user start local-ai-model-downloads-phase3.service  # image editing: Qwen Image Edit
systemctl --user start local-ai-model-downloads-phase4.service  # researched candidates
```

Queues are named here by what they install. The unit, queue, state and stamp
files keep their original `phaseN` names because those names appear in completion
stamps that record what was actually downloaded; renaming them would rewrite
evidence. Each queue waits for its predecessor's completion stamp to match an
exact byte total. Transfers resume into `.partial` siblings, verify exact size and any
published SHA-256 recorded in the queue,
and promote only verified bytes with `os.replace`. Independent files run
concurrently; model loading and qualification stay serialized. Interrupting is
safe — the next run revalidates and continues. Details:
[operations → downloading weights](reference/OPERATIONS.md#downloading-weights).

## Everyday chat

Local chat, writing and coding through the router are usable as soon as the
router is up and a GGUF is on disk. You do **not** need to wait for the full
qualification mission: grounding, ComfyUI, TTS and WeMM retrieval are separate
lanes and none of them gates ordinary chat.

Cloud remains the Hermes default. To use a local model, pick it in the Desktop
model picker or type `/model <alias>` in a new chat:

```text
/model heretic
/model obliterated
/model ridge
/model fable
/model phr00ty
/model qwen3-coder-next
/model gemma4-heretic
```

The first reply after a switch waits for the GGUF to load, because the router
keeps at most one large model resident and selecting another evicts the previous
one. Later requests avoid that initial model-load step; response speed is not qualified here. Start a fresh chat when switching model
families: handing a local model a long history created under a different model's
hidden assumptions produces bad results for reasons that have nothing to do with
the model.

Desktop, TUI and CLI all read `~/.hermes/config.yaml`; there is no separate
Desktop model registry.

One-shot check from a shell:

```bash
hermes -z 'Reply with exactly pong.' --provider llamacpp-local -m heretic \
  --reasoning none --ignore-rules
```

Direct against the router, no Hermes involved:

```bash
python3 scripts/verify-router-models.py ridge heretic
```

## Picking a model

| Alias | Weight file it loads | Use it for | Trade-off |
|---|---|---|---|
| `heretic` | `RVN-Q6_K-multilingual-mtp.gguf` | recommended uncensored daily driver: private code review, offline debugging, general work | near-stock Qwen3.8 behaviour, still below frontier agents |
| `obliterated` | `Qwen3.8-27B-OBLITERATED-Q6_K.gguf` | when refusal and soft-deflection removal matters more than staying close to stock | publisher reports MMLU 82.3% vs 84.5% stock; STEM most affected |
| `ridge` | `Qwen3.8-27B-Ridge-3.7bpw.gguf` | compressed baseline option | quality and speed require separate qualification |
| `fable` | `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-Q6_K.gguf` | unrestricted fantasy writing where plot logic, continuity and instruction following all matter | Qwen3.6 rather than Qwen3.8; 65,536-token preset |
| `phr00ty` | `Phr00tyMix-v4-32B-imat-Q6_K.gguf` | voice, scene texture and prose-first roleplay | 65,536-token preset, no MTP, `temp 1.5` |
| `qwen3-coder-next` | `Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf` | local repository-agent and tool work | 80B-A3B; 46.77 GiB across four shards is the only preset that needs all three cards — the two RX 6900 XTs fill first and the remainder goes to the V620 |
| `gemma4-heretic` | `Gemma-4-12B-it-heretic-Q6_K.gguf` | multimodal *reading* | low-privilege by policy: never give it executable tools |

The alias is what you type; the weight file is what loads. They are not the same
string and, for `heretic`, not even the same name — the release is published as
Qwen3.8-27B-Heretic-Abliterated-Uncensored and the artifact on disk is
`RVN-Q6_K-multilingual-mtp.gguf`.

`obliterated-vision` and `gemma4-heretic-vision` are the projector-bearing
variants of their text presets: they load the same weight file plus
`Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf` and
`mmproj-Gemma-4-12B-it-BF16.gguf` respectively.
`python3 scripts/serve-model.py --list` prints this mapping for every preset.

Every alias, with its exact model file and runtime flags, is in
[configuration → router presets](reference/CONFIGURATION.md#router-presets).

**Abliteration is not fine-tuning.** It edits refusal behaviour; it does not add
knowledge, and it is not a safety property. Model willingness is never
authorization.

## Tools and structured output

The router speaks the OpenAI-compatible chat API, including native
`message.tool_calls[]`. A model that writes `<tool_call>{…}</tool_call>` into
`message.content` has only emitted tool-shaped text, not a usable API tool call, because
the transport never parses it — `gate_native_tool_use.py` fails that case
explicitly and also checks the second half of the loop, that a final answer
follows the tool result rather than another call.

Tool authorization is policy, not convention. Three tiers:

| Tier | Aliases | May receive tools |
|---|---|---|
| tool-using | `ridge`, `heretic`, `obliterated`, `obliterated-vision`, `fable`, `phr00ty`, `qwen3-coder-next` | yes |
| low | `gemma4-heretic`, `gemma4-heretic-vision` | never |
| read-only | `qwen3-embedding-8b`, `qwen3-reranker-8b`, `qwen25-coder-7b-fim` | never |

An alias the policy does not know is refused, not defaulted. Adding a preset
without a privilege decision fails the router gate — that is the intended
behaviour, not an obstacle.

## Sharing the host with the qualification mission

The mission is serialized and fail-closed, but it does not pause when you start
using the router, and it does not require the router to be stopped between steps.
Sharing the host is normal. What follows is how to read the risk, not a
guarantee that any particular moment is free.

Ask the status helper rather than a hand-written state read:

```bash
scripts/local-model-status.sh --host-sharing
```

It reports one of three verdicts, and none of them means "safe":

| Verdict | What it means |
|---|---|
| `mission-step-running` | The supervisor's last write says a step was executing. Treat it as potentially active until process state is checked. |
| `mission-may-take-the-gpu` | The last write records preparation for a step, **or** state/host inspection is unavailable. This is not proof of a live supervisor. |
| `mission-idle-per-last-write` | The last thing the supervisor durably wrote was a terminal status. Nothing is reserved by that. |

The command reads the router over loopback, the mission's state file and the
process table. It starts nothing, stops nothing and loads nothing, and it exits
`0` when it produces a report, regardless of verdict. **Do not** chain it into
`… && load-a-model`: that would launch the model even when conflicts are reported.
JSON conflict details are capped at eight samples per category; the accompanying
counts report the complete classified totals.

**Do not read `current_step` yourself.** It is the field that looks like the
answer and is not: it is set only after the supervisor's quiet-host wait has
already succeeded, it is never cleared when an individual step ends, and it does
not exist at all in a freshly initialised state. So it is empty exactly when the
supervisor is starting up and closest to claiming the GPU, and it goes on naming
a step for as long as the mission has nothing else to write. A missing,
truncated or unreadable state file means *unknown*, never *idle* — the file is
untracked runtime state and a supervisor killed outright never gets to correct
it. The mechanism is in
[operations → sharing the host](reference/OPERATIONS.md#sharing-the-host-with-the-mission).

**The status is advisory, not an admission ticket.** It describes the instant it
was read. The mission unit is `WantedBy=default.target`, so it can start — or
reach the end of its own quiet-host wait — in the gap between the check and your
next request. Nothing you can read reserves the GPU.

To use a local model alongside the mission:

1. Keep the router the only model owner. Do not run a second `llama-server`, a
   benchmark, a download, or a ComfyUI/TTS/grounding gate alongside it. A
   sidecar left resident by a failed gate counts as a second owner.
2. Start a fresh chat after switching families; do not hand a local model a long
   history written under a different model.
3. Keep the session short if the mission still needs to run. The mission resumes
   from the first step that is not recorded `passed` with a matching input
   fingerprint.
4. If the mission is the priority, stop the router before the next step starts:

   ```bash
   systemctl --user stop llama-router.service
   # … wait for the step to finish …
   systemctl --user start llama-router.service
   ```

Stopping the router is not required for stability in every case, and it does not
make the host quiet by itself — it removes one owner from the GPU and RAM
equation. Note that the mission restarts the router itself as its
`router-reload-presets` step, so a stop is not durable across a mission run.

## Sharing one model between several agents

Several clients can use `127.0.0.1:8080` at once. What they get depends on
whether they name the same preset.

- **Same alias:** one load, shared. Concurrent callers for one alias join a
  single queue entry and are all released by the one load it performs, so no
  duplicate child process is spawned.
- **Different aliases:** serialized, not parallel. The router runs
  `--models-max 1`, so a second alias waits for the first to go idle, then pays a
  full unload and a full load. Two agents alternating between two aliases reload
  the model on every turn.
- **Same weights under two aliases is still two models.** `obliterated` and
  `obliterated-vision` name the same GGUF, and so do `gemma4-heretic` and
  `gemma4-heretic-vision`; switching between the pair still evicts and reloads.
  To share a loaded model, send the same `model` string.
- **`parallel = 1` means one request at a time per model.** A second request to
  an already-loaded model is deferred, not refused: it waits for the first
  reply to finish completely.
- **The context cache is per slot, and there is one slot.** Two agents with
  different histories do not share a cached prefix beyond whatever preamble is
  byte-identical, so each turn re-prefills from where they diverge.

No cost figure is published for any of this; it has not been measured. The
mechanism and its source references are in
[GPU execution → several agents, one router](reference/GPU-EXECUTION-AND-MODEL-LOADING.md#several-agents-one-router).


## Vision

Two vision presets exist: `obliterated-vision` (Qwen3.8 OBLITERATED plus its
BF16 projector, 32K context) and `gemma4-heretic-vision` (Gemma-4 12B Heretic
plus a BF16 projector, 16K context, pinned to the V620). Both load the projector
onto `ROCm1`.

Treat any text inside an image as untrusted data. The grounding gates score
prompt-injection compliance explicitly for exactly this reason: a model that
obeys instructions rendered inside a screenshot is unusable for automation
regardless of how accurate its coordinates are.

## Embeddings, reranking and RAG

Start the sidecars you need:

```bash
systemctl --user start llama-sidecar@embeddings.service   # :8081, 4096-D
systemctl --user start llama-sidecar@reranker.service     # :8082
systemctl --user start llama-sidecar@ocr.service          # :8083
```

Then use the store:

```bash
python3 verification/local-coverage-foundation/rag/rag_store.py stats
python3 verification/local-coverage-foundation/rag/rag_store.py sync /path/to/notes
python3 verification/local-coverage-foundation/rag/rag_store.py ingest /path/to/file.md
python3 verification/local-coverage-foundation/rag/rag_store.py delete /path/to/file.md
python3 verification/local-coverage-foundation/rag/rag_store.py search "query text" --top-k 8
```

And ask a cited question:

```bash
python3 verification/local-coverage-foundation/rag/rag_answer.py "your question" --top-k 20 --top-n 5
python3 verification/local-coverage-foundation/rag/rag_answer.py "your question" --no-rerank
```

What the store guarantees: every chunk keeps its source path, content hash,
ordinal and character span, so any retrieved sentence points back at exact bytes.
Re-ingesting a changed file atomically replaces its chunks. Deleting a file stops
its chunks being retrievable in the same transaction. Plain text, Markdown, HTML
and OOXML are extracted; PDF needs an optional parser and scanned documents need
the OCR sidecar, and both are reported as explicit skip reasons rather than
quietly dropped files.

The answer path frames retrieved text as untrusted data. This is a prompt-level mitigation, not a guarantee that a model will ignore injected instructions; executable authority must remain separately constrained; every `[n]` the model emits is checked against the passages that were
actually in the context, and an invented citation is reported rather than
rendered as fact; an unreachable reranker degrades to dense order with a recorded
status instead of silently pretending it reranked.

WeMM multimodal embeddings are a **separate 2048-D space** in a separate database
under a separate alias. Policy refuses any configuration that would merge them
with the 4096-D text index — mixing two vector spaces in one store returns
plausible nonsense rather than an error.

## Code completion

The FIM sidecar serves llama.cpp's `/infill` endpoint on `:8084` using exactly
the artifact `llama-server --fim-qwen-7b-default` resolves to, so the
prefix/suffix/middle special tokens are known present rather than assumed. No
chat template is involved.

```bash
systemctl --user start llama-sidecar@fim.service
```

Keep this lane separate from `qwen3-coder-next`. Low-latency keystroke completion
and long-horizon agentic coding are different jobs, and the candidate policy
records `qwen25-coder-7b-fim` as a `must_retain` for exactly that reason.

## Images, editing and music

ComfyUI is not a managed service. It is launched by a gate runner for the
duration of a run, on `127.0.0.1:8188`, with only the two headless GPUs visible
and the 32 GiB V620 selected.

Before any heavy media work, give the router's VRAM back:

```bash
systemctl --user stop llama-router.service
# … media work …
systemctl --user start llama-router.service
```

The runners do this themselves and restore the router on exit, including on a
forwarded signal.

```bash
bash verification/generative-media/run_schema_serialized.sh      # node/model discovery, no generation
bash verification/generative-media/run_functional_serialized.sh  # real image, music and edit outputs
python3 verification/generative-media/preflight.py               # static; imports no torch, loads no model
```

The pinned lanes are Z-Image Turbo (text-to-image), ACE-Step 1.5 (music), Qwen
Image Edit 2511 with a Lightning LoRA (editing), and FLUX.2 Klein 4B (pinned but
not submitted by the functional gate). The preflight resolves every bare filename
in those graphs through the categories in `extra_model_paths.yaml`, so a
reference that resolves to nothing, to two files, or to an uninventoried file
fails before a live run rejects the graph.

Weights being present is not a working workflow. Every media workflow currently
reports `functionally_proven: false` until a live run says otherwise.

## Speech

ASR (Qwen3-ASR-1.7B) and TTS (Qwen3-TTS-12Hz-1.7B-Base) run in the isolated
`venvs/tts` environment, not through the router.

```bash
bash verification/generative-media/setup_tts.sh                       # once
python3 verification/local-coverage-foundation/validators/gate_asr.py
bash verification/tts-local/run_serialized.sh
```

TTS admission requires an ASR round trip, not merely an emitted audio file. The
gate clones a voice from a LibriSpeech clip the ASR gate already validated, then
synthesises sentences the reference never contained, then feeds the waveform back
through the admitted ASR model and compares the transcript to the requested text.
Silence, noise and a truncated tail all fail that — which is exactly what a
duration or RMS check misses.

## Screen grounding and computer use

Grounding models are downloaded and the shared scoring contract is written; there
is **no end-to-end computer-control service**, and that gap is a service and
scaffold decision rather than a missing model.

```bash
# Host check only: loads nothing, touches no GPU memory
python3 verification/computer-use-grounding/gate_computer_use.py --preflight-only --max-load 6.0

# The full gate: ~15.5 GiB across three GPUs
bash verification/computer-use-grounding/run_when_idle.sh
```

Actions are applied to a pure-Python model of the same fixture pixels the model
saw. Nothing synthesises input or touches a display server, and the real desktop
is never driven.

Two standing rules: a GUI actor is **never** abliterated — ablation is refused
structurally for anything that acts on the host — and desktop-control authority
requires a passing grounding verdict, where "no verdict yet" is refused exactly
as a failure is.

## Knowing what actually works

Four states, and they are not interchangeable:

**researched** → **downloaded** → **policy-admitted** → **functionally qualified**

Downloaded means pinned bytes arrived intact. Policy-admitted means a
non-inference gate checked inventory, privileges, index separation and reviewed
digests. Only *functionally qualified* means something loaded, behaved and
unloaded. A passing policy gate is not a functional verdict.

Get the current machine reading — it loads no model and touches no GPU:

```bash
python3 verification/local-coverage-foundation/build_capability_ledger.py --print-only
```

On a fresh clone every capability reads `downloaded` or `researched`, because
gate evidence is untracked runtime state.

Read [the capability matrix](reference/CAPABILITY-MATRIX.md) for what each
capability is backed by, what would decide it, and what is genuinely absent —
including the missing model-characterization harness, which is the reason no
document here contains a locally measured model number.

## Qualifying the stack

```bash
systemctl --user start local-ai-functional-mission.service
tail -f verification/mission-supervisor/mission.log
```

The mission is serialized, resumable and fail-closed. It waits for all four
download stamps, then for a quiet host, then runs twelve steps starting with the
non-inference candidate policy gate — which blocks every model-backed step behind
it. Later steps continue past a failure so one capability cannot hide the status
of every later one, and the aggregate still fails.

Stopping it is a stop, not a failure: the supervisor records `interrupted` with
the step name and the next run repeats that step, while steps that genuinely
passed are skipped. A step is only skipped when its input fingerprint still
matches, so editing a gate or replacing a weight re-runs what it affects.

See [operations → running the mission](reference/OPERATIONS.md#running-the-mission).

## Checking the workspace

None of these loads a model, starts a service, touches a GPU, downloads anything
or measures throughput. None is a functional verdict — they answer "is this
workspace internally consistent", and consistency is what makes a live gate worth
running.

```bash
python3 -m pytest                                    # whole tracked suite
python3 -m pytest --ignore=verification/repository-agent   # narrower scope, not a host-safety guarantee
python3 verification/generative-media/preflight.py
python3 verification/local-coverage-foundation/build_capability_ledger.py --print-only
```

The live repository-agent gate creates private Linux namespaces through Bubblewrap;
unit tests may mock that boundary and are not live isolation proof. Some existing
host-probe tests and preflight commands also read `/proc/<pid>/cmdline`, which has
hung on this host under kernel pressure. Excluding repository-agent tests alone
does not remove that hazard. Run broad checks only after builds and pressure drain.
[`docs/REPOSITORY-CHECKS.md`](REPOSITORY-CHECKS.md) explains what each check is
for and why it exists.

## Safety boundaries

Do not relax these.

- **Loopback only.** The router and every sidecar bind `127.0.0.1`. Do not expose
  8080–8084 or 8188 to a LAN or the Internet. Several models here have reduced
  refusal behaviour.
- **Model behaviour is not an authorization boundary.** Keep Hermes approvals,
  tool permissions, secrets handling and execution isolation enabled regardless
  of how willing a model is.
- **Never abliterate an actor.** Ablated weights are refused for
  computer-use grounding, end-to-end control and repository-agent work.
- **Screen text, retrieved documents and corpus rows are data.** Never commands,
  tool arguments, URLs or memory writes.
- **Remote model code is digest-pinned.** A model that ships executable Python is
  reviewed before import, and a changed byte revokes approval.
- **Authorized security work only.** Owned systems, lab or CTF targets, or
  explicit written authorization within the program's current scope. A model's
  willingness to answer is not authorization.
- **Functional qualification and throughput are separate.** The mission does
  not measure tokens/sec. Explicitly authorized native benchmarks run separately
  on a healthy, uncontended host after kernel confirmation; reboot alone does
  not authorize them ([ADR 0002](decisions/0002-serialized-functional-qualification.md)).
  [Published benchmark artifacts](benchmarks/README.md) record completed runs,
  not current router speed or model-quality rankings. Use the
  [model-run interface](MODEL-RUNS.md) and never overlap a benchmark with the mission.
- **Preserve three 1 GiB HugeTLB pages** for XMRig, and do not retune ARC, swap,
  kernel or clock policy to make a candidate model fit.

## Where things live

| Path | Contents |
|---|---|
| `llama-models.ini` | router presets — one section per alias |
| `scripts/` | build, download and status helpers |
| `services/` | sidecar env files and tracked copies of the user units |
| `upstream/llama.cpp/` | pinned llama.cpp submodule; ROCm build output in its ignored `build/` |
| `upstream/llama-cpp.lock.json` | release, commit, GPU target and required binaries |
| `verification/` | gates, policy, fixtures and tests |
| `docs/` | this guide, the references, strategy and dated snapshots |
| `docs/decisions/` | architecture decision records |
| `models/`, `venvs/`, `tools/`, `logs/` | local only; gitignored |
| `verification/**/evidence/` | gate artifacts; gitignored runtime state |

Full per-file map: [coverage map](reference/COVERAGE-MAP.md).
