# Operations runbook

> Operator-interface update: [Model runs](../MODEL-RUNS.md) is the current reference
> for standalone serving, qualification and native benchmark scripts. The former
> standalone Ridge environment overrides are replaced by shared presets/configs.
> Any older statement below about absent benchmark tooling predates this interface;
> native throughput tooling exists, but comparative agent-quality evaluation remains absent.

Undated procedures. Runtime install state changes; this file documents what to
do, not what is currently running. To find out what is running, run the commands
in [checking the backend](#checking-the-backend) — do not read a status sentence
in a document as live truth.

Related: [user guide](../USER-GUIDE.md) · [configuration](CONFIGURATION.md) ·
[troubleshooting](TROUBLESHOOTING.md)

## Contents

- [First-time setup](#first-time-setup)
- [Installing the user units](#installing-the-user-units)
- [Checking the backend](#checking-the-backend)
- [Service control](#service-control)
- [Sidecars](#sidecars)
- [GPU handoff](#gpu-handoff)
- [Downloading weights](#downloading-weights)
- [Running the mission](#running-the-mission)
- [Running one gate](#running-one-gate)
- [Repository checks](#repository-checks)
- [Upgrading llamacpp](#upgrading-llamacpp)
- [Rollback](#rollback)
- [Adding or removing a model](#adding-or-removing-a-model)
- [Reclaiming disk](#reclaiming-disk)
- [Exit codes](#exit-codes)

## First-time setup

Prerequisites on the host: a ROCm toolchain with `hipconfig` on `PATH`, CMake,
Ninja, Python 3, `curl`, and — for the repository-agent gate — `bubblewrap`.
`aria2c` is optional and only accelerates downloads.

```bash
git clone git@github.com:TyphooN-/frankenstein-llm.git /home/typhoon/git/frankenstein-llm
cd /home/typhoon/git/frankenstein-llm
git submodule update --init --recursive upstream/llama.cpp
scripts/build-llama-cpp.sh
```

The build script derives its job count from `nproc`, so it will use every logical
CPU. Run it when no other optimized build is active — see
[ADR 0002](../decisions/0002-serialized-functional-qualification.md). It refuses
to start when the submodule `HEAD` is not the locked commit, when the submodule
worktree is dirty, when the lock names an unexpected repository or GPU target, or
when `nproc` does not return a positive integer, and it ends by printing
`llama-server --version`.

Weights are not in the clone. See [downloading weights](#downloading-weights).

## Installing the user units

The tracked units under `services/systemd/` are copies. Installing them is a
deliberate step, and it must not be done while a build is running.

```bash
install -Dm644 services/systemd/llama-router.service   ~/.config/systemd/user/llama-router.service
install -Dm644 services/systemd/llama-sidecar@.service ~/.config/systemd/user/llama-sidecar@.service
systemctl --user daemon-reload
systemctl --user start llama-router.service
```

Install the other units the same way when you need them. Two have no `[Install]`
section on purpose and are always started by hand:
`local-ai-computer-use-gate.service` (it places ~15.5 GiB across all three GPUs)
and the three gate units it chains to.

After installing, confirm the unit is executing the submodule build rather than a
path that no longer exists:

```bash
systemctl --user cat llama-router.service | grep ExecStart
```

It must name `upstream/llama.cpp/build/bin/llama-server`. A unit still pointing at
`~/.local/bin/llama-server` or `/home/typhoon/src` fails with `status=203/EXEC`
and restarts on a five-second timer.

## Checking the backend

```bash
systemctl --user is-active llama-router.service
scripts/local-model-status.sh
scripts/local-model-status.sh --json     # bounded machine-readable summary
```

`local-model-status.sh` execs `scripts/local_model_status.py`. It accepts
`--json` and `--timeout SECONDS` (default 5, maximum 30), reads only `/health`
and `/models` over loopback, caps the response at 2 MiB, never loads a model, and
never prints router launch arguments. It exits 1 with `local router unavailable:`
on stderr when the router is not answering.

Each router model id is an alias, so the summary resolves it against
`llama-models.ini` and `config/model-catalog.json` and leads each entry with the
weight filename and use case, putting the state and the handle on the line
beneath (`description` in `--json`, plus `loaded_descriptions`).
An id this checkout does not configure is reported as `not a configured preset`
rather than shown bare; unreadable local configuration degrades the description
but never fails the status read, which describes a live service rather than
gating configuration.

It classifies each model by llama.cpp's own six-state vocabulary rather than
guessing. `loading`, `loaded` and `sleeping` occupy the single resident slot;
`downloading`, `downloaded` and `unloaded` do not — `downloaded` only means
weights reached local disk. A state this pin does not define is an error rather
than a guess.

```bash
systemctl --user status llama-router.service
journalctl --user -u llama-router.service -f
```

## Service control

```bash
systemctl --user restart llama-router.service   # reload llama-models.ini
systemctl --user stop    llama-router.service
systemctl --user start   llama-router.service
```

A preset edit takes effect on restart. If the router refuses to come back after
an edit, suspect an unknown preset key: llama.cpp treats one as fatal and the
whole router fails to start. Run the pin suite, which checks every key against
the installed build's `--help`:

```bash
python3 -m pytest verification/upstream-pin/test_llama_cpp_pin.py
```

## Sidecars

```bash
systemctl --user start llama-sidecar@embeddings.service
systemctl --user start llama-sidecar@reranker.service
systemctl --user start llama-sidecar@ocr.service
systemctl --user start llama-sidecar@fim.service
systemctl --user stop  llama-sidecar@embeddings.service
```

Sidecars are independent of the router by design, so restarting the router does
not cycle them, and a sidecar left resident by a failed gate is exactly what can
keep the mission's quiet-host wait from ever clearing. Check for stragglers with
`systemctl --user list-units 'llama-sidecar@*'`.

## GPU handoff

The router's resident model uses all three GPUs. Before heavy ComfyUI work, give
the VRAM back:

```bash
systemctl --user stop llama-router.service
# … ComfyUI / media gate …
systemctl --user start llama-router.service
```

The media runners do this themselves, including restoring the router on exit and
on a forwarded signal, and they assert afterwards that the V620 released its
residency within a 768 MiB tolerance.

## Downloading weights

### One-off scripts

Small, self-contained, publisher-hash-gated:

```bash
scripts/download-ridge.sh                    # LLM_MODELS_DIR, RIDGE_FILE, RIDGE_MMPROJ=1
scripts/download-uncensored-models.sh        # obliterated + heretic
scripts/download-writing-models.sh [all|fable|phr00ty]
scripts/download-heretic-clean.sh            # range resume, full SHA-256, atomic promote
scripts/redownload-heretic-after-crash.sh    # same, and clears a bad .partial
```

All of them verify a publisher SHA-256 before renaming into place.
`download-writing-models.sh` additionally holds a `flock` on the partial.

### The chained download queues

The four queues are the supported path for everything else, one systemd unit
each, each waiting for its predecessors' stamps. They are listed by content;
their unit and file names retain the original `phaseN` spelling because the
completion stamps that record what was downloaded use it, and renaming them would
rewrite evidence rather than clarify it.

| Queue unit | Installs |
|---|---|
| `local-ai-model-downloads.service` | core capabilities: embeddings, reranker, OCR, ASR, TTS, music, image, FIM |
| `local-ai-model-downloads-phase2.service` | computer-use grounding: UI-TARS |
| `local-ai-model-downloads-phase3.service` | image editing: Qwen Image Edit set |
| `local-ai-model-downloads-phase4.service` | researched candidates: Qwen3-Coder-Next, Gemma-4 Heretic, UI-Mate, WeMM, FLUX.2 Klein |

All four completed on this host; starting one revalidates rather than refetches.

```bash
systemctl --user start local-ai-model-downloads.service
systemctl --user start local-ai-model-downloads-phase2.service
systemctl --user start local-ai-model-downloads-phase3.service
systemctl --user start local-ai-model-downloads-phase4.service

journalctl --user -u local-ai-model-downloads.service -f
```

To run one by hand, set the environment rather than passing arguments — the
program takes none:

```bash
HERMES_DOWNLOAD_QUEUE=$PWD/verification/local-coverage-foundation/download-queue-phase3.json \
HERMES_DOWNLOAD_STATE=$PWD/verification/local-coverage-foundation/download-state-phase3.json \
HERMES_DOWNLOAD_LOCK=$PWD/verification/local-coverage-foundation/download-queue-phase3.lock \
HERMES_DOWNLOAD_STAMP=$PWD/verification/local-coverage-foundation/downloads-phase3-complete.ok \
HERMES_DOWNLOAD_LOG=$PWD/verification/local-coverage-foundation/downloads-phase3.log \
python3 verification/local-coverage-foundation/download_queue.py
```

Interrupting is safe. The queue resumes into `.partial` siblings, promotes only
after an exact size and digest match, and revalidates finished files on the next
run. Progress is in `downloads*.log` and `download-state*.json`.

### Read-only reconciliation

```bash
python3 verification/computer-use-grounding/reconcile_downloads.py   # stat only; safe on a busy host
python3 verification/computer-use-grounding/reconcile_full.py        # re-hashes every declared digest
```

`reconcile_full.py` also enumerates quarantined `.bad-*` and stale `.partial`
files with byte accounting, and cross-checks that on-disk model trees contain no
unaccounted large files. Neither script downloads, moves or deletes anything; a
discrepancy is surfaced, never silently repaired.

## Running the mission

```bash
systemctl --user start local-ai-functional-mission.service
tail -f verification/mission-supervisor/mission.log
```

The supervisor waits, without a deadline, for all four download stamps. It then
waits for a quiet host — two consecutive polls with `MemAvailable ≥ 32 GiB`, load
≤ 6.0 and no conflicting build, transfer or unmanaged model server — bounded by
`HERMES_MISSION_QUIET_TIMEOUT` (default six hours). Its per-step logs are
`verification/mission-supervisor/<step>.log`.

To read progress, use the durable state:

```bash
python3 -c "import json;s=json.load(open('verification/mission-supervisor/mission-state.json'));\
print(s['status']);[print(f\"{k:28s} {v['status']}\") for k,v in sorted(s['steps'].items())]"
```

Stopping is a stop, not a failure: `systemctl --user stop` forwards `SIGTERM`,
the supervisor records `interrupted` with the step name, and the next run repeats
that step. A step that had already exited 0 is kept, so hours of GPU work are not
repeated.

Re-running skips steps recorded `passed` whose `input_fingerprint` still matches.
Editing a gate, replacing a weight or reordering the step list changes the
fingerprint and re-runs the affected steps.

Setting a shorter quiet timeout for an experiment:

```bash
systemctl --user set-environment HERMES_MISSION_QUIET_TIMEOUT=900
```

An unparseable or non-positive value raises rather than silently defaulting.

## Running one gate

Every gate can run outside the mission. Each publishes its own evidence artifact
and each is fail-closed on its own terms.

```bash
# Non-inference policy preflight
python3 verification/candidate-qualification/gate_candidate_policy.py

# Router presets (needs the router)
python3 verification/router-functional/gate_router_models.py

# Sidecar-backed capability gates (each manages its own sidecar)
python3 verification/local-coverage-foundation/validators/gate_embeddings.py
python3 verification/local-coverage-foundation/validators/gate_reranker.py
python3 verification/local-coverage-foundation/validators/gate_ocr.py
python3 verification/local-coverage-foundation/validators/gate_fim.py
python3 verification/local-coverage-foundation/validators/gate_asr.py
python3 verification/local-coverage-foundation/validators/gate_native_tool_use.py
python3 verification/local-coverage-foundation/validators/gate_vision_grounding.py

# RAG
python3 verification/local-coverage-foundation/validators/gate_rag.py --structural   # offline, no GPU
python3 verification/local-coverage-foundation/validators/gate_rag.py --live
python3 verification/local-coverage-foundation/validators/gate_rag_behavioral.py     # offline, scripted answerer

# Heavy, serialized lanes — prefer their runners, which handle the router handoff
bash verification/computer-use-grounding/run_when_idle.sh
bash verification/tts-local/run_serialized.sh
bash verification/repository-agent/run_serialized.sh
bash verification/generative-media/run_schema_serialized.sh
bash verification/generative-media/run_functional_serialized.sh
```

Useful narrower entry points:

```bash
# Host check only: no model is loaded and no GPU memory is touched
python3 verification/computer-use-grounding/gate_computer_use.py --preflight-only --max-load 6.0

# One repository-agent lane; --model accepts only heretic or qwen3-coder-next
python3 verification/repository-agent/gate_repo_agent.py --model qwen3-coder-next

# Static media preflight: does not import torch and does not load a model
python3 verification/generative-media/preflight.py

# GGUF header metadata without putting the model on a GPU
python3 verification/local-coverage-foundation/scripts/inspect_gguf.py <file.gguf>
```

The four serialized runners each begin by re-running the computer-use preflight
with `--max-load 6.0` and exit **75** if the host is no longer idle, so a gate
started into a newly busy machine refuses rather than producing a confounded
result.

### Isolated runtimes

Some gates need their own Python environment, all ignored by Git:

```bash
bash verification/candidate-qualification/setup_runtime.sh   # venvs/candidates (uv, hash-locked, ROCm Torch)
bash verification/generative-media/setup_comfy.sh            # venvs/comfy + tools/ComfyUI
bash verification/generative-media/setup_tts.sh              # venvs/tts
```

`setup_runtime.sh` recreates the venv when its Python is not 3.14, syncs the
hash-locked requirements, and then asserts exact package versions and that the
reused host Torch is a ROCm build — it refuses a non-ROCm Torch rather than
installing NVIDIA CUDA packages beside the AMD runtime. Both ComfyUI and TTS
setups reuse the system ROCm Torch through `--system-site-packages` and never
upgrade it.

## Repository checks

Offline, no model, no service, no GPU, no download, no throughput. See
[`docs/REPOSITORY-CHECKS.md`](../REPOSITORY-CHECKS.md) for what each one is for.

```bash
python3 -m pytest                                    # whole tracked suite
python3 -m pytest verification/upstream-pin/         # the llama.cpp pin
python3 -m pytest verification/mission-supervisor/   # supervisor contracts
python3 -m pytest verification/docs/                 # documentation map and links
python3 verification/generative-media/preflight.py   # static media preflight
python3 verification/local-coverage-foundation/build_capability_ledger.py --print-only
```

The live repository-agent gate creates private Linux namespaces through Bubblewrap; unit tests may mock subprocess boundaries. Broad tests can still invoke host probes reading /proc/<pid>/cmdline, a known hang hazard under kernel pressure.
Run the complete suite only after builds and other host pressure have drained; a
busy or RCU-stalled host is not a valid sandbox test environment. To run
a narrower set (this does not guarantee safety under host pressure):

```bash
python3 -m pytest --ignore=verification/repository-agent
```

`pytest.ini` scopes collection to `verification/` and excludes
`verification/repository-agent/fixture/`, whose three failing tests are the
*oracle* for the repository-agent gate and are defective on purpose.

## Upgrading llama.cpp

Per [ADR 0005](../decisions/0005-track-llama-cpp-submodule.md). Do not skip
step 4: staging picks up whatever the worktree is on, so a fetch left checked out
on `master` pins `master` for every later clone while every other record still
reads the release.

1. Review the upstream release and resolve its tag to an exact commit.
2. In the submodule, fetch tags and check out that commit detached:
   ```bash
   git -C upstream/llama.cpp fetch --tags origin
   git -C upstream/llama.cpp checkout --detach <commit>
   ```
3. Update `upstream/llama-cpp.lock.json` to the same tag and commit.
4. Stage the submodule and confirm the gitlink agrees:
   ```bash
   git add upstream/llama.cpp
   git ls-files -s upstream/llama.cpp
   python3 -m pytest verification/upstream-pin/test_llama_cpp_pin.py
   ```
5. Run `scripts/build-llama-cpp.sh`, only when no other optimized build is active.
6. Run functional qualification. No token-rate or latency measurement.
7. Commit the outer gitlink, lock, service paths, documentation and verification
   together.

A version update is not accepted until binary identity, required CLI flags,
router startup, model discovery, deterministic text/structured/tool behaviour,
relevant Gemma-4 vision behaviour and unload paths all pass.

Restoring a checkout at any commit:

```bash
git submodule update --init --recursive upstream/llama.cpp
```

## Rollback

**llama.cpp.** Check out the prior outer-repository commit, run
`git submodule update --init --recursive`, rebuild the pinned prior gitlink with
`scripts/build-llama-cpp.sh`, then reinstall the tracked user units. Do not keep
an untracked second production checkout as a rollback mechanism — that is the
failure ADR 0005 exists to prevent.

**Router.** `llama-ridge.service` runs `scripts/serve-ridge.sh` as a single-model
fallback. It must never run at the same time as `llama-router.service`: both bind
port 8080 and own the same GPUs.

```bash
systemctl --user stop  llama-router.service
systemctl --user start llama-ridge.service
```

Note it uses the historical `--tensor-split 1,2,1`, not the placement the router preset carries (`1,0,1`).

**A preset change.** Presets are plain text in `llama-models.ini`; revert the
file and restart the router. Nothing caches a preset elsewhere.

**A model.** Weights are untracked, so rollback is deleting the file and
re-running the queue or script that installs it; the queue revalidates size and
digest before promotion.

**Evidence.** Gate artifacts are ignored runtime state. Deleting one does not
roll a capability back to "qualified" — the ledger reads an absent artifact as no
verdict.

## Adding or removing a model

See [developer guide → adding a capability](DEVELOPER-GUIDE.md#adding-a-capability)
for the full path. The short operational version:

1. Collect exact metadata (`research/collect_hf_metadata.py`) and build a queue
   entry from it. Never hand-type a size or digest.
2. Update the queue's `total_bytes` **and** every copy of it — the phase runner
   and the mission supervisor — then run `test_queue_manifests.py`.
3. Download through the queue.
4. Add the router preset or sidecar env file.
5. Give it a privilege tier in `candidate_policy`, or the router gate will fail
   on an uncovered preset.
6. Declare its evidence artifact in `build_capability_ledger.CAPABILITY_EVIDENCE`
   and `EXPECTED_EVIDENCE_GATES`.
7. Qualify it. Downloaded is not admitted.

Removing is the same list in reverse. Leaving a preset in the INI whose file is
gone means the alias fails at load time, not at startup.

## Reclaiming disk

`reconcile_full.py` reports quarantined `.bad-*` files and stale `.partial`
files as reclaimable, with byte accounting, and never removes them — that is an
operator decision. Review the list before deleting; a `.bad-*` file is the
evidence of a failed transfer.

After an unclean power loss, do not trust a model because its byte size is
complete. Re-run the publisher SHA-256 check before loading it. ZFS has
previously rejected a full-size replacement with `Input/output error` and a
different full-size pre-crash copy failed its digest.

```bash
sudo zpool status -v zroot     # privileged listing of affected paths
```

Do not run `zpool clear` until the scrub result and affected paths have been
reviewed. Model artifacts are replaceable; unrelated user data needs a restore
decision rather than blind deletion.

## Exit codes

| Code | Meaning | Where |
|---|---|---|
| 0 | pass / complete | all |
| 1 | failure with a verdict | gates, downloader, ledger (`problems` non-empty) |
| 2 | bad usage, or a refused precondition | `download_queue.py`; `build-llama-cpp.sh` lock/worktree refusals |
| 3 | another instance already running | `gate_computer_use.py`; `post_reboot_gate.py` uses 3 for "not ready" |
| 4 | interrupted by a handled signal | `gate_computer_use.py` |
| 5 | ran but produced no usable artifact | `run_when_idle.sh` (`EXIT_NO_ARTIFACT`) |
| 75 | another writer holds the lock, or the host is no longer idle | `download_queue.py`, mission supervisor, serialized runners |
| 128+N | terminated by signal N | mission supervisor |

`local-ai-computer-use-gate.service` excludes 1, 3 and 5 from `Restart=`
precisely because retrying them cannot change the outcome; exit 4 and death by
signal do restart, and an operator `stop` never triggers a restart.
