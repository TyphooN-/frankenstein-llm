# Troubleshooting

Symptom → likely cause → what to do. Every diagnostic here is read-only unless it
says otherwise.

Related: [operations](OPERATIONS.md) · [configuration](CONFIGURATION.md) ·
[capability matrix](CAPABILITY-MATRIX.md)

## Contents

- [Quick triage](#quick-triage)
- [Router will not start](#router-will-not-start)
- [Router starts but a model will not load](#router-starts-but-a-model-will-not-load)
- [A card is missing and the ROCm indices moved](#a-card-is-missing-and-the-rocm-indices-moved)
- [Model loads but never answers](#model-loads-but-never-answers)
- [Model answers but never calls a tool](#model-answers-but-never-calls-a-tool)
- [Hermes cannot see local models](#hermes-cannot-see-local-models)
- [Build failures](#build-failures)
- [Download problems](#download-problems)
- [Qualification never starts](#qualification-never-starts)
- [Stopping the qualification unit times out](#stopping-the-qualification-unit-times-out)
- [Every boot repeats the whole qualification](#every-boot-repeats-the-whole-qualification)
- [A gate's receipt is invalidated every boot](#a-gates-receipt-is-invalidated-every-boot)
- [A refused router qualification revokes a pass](#a-refused-router-qualification-revokes-a-pass)
- [The qualification dies reading its own inputs](#the-qualification-dies-reading-its-own-inputs)
- [The policy gate blocks every gate over a file that is still arriving](#the-policy-gate-blocks-every-gate-over-a-file-that-is-still-arriving)
- [A gate reports VRAM still held after unload](#a-gate-reports-vram-still-held-after-unload)
- [One preset fails only its tool-call check](#one-preset-fails-only-its-tool-call-check)
- [Gate problems](#gate-problems)
- [A failed process outranks a passing artifact](#a-failed-process-outranks-a-passing-artifact)
- [A gate fails while the kernel is faulting](#a-gate-fails-while-the-kernel-is-faulting)
- [Ledger says something unexpected](#ledger-says-something-unexpected)
- [Test suite problems](#test-suite-problems)
- [Storage and power loss](#storage-and-power-loss)

## Quick triage

```bash
systemctl --user is-active llama-router.service
scripts/local-model-status.sh
systemctl --user status llama-router.service
journalctl --user -u llama-router.service -n 100 --no-pager
python3 -m pytest --ignore=verification/repository-agent -q
```

The last one answers "is this workspace internally consistent". It is never a
functional verdict — consistency is what makes a live gate worth running.

## Router will not start

| Symptom | Cause | Action |
|---|---|---|
| `status=203/EXEC`, restart every 5 s | The installed unit execs a path that no longer exists — typically a `~/.local/bin` shim whose target went away with an external checkout | Reinstall the tracked unit, which execs `upstream/llama.cpp/build/bin/llama-server`. See [operations](OPERATIONS.md#installing-the-user-units) |
| Exits immediately after a preset edit | An unknown key in `llama-models.ini`. llama.cpp makes unknown preset keys fatal, so one bad key stops the whole router | `python3 -m pytest verification/upstream-pin/test_llama_cpp_pin.py`; the preset-compatibility test names the offending key |
| `Address already in use` | `llama-ridge.service` or a hand-started `llama-server` already owns 8080 | `systemctl --user stop llama-ridge.service`; they must never run together |
| Binary missing | The submodule was never built, or `build/` was cleaned | `git submodule update --init --recursive upstream/llama.cpp && scripts/build-llama-cpp.sh` |

The preset test and the version check skip rather than fail when the submodule
has not been built, so a green pin suite does not by itself mean a binary exists.
Check directly:

```bash
upstream/llama.cpp/build/bin/llama-server --version
```

## Router starts but a model will not load

| Symptom | Cause | Action |
|---|---|---|
| First request after a switch hangs for a long time | Expected. `--models-max 1` evicts the previous model and loads the requested GGUF; later requests avoid the initial model-load step | Wait; do not restart mid-load |
| One alias fails, others work | The GGUF path in that preset is wrong or the file is gone. Weights are untracked, so a fresh clone has none | Check the `model =` path; re-run the download that installs it |
| Load fails on a 16 GiB card | A `tensor-split` that does not reflect the model's size, its KV cache at the configured context, or a projector pinned by `mmproj-device` | Run `python3 scripts/gpu_placement.py` — it sizes every preset and says which device is over budget. Change `config/gpu-placement.json` and regenerate rather than editing the INI by hand |
| Odd sampling behaviour on Qwen3.8 | `split-mode = tensor` was added. Qwen3.8 MTP backend sampling is incompatible with that path | Remove it |
| Vision alias returns text-only behaviour | The `mmproj` file is missing or the projector device is wrong | Check `mmproj` and `mmproj-device` in the preset |

## A card is missing and the ROCm indices moved

| Symptom | Cause | Action |
|---|---|---|
| `scripts/gpu_vram.py` lists two devices where it listed three | A card did not enumerate on this boot | Reboot. Confirm with `lspci -nn \| grep -Ei 'vga\|display'` and `journalctl -b -k \| grep 'initializing kernel modesetting'`, which names every card the driver bound |
| The display card answers to a lower ROCm index than it used to | A ROCm index is a position in a list. Drop a card out of the middle and every card behind it moves up one | Do not retune `tensor-split` or `config/gpu-placement.json` to the short topology. Restore the card |
| `gpu_placement.py` reports `DOES NOT FIT` for presets that fit yesterday | The same thing, measured: budgets are read from the cards actually present | Compare the device list at the top of its own output against the three the host is supposed to have |

Observed on 2026-09-06: the V620 at `0000:07:00.0` was absent from one boot, and
the second RX 6900 XT moved from `0000:0a:00.0` to `0000:08:00.0` as PCI
renumbering closed the gap behind it. A later reboot brought all three back at
their usual addresses. Nothing was reconfigured in between.

The tools report this rather than absorb it, which is the intended behaviour and
not a second fault to chase: `gpu_placement.py` recomputes against whatever is
present and says which presets no longer fit,
`configured_split` reads `None` when the INI's split has more positions than the
host has devices, and the comparison test in
`verification/gpu-placement/test_gpu_placement.py` skips with `needs the
three-GPU host` instead of comparing against a topology nobody planned for. A
positional `--tensor-split` is only meaningful against the device list it was
sized for.

## Model loads but never answers

`local-model-status.sh` shows the model `loaded`, VRAM is allocated on every
card, and the first request never returns.

Check what the server process is actually waiting on:

```bash
pid=$(pgrep -n llama-server)
grep ^State /proc/$pid/status
cat /proc/$pid/wchan; echo
cat /proc/pressure/memory /proc/pressure/io
```

`State: D` together with `amdgpu_amdkfd_gpuvm_map_memory_to_gpu` means the ROCm
KFD mapping call has not returned. Observed on this host while a 13 GB `ld.lld`
held memory: one thread parked in that call, 48 threads idle, generation never
starting, memory pressure `some avg10≈18` and I/O pressure `some avg10≈34`.

**Wait for the competing build to finish.** A `D`-state thread does not take
SIGTERM, so killing the process does not release it any sooner, and restarting
the router or the driver to force it is the wrong response — it risks the GPU
state rather than fixing it. This is the same contention the qualification gates
refuse to run through; see
[GPU execution and model loading](GPU-EXECUTION-AND-MODEL-LOADING.md#a-stall-this-host-can-produce).

## Model answers but never calls a tool

The preset returns `PONG`, returns valid JSON for a schema request, and returns
an empty `tool_calls` array for a request that carries `tools`. The content is
coherent, so it looks like a model that understood the tools and declined.

**Check the template before you suspect the weights or the sampler.** llama.cpp
renders the tools payload through the chat template stored in the GGUF. If that
template has no `tools` branch, the function signatures are dropped before the
model is ever prompted, and no sampler setting and no amount of instruction will
produce a call. Ask the server what it actually loaded:

```bash
curl -s 'http://127.0.0.1:8080/props?model=<alias>' |
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("chat_template") or "")); print(d.get("chat_template_caps"))'
```

`supports_tools: false` is the answer. Read the template out of the weights
themselves to confirm it is the file and not the server:

```bash
PYTHONPATH=upstream/llama.cpp/gguf-py python3 -c '
from gguf import GGUFReader
f = GGUFReader("models/<file>.gguf", "r").fields["tokenizer.chat_template"]
t = str(bytes(f.parts[f.data[0]]), "utf-8"); print(len(t)); print(t)'
```

This is a real defect in published community merges, not a rare one. Two of this
catalog's presets shipped with it, and both were serving with it:

| Preset | Shipped template | What it did |
|---|---|---|
| `obliterated` | 506 chars | Looped messages with no `tools` branch and no `tool` role branch, so tool definitions and tool results were both discarded |
| `phr00ty` | 88 chars | Rendered `messages[0]` only and dropped every later turn, never emitted `<\|im_end\|>` although this model's EOS *is* `<\|im_end\|>`, and ignored `add_generation_prompt` |

Both models were tool-trained: the `<tool_call>`, `</tool_call>`,
`<tool_response>` and `</tool_response>` tokens are in their vocabularies. The
fix is `chat-template-file` in the preset, pointing at a template taken from
weights of the same architecture rather than one written by hand — see
[configuration](CONFIGURATION.md#chat-templates). Do not reach for `temp` first:
`phr00ty` serves at `temp = 1.5` and emits a correct native call at that setting
once the template can express one.

## Hermes cannot see local models

1. Confirm the backend answers:
   ```bash
   scripts/local-model-status.sh
   ```
   It exits 1 with `local router unavailable:` when it cannot reach `/health` or
   `/models`.
2. Confirm the aliases you expect are listed. The script prints each model's id,
   state and modalities.
3. Restart the Hermes front end. Desktop, TUI and CLI all read
   `~/.hermes/config.yaml`; there is no separate Desktop model registry, so a
   model visible to one and not another is a front-end restart problem.
4. Start a fresh chat after switching model families. Switching a long
   conversation midstream hands the local model a history created under different
   hidden assumptions.

## Build failures

`scripts/build-llama-cpp.sh` refuses before CMake runs when a precondition fails.
All of these exit 2 with a one-line reason:

| Message | Meaning |
|---|---|
| `missing llama.cpp submodule` | run `git submodule update --init --recursive upstream/llama.cpp` |
| `llama.cpp HEAD does not match <tag>` | the worktree is on a different revision than the lock |
| `llama.cpp tracked worktree is dirty` | uncommitted changes in the submodule |
| `unexpected llama.cpp repository in lock` / `unexpected GPU target in lock` | the lock was edited to something this stack does not support |
| `nproc did not return a positive integer` | the environment cannot report a CPU count |

If CMake itself fails, check that `hipconfig` is on `PATH` — the script derives
`HIPCXX` and `HIP_PATH` from it.

Do not run this concurrently with another optimized build. It uses every logical
CPU.

## Download problems

| Symptom | Cause | Action |
|---|---|---|
| Unit restarts forever, transfers nothing | Historically: a `.partial` that is already the whole file, resumed at EOF, answered HTTP 416. The current queue promotes or quarantines that case instead | Confirm you are running the tracked `download_queue.py`; check `downloads*.log` |
| `SHA-256 mismatch` | Corrupt or truncated transfer. The partial is quarantined to `.bad-<digest prefix>` and the transfer restarts from zero | Nothing to do; it self-heals. Investigate if it repeats |
| `quarantined invalid final` | A file already at the destination did not match size or digest | Expected after a bad copy or a power loss; the queue re-fetches |
| `duplicate destination in queue` | Two queue entries resolve to the same real path | Fix the queue; per-file ownership is what makes concurrency safe |
| Exit 75 | Another writer holds the phase lock | Wait, or stop the other unit |
| Exit 2 | Arguments were passed. The program takes none | Set `HERMES_DOWNLOAD_*` instead |
| A phase never starts | The previous phase's state is not `complete`, or its stamp does not equal the expected byte total | `cat verification/local-coverage-foundation/downloads*-complete.ok` and compare against the queue's `total_bytes`; run `test_queue_manifests.py` |
| `phase-one stamp mismatch` | The queue was edited without updating every copy of its byte total | Update queue, phase runner and qualification supervisor together |

Read-only reconciliation, safe on a busy host:

```bash
python3 verification/computer-use-grounding/reconcile_downloads.py
```

## Qualification never starts

The supervisor logs why it is waiting to
`verification/qualification-supervisor/qualification.log` and mirrors it into
`qualification-state.json`.

| `status` | Meaning | Action |
|---|---|---|
| `waiting-artifacts` | One or more phases are not `complete`, or a stamp mismatches | Finish the downloads; the log names the file and the reason |
| `waiting-safe-host` | A **blocking** conflict (`kernel-build`, `inference` or `download-queue`) or `MemAvailable < 32 GiB`. Load average is not read at all | The log line names counts by reason and up to eight sample tasks |
| `blocked-policy` | The candidate policy gate failed; nothing model-backed ran | Read `candidate-qualification/evidence/candidate-policy.json` → `problems` |
| `interrupted` | An operator stop or a signal | Re-run; passed steps are skipped, the interrupted step repeats |
| `failed` with `host did not become quiet` | The bounded wait expired | The message lists the blockers actually observed |
| `inputs-unreadable` | `git ls-files`/`git diff` over `verification` did not answer within the retry budget | Let the disk quiet down; the run exits 75 having attempted nothing |
| `admission-blocked` | Every incomplete gate refused admission; none failed | Re-run when the queues are idle. `blocked_steps` names them; `failed_steps` is empty |

Three classified reasons actually hold the qualification
(`QUALIFICATION_BLOCKING_REASONS`): a build whose `cwd` is inside a kernel tree
(`kernel-build`), any inference process that is not the managed router
(`inference`) — a **sidecar left resident by a failed gate**, or `llama-server`
started by hand — and any process in the `local-ai-model-downloads` slice
(`download-queue`). The router is excused by its cgroup, not its name, so a
hand-started `llama-server` blocks even though the unit does not.

`download-queue` is matched by cgroup before the command tests, because a queue
runs `python3` and `aria2c` and would otherwise classify as workspace Python or
a transfer, neither of which blocks. It blocks for one specific reason: the
router gate refuses a qualification that competing work would confound, so a
qualification that starts it beside a queue records nine model `FAIL` rows in under a
second and none of them is a verdict about a model. See
[every boot repeats the whole qualification](#every-boot-repeats-the-whole-qualification).

Everything else the classifier names is reported and not gating: a generic
compiler or `makepkg` outside a kernel tree, a transfer such as `aria2c` or
`curl`, and workspace Python. Deliberately so — those are not a missing kernel
and must not hold a qualification after a new-kernel boot. Seeing them in a
`waiting-safe-host` log line does not explain the wait; look for a `kernel-build`
or `inference` count, or for `MemAvailable`.

```bash
scripts/local-model-status.sh --host-sharing   # separates blocking from advisory
```

Exit 75 from the unit means another supervisor holds the lock.

## Stopping the qualification unit times out

Symptom: `systemctl --user stop local-ai-qualification.service` hangs for the
unit's whole stop timeout and the unit ends `failed` with `MainPID=0` and no
cgroup. `systemctl --user list-jobs` during the hang shows a `llama-router.service`
start job queued behind the qualification stop job.

Cause: a serialized runner stops the router for exclusive GPU access and restarts
it from its shell EXIT trap. That trap runs *inside* the unit's stop job, and
the gate units are ordered `After=llama-router.service`, so systemd schedules
the qualification stop ahead of the router start in the same transaction. A
blocking `systemctl start` in the trap then waits for a job that is waiting for
the trap to return. Observed on 2026-09-09 against the pre-rename unit; the
runner was killed after the stop timeout, which loses the gate's real exit code.

Fix, already applied: `run_serialized.sh` (TTS), `run_functional_serialized.sh`
(media) and `run_when_idle.sh` (computer use) enqueue the restart with
`systemctl --user --no-block start`, which returns as soon as the job is
accepted. The router is therefore *requested*, not observed ready — the runner
logs say `router restore queued; readiness not verified`, and the computer-use
runner result records `router_restart_requested` rather than a restored router.
Callers that need a live router start it and poll for it themselves, as
`verification/repository-agent/run_serialized.sh` does.

Check the router actually came back after the stop completes:

```bash
systemctl --user is-active llama-router.service
```

## Every boot repeats the whole qualification

Symptom: every invocation logs `passed step inputs changed; rerunning` for the
first step and works forward from there, so gates that passed an hour ago are run
again and the qualification never reaches its last step. On a host that is also
rebooting this looks like the kernel problem below, and it is a separate cause
that survives fixing the kernel.

The qualification skips a step only when its recorded `input_fingerprint` equals the
one computed at startup. `qualification_inputs_fingerprint` hashes each
`download-state*.json` document, and the download units start at boot too and
rewrite those documents with fresh `started_at`/`completed_at` timestamps while
they re-verify files that are already present and correct. Different bytes,
different fingerprint, every pass invalidated.

```bash
python3 -c "import importlib.util as u;s=u.spec_from_file_location('m','verification/qualification-supervisor/run_qualification.py');m=u.module_from_spec(s);s.loader.exec_module(m);print(m.qualification_inputs_fingerprint())"
python3 -c "import json;print(json.load(open('verification/qualification-supervisor/qualification-state.json'))['input_fingerprint'])"
```

If those disagree while `git status` is clean, compare the queue documents
against the last pass. A queue that only re-verified is not a changed input;
`durable_queue_state` is what keeps the run timestamps out of the digest. A
changed `repository`, `revision` or file count **is** a changed input and is
supposed to invalidate the passes that used it.

Related: `router-models` failing in well under a second with every model `FAIL`
is not nine model failures. `gate_router_models` refuses a qualification that
would be confounded by competing work, and a download queue re-verifying ~130 GB
is competing work. The refusal is correct; the qualification starting the gate anyway
is not. Confirm from the artifact rather than the log, which has no run
boundaries:

```bash
python3 -c "import json;d=json.load(open('verification/router-functional/evidence/router-functional.json'));print(d['models'][0]['problems'])"
systemctl --user list-units 'local-ai-model-downloads*'
```

`refusing confounded qualification; active workloads:` naming `download-queue`
means wait for the queues and re-run the step. It is not a weights, template or
placement problem, and the tolerances must not be moved for it.

## A gate's receipt is invalidated every boot

Symptom: `qualification-cache/` holds a `passed` receipt for a gate, the gate's
code and virtualenv have not been touched, and it still re-runs. `--plan` says
`run` where you expected `reuse`.

The receipt key includes each model artifact the gate depends on, and a large
weight is identified by size, mtime and ctime rather than by re-reading its
bytes. So the question is not "did the weights change" but "were the files
rewritten". On this host they are, repeatedly, and for a reason that has nothing
to do with the gate:

```bash
grep -c quarantined verification/local-coverage-foundation/downloads*.log
find models -name '*.bad-*' -printf '%TF %f\n' | sort | tail
```

A queue that re-reads an artifact it already promoted and computes a different
SHA-256 quarantines it as `*.bad-<epoch>` and downloads it again. The re-promoted
file has a new mtime and ctime, so every receipt and the whole-qualification
`input_fingerprint` that named it are invalidated — correctly, by their own
rules, because from the cache's point of view the input really did change.

**Read the promotion lines before blaming the cache.** If a file is quarantined
and then re-promoted with the *same* SHA-256 it had before, the replacement
matches the previously expected artifact. That alone does not establish whether
the intervening corruption was on disk, in memory, or in a read path:

```bash
grep -E 'quarantined|promoted' verification/local-coverage-foundation/downloads.log | tail -20
```

Investigate this alongside the kernel and storage evidence described in
[a gate fails while the kernel is faulting](#a-gate-fails-while-the-kernel-is-faulting),
without attributing a hardware or mirror cause from hashes alone. Fix the host
integrity issue first. Raising no tolerance, disabling no
verification and rebuilding no cache will stop this loop while reads keep coming
back wrong: the queue re-downloads, the fingerprint moves, and the qualification
restarts from the top for as long as it continues.

## A refused router qualification revokes a pass

**Repaired in the qualification admission/receipt update.** Previously a host refusal
inside the cached callable overwrote a valid pass with a failure. Admission is
now checked before revocation, while the inner guard still covers a workload
appearing between outer admission and dispatch. A pre-dispatch refusal restores
the exact old receipt; a late conflict after dispatch is inconclusive and does
not restore it. Real failures and unload evidence are retained.

So a preset that genuinely passed loses its receipt because a download queue
happened to be running, and the artifact records nine model `FAIL` rows that are
not verdicts about any model. Confirm from the artifact, which keeps the reason:

```bash
python3 -c "import json;d=json.load(open('verification/router-functional/evidence/router-functional.json'));print(d['models'][0]['problems'])"
```

The router stops dispatching on refusal and returns 75 unless an earlier model
actually failed. No model FAIL row is fabricated for the refused preset. The
repository-agent gate and serialized runner follow the same contract. The
supervisor waits for admission before revoking a receipt and distinguishes 75
(blocked) from 76 (inconclusive). A failed forced retest still invalidates reuse.

## The qualification dies reading its own inputs

Symptom: `qualification.log` ends in

```
fatal TimeoutExpired: Command '['git', 'diff', '--binary', 'HEAD', '--', 'verification', 'llama-models.ini', 'scripts']' timed out after 30 seconds
```

or the same command with `returned non-zero exit status 128`, and no gate ran at
all. On 2026-09-08 this ended twenty invocations between 13:23 and 23:35.

`qualification_inputs_fingerprint()` asks git what the checkout looks like.
`git diff --binary HEAD` walks every tracked file under `verification/`, which
takes minutes while a download queue is saturating the disk; `128` is git
refusing over a contended `index.lock` or an unreadable object. Both are
statements about the host at that moment, not about the qualification, and both used to
escape as unhandled exceptions into the module-level handler, which recorded
`status: failed` and exited.

Each query is now retried (`FINGERPRINT_ATTEMPTS`, `FINGERPRINT_TIMEOUT_SECONDS`)
and a persistent failure records `inputs-unreadable` and exits 75 with the
previous `input_fingerprint` left in place, so the next run still compares
against what actually ran.

```bash
grep -c 'input fingerprint attempt' verification/qualification-supervisor/qualification.log
grep -n 'inputs unreadable' verification/qualification-supervisor/qualification.log | tail
```

Persistent `128` is not a timing problem. Check the checkout itself, and read it
against [a gate fails while the kernel is faulting](#a-gate-fails-while-the-kernel-is-faulting).

## The policy gate blocks every gate over a file that is still arriving

Symptom: `candidate-policy` fails with a single problem naming one weight file,
and all eleven model-backed gates are recorded `blocked-policy`:

```
"problems": ["uncensored-multimodal-gemma4-heretic-q6k: Gemma-4-12B-it-heretic-Q6_K.gguf missing or wrong size"]
```

Check whether the file arrived shortly afterwards before treating it as a
missing artifact:

```bash
ls -la --time-style=+%F\ %T models/gemma4-heretic/
grep -E 'quarantined|promoted' verification/local-coverage-foundation/downloads.log | tail
```

On 2026-09-09 the gate ran at 00:01:15 and the file was promoted at 00:17 --
sixteen minutes of the qualification's whole run refused over a transfer in progress.
A completion stamp records that a queue finished once; it does not survive the
quarantine-and-re-download loop, because the queue verifies SHA-256 on
promotion, moves a mismatching file aside as `<name>.bad-<stamp>` and fetches it
again while the stamp still reads complete.

`wait_for_inputs` now also holds while any queue destination is absent or the
wrong size **and** has a `.partial` sibling, which is the signature of a transfer
staging bytes right now. A file that is simply absent, with nothing fetching it,
is still reported by the policy gate rather than waited for -- otherwise a real
missing artifact would become a hang.

```bash
python3 -c "import sys;sys.path.insert(0,'verification/qualification-supervisor');import run_qualification as m;print(m.artifacts_in_flight())"
```

The same loop makes a preset's identity unreadable, because `preset_identity`
stats the weights the preset names. The router and repository-agent gates now
refuse that preset (`qualification-inputs-unreadable`, exit 75) instead of
dying on `FileNotFoundError` and exiting 1, which every runner reads as a model
failure. Chat-versus-vision classification reads the preset's settings only, so
one in-flight download no longer aborts the gate before any preset is considered.

## A gate reports VRAM still held after unload

Symptom: a gate passes every functional section and fails only `unload`:

```
"problems": ["VRAM still held after unload: {'card0': 845180928}"]
```

Check what the residue is proportional to before treating it as a leak. In the
2026-09-08 grounding run the model held 6.80 GiB on card0, 6.99 GiB on card1 and
2.80 GiB on card2, and the residue was 498 MiB, 300 MiB and 298 MiB. Residue that
does not scale with what the card carried is not un-freed weights.

The sidecar gates stop a systemd unit, so the process holding the weights exits
and a card-level before/after reading really is a reading about the model. The
TTS, grounding and vision gates load the model inside their own process, which
keeps its HIP context, compiled kernels and BLAS workspaces on every device it
touched until the process exits. That floor was being scored as retained model
memory, and the three gates had each absorbed it into a different constant --
256, 512 and 768 MiB.

`unload_verdict` now also takes `allocator_report()`. Any allocated or reserved
tensor memory fails even below the card threshold. A zero allocator reading does
not identify the owner of device-wide residue: direct allocations, other
processes and driver allocations remain possible. Above-threshold residue stays
failed and is recorded as `unattributed_device_bytes`. Attribution requires an
independent runtime baseline or process-exit check; no threshold is waived.

```bash
python3 -c "import json;d=json.load(open('verification/tts-local/evidence/gate-tts.json'));print(json.dumps(d['unload'],indent=1))"
```

Use the complete `unload.pass` verdict, not one empty counter, to decide whether
release was proven. Allocator counters are diagnostic evidence, not proof that
every residual device allocation is harmless runtime context.

## One preset fails only its tool-call check

Observed on 2026-09-08, boot `b9e45a0d`: `router-models` ran for six minutes with
the queues idle and returned eight passes and one failure. `obliterated` answered
`PONG` for coherence and returned valid JSON for structured output, then produced
160 `/` characters for the tool-call prompt and stopped on `finish_reason:
length` with `tool_calls: []`.

Rule the template out first, because it was the previous cause here and it is
recorded in the artifact:

```bash
python3 -c "import json;d=json.load(open('verification/router-functional/evidence/router-functional.json'));m=[x for x in d['models'] if x['model']=='obliterated'][0];print(m['checks']['tool_call']['template'])"
```

`chat_template_length` 8952 with `supports_tool_calls: true` records template
metadata, not proof of the exact rendered payload. The repeated character is
degenerate generation; inspect rendering, effective sampling, runtime and weight
integrity before assigning a cause.

Compare presets before suspecting the weights. `ridge`, `heretic` and `fable` all
run `spec-type = draft-mtp` with `spec-draft-n-max = 2` and pass, but that does not
exclude a model-specific speculative-decoding defect. One candidate difference is
`repeat-penalty = 1.15` with `temp = 0.2`: a tool-call prompt repeats its JSON
scaffolding, a repetition penalty pushes exactly those tokens down, and a
low temperature may reinforce a degenerate continuation. This mechanism is
unproven here and must not be reported as the established cause.

**This is a hypothesis, not a finding.** Testing it means one bounded run of that
preset with the penalty at the default, and that is a GPU workload: do it when
the qualification is not running, and change nothing in `llama-models.ini` until a run
shows the check passing. `phr00ty` passing at `temp = 1.5` has already closed the
older theory that its temperature was the cause of its tool-call failures.

## Gate problems

| Symptom | Cause | Action |
|---|---|---|
| A runner exits 75 immediately | Its opening `gate_computer_use.py --preflight-only` check found a **hard** blocker. Only the available-RAM floor is hard; load and heavy processes are advisory and pass unless `--require-quiet-host` was given | Read `hard_blockers` in `computer-use-grounding/evidence/computer-use-preflight.json`; free RAM and re-run |
| Repository-agent gate refuses to start | `bwrap` is missing or not executable. There is deliberately no unisolated fallback | Install bubblewrap |
| Repository-agent aborts mid-run with a sandbox error | The boundary failed, which is a host problem, not a candidate mistake. It aborts rather than returning a retryable tool error | Check namespace limits and host pressure; the in-progress artifact remains fail-closed |
| `oracle_tampered` in the artifact | The fixed test suite was modified or removed during the run | The run does not count. Investigate the candidate's behaviour |
| Computer-use gate exits 3 | Another instance holds the flock | Wait; do not start a second worker |
| Computer-use gate exits 4 | Interrupted by a handled signal | Re-run; an unfinished run is not a verdict |
| Runner exits 5 | Python exited 0 but the artifact is missing, stale or unreadable | Fail-closed by design; check the gate log |
| Media runner reports `V620 retained excess VRAM after unload` | ComfyUI did not release residency within the 768 MiB tolerance | Check for a surviving ComfyUI process before re-running |
| A gate passes but the ledger disagrees | The artifact's `gate` field does not match the expected identity, or `sections_missing` is non-empty | Compare against [the evidence contract](DEVELOPER-GUIDE.md#writing-an-evidence-artifact) |
| Router gate reports `reclaimable RAM remained N bytes below baseline` | Host memory did not come back after the model was unloaded, and the ZFS ARC does not account for it | A real shortfall. Note this is measured as `MemAvailable + (ARC size - ARC c_min)`, because MemAvailable alone does not credit the ARC and charged whichever model was under test for cache the router had filled reading earlier presets. `ram_recovery` in the artifact separates the two deltas. A shortfall is real memory, which is not the same as the model's: confirm the host is not retiring corrupt pages first, see [a gate fails while the kernel is faulting](#a-gate-fails-while-the-kernel-is-faulting) |
| A gate exits -11 or 139 after its artifact says `"pass": true` | Two causes look identical here and only one is benign: ROCm's HSA runtime segfaulting in its own process-exit teardown *after* the verdict was written, or a crash *during* the gate that left an artifact from an earlier run in place | The run failed until you prove otherwise; see [a failed process outranks a passing artifact](#a-failed-process-outranks-a-passing-artifact) |
| A ComfyUI node fails with `HIPBLAS_STATUS_INVALID_VALUE` from `hipblasLtMatmulAlgoGetHeuristic` | hipBLASLt has no algorithm for that problem shape on this architecture. These cards are gfx1030 and torch's own hipBLASLt support list is gfx9 | Not a model or workflow fault, and not a tolerance question. The media runner exports `TORCH_BLAS_PREFER_HIPBLASLT=0` to route matmuls through hipBLAS instead |
| `router-models` exits 1 in under a second with every model `FAIL` | Not a model verdict. The gate refused a qualification confounded by competing work, usually a download queue re-verifying at boot | [Every boot repeats the whole qualification](#every-boot-repeats-the-whole-qualification) |
| A gate shows `running` long after the host rebooted | A crash or hard kill left the record behind; no signal handler ran to mark it interrupted | `python3 scripts/qualification_status.py` reports it as `interrupted` rather than trusting the record |
| Grounding scores look confidently wrong | A geometry or action-space mismatch, not the model. This is exactly what `groundlib.py` exists to prevent | `python3 -m pytest verification/computer-use-grounding/test_grounding_contract.py` |

## A failed process outranks a passing artifact

A gate that exits non-zero or on a signal has **failed**. That is the default and
it is not negotiable by reading the artifact, because the artifact is a file that
persists between runs and a crashed run does not necessarily overwrite it. This is
the same rule `Runner exits 5` already enforces: an artifact that is missing,
stale or unreadable fails closed rather than being interpreted.

The one benign explanation on this host is ROCm's HSA runtime segfaulting inside
its own interpreter finalization, after the gate has already computed and written
its verdict. It was observed three times on boot `1669f3ad`, always
`libhsa-runtime64.so.1.18.0`, always at finalization. Three things must all hold
before that explanation applies:

1. **The artifact is from this run.** Its `recorded_at` falls between the step's
   `started_at` and `finished_at` in `qualification-state.json`, and its `boot_id`, where
   the gate records one, is the current `/proc/sys/kernel/random/boot_id`. An older
   timestamp means you are reading a previous verdict and the crash destroyed this
   one.
2. **The only fault is that userspace segfault.** `journalctl -k -b` must show the
   `libhsa-runtime64` segfault and nothing else for the window. No
   `BUG: Bad page state`, no general protection fault, no GPU ring timeout or
   reset, no RCU stall, no OOM kill, no MCE. Any of those makes every result from
   that window suspect, the passing ones included.
3. **The gate does not already route around it.** `gate_asr.py` and
   `gate_computer_use.py` exit through `gatelib.exit_after_verdict`, which returns
   the computed code by design. If one of *those* still exits -11 or 139, the crash
   may have landed before or after the artifact write but before process exit.
   The artifact is not necessarily old; the run still failed.

If any of the three does not hold, re-run the gate on a healthy host. Do not edit
the state file, and do not carry the artifact forward as a verdict.

## A gate fails while the kernel is faulting

Symptoms that arrive together: gates that pass and fail across consecutive runs
with no change to weights, configuration or presets; the router gate reporting
`reclaimable RAM remained N bytes below baseline` for whichever preset happened to
be under test; the qualification restarting from the top every few minutes.

Rule out the harness first for that last one: a changed input fingerprint
restarts the qualification from the top on a perfectly healthy host. See
[every boot repeats the whole qualification](#every-boot-repeats-the-whole-qualification).

Check the host before the model:

```bash
uptime
journalctl --list-boots | tail -12
journalctl -k -b | grep -E "BUG: Bad page state|general protection|Tainted:|Oops"
```

`BUG: Bad page state` and general protection faults establish a kernel-integrity
problem, not its root cause. Hardware instability, firmware configuration, and
kernel or module defects remain possible. Neither these messages nor the
`CPU_OUT_OF_SPEC` taint alone proves an overclock or explains a measured RAM
shortfall. Record the first fault and surrounding boot messages before assigning
causality. Host-wide memory deltas also do not isolate a particular model leak.

**Do not raise tolerances to hide the failure.** Preserve the gate and kernel
evidence, investigate the host-integrity problem, and qualify models again on a
healthy host. Results from the faulting window remain useful diagnostic evidence,
but cannot establish reliable model admission or isolate a model defect.

## Ledger says something unexpected

| Reading | Meaning |
|---|---|
| `evidence-stale` | The gate artifact is older than the newest weight file for that capability. It is a statement about coverage, not about the model. Re-run the gate. mtime also moves when a verified file is re-promoted rather than re-fetched |
| `evidence-interrupted` | A run was cut short. Re-run it to completion |
| `downloaded` with "no evidence artifact is declared" | Intentional for `image-generation-editing` and `uncensored-multimodal`; see [the matrix](CAPABILITY-MATRIX.md#the-matrix) |
| `download-incomplete` | A declared file is missing or the wrong size. Re-run the queue |
| `problems: [… reports measured throughput]` | An artifact claims a measurement this workspace forbids. Find and remove it |
| `problems: [… declared for a capability no queue owns]` | `CAPABILITY_EVIDENCE` names a capability no queue provides. Fix the map or the queue |
| Everything reads `downloaded` on a fresh clone | Correct. Evidence is untracked runtime state |

## Test suite problems

| Symptom | Cause | Action |
|---|---|---|
| Dozens of collection errors | `pytest` was run without this repository's `pytest.ini`, so it walked into `tools/` and `venvs/` | Run from the repository root |
| Three failures in `verification/repository-agent/fixture/` | The fixture is the gate's oracle and is defective on purpose | It is excluded by `norecursedirs`; do not "fix" it |
| Sandbox tests hang or fail on a loaded host | A busy or RCU-stalled host is not a valid namespace test environment | `python3 -m pytest --ignore=verification/repository-agent`, then run the full suite when the host drains |
| Preset-compatibility test skips | The submodule has not been built, so there is no `--help` to check against | Build it |
| `verification/docs` fails on a new file | A tracked file has no coverage-map row, or a link/anchor does not resolve | Update [the coverage map](COVERAGE-MAP.md) or fix the link |

## Storage and power loss

After an unclean shutdown, a model whose byte size is complete is not
trustworthy. Re-verify the publisher SHA-256 before loading it. Both failure
shapes have been seen here: ZFS rejecting a full-size replacement with
`Input/output error`, and a different full-size pre-crash copy failing its digest.

```bash
python3 verification/computer-use-grounding/reconcile_full.py   # re-hashes every declared digest
sudo zpool status -v zroot                                      # privileged listing of affected paths
```

Do not run `zpool clear` until the scrub result and affected paths have been
reviewed. Model artifacts are replaceable; unrelated user data needs a restore
decision.

Crash residue you may see, all gitignored: `*.partial`, `*.recovered`,
`*.bad-*`, `*.tmp`, `*.tmp.<pid>`. `reconcile_full.py` accounts for reclaimable
bytes and never deletes them — that is an operator decision.
