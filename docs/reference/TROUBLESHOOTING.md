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
- [Mission never starts](#mission-never-starts)
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
| `phase-one stamp mismatch` | The queue was edited without updating every copy of its byte total | Update queue, phase runner and mission supervisor together |

Read-only reconciliation, safe on a busy host:

```bash
python3 verification/computer-use-grounding/reconcile_downloads.py
```

## Mission never starts

The supervisor logs why it is waiting to
`verification/mission-supervisor/mission.log` and mirrors it into
`mission-state.json`.

| `status` | Meaning | Action |
|---|---|---|
| `waiting-artifacts` | One or more phases are not `complete`, or a stamp mismatches | Finish the downloads; the log names the file and the reason |
| `waiting-safe-host` | A **blocking** conflict (`kernel-build` or `inference`) or `MemAvailable < 32 GiB`. Load average is not read at all | The log line names counts by reason and up to eight sample tasks |
| `blocked-policy` | The candidate policy gate failed; nothing model-backed ran | Read `candidate-qualification/evidence/candidate-policy.json` → `problems` |
| `interrupted` | An operator stop or a signal | Re-run; passed steps are skipped, the interrupted step repeats |
| `failed` with `host did not become quiet` | The bounded wait expired | The message lists the blockers actually observed |

Only two classified reasons actually hold the mission
(`MISSION_BLOCKING_REASONS`): a build whose `cwd` is inside a kernel tree
(`kernel-build`), and any inference process that is not the managed router
(`inference`) — a **sidecar left resident by a failed gate**, or `llama-server`
started by hand. The router is excused by its cgroup, not its name, so a
hand-started `llama-server` blocks even though the unit does not.

Everything else the classifier names is reported and not gating: a generic
compiler or `makepkg` outside a kernel tree, a transfer such as `aria2c` or
`curl`, and workspace Python. Deliberately so — those are not a missing kernel
and must not hold a mission after a new-kernel boot. Seeing them in a
`waiting-safe-host` log line does not explain the wait; look for a `kernel-build`
or `inference` count, or for `MemAvailable`.

```bash
scripts/local-model-status.sh --host-sharing   # separates blocking from advisory
```

Exit 75 from the unit means another supervisor holds the lock.

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
   `started_at` and `finished_at` in `mission-state.json`, and its `boot_id`, where
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
be under test; the mission restarting from the top every few minutes.

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
