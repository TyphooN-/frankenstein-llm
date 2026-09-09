# User-operated model serving, qualification and benchmarks

These entry points are for both humans and AI agents. Preview is the default;
`--execute` is required to launch a model or qualification. No service is silently stopped,
no GPU topology is auto-rewritten, and no dependency is automatically downloaded.

## Normal hardware and configuration

The intended host has **three GPUs**, not two: ROCm0 RX 6900 XT 16 GiB
(headless), ROCm1 Radeon Pro V620 32 GiB (headless), ROCm2 RX 6900 XT 16 GiB
(display). Temporary device absence does not change this design. Shared serving
presets retain `ROCm0,ROCm1,ROCm2` and an equal `1,1,1` default; almost every
preset overrides the proportions with a placement computed by
`scripts/gpu_placement.py` (see [ADR 0006](decisions/0006-placement-policy-prefers-the-rx-6900-xts.md)).
Benchmark defaults use the same roles with native bench syntax
`ROCm0/ROCm1/ROCm2`, at the same proportions the `[*]` section configures — only
the ratio matters, so `6/6/6` and `1,1,1` are the same placement, and a test
compares the two files by ratio rather than by literal. These are allocation
requests, not measured residency or memory-fit proof. A missing required device should cause a failed run,
not silently establish a smaller machine as the default.

Editable sources of truth:

| File | Owns |
|---|---|
| `llama-models.ini` | Model files, context, KV precision, sampling, devices, splits and model-specific serving options |
| `config/serving.json` | Standalone loopback host and port |
| `config/model-runs.json` | Benchmark workload, device list, split, layers, repetitions and run timeouts |
| `services/*.env` | Existing dedicated sidecar configuration |

CLI options override config. Copy a config and pass `--config path.json` for a
local experiment. Copy the INI and use `--presets path.ini` for serving experiments.
Execution permission and kernel confirmation cannot be stored in tuning config.
Unknown config keys fail rather than being silently ignored. Threads default to
available logical CPUs; add an integer `threads` to the benchmark config to tune.

## Serve any configured model

The default listing leads with the **full weight filename(s) and (purpose/use
case)**, and tags the short handle behind them as
`[compatibility alias: <name>]`. The filename is the primary label because it is
the artifact `llama-server` actually opens; a vision preset also names its
projector, because it shares the weight file with its text-only sibling.

A handle like `heretic` is not a name for anything on disk — it identifies no
file, and nothing stops two different weight files wearing it across a
reconfiguration. It is retained solely so existing commands, scripts and API
callers keep working.

`config/model-catalog.json` holds two fields per alias. `purpose` is the intended
use case shown in parentheses; it is a description, not a comparative quality
claim. `source` is the publisher's name for the release the bytes came from, and
it is provenance rather than identity — `heretic` loads
`RVN-Q6_K-multilingual-mtp.gguf` from a release published as
`Qwen3.8-27B-Heretic-Abliterated-Uncensored`, so printing the source alone would
name a file that is not on this disk. `--list --details` adds `source` and the
full weight/projector paths. Aliases remain the stable command and API
identifiers and are never renamed; they trail the description as labelled
compatibility metadata, never lead it. `scripts/model_catalog.py` owns these strings so the serving
listing, the router status summary and the direct-API acceptance suite cannot
drift apart. A preset with no catalog entry fails the listing rather than
printing a blank use case.

```bash
cd /home/typhoon/git/frankenstein-llm
python3 scripts/serve-model.py --list
bash scripts/serve-heretic.sh                 # preview only
bash scripts/serve-qwen3-coder-next.sh         # preview only
bash scripts/serve-heretic.sh --execute       # foreground server; Ctrl-C stops it
```

Every current INI alias has a `scripts/serve-<alias>.sh` convenience wrapper,
including text, vision, FIM, embedding and reranker presets. They all delegate to
`serve-model.py`; none duplicates tuning values. New INI aliases immediately work
with `python3 scripts/serve-model.py NEW_ALIAS`; add a wrapper if desired.

The shared router remains the normal multi-model Hermes entry point. Standalone
serving is an alternative, not an additional server to launch onto an occupied
port. Stop the router yourself before using port 8080, or choose a distinct port
with `--port 8090` and explicitly configure the client. A different port does not
make overlapping GPU workloads safe. The launcher checks the port but never
terminates the process holding it. It also checks that required weight/projector
paths and the runtime binary exist before execution.

`serve-ridge.sh` now delegates to the same presets and requires `--execute`.
Its former RIDGE_MODEL/LLAMA_CTX/LLAMA_HOST and related environment tuning is
replaced by the INI, serving JSON and documented CLI flags. The tracked
`llama-ridge.service` has been updated accordingly; installing updated unit copies
and daemon-reloading remains an explicit operator step. Do not run standalone
Ridge and the router on the same port.

Serving does not grant tools. The launcher is not an authorization engine: Hermes
and candidate-policy tool restrictions still apply, especially reader-only Gemma.
Foreground output goes to the terminal unless the caller redirects it or uses a unit.

## Run functional qualification

```bash
bash scripts/qualify-models.sh                # plan only
# Stop a previously active qualification before starting another owner:
systemctl --user stop local-ai-qualification.service
bash scripts/qualify-models.sh --execute
```

This runs the existing serialized, resumable full qualification, including its policy
prerequisite, artifact readiness checks, model gates and capability evidence. It
is not a new lightweight single-model gate. The qualification may restart the router
and hand GPUs between capability lanes. Do not run it while using Hermes on that
same local backend; switch to a cloud backend or close the local chat first.
Existing qualification progress remains under `verification/qualification-supervisor/`.
The configured total timeout includes waiting for downloads and host readiness.

## Build and run native benchmarks

Build the optional benchmark binary once, on an idle host:

```bash
bash scripts/build-model-bench.sh
```

This validates/builds the pinned runtime using the existing build script, then
builds `llama-bench` with `nproc` parallelism. It does **not** execute a benchmark.
Do not overlap this build with a kernel or other optimized build.

Preview and run an example:

```bash
bash scripts/benchmark-model.sh --model models/Qwen3.8-27B-Ridge-3.7bpw.gguf
# Deliberate exclusive handoff; never stop a backend hosting the active agent:
systemctl --user stop local-ai-qualification.service llama-router.service
bash scripts/benchmark-model.sh \
  --model models/Qwen3.8-27B-Ridge-3.7bpw.gguf \
  --confirm-kernel --execute
# Restore normal service when finished:
systemctl --user start llama-router.service
```

`--confirm-kernel` means the operator has verified the intended boot/kernel and
explicitly authorizes this performance run. Reboot alone is not authorization.
The config defaults request GPU offload; `--gpu-layers 0` requests a CPU baseline.
For split GGUFs pass the first shard. Native results measure prompt processing and
token generation, **not** coding quality or Hermes agent reliability. They do not
apply the router's full chat/MTP preset: the recorded argv is the benchmark
configuration. Use identical workload, quantization, devices and settings for
meaningful comparisons; do not label different configurations a model ranking.

## Preflight, ownership and reports

Qualification and benchmarks refuse execution if the qualification lock is busy, ZFS
cannot be positively identified as healthy, recognized compiler/download
processes are active, or available RAM is below 16 GiB. Benchmarks additionally
refuse live `llama-*` processes. The RAM threshold is only a minimum, not proof
that the selected model fits. Process inspection across this repository reads
`stat`, `comm`, `cgroup` and the `cwd` symlink, never `cmdline`: Linux serves
`cmdline` through `access_remote_vm`, so inspecting a compiler that holds its own
mmap write lock can block the inspector. Classification from a task name is
coarser than an argument vector and therefore blocks on more, which is the safe
direction for a refusal check.
This is conservative preflight, not system-wide exclusion: unrelated Python/GPU
work or a workload starting later still requires operator discipline. Do not run
other model clients, downloads, mining-heavy tests or builds during measurement.

The known ZFS data error must be resolved before these performance/qualification
entry points will run. Do not clear error records just to bypass admission.
Serving remains a separate user decision; it does not assert storage health.

Every admitted run gets a unique ignored directory in `logs/model-runs/`:

- `report.json`: command, mode, boot/kernel identity, outcome and exit status;
- `stdout.log` and `stderr.log`: native output;
- for a successful benchmark, `stdout.log` contains native JSON metrics.

Qualification leaves its detailed gate evidence in existing qualification locations.
No qualification pass is fabricated from a process merely starting. Timeouts and
interrupts terminate the owned child process group; interrupted/failed reports
remain available. Do not interpret an interrupted benchmark as a measured result.
Failed preflight prints a reason and exits before creating a run directory.

Exit status: 0 success or valid preview, 2 invalid configuration/acknowledgement,
75 preflight/lock refusal, 124 timeout, 130 interrupt, otherwise native child
failure (or 1 for report/cleanup failures). Reports never overwrite prior runs.

## Publish measured throughput as markdown

Throughput is deliberately not part of `local-ai-qualification.service`,
which records `benchmark_performed: false`. The qualification answers "does this
preset work"; this lane answers "how fast was it in one measured run", and the
two are kept apart so a rate can never stand in for a gate.

`scripts/bench_report.py` turns run directories that already exist under
`logs/model-runs/` into one markdown file per weight file under
[`docs/benchmarks/`](benchmarks/README.md). It never loads a model, never
touches a GPU and never starts or stops a service:

```bash
bash scripts/bench-reports.sh \
  logs/model-runs/benchmark-XXXXXXXX \
  gemma4-heretic-vision=logs/model-runs/benchmark-YYYYYYYY
```

A run directory usually identifies its own preset from the weight path in the
recorded argv. `alias=directory` is required only where two presets share one
GGUF, which is every vision sibling: the projector is the difference, and
`llama-bench` does not load it.

The writer refuses rather than approximating:

- **No native JSON, no artifact.** A run whose `report.json` is missing, is not
  `passed`, recorded no `native_results`, or whose `stdout.log` does not parse
  into complete rows produces a refusal and exit status 2. Nothing is estimated
  from a wall clock, and an interrupted run stays a recorded interruption.
- **No ranking across serving-MTP classes.** `ridge`, `heretic`, `obliterated`
  and `fable` serve with `spec-type = draft-mtp`; `llama-bench` applies no draft
  model to any of them. Ordering an MTP-served preset against a non-MTP-served
  one by rates that were all measured without MTP would read as a ranking of the
  weights, so it is refused. The index orders only within one class, and its
  table is alphabetical.
- **No two runs under one alias.** The surviving file would carry the other
  run's argv.
- **No placement the alias does not serve.** The `-ts` in the recorded argv is
  compared with `llama-models.ini` by ratio, so `6/6/6` and `1,1,1` agree while
  `1,1,1` and `5,8,4` do not.

Usability notes are read from `verification/router-functional/evidence/` as
pass/fail with the gate's own timestamp. No quality score is computed, a gate
failure is never restated as a pass, and a model absent from that evidence is
reported as absent rather than as passing.

VRAM and RAM are reported only when they were measured. Sample them beside a run
by pointing the sampler at the report directory the benchmark printed:

```bash
python3 scripts/bench_report.py sample --into logs/model-runs/benchmark-XXXXXXXX &
# ... the benchmark runs ...
kill -TERM %1
```

It republishes peaks atomically on every tick, so stopping it at any moment
leaves a valid file. Without it, an artifact says the residency was not measured
rather than reporting the allocation request as though it were.

Artifacts are tracked, so a newly benchmarked model needs a row in
[`docs/reference/COVERAGE-MAP.md`](reference/COVERAGE-MAP.md) before the
documentation suite passes.

## Asking an agent to operate this

Examples:

- "Preview qualification with scripts/qualify-models.sh; do not execute."
- "Run scripts/qualify-models.sh --execute and summarize the report and failed gates."
- "I verified the intended kernel and authorize a benchmark. Run benchmark-model.sh
  for this GGUF using config/model-runs.json and report the native results."
- "Tune a copied benchmark config, show me the plan, and wait before execution."

An agent must honor the same lock, preflight and approval boundaries. Scripts do
not grant permission to stop services or override system security. Prefer one
coherent run and saved evidence over reimplementing the workflow in a chat.
