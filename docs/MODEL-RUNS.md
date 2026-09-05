# User-operated model serving, qualification and benchmarks

These entry points are for both humans and AI agents. Preview is the default;
`--execute` is required to launch a model or mission. No service is silently stopped,
no GPU topology is auto-rewritten, and no dependency is automatically downloaded.

## Normal hardware and configuration

The intended host has **three GPUs**, not two: ROCm0 RX 6900 XT 16 GiB
(headless), ROCm1 Radeon Pro V620 32 GiB (headless), ROCm2 RX 6900 XT 16 GiB
(display). Temporary device absence does not change this design. Shared serving
presets retain `ROCm0,ROCm1,ROCm2` and `3,6,2`; individual models may override
proportions. Benchmark defaults use the same roles with native bench syntax
`ROCm0/ROCm1/ROCm2` and `3/6/2`. These are allocation requests, not measured
residency or memory-fit proof. A missing required device should cause a failed run,
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

The default listing displays **Full model name (purpose/use case)**. Human-readable
names and intended roles live in `config/model-catalog.json`; they are descriptions,
not comparative quality claims. Use `--list --details` to additionally expose aliases,
weight paths and projectors. Aliases remain stable command/API identifiers.

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
# Stop a previously active mission before starting another owner:
systemctl --user stop local-ai-functional-mission.service
bash scripts/qualify-models.sh --execute
```

This runs the existing serialized, resumable full mission, including its policy
prerequisite, artifact readiness checks, model gates and capability evidence. It
is not a new lightweight single-model gate. The mission may restart the router
and hand GPUs between capability lanes. Do not run it while using Hermes on that
same local backend; switch to a cloud backend or close the local chat first.
Existing mission progress remains under `verification/mission-supervisor/`.
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
systemctl --user stop local-ai-functional-mission.service llama-router.service
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

Qualification and benchmarks refuse execution if the mission lock is busy, ZFS
cannot be positively identified as healthy, recognized compiler/download
processes are active, or available RAM is below 16 GiB. Benchmarks additionally
refuse live `llama-*` processes. The RAM threshold is only a minimum, not proof
that the selected model fits. Process inspection reads `stat`, never `cmdline`.
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

Qualification leaves its detailed gate evidence in existing mission locations.
No qualification pass is fabricated from a process merely starting. Timeouts and
interrupts terminate the owned child process group; interrupted/failed reports
remain available. Do not interpret an interrupted benchmark as a measured result.
Failed preflight prints a reason and exits before creating a run directory.

Exit status: 0 success or valid preview, 2 invalid configuration/acknowledgement,
75 preflight/lock refusal, 124 timeout, 130 interrupt, otherwise native child
failure (or 1 for report/cleanup failures). Reports never overwrite prior runs.

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
