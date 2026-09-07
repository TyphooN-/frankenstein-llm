# frankenstein-llm

## Scripted operation

See [Model runs](docs/MODEL-RUNS.md) for per-model serve scripts, editable configs,
user/agent-invoked qualification, native benchmarks and saved reports. Commands
preview by default; the intended GPU topology remains all three cards.

## Documentation

Start with the [User guide](docs/USER-GUIDE.md) for installation and everyday use.
The [documentation index](docs/README.md) links implementation architecture,
configuration, operations, troubleshooting, capability status, development and
tracked-file coverage. Downloaded weights alone are not functional proof.

Private local-AI workspace for the X99 host: llama.cpp routing, capability qualification, ComfyUI media, and serialized download/gates.

**This repository is AMD ROCm only.** Every GPU in it is an AMD card, the pinned
llama.cpp submodule is built for the HIP/ROCm backend by `scripts/build-llama-cpp.sh`,
device selectors are spelled `ROCm0,ROCm1,ROCm2` throughout the presets, and
ComfyUI is launched with `HIP_VISIBLE_DEVICES`. There is no CUDA path here and
none is planned; NVIDIA-specific advice does not transfer. See
[GPU execution and model loading](docs/reference/GPU-EXECUTION-AND-MODEL-LOADING.md).

Canonical checkout: `/home/typhoon/git/frankenstein-llm`

This repository tracks source, configuration, verification, and documentation. It does not track model weights.

## Layout

- `llama-models.ini` — llama.cpp router presets
- `scripts/` — download and verification helpers
- `services/` — sidecar env files and systemd unit copies
- `upstream/llama.cpp/` — tracked llama.cpp submodule; production source and local ROCm build
- `upstream/llama-cpp.lock.json` — release, commit, GPU target, and required binary lock
- `verification/` — functional gates; no tokens/sec measurements
- `verification/prompt-corpus-admission/` — pinned, inert safety-corpus policy
- `docs/` — strategy and operating notes
- `docs/decisions/` — architecture decision records
- `models/` — local weights only; gitignored
- `venvs/` and `tools/` — local runtimes; gitignored

## Hardware

- GPU0: RX 6900 XT 16 GiB, headless, preferred compute
- GPU1: Radeon Pro V620 32 GiB, headless, preferred large-model/media
- GPU2: RX 6900 XT 16 GiB, display, minimize compute
- One resident llama.cpp model at a time
- Preserve three 1 GiB HugeTLB pages for XMRig

## Qualification policy

Functional load, coherence, memory-fit, and clean unload are in scope. Benchmarking and tokens/sec require explicit user authorization after confirmation of the intended kernel; reboot alone is insufficient.

See:

- `docs/decisions/0001-git-tracked-workspace-without-weights.md`
- `docs/decisions/0002-serialized-functional-qualification.md`
- `docs/decisions/0003-hardware-allocation-and-memory-policy.md`
- `docs/decisions/0004-prompt-corpus-admission.md`
- `docs/decisions/0005-track-llama-cpp-submodule.md`
- `docs/LOCAL-AI-MODEL-STRATEGY.md`
- `docs/CANDIDATE-MODEL-REVIEW-2026-09-02.md` — 2026-09-02 research snapshot
- `docs/CANDIDATE-STATUS-2026-09-03.md` — current download/policy/runtime/functional state

## Downloads

The download queues preserve one writer per resolved artifact, resume into an
ignored `.partial` sibling, verify exact size and SHA-256, and durably promote
only verified bytes. Independent files run concurrently according to
`HERMES_DOWNLOAD_WORKERS` (default 4, service policy 16).

When `aria2c` is installed, each file can also use bounded HTTP range
connections. `HERMES_DOWNLOAD_CONNECTION_BUDGET` is shared across active files
(default and general service policy 32; the image-editing and researched-candidate
queues 64), with at most 16
connections assigned to one
file. If aria2 is unavailable or only one connection is assigned, the portable
curl continuation path is used. Model loading and functional qualification
remain serialized even though transfers are parallel.

The researched-candidate queue (`download-queue-phase4.json`, kept under its
original name because its completion stamp records what was downloaded) installs
Qwen3-Coder-Next Q4_K_M, Gemma-4 Heretic Q6_K with its projector, UI-Mate-9B,
WeMM-Embedding-2B, and the non-duplicated FLUX.2-klein-4B runtime files.
Qwen3-ASR-1.7B is not duplicated because the same pinned revision was already
verified by the core-capability queue. The mission supervisor treats that queue's
successful completion as a prerequisite for subsequent serialized functional
qualification. It then runs the non-inference candidate policy gate before loading
anything. That gate verifies manifest-backed inventory, the core-capability ASR
deduplication claim, distinct text/multimodal vector spaces, privilege boundaries,
and the exact reviewed WeMM remote-code digests.
See `verification/candidate-qualification/README.md`. A passing policy gate is
not a functional model verdict: UI-Mate grounding, WeMM retrieval, FLUX.2
workflows, and all candidate load/behavior/unload checks still require their live
serialized gates.

## Runtime

User systemd units live in `~/.config/systemd/user/` and are copied here under `services/systemd/`. After clone or path changes:

```
git submodule update --init --recursive upstream/llama.cpp
scripts/build-llama-cpp.sh
install -Dm644 services/systemd/llama-router.service ~/.config/systemd/user/llama-router.service
install -Dm644 services/systemd/llama-sidecar@.service ~/.config/systemd/user/llama-sidecar@.service
systemctl --user daemon-reload
systemctl --user start llama-router.service
```

The router listens on `127.0.0.1:8080`.
