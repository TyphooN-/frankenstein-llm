# frankenstein-llm

Private local-AI workspace for the X99 host: llama.cpp routing, capability qualification, ComfyUI media, and serialized download/gates.

Canonical checkout: `/home/typhoon/git/frankenstein-llm`

This repository tracks source, configuration, verification, and documentation. It does not track model weights.

## Layout

- `llama-models.ini` — llama.cpp router presets
- `scripts/` — download and verification helpers
- `services/` — sidecar env files and systemd unit copies
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

Functional load, coherence, memory-fit, and clean unload are in scope. Benchmarking and tokens/sec wait until an explicit reboot into the intended kernel.

See:

- `docs/decisions/0001-git-tracked-workspace-without-weights.md`
- `docs/decisions/0002-serialized-functional-qualification.md`
- `docs/decisions/0003-hardware-allocation-and-memory-policy.md`
- `docs/decisions/0004-prompt-corpus-admission.md`
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
(default and general service policy 32; phases three and four 64), with at most 16
connections assigned to one
file. If aria2 is unavailable or only one connection is assigned, the portable
curl continuation path is used. Model loading and functional qualification
remain serialized even though transfers are parallel.

Phase four is the pinned research-candidate queue. It installs Qwen3-Coder-Next
Q4_K_M, Gemma-4 Heretic Q6_K with its projector, UI-Mate-9B, WeMM-Embedding-2B,
and the non-duplicated FLUX.2-klein-4B runtime files. Qwen3-ASR-1.7B is not
duplicated because the same pinned revision was already verified by phase one.
The mission supervisor treats successful phase-four completion as a prerequisite
for subsequent serialized functional qualification. It then runs the non-inference
candidate policy gate before loading anything. That gate verifies manifest-backed
inventory, the phase-one ASR deduplication claim, distinct text/multimodal vector
spaces, privilege boundaries, and the exact reviewed WeMM remote-code digests.
See `verification/candidate-qualification/README.md`. A passing policy gate is
not a functional model verdict: UI-Mate grounding, WeMM retrieval, FLUX.2
workflows, and all candidate load/behavior/unload checks still require their live
serialized gates.

## Runtime

User systemd units live in `~/.config/systemd/user/` and are copied here under `services/systemd/`. After clone or path changes:

```
systemctl --user daemon-reload
systemctl --user start llama-router.service
```

The router listens on `127.0.0.1:8080`.
