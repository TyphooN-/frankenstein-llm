# frankenstein-llm

Private local-AI workspace for the X99 host: llama.cpp routing, capability qualification, ComfyUI media, and serialized download/gates.

Canonical checkout: `/home/typhoon/git/frankenstein-llm`

This repository tracks source, configuration, verification, and documentation. It does not track model weights.

## Layout

- `llama-models.ini` — llama.cpp router presets
- `scripts/` — download and verification helpers
- `services/` — sidecar env files and systemd unit copies
- `verification/` — functional gates; no tokens/sec measurements
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
- `docs/LOCAL-AI-MODEL-STRATEGY.md`
- `docs/CANDIDATE-MODEL-REVIEW-2026-09-02.md` — reviewed but not downloaded or qualified

## Downloads

The download queues preserve one writer per resolved artifact, resume into an
ignored `.partial` sibling, verify exact size and SHA-256, and durably promote
only verified bytes. Independent files run concurrently according to
`HERMES_DOWNLOAD_WORKERS` (default 4, service policy 16).

When `aria2c` is installed, each file can also use bounded HTTP range
connections. `HERMES_DOWNLOAD_CONNECTION_BUDGET` is shared across active files
(default and service policy 32), with at most 16 connections assigned to one
file. If aria2 is unavailable or only one connection is assigned, the portable
curl continuation path is used. Model loading and functional qualification
remain serialized even though transfers are parallel.

## Runtime

User systemd units live in `~/.config/systemd/user/` and are copied here under `services/systemd/`. After clone or path changes:

```
systemctl --user daemon-reload
systemctl --user start llama-router.service
```

The router listens on `127.0.0.1:8080`.
