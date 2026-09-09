# 0002. Serialized functional qualification, no benchmarks until kernel confirmation

- Status: accepted
- Date: 2026-09-02

## Policy update

The operator now authorizes passive tok/s and modality-appropriate observations
during existing qualification runs. This supersedes the original no-timing rule
below, not host-admission or correctness requirements. See
[passive performance](../reference/QUALIFICATION-PERFORMANCE.md).

## Context

The host is an X99 workstation with three AMD GPUs and user-owned kernel compilation. Functional load/coherence/unload checks are authorized. Tokens/sec, throughput rankings, and other benchmark measurements are not authorized until the user confirms a reboot into the intended kernel.

Concurrent downloads, ComfyUI, llama.cpp loads, and kernel builds confound memory-fit evidence.

## Decision

- Qualify models one at a time through the llama.cpp router with explicit load and unload.
- Record architecture/template, usable context, GPU allocation, memory fit, representative behavior, and clean unload.
- Refuse confounded runs when download queues, makepkg, or other heavy writers are active.
- Record passive performance observations from existing inference; do not add benchmark work or make speed an acceptance criterion.
- Download additional weights only to close a missing use case or materially improve an existing one.

## Consequences

- The functional qualification supervisor waits for verified artifact queues and a quiet host.
- UI-TARS grounding remains a prerequisite for end-to-end computer control.
- TTS admission requires an ASR intelligibility round-trip, not merely an emitted audio file.
