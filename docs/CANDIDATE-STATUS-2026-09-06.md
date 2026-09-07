# Candidate and capability status — 2026-09-06

This is the current operational reading. Dated 2026-09-02 and 2026-09-03
documents remain historical snapshots; where they disagree with this file, this
file wins. The coverage roadmap is
[LOCAL-HERMES-CAPABILITY-COVERAGE-PLAN.md](LOCAL-HERMES-CAPABILITY-COVERAGE-PLAN.md).
Candidate research that is not admitted lives in
[reference/CANDIDATE-RESEARCH-CLOSEOUT.md](reference/CANDIDATE-RESEARCH-CLOSEOUT.md).

States remain distinct:

- **researched** — named and pinned in a review
- **downloaded** — queued exact size (and any recorded SHA-256) verified and promoted
- **policy-admitted** — non-inference candidate policy passed
- **functionally qualified** — a live gate observed load, behaviour, and unload

A later state never rewrites an earlier document. Comparative tokens/sec remain
unauthorized until the user explicitly authorizes performance testing on the
intended kernel. Qualitative characterization (`verification/qualitative-characterization`)
saves inspectable outputs; it is not a numeric scoring harness.

## How to refresh

```bash
python3 verification/local-coverage-foundation/build_capability_ledger.py --print-only
```

The reading below was taken 2026-09-06. It loaded no model and measured no
throughput. SHA-256 was not re-hashed; presence and exact size matched the
pinned queues. `sha256_reverified` is false.

## Downloads

Phases one through four are complete by size. Transfer debris was last reported
zero; this file does not repeat that scan.

Installed and size-complete (not all qualified):

- core chat GGUFs: Ridge, RVN/`heretic`, OBLITERATED plus BF16 projector, Fable, Phr00ty
- Qwen3-Embedding-8B Q6_K, Qwen3-Reranker-8B source, HunyuanOCR BF16 + mmproj
- Qwen2.5-Coder-7B FIM Q8_0
- Qwen3-ASR-1.7B (`bcd2b5b7`), Qwen3-TTS-12Hz-1.7B-Base
- Z-Image Turbo, ACE-Step 1.5, Qwen Image Edit 2511 stack, FLUX.2-klein-4B
- Qwen3-Coder-Next Q4_K_M (`b82fb738`)
- Gemma-4-12B Heretic Q6_K plus BF16 mmproj (`efa14611`)
- UI-TARS-1.5-7B and UI-Mate-9B
- WeMM-Embedding-2B (`bbd6cd4b`)

Research-only candidates from the 2026-09-06 closeout are **not** in these
queues and must not be treated as downloaded.

## llama.cpp runtime

- Production source: submodule `upstream/llama.cpp`, tag v0.4.0, commit
  `5266f24`. HIP/ROCm `gfx1030` build via `scripts/build-llama-cpp.sh`.
- Binary reports `0.4.0-dev` (build 10809, commit `5266f24da`).
- Presets in `llama-models.ini`: `ridge`, `heretic`, `obliterated`,
  `obliterated-vision`, `fable`, `phr00ty`, `qwen3-embedding-8b`,
  `qwen3-reranker-8b`, `qwen25-coder-7b-fim`, `qwen3-coder-next`,
  `gemma4-heretic`, `gemma4-heretic-vision`.
- GLM-5.3-Flash remains experimental: v0.4.0 has `qwen4exp` but no `glm5next`.
  Any GLM work uses an isolated worktree and never replaces the production
  gitlink or production port.
- Units in `services/systemd/` install to `~/.config/systemd/user/`. Whether
  those units are currently running is host state; use
  `scripts/local-model-status.sh` / `systemctl --user status llama-router.service`.
  A dated note that the unit still pointed at a missing `~/.local/bin/llama-server`
  is not this file's claim.

## Ledger states (2026-09-06)

Functionally qualified:

- `asr`
- `multimodal-embeddings` (WeMM **text-only**; image/video not claimed)
- `native-tool-use`
- `rag` (structural, behavioural, and live artifacts all present)
- `vision-grounding`

Evidence stale (2026-09-01 gates older than later weight installs):

- `embeddings`, `fim`, `ocr`, `reranking`

Evidence interrupted:

- `computer-use-grounding` (UI-TARS gate cut by SIGTERM; UI-Mate A/B not started)

Downloaded, no passing functional artifact:

- `image`, `image-editing`, `music` (shared `media-functional.json` absent)
- `image-generation-editing` (FLUX.2 Klein; no evidence artifact declared)
- `tts` (`gate-tts.json` absent)
- `repository-agent` (both heretic and Coder-Next gate artifacts absent)
- `uncensored-multimodal` (Gemma-4; no durable gate; 2026-09-03 hand checks only)

Hand checks from 2026-09-03 (confounded host, not idle-host memory-fit, no
throughput, no durable Gemma gate):

- `gemma4-heretic`: exact `PONG`; no tools payload; unloaded
- `gemma4-heretic-vision`: exact `Run Gate` on the screen fixture; audio/video
  untested
- `qwen3-coder-next`: exact `PONG` and `lookup_ticket` / `{"ticket_id":4172}`;
  unloaded. Those replies are not the repository-agent gate.

## Still not product-complete

- Re-run stale embedding/FIM/OCR/reranker gates against current bytes
- Finish UI-TARS grounding, then UI-Mate A/B
- Computer-control **service** (screenshot → action → state readback)
- ComfyUI workflows: Z-Image, ACE-Step, Qwen Image Edit, FLUX.2
- TTS → ASR intelligibility
- Repository-agent A/B of Coder-Next vs `heretic` on real fixtures
- Gemma-4 audio/video; durable Gemma text/vision gate
- WeMM image/video (no ROCm torchvision) and the 2048-D index file
- GLM-5.3-Flash 32K on an isolated binary, idle host
- Live three-GPU placement qualification (ADR 0006: computed, not exercised)
- Isolated evaluation of researched upgrades (122B/35B/Omega/Flash-Next/Ling)
  only after the downloaded workflows above, on a healthy idle host

Do not download another general chat, coding, OCR, embedding, image, or audio
model while those weights remain unproven as product workflows.

## Hardware (observed 2026-09-06, not idle-host certification)

Headless RX 6900 XT 16,368 MiB, V620 30,704 MiB, display RX 6900 XT 16,368 MiB.
System RAM 101,109,370,880 bytes. Placement still prefers the two 6900 XTs when
a model fits, with V620 for remainder, per model from measured facts.
