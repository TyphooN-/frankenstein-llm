# Candidate and capability status — 2026-09-03

This is the current operational reading. Dated 2026-09-02 documents remain
historical snapshots; where they disagree with this file, this file wins.

States remain distinct:

- **researched** — named and pinned in a review
- **downloaded** — exact size/SHA verified and promoted
- **policy-admitted** — non-inference candidate policy passed
- **functionally qualified** — a live gate or equivalent observed load, behavior, and unload

A later state never rewrites an earlier document. Benchmarking and tokens/sec
are still unauthorized.

## Downloads

Phases one through four are complete and verified. Transfer debris count was
zero at the last full scan.

Phase four installed:

- official Qwen3-Coder-Next Q4_K_M (`b82fb738`)
- Gemma-4-12B Heretic Q6_K plus BF16 mmproj (`efa14611`)
- UI-Mate-9B
- WeMM-Embedding-2B (`bbd6cd4b`)
- FLUX.2-klein-4B (native ComfyUI root plus encoder/tokenizer/VAE)
- Qwen3-ASR-1.7B already satisfied by phase one at `bcd2b5b7`

Phase three installed the Qwen Image Edit stack. Image editing is therefore
**downloaded, not missing**. It is not functionally proven until ComfyUI
workflows actually run.

## llama.cpp runtimes

Do not rebuild production llama.cpp for these candidates.

- Production: `/home/typhoon/.local/bin/llama-server` ->
  `/home/typhoon/src/llama.cpp/build/bin/llama-server`, HIP/ROCm gfx1030,
  commit `50f068f`. Already includes `--models-preset`, `draft-mtp`,
  `qwen3next`, `gemma4`, and Gemma 4 vision/audio projectors.
- Isolated GLM-only: worktree
  `/home/typhoon/src/llama.cpp/.claude/worktrees/glm53flash-local`,
  branch `pr27752-glm53flash`. Keep off `:8080`. The on-disk
  `build-rocm` / `build-rocm-shim` binaries report commit `7152e9bf2`,
  older than worktree HEAD `c9ddd6821`. Rebuild that isolated tree only
  if a GLM load fails; never replace production `50f068f` with it.

## Observed 2026-09-03 functional checks

These used the production router. They are not idle-host memory-fit evidence:
linux-tkg and cargo were active. No throughput was recorded.

- `gemma4-heretic`: exact `PONG`; no tools payload; unloaded.
- `gemma4-heretic-vision`: exact `Run Gate` on the screen fixture; metadata
  reports text/image/audio in, text out; unloaded. Audio and video were not
  tested.
- `qwen3-coder-next`: exact `PONG` and exact `lookup_ticket` /
  `{"ticket_id":4172}`; unloaded. Preset `tensor-split = 13,26,6` still
  spends display-GPU2 VRAM; bias GPU0+GPU1 on a later quiet-host pass.

The mission `router-models` failure at 23:15 was confounder admission
(`makepkg` started after `ridge` passed), not an architecture miss.

Candidate Transformers runtime: `venvs/candidates` on host Python 3.14.7,
transformers 5.5.4, host ROCm Torch 2.13.0 (`hip=7.2.53211`). Setup must
pass `--python 3.14`; a 3.11 venv cannot install the hashed lock.

## Still not functionally qualified

- UI-TARS grounding (interrupted historically); UI-Mate A/B after that
- WeMM live embeddings and a separate 2048-D index
- ComfyUI: Z-Image, ACE-Step, Qwen Image Edit, FLUX.2
- TTS -> ASR intelligibility
- GLM-5.3-Flash 32K on the isolated binary
- repository-agent A/B of Coder-Next vs `heretic` on real fixtures
- Gemma 4 audio/video modalities
- `obliterated` / `phr00ty` automatic tool selection (earlier clean-run miss)

Computer control remains **missing as a service**, not as a model. Do not
abliterate GUI/browser actors.

## Do not download next

No additional general chat, coding, OCR, embedding, image, or audio model is
justified while the weights above are unproven as product workflows.

Watch only: Qwen3.8-Flash-Next, VibeVoice-ASR-Streaming-1.5B, Fara1.5-4B,
Muse Glimmer. Skip Hy4 Preview, Lily (Apple Silicon), DeepSeek V4-class
150 GiB+ local quants, and Coder-Next abliterated PoCs.
