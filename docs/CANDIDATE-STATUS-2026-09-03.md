# Candidate and capability status — 2026-09-03

> **Historical snapshot from 2026-09-03.** Keep the hand-check notes. Do not
> read this file as current host state. Current operational status:
> [CANDIDATE-STATUS-2026-09-06.md](CANDIDATE-STATUS-2026-09-06.md).

Dated 2026-09-02 documents remain earlier snapshots.

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

- Production source: tracked submodule `upstream/llama.cpp`, tag v0.4.0,
  commit `5266f24`, with a HIP/ROCm `gfx1030` build produced by
  `scripts/build-llama-cpp.sh`. Services execute its build directly; there is
  no required checkout under `/home/typhoon/src` or runtime symlink under
  `~/.local/bin`.
- GLM-5.3-Flash remains experimental. v0.4.0 has `qwen4exp` but no
  `glm5next` architecture identifier. Any renewed GLM work must use an isolated
  worktree below the tracked submodule and must never replace the production
  v0.4.0 gitlink or bind the production port.
- Not yet installed as of 2026-09-04. The tracked units point at the submodule
  build, but `~/.config/systemd/user/llama-router.service` still execs
  `~/.local/bin/llama-server`, whose symlink target went away with
  `/home/typhoon/src`. The unit therefore fails with `status=203/EXEC` and
  restarts on a five-second timer; the hand-started server that was answering
  `:8080` has since exited, so nothing is listening there now. The tracked
  binaries exist and report `0.4.0-dev (build 10809, commit 5266f24da)`.
  Installing the units as `README.md` describes is the remaining runtime step,
  and it must not be done while a build is active.

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

The qualification `router-models` failure at 23:15 was confounder admission
(`makepkg` started after `ridge` passed), not an architecture miss.

Candidate Transformers runtime: `venvs/candidates` on host Python 3.14.7,
transformers 5.5.4, host ROCm Torch 2.13.0 (`hip=7.2.53211`). Setup must
pass `--python 3.14`; a 3.11 venv cannot install the hashed lock.

## Still not functionally qualified

- UI-TARS grounding (interrupted historically); UI-Mate A/B after that
- WeMM live embeddings and a separate 2048-D index
  Text-only WeMM gate passed 2026-09-03 on cuda:1 (2048-D, semantic order).
  Image/video not claimed (no ROCm torchvision). Index file not built yet.
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
