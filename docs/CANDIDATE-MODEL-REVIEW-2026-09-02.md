# Candidate model review — 2026-09-02

Status: **historical research snapshot from 2026-09-02.** Keep the matrix and
pins. Do not read the original “nothing downloaded” banner as current state.

Current operational status: `docs/CANDIDATE-STATUS-2026-09-03.md`.

This is the decision/backlog record for the 2026-09-02 candidate sweep. It sits
under `docs/LOCAL-AI-MODEL-STRATEGY.md`, which remains the portfolio decision
record; the coverage roadmap is `docs/LOCAL-HERMES-CAPABILITY-COVERAGE-PLAN.md`.

## What "adopt" means here

`ADOPT` in the table below means *promoted to the download/qualification
backlog*, not installed. Nothing enters a runtime queue
(`verification/local-coverage-foundation/download-queue*.json`) until exact
file-level size and SHA-256 metadata have been collected independently, the way
`research/collect_hf_metadata.py` and `research/build_download_queue.py` already
do it. The revision hashes below pin *what was reviewed*; they are not a
substitute for per-file pins.

Every retained candidate still has to pass the existing local gates — load,
coherence, structured output, workflow quality, memory safety, and clean unload
— before it is admitted. Publisher refusal rates and benchmark claims are not
admission proof.

## Hardware boundary applied

Unchanged from the strategy document: ROCm0 RX 6900 XT 16 GiB, ROCm1 Radeon Pro
V620 32 GiB, ROCm2 RX 6900 XT 16 GiB, ~64 GiB aggregate VRAM with no automatic
aggregation, ~94 GiB RAM, and one resident large model behind `--models-max 1`.
"It fits on disk" is not the test; KV cache and runtime overhead come out of the
same pool.

## Recommendation matrix

Links are pinned to the Hugging Face commit observed on 2026-09-02.

| # | Candidate (pinned revision) | Release / license / shape | Why it is a real gain | Compatibility | Verdict |
|---|---|---|---|---|---|
| 1 | [Qwen3-Coder-Next-GGUF `b82fb738`](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF/tree/b82fb7382639d97b38fa7672e526c760c2fb358e) | 2026-02-02, Apache-2.0. 80B MoE, 3B active, 262K context. Official Q4_K_M is ≈48.4 GB. | Clearest upgrade for the repository-agent lane: post-trained for agentic coding and tool calls. Q4_K_M uses GPU0+1 with bounded host/display-GPU spill as required by cache and runtime overhead. | Official llama.cpp/GGUF, Transformers, vLLM, SGLang. No ComfyUI role. | **ADOPT (official)** for repo agents. **Keep Qwen2.5 Coder for FIM** — do not replace the completion model with this. |
| 2 | [SC117/Gemma-4-12B-it-heretic-GGUF `efa14611`](https://huggingface.co/SC117/Gemma-4-12B-it-heretic-GGUF/tree/efa14611b0b04ab1ab1e38356596ac8d673a619a) | 2026-06-06 (refreshed 06-07), Apache-2.0 metadata. Dense 11.95B, 256K. Q4_K_M 7.38 GB, Q6_K 9.79 GB, Q8 12 GB. | The one genuinely distinct uncensored addition: native text/image/audio/video understanding rather than another Qwen chat merge. Heretic v2 ARA+LoRA, card reports KL 0.055 and 15/100 residual refusals — less censored, not "zero refusal". | GGUF/llama.cpp packaged; BF16 base in recent Transformers. Media paths need a Gemma-4-capable llama.cpp build and must be validated modality by modality. | **TRIAL** as the lightweight uncensored multimodal/audio preset. Keep it away from high-privilege tool execution. |
| 3 | [tencent/UI-Mate-9B `05dd5f29`](https://huggingface.co/tencent/UI-Mate-9B/tree/05dd5f2975195a5bb03d4363e8767f12158c8421) | 2026-08-14, Apache-2.0. Qwen3.5-based 9B multimodal GUI policy, BF16 ≈18.82 GB — a good V620 fit. | Long-horizon native desktop interaction, structured mouse/keyboard actions, in-context demonstrations. Directly challenges the already-downloaded UI-TARS-1.5-7B. | Official Transformers, vLLM, SGLang with the official prompt/parser/harness. Community GGUFs exist but llama.cpp is not the reference action path. | **A/B AGAINST UI-TARS.** Do **not** abliterate a GUI actor — action authorization and confirmations are a safety boundary, not friction. |
| 4 | [tencent/WeMM-Embedding-2B `bbd6cd4b`](https://huggingface.co/tencent/WeMM-Embedding-2B/tree/bbd6cd4bf52cfc6716f752a2df80b2706720bd95) | 2026-08-25. This refreshed pin includes an explicit Apache-2.0 `LICENSE`; its only change from reviewed `df8094e5` is `README.md`, and runtime objects are byte-identical. 2.72B, ≈5.44 GB BF16, 2048-D Matryoshka. | Joint retrieval over text, images, video, visual documents and interleaved inputs — a capability the Qwen3 text embedder does not have. | Transformers 5.2 with remote code, SentenceTransformers, vLLM pooling, SGLang. No verified llama.cpp path. | **ADOPT** if screenshot/document/image/video retrieval is wanted. Build a **separate** index: vector spaces are not interchangeable, so keep Qwen3 embedding/reranker for text-only. |
| 5 | [black-forest-labs/FLUX.2-klein-4B `e7b7dc27`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) | 2026-01-14, Apache-2.0. 3.88B rectified flow; card targets ≈13 GB VRAM and four-step generation. | Unified text generation, image editing and multi-reference editing at low latency. Complements Z-Image Turbo rather than replacing it, and speaks directly to the open image-editing gap. | Native ComfyUI and Diffusers. Not llama.cpp. | **ADOPT** as the fast editing / multi-reference lane; retain Z-Image for its established quality lane. An "uncensored text encoder" repack is not worth adopting without a prompt-refusal test showing a real limitation. |
| 6 | [Qwen/Qwen3-ASR-1.7B-hf `bcd2b5b7`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/tree/bcd2b5b7f32b480ab5790554cfa8347f246a14f3) | 2026-06-26, Apache-2.0. 2.04B, ≈4.08 GB BF16. | Useful refresh *if* the installed ASR lacks streaming/offline unification, hotword/context prompting, singing/BGM handling, or the documented 52 languages. Pair `Qwen3-ForcedAligner-0.6B-hf` only when timestamps are needed. | Native Transformers ≥5.13. Community GGUFs exist; the supported path is Transformers. | **WATCH** — adopt only after a WER/RTF comparison against the installed ASR on the real microphones and accents. |
| 7 | [mistralai/Mistral-Small-4-119B-2603 `a11f36be`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/tree/a11f36bebf709121056b1dbcc943d1c6afbe494d) · [Huihui abliterated-v2](https://huggingface.co/huihui-ai/Huihui-Mistral-Small-4-119B-2603-BF16-abliterated-v2-GGUF) | 2026-01-23 (card updated July), Apache-2.0. 119B MoE, 6.5B active, 256K, vision + tool use. IQ4 GGUFs ≈58–59 GB. | Possible fast-MoE alternate with integrated reasoning/vision. Cannot sit entirely on GPU0+1: ~10+ GB host spill before cache. The abliterated v2 touches every layer but the first and publishes no KL, perplexity or task-retention evidence. | Base: llama.cpp GGUF, vLLM, current Mistral/Transformers. Abliterated repo ships split GGUF and its own tool-call template. | **WATCH.** Not a priority over Qwen3.8 plus Coder-Next; adopt only if a real agent benchmark offsets the memory and provenance cost. |
| 8 | [microsoft/Fara1.5-4B `776a33ae`](https://huggingface.co/microsoft/Fara1.5-4B/tree/776a33ae5b2ad503796a97ae20fdc66f61d2feea) | 2026-07-17, MIT. 4.54B, 262K, ≈9.08 GB BF16. | Strong compact **browser-only** computer-use agent: screenshots in, XML tool calls out. Smaller than UI-Mate but narrower. | Transformers ≥5.2, vLLM ≥0.19.1, Microsoft Fara/MagenticLite harness. llama.cpp is not the reference deployment. | **WATCH / browser fallback.** Prioritise UI-Mate for desktop-wide automation. No uncensored variant is desirable for an actor touching live websites. |
| 9 | [microsoft/VibeVoice-ASR-Streaming-1.5B `94efa5c0`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B/tree/94efa5c0d363b47fb552095c8e81edc1c88f1e45) | Released 2026-09-02, MIT. Named 1.5B but Hub safetensors metadata reports 2.814B — an unresolved packaging/parameter discrepancy. | Speaker-attributed streaming transcription with hotwords is valuable for meetings, but this is a day-zero release with no download history and ten languages. | Microsoft's custom VibeVoice code is the documented path; no verified llama.cpp or ComfyUI support. | **WATCH** until packaging, latency, diarization stability and ROCm support settle. |
| 10 | [Qwen/Qwen-AgentWorld-35B-A3B `60d2b043`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B/tree/60d2b0434a53d2e62a7c00a489586815d94ebffb) | 2026-06-22, Apache-2.0. 35B/3B active, 262K. | Frequently mistaken for an agent checkpoint. It predicts the **next environment state**: a world simulator/evaluator, not a better repository or desktop actor. Card recommends ≥128K context. | Transformers, vLLM, SGLang (vLLM needs `--language-model-only`). | **SKIP for serving.** Consider only for synthetic agent environments or rollout evaluation. |
| 11 | [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp `6821d6ad`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tree/6821d6ad3681a4b137b066b76094fa82ebd0a380) · [day-zero uncensored GGUF `e262f88e`](https://huggingface.co/orcarouter/DeepSeek-V4-Flash-Vision-Uncensored-GGUF/tree/e262f88e8b1fdd390a2c0cad2af436dce4c8fa0e) | 2026-08-31 / 2026-09-02, MIT. 304.65B parameters; idealised 4-bit weights alone ≈152.3 GB. | Exceeds 96 GiB before runtime overhead. The uncensored conversion appeared 2026-09-02 with no independent validation; experimental vision plus fresh abliteration compounds failure risk. | Not a practical Transformers/llama.cpp/ComfyUI deployment on this host. | **SKIP** on this hardware. |

## Backlog, in order

1. **Qwen3-Coder-Next (official)** — phase four pins the official Q4_K_M shards;
   A/B against `heretic` on repository tasks after verification. Keep the
   Qwen2.5 Coder FIM model in place regardless of the outcome.
2. **UI-Mate-9B vs UI-TARS-1.5-7B** — blocked until the
   `computer-use-grounding` gate produces a verdict at all. Per
   `verification/local-coverage-foundation/docs/GAP-DECISIONS-2026-09-02.md`,
   that gate was killed mid-run and never finished, so there is no baseline to
   A/B against yet.
3. **FLUX.2-klein-4B** — phase four installs the root checkpoint plus the
   official encoder, tokenizer, and VAE while excluding the duplicate Diffusers
   transformer packaging. It still requires a real ComfyUI generation/edit gate.
4. **WeMM-Embedding-2B** — the explicit Apache-2.0 license is resolved at the
   refreshed pin. Qualification must use a separate multimodal index and audit
   the pinned remote-code implementation before execution.
5. **Gemma-4-12B Heretic** — production `50f068f` already loads the text and
   vision presets. Remaining work is audio/video, quiet-host memory-fit, and
   keeping it low-privilege. Do not rebuild llama.cpp for this candidate.
6. **Qwen3-ASR-1.7B** — already downloaded and verified at this reviewed
   revision by phase one; only functional comparison remains.

## Standing rules this sweep reinforces

- **Do not abliterate a privileged actor.** GUI and browser agents take real
  actions; refusal and confirmation behaviour is part of the safety boundary,
  not friction to remove.
- **Skip day-zero "uncensored" uploads** that ship a label and no measurements,
  especially for large or action-taking models.
- **Prefer measured ablations.** Gemma-4 Heretic is preferred over the Huihui
  Qwen3-Coder-Next and Mistral-Small-4 derivatives because it publishes KL and
  residual-refusal numbers; Huihui's own card calls its method a "crude,
  proof-of-concept" refusal-direction removal.
- **Size is not fit.** Anything that cannot sit on GPU0+1 with room for KV cache
  is a host-spill decision, not a download decision.

## Collection caveats

- All revisions and sizes are as observed on 2026-09-02 and must be re-checked
  before any download; publishers re-upload and re-quantize.
- Metadata access to the gated Huihui Mistral repository returned HTTP 401
  through the Hub API, so its ablation details come from the publicly rendered
  model card and its exact quant file sizes could not be independently
  enumerated.
- Community-reported metrics (Heretic KL/refusal counts, publisher benchmark
  tables) are directional only. Local A/B evidence is what promotes a model.

## Sources

- Qwen3-Coder-Next: https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF
- Huihui Qwen3-Coder-Next abliterated (BF16): https://huggingface.co/huihui-ai/Huihui-Qwen3-Coder-Next-abliterated
- Bartowski GGUF of the above: https://huggingface.co/bartowski/huihui-ai_Qwen3-Coder-Next-abliterated-GGUF
- Gemma-4-12B Heretic GGUF: https://huggingface.co/SC117/Gemma-4-12B-it-heretic-GGUF
- UI-Mate-9B: https://huggingface.co/tencent/UI-Mate-9B
- WeMM-Embedding-2B: https://huggingface.co/tencent/WeMM-Embedding-2B
- FLUX.2-klein-4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Qwen3-ASR-1.7B: https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf
- Qwen3-ForcedAligner-0.6B: https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf
- Mistral-Small-4-119B-2603: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
- Huihui Mistral-Small-4 abliterated v2 (gated): https://huggingface.co/huihui-ai/Huihui-Mistral-Small-4-119B-2603-BF16-abliterated-v2-GGUF
- Microsoft Fara1.5-4B: https://huggingface.co/microsoft/Fara1.5-4B
- Microsoft VibeVoice-ASR-Streaming-1.5B: https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B
- Qwen-AgentWorld-35B-A3B: https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B
- DeepSeek-V4-Flash-Vision-Exp: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
- OrcaRouter DeepSeek-V4-Flash-Vision uncensored GGUF: https://huggingface.co/orcarouter/DeepSeek-V4-Flash-Vision-Uncensored-GGUF
