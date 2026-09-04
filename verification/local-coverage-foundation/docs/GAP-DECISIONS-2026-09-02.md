# Gap decisions and phase-3 verdict — 2026-09-02

**Historical.** After this snapshot, phase three installed Qwen Image Edit and
phase four installed FLUX.2-klein-4B, UI-Mate-9B, WeMM, Coder-Next, and Gemma-4
Heretic. Image editing is no longer a missing-weight gap; it is an unproven
workflow. Current status: `docs/CANDIDATE-STATUS-2026-09-03.md`.

## Verdict

**No phase-3 download queue was created.** The precondition was that existing
pinned metadata *conclusively establishes* a missing, non-duplicative artifact.
It does not. Creating a queue anyway would mean inventing a pin from a guess,
which is the failure mode the precondition exists to prevent.

Nothing was downloaded, and no queue, runner, or lock file was added.

## How that was checked

`research/hf/` holds 19 collected metadata files. Nine describe repositories that
are not in either download queue. Every one of them is duplicative:

| Pinned but not downloaded | Why it is duplicative |
|---|---|
| `Comfy-Org/z_image` | Another **generation** checkpoint (`z_image_bf16`), not an editor. Its `qwen_3_4b.safetensors` and `ae.safetensors` are byte-identical (same SHA-256) to files already on disk. |
| `Qwen/Qwen3-TTS-Tokenizer-12Hz` | Its `model.safetensors` is 682,293,092 bytes — the exact file already vendored at `Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer/model.safetensors`. |
| `tencent/HunyuanOCR` | Source weights for the OCR model already downloaded as `ggml-org/HunyuanOCR-GGUF`. |
| `Qwen/Qwen3-ASR-0.6B-hf` | Smaller tier of the admitted ASR model. |
| `Qwen/Qwen3-Embedding-0.6B-GGUF`, `Qwen/Qwen3-Embedding-4B-GGUF` | Smaller tiers of the admitted embedding model. |
| `Qwen/Qwen3-Reranker-4B` | Smaller tier of the admitted reranker. |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Voice variant of a TTS model whose base gate has not run yet. |
| `ggml-org/Qwen2.5-Coder-3B-Instruct-Q8_0-GGUF` | Smaller tier of the admitted FIM model. |

A search across `research/` and both queues for `kontext`, `image-edit`,
`image_edit`, `inpaint`, and `qwen-image` returns nothing. There is no pinned
image-editing artifact to queue.

## Gap 1 — image editing

**Z-Image generation is not image editing.** Z-Image Turbo is a text-to-image
generation checkpoint: prompt in, new image out. Editing is instruction-following
modification of an existing image, and the coverage plan already treats them as
two steps ("one image workflow, then one editing workflow", Phase E).

Nothing installed covers editing, and no candidate is pinned.

### The decision required before anything is added

Two steps, in order. Neither is a download.

1. **Functional, not research.** Determine whether a ComfyUI img2img / inpaint
   graph over the already-present Z-Image Turbo checkpoint meets the actual
   editing need. This costs nothing to try and may close the gap with zero new
   bytes. It cannot be answered statically, and it cannot be attempted until the
   image-generation gate itself has passed — which it has not.
2. **Only if step 1 fails.** Run a metadata-collection pass for a dedicated
   instruction-following editor and pin exact repo, revision, files, sizes, and
   SHA-256 before proposing any queue. The 2026-09-02 candidate sweep names
   FLUX.2-klein-4B as the leading editor candidate for that pass
   (`docs/CANDIDATE-MODEL-REVIEW-2026-09-02.md`); it is a *starting point for
   metadata collection*, not a pin, and step 1 still comes first.

Budget constraint to carry into step 2: the Z-Image UNet is 12.3 GB bf16 and its
text encoder 8.0 GB. A second editor family adds roughly another 20 GB competing
for the same 32 GiB V620, so "it fits on disk" is not the test.

## Gap 2 — end-to-end computer control

**UI-TARS grounding is not a computer-control service.** UI-TARS-1.5-7B is a
grounding *model*, and it is already downloaded and integrity-checked. What is
absent is the end-to-end screenshot → action → state-readback control *service*.
That is an application and a scaffold decision, not a model download; the
upstream UI-TARS Desktop referenced in the coverage plan is a GitHub application,
not a Hugging Face artifact.

### The decision required before anything is added

**Re-run the `computer-use-grounding` gate to completion first.** It was killed
mid-`grounding` on the previous boot and never produced a verdict, so the
question "do we need an end-to-end control service on top of this model" is not
yet answerable — we do not know whether the grounding underneath it works.

Note also that the Qwen3.8 vision gate passed and recorded
`conditional_ui_tars_required: false` for plain coordinate pointing. UI-TARS was
downloaded to cover typed GUI action selection, which is a different claim and is
exactly the part of its gate that never ran.

If the gate passes, the follow-on is a bounded scaffold in the shape of the
existing repository-agent and Strix scaffolds — deny-by-default, state readback
after every action, screen text treated as untrusted data — not a new model.

A separate 2026-09-02 sweep does name a candidate *model* worth A/B-testing
against UI-TARS for typed GUI action selection — Tencent UI-Mate-9B, see
`docs/CANDIDATE-MODEL-REVIEW-2026-09-02.md`. That comparison is blocked behind
the same precondition: without a grounding verdict there is no baseline to
compare against. That review also records the standing rule that a GUI actor is
never abliterated.

## Gaps deliberately not pursued

- **Secondary sound-effect model.** The coverage plan defers it until after image
  generation is stable. Image generation has not run. Not a gap today.
- **TTS CustomVoice.** Duplicative until the Base gate has produced a verdict.
- **Smaller model tiers.** Every one is a latency/quality tradeoff against a
  model that already passed its gate. Downloading one is a benchmarking decision
  that needs a measured reason, and no such measurement exists.
