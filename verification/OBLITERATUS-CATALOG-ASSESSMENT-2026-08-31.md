# Pliny OBLITERATUS / OBLITERATED Hugging Face Catalog Assessment

Assessment date: 2026-08-31

Evidence source: `obliteratus-hf-inventory-2026-08-31-live.json`

Inventory integrity: 11 live IDs, 11 union IDs, 11 records, 0 collection errors. Repository revisions, model metadata, trees, README files, and OBLITERATUS metadata were captured live. Publisher benchmarks and generated `abliteration_metadata.json` scores are evidence of publisher claims, not independent local validation.

## Decision summary

Do not bulk-download this catalog. The installed `OBLITERATUS/Qwen3.8-27B-OBLITERATED` Q6_K already represents the strongest, newest, broadest OBLITERATUS role and overlaps most of the catalog.

Immediate action:

1. Keep and functionally validate the installed Qwen3.8 27B Q6_K.
2. Download its matching 931,145,888-byte BF16 projector and validate vision, structured/tool output, coding, authorized security prompts, MTP, long-context behavior, and clean unload.
3. Do not download the Qwen3.6 27B, older 7B/8B text models, GPT-2 XL, or the damaged-looking Gemma E4B model.
4. Retain Ornith 1.5 9B Q6_K plus projector as the only near-term conditional addition: it could be a fast, low-memory multimodal/agentic worker if Qwen3.8 is too slow to keep responsive or cannot satisfy computer-use vision needs.
5. Retain the structured-output Qwen3 4B as a conditional conversion/test candidate only if a dedicated low-latency JSON/tool router is operationally valuable. It has no publisher GGUF.
6. Retain Gemma 4 12B only as a conditional cross-family multimodal/reviewer candidate. It does not displace Qwen3.8 and its repository exposes no separately named projector.

## Hardware interpretation

Current machine: approximately 94.16 GiB RAM and 61.55 GiB aggregate VRAM (V620 32 GiB plus two RX 6900 XT 16 GiB). Planned machine: 3–4 V620 GPUs, approximately 96–128 GiB aggregate V620 VRAM.

All published GGUFs in this catalog fit the current aggregate VRAM by file size. The 4B–12B quants fit one V620; the 27B Q6_K files fit one 32 GiB V620 with room for runtime/KV at practical contexts. Fit is not admission: one-resident-model serving means every duplicate adds storage and test/maintenance cost without adding simultaneous capacity.

## Model-by-model assessment

### 1. OBLITERATUS/DeepSeek-R1-Distill-Llama-8B-OBLITERATED

- Revision: `24fc7e6b8ffc8020e889fc87f49c3bf3104a1334`
- Provenance: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Architecture: `LlamaForCausalLM`
- Artifacts: safetensors only, approximately 14.96 GiB; no publisher GGUF or projector.
- Publisher metadata: refusal rate 0.0, perplexity 26.80, coherence 0.0 from only 5 harmful and 5 harmless prompts.
- Fit: weights fit one V620 after conversion/quantization.
- Overlap: weaker, older reasoning role already covered by the installed 27B models and the pending GLM specialist.
- Decision: reject. Coherence 0.0 is a hard warning, evidence volume is tiny, and conversion work would buy no distinct workflow.

### 2. OBLITERATUS/Gemma-4-12B-OBLITERATED

- Revision: `f81b0cbd28a3650138635823bc101adb56a0bc4a`
- Provenance: `google/gemma-4-12B-it`
- Architecture: `Gemma4UnifiedForConditionalGeneration`; repository tags it image-to-text/text-generation.
- Artifacts: BF16 GGUF 23,832,065,088 bytes; Q4_K_M 7,381,382,208 bytes; v2 Q8_0 12,669,645,888 bytes; safetensors approximately 22.28 GiB. No separately named `mmproj` artifact in the captured tree.
- Publisher claim: 0/842 refusals and 46/70 MMLU-Pro, equal to its stock comparison. These results are not locally reproduced.
- Fit: Q4/Q8 easily fit one V620.
- Overlap: broad multimodal assistant/reviewer overlaps installed Qwen3.8 plus projector.
- Possible distinct role: cross-family Gemma reviewer or fallback multimodal model, reducing correlated Qwen failure modes.
- Decision: defer. Consider only after Qwen3.8 multimodal proof, and only if an independent cross-family reviewer is worth another runtime/configuration path.

### 3. OBLITERATUS/Mistral-7B-v0.3-OBLITERATED

- Revision: `0d433cc423d6de6e481e74f5c42f8c087be257bf`
- Provenance: `mistralai/Mistral-7B-v0.3`
- Architecture: `MistralForCausalLM`
- Artifacts: safetensors only, approximately 13.50 GiB; no publisher GGUF or projector.
- Publisher metadata: perplexity 3.76, coherence 1.0, refusal rate 3.33%, one degenerate sample, based on 33+33 prompts.
- Fit: trivial after conversion/quantization.
- Overlap: superseded small text assistant; weaker than current 27B daily/security models.
- Decision: reject. It adds neither multimodality, modern agent behavior, structured-output specialization, nor repository-scale coding capability.

### 4. OBLITERATUS/Ornith-1.5-9B-OBLITERATED

- Revision: `82df4702cc5e716cf7573a8942fee28ae8f65088`
- Provenance: `ornith-ai/Ornith-1.5-9B`
- Architecture: `Qwen3_5ForConditionalGeneration`, with image/video-aware chat template.
- Artifacts: Q2_K 3,914,968,352; Q3_K_M 4,737,608,992; IQ4_XS 5,357,874,464; Q4_K_M 5,780,090,144; Q5_K_M 6,642,543,904; Q6_K 7,558,901,024; Q8_0 9,786,060,064; BF16 18,407,320,864 bytes. BF16 projector: 921,704,384 bytes.
- Publisher claim: 15/16 liberation, 6/6 cyber, 2/2 capability; MMLU sample falls from 78.82% stock to 74.82%; long-context coherence 5/6 versus stock 4/6. This is a small publisher test, not local proof.
- Fit: Q6_K plus projector fits comfortably on one V620 and could remain responsive.
- Overlap: same broad text/vision/agent class as installed Qwen3.8, but materially smaller.
- Possible distinct role: low-latency multimodal GUI/OCR/browser worker that leaves more VRAM free than 27B.
- Decision: conditional shortlist. Download Q6_K plus projector only if Qwen3.8 vision latency/residency is unsuitable or a dedicated fast computer-use worker is needed. This is the strongest non-duplicate candidate in the catalog.

### 5. OBLITERATUS/Qwen2.5-Coder-7B-Instruct-OBLITERATED

- Revision: `51b8a8f30c58846c6898d2126a9b8add136c0442`
- Provenance: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Architecture: `Qwen2ForCausalLM`
- Artifacts: safetensors only, approximately 14.19 GiB; no publisher GGUF.
- Publisher metadata: perplexity 3.02, coherence 1.0, refusal rate 3.33% on 33+33 prompts.
- Fit: easy after conversion.
- Overlap: older 7B coding model is dominated by installed Qwen3.8/Ridge/Fable and does not establish a modern repository-agent win.
- Decision: reject. A dedicated local autocomplete model should be selected on measured fill-in-the-middle latency/quality, not merely uncensored coding provenance.

### 6. OBLITERATUS/Qwen3-4B-OBLITERATED

- Revision: `2b3b4f58e8a9659f6640c9f7b859ca076b8ec55d`
- Provenance: `Qwen/Qwen3-4B`
- Architecture: `Qwen3ForCausalLM`
- Artifacts: safetensors only, approximately 7.49 GiB; no publisher GGUF.
- Publisher metadata: perplexity 9.91, coherence 1.0, refusal rate 20%.
- Fit: easy after conversion/quantization.
- Overlap: small daily chat/tool role, but this generic variant is inferior to the structured-output sibling for a distinct fast-router use case.
- Decision: reject. The 20% reported refusal rate also means the stated transformation did not achieve its core goal reliably.

### 7. OBLITERATUS/Qwen3.6-27B-OBLITERATED

- Revision: `b85c6304150be44ee02ab130a733f65bc562bbc6`
- Provenance: `Qwen/Qwen3.6-27B`
- Architecture: `Qwen3_5ForCausalLM`
- Artifacts: Q4_K_M 16,547,399,872; Q5_K_M 19,231,099,072; Q6_K 22,082,529,472; Q8_0 28,595,763,392 bytes; safetensors approximately 50.10 GiB. No projector in the captured tree.
- Publisher metadata: perplexity 3.85, coherence 1.0, refusal rate 0.0, four degenerate samples, KL 0.107, spectral status RED. README claims 95.84% non-refusal and 93.94% quality pass over 842 prompts.
- Fit: every GGUF fits one V620, including Q8_0 by file size.
- Overlap: directly superseded by installed Qwen3.8 27B Q6_K, which also has a matching vision projector and newer capability evidence.
- Decision: reject as redundant. Do not retain both generations without a measured workflow where 3.6 wins.

### 8. OBLITERATUS/Qwen3.8-27B-OBLITERATED

- Revision: `a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8`
- Provenance: `Qwen/Qwen3.8-27B`
- Architecture: `Qwen3_5ForConditionalGeneration`, multimodal.
- Artifacts: Q2_K 10,864,583,712; Q3_K_M 13,500,728,352; IQ4_XS 15,420,441,632; Q4_K_M 16,810,705,952; Q5_K_M 19,535,692,832; Q6_K 22,430,991,392; Q8_0 29,047,075,872 bytes. BF16 projector: 931,145,888 bytes.
- Local state: Q6_K is installed at `/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-Q6_K.gguf`; verified SHA-256 `3535d4a15b75840fb391138ce04e6c73fde709320c6f5fe788cdbd586ee08e3a`; served as alias `obliterated`.
- Publisher metadata: perplexity 4.26, coherence/capability 1.0, six listed capability checks true, refusal rate 0.0, KL 0.962, spectral status RED. README reports MMLU 82.3% versus 84.5% stock and recommends greedy decoding, repetition penalty 1.15, empty system prompt, and bundled Jinja template.
- Fit: Q6_K fits one V620 with room for projector and practical KV/cache allocations.
- Distinct role: strongest installed unrestricted generalist for authorized security, difficult tool use, multimodal work, and non-refusing local workflows.
- Decision: keep and fully validate. This is the catalog representative; no second 27B OBLITERATUS download is justified before its full test suite completes.

### 9. OBLITERATUS/gemma-4-E4B-it-OBLITERATED

- Revision: `d8678bbb9e0d4f5729c115087485a4e25ba89d65`
- Provenance: `google/gemma-4-E4B-it`
- Architecture: `Gemma4ForConditionalGeneration`, multimodal.
- Artifacts: Q4_K_M 5,335,289,792; Q5_K_M 5,762,912,192; Q8_0 8,031,240,128 bytes. F16 projector: 990,372,288 bytes. Safetensors approximately 14.89 GiB.
- Publisher conflict: README/eval files claim 0% hard refusal and 97.5% compliance, but `abliteration_metadata.json` reports perplexity 31,539.25, coherence 0.1, KL 12.65, and spectral status RED. The repository itself warns of earlier architecture/tool breakage.
- Fit: easily fits one V620.
- Overlap: fast multimodal role overlaps Ornith 9B and Qwen3.8.
- Decision: reject. The internal quality diagnostics are too alarming to justify local admission despite positive README claims.

### 10. OBLITERATUS/gpt2-xl-OBLITERATED

- Revision: `2d81b0605c0dbf919253c8f2fa120954f0e6de59`
- Provenance: `openai-community/gpt2-xl`
- Architecture: `GPT2LMHeadModel`
- Artifacts: safetensors only, approximately 2.90 GiB; no publisher GGUF.
- Publisher metadata: perplexity 9.56, coherence 0.9, refusal rate 0.0, three degenerate samples.
- Fit: trivial.
- Overlap: none in an architectural-history sense, but it fills no useful Hermes workflow: no modern instruction following, tools, vision, long context, or competitive coding/reasoning.
- Decision: reject except as an alignment-research artifact. Historical novelty is not production capability coverage.

### 11. OBLITERATUS/qwen3-4b-structured-output-merged-stage-a-OBLITERATED

- Revision: `9fca8324ba3d696746a0b7c443a691a751809f13`
- Provenance: `cyumizou/qwen3-4b-structured-output-merged-stage-a`
- Architecture: `Qwen3ForCausalLM`; chat template explicitly supports tool calls.
- Artifacts: safetensors only, approximately 7.49 GiB; no publisher GGUF.
- Publisher metadata: perplexity 3.84, coherence 1.0, refusal rate 0.0, one degenerate sample, based on 33+33 prompts.
- Fit: easy after GGUF conversion/quantization; likely suitable for a small always-hot worker on one GPU.
- Overlap: installed 27B models already support structured output, but with greater residency and latency.
- Possible distinct role: dedicated low-latency JSON/schema/tool-call router or extraction worker.
- Decision: conditional shortlist, behind Ornith. Convert and test only if measured router latency or concurrency demonstrates that a dedicated 4B worker materially improves the local Hermes control plane.

## Admission order

1. Complete installed Qwen3.8 Q6_K plus projector/MTP functional validation.
2. If fast multimodal/computer-use remains a gap, test Ornith 1.5 9B Q6_K plus its projector.
3. If structured-output routing remains a latency bottleneck, convert/test the structured-output Qwen3 4B.
4. If correlated Qwen errors justify cross-family redundancy, test Gemma 4 12B Q4_K_M.
5. Reject the remaining seven for the production stack unless new independent evidence establishes a named workflow win.

This ordering maximizes coverage while avoiding a shelf of near-duplicate uncensored text models.