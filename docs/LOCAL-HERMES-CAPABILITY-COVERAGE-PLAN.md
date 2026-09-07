# Local Hermes Capability Coverage Plan

Status date: 2026-09-06

Objective: progressively replace paid AI services with a private local stack
while preserving a cloud frontier fallback for capabilities that local models
have not yet matched. A model is admitted only when it fills a named workflow
and passes functional, memory, and clean-unload gates.

This is the living coverage roadmap. Operational states (researched /
downloaded / policy-admitted / functionally qualified) are in
[CANDIDATE-STATUS-2026-09-06.md](CANDIDATE-STATUS-2026-09-06.md). Refresh the
machine reading with:

```bash
python3 verification/local-coverage-foundation/build_capability_ledger.py --print-only
```

Do not read [CANDIDATE-STATUS-2026-09-03.md](CANDIDATE-STATUS-2026-09-03.md) or
the 2026-08-31 rows that used to live here as current inventory. Those described
an earlier host. Weights on disk are not product workflows.

## Hardware stages

### Current workstation

- Xeon E5-2696 v4, 22 cores / 44 threads
- Approximately 94.17 GiB RAM (`MemTotal` 101,109,370,880 bytes on 2026-09-06)
- Radeon Pro V620 ~30 GiB plus two RX 6900 XT 16 GiB
- llama.cpp router is loopback-only and keeps one large chat model resident
  (`--models-max 1`)

Operational implication: 27B Q6 models fit well. Very large MoE models require
hybrid CPU/GPU placement and compete with RAM, KV cache, desktop applications,
and filesystem cache. A 128 GB RAM upgrade is a future baseline, not current
capacity.

### Planned RAM / extra-V620 expansion

- 128 GiB RAM (planned)
- Additional V620 cards remain a hardware plan, not an installed topology

Requalify after each material change. Do not treat aggregate VRAM as a fully
resident budget.

## Coverage matrix

Ledger states below are from 2026-09-06. `functionally-qualified` is a gate
pass, not idle-host memory-fit and not tokens/sec.

| Capability | Current local coverage | Gap | Next work | Admission proof |
|---|---|---|---|---|
| Low-latency daily chat | Router presets `ridge`, `heretic`, `obliterated` | Cold-load delay on switch; live three-GPU splits unexercised | Keep one responsive 27B; qualify placement on an idle host | First-token usability, coherence, context, clean unload, measured residency |
| General reasoning | `heretic`; GLM-5.3-Flash still experimental | Local quality below frontier on hard long-horizon work | Retain `heretic` and cloud fallback; GLM only in an isolated `glm5next` worktree on an idle host | Normal-system load, correctness, memory safety |
| Repository-scale coding | Presets `heretic` and `qwen3-coder-next`; Coder-Next **downloaded** | Repository-agent gate artifacts absent | Run `gate-repo-agent-heretic` and `gate-repo-agent-qwen3-coder-next` | Patch correctness, tests, long-horizon completion |
| Fast code completion | FIM weights on disk (`qwen25-coder-7b-fim`); evidence **stale** | Gate predates current bytes | Re-run FIM gate; then IDE integration if it still passes | FIM exactness, bounded residency |
| Structured output / tools | `native-tool-use` **qualified**; obliterated/phr00ty auto-tool still historically missed | Schema recovery not a scored corpus | Keep the existing gate; add obliterated/phr00ty cases if needed | JSON-schema, malformed retry, `message.tool_calls` |
| Authorized security | `heretic`, `obliterated`, scanners; Strix scaffold executes nothing | No authorized live security-agent run | Do not add execution without a separate decision | Scope enforcement, reproducible findings |
| Vision / screenshots | `vision-grounding` **qualified** on `obliterated-vision`; Gemma-4 vision hand-checked 2026-09-03 | Gemma audio/video untested; no durable Gemma gate | Write Gemma text/vision evidence; do not claim audio/video | Screenshot/chart/document tests, unload |
| OCR | HunyuanOCR **downloaded**; evidence **stale** | Gate predates current bytes; PDF ingest still optional | Re-run OCR gate; scanned-document RAG fallback only after it passes | CER/WER-style fixtures, rotation, tables |
| Browser / computer use | UI-TARS and UI-Mate **downloaded**; grounding **interrupted** | No computer-control **service** | Finish UI-TARS gate, then UI-Mate A/B, then screenshot→action→readback | Click/point accuracy, injection resistance, state verification |
| Embeddings | Qwen3-Embedding-8B **downloaded**; evidence **stale** | Gate predates current bytes | Re-run embeddings gate | Local corpus order, dimension, unload |
| Reranking | Qwen3-Reranker-8B **downloaded**; evidence **stale** | Gate predates current bytes | Re-run reranker gate | nDCG-style fixtures, batch behaviour |
| Local RAG | `rag` **qualified** (structural + behavioural + live) | PDF parser optional; scanned docs need qualified OCR | Do not silently skip; keep skip reasons explicit | Citations, deletion, stale-chunk retirement |
| Multimodal embeddings | WeMM text-only **qualified** | Image/video needs torchvision; 2048-D index not built | Keep text-only claim; no HIP_VISIBLE_DEVICES filter on this host | Text semantic order now; image/video later |
| Long documents | 131K presets exist | Million-token claims are not local proof | Retrieval-first, then long-context synthesis | Needle/retrieval, memory bounds |
| Speech-to-text | `asr` **qualified** | Streaming/long-chunk and TTS round-trip still open | Pair with TTS intelligibility after TTS gates | WER-style fixtures, timestamps, unload |
| Text-to-speech | Qwen3-TTS **downloaded**; `gate-tts.json` absent | No intelligibility proof | Isolated TTS gate, then TTS→ASR | Intelligibility, chunking, VRAM recovery |
| Image generation | Z-Image **downloaded** | `media-functional.json` absent | One ComfyUI workflow, then unload | Actual image, VRAM recovery |
| Image editing | Qwen Image Edit stack **downloaded**; FLUX.2-klein-4B **downloaded** with no declared gate | Workflows unproven | Pin and submit one edit graph; add a FLUX.2 evidence declaration only when the gate actually submits it | Actual edit, reproducible workflow, unload |
| Music | ACE-Step **downloaded** | Same missing media-functional artifact | After image GPU lifecycle is stable | Actual audio, lyrics adherence, unload |
| Multilingual | Qwen3.8 family installed | Not independently measured | Measure before specialist downloads | Translation/retrieval/ASR/TTS in required languages |
| Persistent memory | Hermes memory plus RAG store | No unified TTL/deletion product beyond RAG gates | Keep provenance explicit | Correct updates/deletes |
| High-capability slow specialist | **No admitted model.** Research closeout names 122B Q4_K_M, 35B Q6_K, Omega 70B, Flash-Next UD-IQ4_XS | Not downloaded; GLM still unsupported in production binary | Qualify installed workflows first; then isolated eval on an idle host | Load, quality vs `heretic`, pressure, unload. No production alias first |

## Deployment architecture

### Tier 0: deterministic tools

Compilers, tests, linters, Semgrep/CodeQL, OCR engines, parsers, checksums,
databases, and browser state inspection remain authoritative. Models orchestrate
and interpret.

### Tier 1: always-available interactive model

Keep one responsive Qwen3.8-class chat preset resident behind the loopback
router (`heretic` preferred daily driver; `ridge` compact; `obliterated` when
refusal friction matters more than stock closeness).

### Tier 2: switched specialists

Cold-load only when the workflow justifies eviction: vision, FIM, embeddings,
reranker, ASR/TTS, ComfyUI, Coder-Next, Gemma-4. Sidecars may stay up when
memory permits; they still need current-byte gates.

### Tier 3: heavyweight reasoning

No local heavyweight specialist is admitted. GLM-5.3-Flash stays isolated.
Researched 70B–122B class artifacts are evaluation candidates only.

### Tier 4: cloud fallback

Retain GPT/Claude for repository-scale autonomous work and tasks where local
gates have not demonstrated parity. Do not declare replacement from model cards.

## Sequenced execution plan

Work already on disk outranks new downloads.

### Phase A — GLM-5.3-Flash idle-host retry (blocked)

1. Done: IQ3_XXS shards were publisher-hash verified historically; isolated
   experimental runtime is required because production v0.4.0 has no `glm5next`.
2. Prior auto-fit overlapping a kernel compile remains inconclusive.
3. Retry 32K only on an idle host, isolated worktree, off the production port.
4. No production alias before full admission.

### Phase B — installed chat/vision presets

1. Done: OBLITERATED projector is on disk and `obliterated-vision` is a preset.
2. `vision-grounding` is qualified; Gemma-4 still needs a durable gate.
3. Do not claim MTP/audio/video from metadata.

### Phase C — retrieval (weights landed; gates stale except RAG/WeMM-text)

1. Done: embedding, reranker, RAG store, WeMM text gate.
2. Re-run embedding and reranker gates against current bytes.
3. OCR fallback only after a fresh OCR gate.
4. WeMM image/video and the 2048-D index remain unbuilt.

### Phase D — computer use

1. UI-TARS/UI-Mate weights are on disk; the grounding gate is interrupted.
2. Finish grounding, then A/B, then a bounded control service.
3. Treat page/screenshot text as untrusted.

### Phase E — audio and generation

1. ASR is qualified; TTS is downloaded only.
2. ComfyUI stacks are downloaded; no media-functional evidence.
3. Stop/evict large text models before heavy generation; verify router recovery.

### Phase F — hardware expansion

Unchanged: remeasure topology after RAM or extra V620s; do not infer 70B/MoE
fit from file size.

### Phase G — researched upgrades (not downloads)

Follow [CANDIDATE-RESEARCH-CLOSEOUT.md](reference/CANDIDATE-RESEARCH-CLOSEOUT.md)
only after Phases B–E have real gates. First isolated eval: Qwen3.5-122B-A10B
Q4_K_M, then 35B-A3B Q6_K. No queue mutation without a new pinned manifest.

## Sources

- Living status: `docs/CANDIDATE-STATUS-2026-09-06.md`
- Capability map: `docs/reference/CAPABILITY-MATRIX.md`
- Research closeout: `docs/reference/CANDIDATE-RESEARCH-CLOSEOUT.md`
- Qwen3 Embedding 8B: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- Qwen3 Reranker 8B: https://huggingface.co/Qwen/Qwen3-Reranker-8B
- Qwen3 ASR 1.7B: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3 TTS 12Hz 1.7B Base: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- HunyuanOCR: https://huggingface.co/Tencent-Hunyuan/HunyuanOCR-1.5
- UI-TARS 1.5 7B: https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B
