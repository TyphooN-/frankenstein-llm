# Local Hermes Capability Coverage Plan

Status date: 2026-08-31

Objective: progressively replace paid AI services with a private local stack while preserving a cloud frontier fallback for capabilities that local models have not yet matched. A model is admitted only when it fills a named workflow and passes functional, memory, and clean-unload gates.

## Hardware stages

### Current workstation

- Xeon E5-2696 v4, 22 cores / 44 threads
- Approximately 94.16 GiB RAM
- Radeon Pro V620 32 GiB plus two RX 6900 XT 16 GiB
- Approximately 61.55 GiB aggregate VRAM
- llama.cpp router is loopback-only and normally keeps one model resident

Operational implication: 27B Q6 models fit well. Very large MoE models require hybrid CPU/GPU placement and compete with RAM, KV cache, desktop applications, and filesystem cache.

### Planned 3–4 V620 system

- 96 GiB aggregate VRAM with three V620s
- 128 GiB aggregate VRAM with four V620s

Operational implication: 27B–35B high-quality models can be fully GPU-resident with substantial context; selected 70B-class quants become practical; approximately 90–120 GiB artifacts may fit only with careful runtime/KV headroom and should not be treated as safely fully resident merely because file size is below aggregate VRAM.

## Coverage matrix

| Capability | Current local coverage | Gap | Preferred next component | Admission proof |
|---|---|---|---|---|
| Low-latency daily chat | `ridge`, `heretic`, `obliterated` | Model switching has cold-load delay | Keep one responsive 27B preset; optionally Ornith 9B only if measured faster | First-token usability, coherence, 64K+ context, clean unload |
| General reasoning | `heretic`; GLM-5.3-Flash IQ3_XXS deferred for idle-host retry | Local quality below frontier on hard long-horizon work | Retain `heretic` and cloud fallback; retry GLM only with no kernel compilation or other heavy workload | Normal-system load, correctness suite, memory safety, workflow quality |
| Repository-scale coding/repair | `heretic` plus local tools | No proven local replacement for Claude Code/GPT frontier agents | Benchmark candidate coding models through real repo tasks; do not infer from model-card scores | Patch correctness, tests, diff review, long-horizon completion |
| Fast code completion | No dedicated always-hot FIM service | General chat models are wasteful/slow for keystroke completion | Select a compact FIM-capable coder after latency/quality bakeoff | FIM exactness, low latency, bounded residency, IDE integration |
| Structured output/tool calls | Installed Qwen3.8 models; not yet systematically tested | Schema adherence and recovery unproven | Test installed `obliterated`; conditional Qwen3 4B structured-output worker | JSON-schema corpus, malformed retry, tool name/argument accuracy |
| Authorized security/red-team | `heretic`, `obliterated`, scanners/tooling | Need repeatable agent scaffold and evidence discipline | Keep Qwen3.8 unrestricted model; evaluate a sandboxed CAI-style scaffold | Scope enforcement, reproducible findings, false-positive rate |
| Vision/screenshots/diagrams | Qwen3.8 text weights installed, projector absent | No verified local image understanding | Download matching Qwen3.8 BF16 projector; conditional Ornith 9B fallback | Screenshot, chart, document, diagram, multi-image tests |
| OCR/document images | No dedicated verified OCR service | General VLMs are not enough for dense layouts | Evaluate PaddleOCR-VL 1.6 and HunyuanOCR 1.5 on local documents | CER/WER, tables, forms, rotation, handwriting, bounded batches |
| Browser/computer use | Hermes desktop control exists; local visual policy unproven | No verified local screen-grounding model | First test Qwen3.8 vision in a sandbox; evaluate UI-TARS only if grounding fails | Click/point accuracy, state verification, prompt-injection resistance |
| Embeddings | No standardized local embedding endpoint | Retrieval cannot be coverage-complete | Qwen3-Embedding-8B, with a smaller tier if latency requires | MTEB-like local corpus recall, dimension/storage cost, multilingual data |
| Reranking | No standardized reranker | Vector-only retrieval quality ceiling | Qwen3-Reranker-8B | nDCG/recall improvement, batch latency, long-query behavior |
| Local RAG | Fragmentary file/session search only | No unified ingestion, chunking, citation, vector lifecycle | Local parser/OCR + embeddings + reranker + Qdrant/SQLite metadata | Incremental ingest, deletion, dedupe, citations, adversarial documents |
| Long documents | 131K router presets on installed models | No end-to-end ingest/retrieval proof; million-token claims not practical proof | Retrieval-first pipeline, then long-context synthesis | Needle/retrieval suite, citation accuracy, memory bounds |
| Speech-to-text | No dedicated verified local ASR stack | Audio input still depends on external/limited paths | Qwen3-ASR-1.7B; retain Whisper-class fallback for comparison | WER, timestamps, accents, noise, long-stream chunking |
| Text-to-speech | Hermes TTS supports configured providers, but local quality path not established | Need private natural voice and streaming | Qwen3-TTS-12Hz-1.7B-Base or a proven lightweight local voice | Intelligibility, latency, long-text chunking, voice consistency |
| Image generation/editing | No deployed pipeline | Text model cannot render/edit images | ComfyUI on ROCm with one known-good FLUX/SDXL-class workflow | Actual image, inpaint/img2img, reproducible workflow, VRAM recovery |
| Music/audio generation | No deployed pipeline | No local full-song generation | ACE-Step 1.5 through ComfyUI; secondary sound-effect model later | Actual song, lyrics adherence, duration, clean GPU handoff |
| Multilingual work | Qwen3.8 family | Not independently measured on user languages | Keep Qwen generalists; measure before specialist downloads | Translation/retrieval/ASR/TTS corpus in required languages |
| Persistent memory | Hermes session/memory plus files | No unified semantic memory with retention/deletion policy | RAG store with explicit provenance, TTL, and deletion | Correct updates/deletes, dedupe, source traceability, privacy |
| High-capability slow specialist | No admitted model; GLM-5.3-Flash retry deferred | Fit remains unproven because the prior pressure run overlapped a Linux kernel compile | Retry once on an otherwise idle current host; requalify after planned RAM/GPU upgrades | Normal-system load, coherence, workflow wins, safe headroom |

## Deployment architecture

### Tier 0: deterministic tools

Use compilers, tests, linters, Semgrep/CodeQL, OCR engines, parsers, checksums, databases, and browser state inspection as authoritative evidence. Models orchestrate and interpret; they do not replace deterministic proof.

### Tier 1: always-available interactive model

Keep one responsive Qwen3.8-class model resident behind the loopback llama.cpp router. This is the default for private chat, code explanation, lightweight tool calls, summarization, and routine authorized security work.

### Tier 2: switched specialists

Cold-load specialists only when their workflow justifies eviction:

- Qwen3.8 OBLITERATED plus projector for unrestricted multimodal/security work
- OCR specialist for dense documents
- embedding and reranking services, preferably separate and persistent if memory permits
- ASR/TTS services
- image/music generation pipelines

### Tier 3: heavyweight reasoning

No local heavyweight specialist is admitted. GLM-5.3-Flash IQ3_XXS remains isolated and may receive one clean 32K retry on the normal 96 GiB/three-GPU system only when Linux kernel compilation and every other heavy workload are absent. Do not tune the workstation around it.

### Tier 4: cloud fallback

Retain GPT/Claude access for repository-scale autonomous work, difficult multimodal reasoning, or time-critical tasks where local tests have not demonstrated parity. Route progressively less traffic to cloud as each local gate passes; do not declare replacement based on model-card benchmarks.

## Sequenced execution plan

### Phase A — GLM-5.3-Flash idle-host retry

1. Done: all 15 IQ3_XXS shards were publisher-hash verified and the isolated PR #27752 runtime was built.
2. The prior auto-fit run is inconclusive because it overlapped a 44-thread Linux kernel compile. Static `3,6,2` placement remains invalid because it exceeded a 16 GiB GPU.
3. Retry 32K auto-fit once only after confirming no kernel compile, linker, downloader, or other heavy workload is active. Do not change ARC, swap policy, or other host settings.
4. Keep 64K, A/B, and router integration blocked until that clean 32K gate passes. No production alias is permitted before full admission.

### Phase B — complete the installed Pliny model

1. Done: local projector `/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf` matches publisher size 931,145,888 and SHA-256 `e484e3b7e907ed0e0644c0de56c3f5929c7ad5c9c6cc84d35a9d8dc08d461545` (revision `a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8`).
2. Test text coherence, coding patch quality, JSON/schema output, tool calls, low-refusal behavior, screenshots/documents/charts, MTP, context handling, and clean unload.
3. Tune from measured behavior. Publisher recommendations are temperature 0, repetition penalty 1.15, empty system prompt, and bundled Jinja template; do not blindly apply them to every Hermes workflow.
4. Test Ornith 9B only if Qwen3.8 cannot provide responsive visual/computer-use behavior.

### Phase C — retrieval foundation

1. Deploy Qwen3-Embedding-8B behind a local embeddings endpoint.
2. Deploy Qwen3-Reranker-8B behind a local reranking endpoint.
3. Build an incremental ingestion pipeline with MIME-aware parsing, OCR fallback, content hashes, source metadata, bounded chunking, and deletion propagation.
4. Use a local vector store plus SQLite metadata/provenance.
5. Gate with a private representative corpus and citation/retrieval tests before connecting it to Hermes memory.

### Phase D — vision, OCR, and computer use

1. Establish Qwen3.8 projector functionality first.
2. Evaluate PaddleOCR-VL 1.6 and HunyuanOCR 1.5 on the same scanned-document corpus; install only the winner or complementary pair.
3. Run browser/desktop tasks in a sandbox with state readback after every action.
4. Treat page/screenshot text as untrusted data, never as operator instructions.
5. Evaluate UI-TARS only if the general VLM cannot meet grounding accuracy.

### Phase E — audio and generation

1. Deploy ASR and TTS in isolated environments with streaming/chunking tests.
2. Deploy ComfyUI on ROCm with one image workflow, then one editing workflow.
3. Add ACE-Step only after image-generation GPU lifecycle is stable.
4. Stop/evict large text models before heavy generation when required; verify router recovery afterward.

### Phase F — 3–4 V620 expansion

1. Re-measure device ordering, peer access, usable VRAM, ROCm stability, and power/thermal limits after each card addition.
2. Prefer role isolation when useful: one always-hot interactive model; one embeddings/reranker/vision service; remaining GPUs for switched heavy specialists.
3. Re-test tensor split rather than assuming equal split is optimal.
4. Reconsider 70B and large MoE models only with measured runtime headroom, not aggregate file-size arithmetic.
5. Preserve bounded queues, one-resident-heavy-model policy where necessary, and explicit GPU admission/backpressure.

## Sources

- Qwen3 Embedding 8B: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- Qwen3 Reranker 8B: https://huggingface.co/Qwen/Qwen3-Reranker-8B
- Qwen3 ASR 1.7B: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3 TTS 12Hz 1.7B Base: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- HunyuanOCR 1.5: https://huggingface.co/Tencent-Hunyuan/HunyuanOCR-1.5
- PaddleOCR-VL 1.6: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
- UI-TARS Desktop: https://github.com/bytedance/UI-TARS-desktop
- UI-TARS 1.5 7B: https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B
- Pliny catalog evidence: `verification/obliteratus-hf-inventory-2026-08-31-live.json`
- Pliny model-by-model assessment: `verification/OBLITERATUS-CATALOG-ASSESSMENT-2026-08-31.md`
