# Candidate research closeout

Research checked 2026-09-06. This is the current decision document. It supersedes
the adopt/wait/reject wording in
[CANDIDATE-ADDITIONS-INVESTIGATION.md](CANDIDATE-ADDITIONS-INVESTIGATION.md)
and the provisional ranking in
[MODEL-UPGRADE-SHORTLIST.md](MODEL-UPGRADE-SHORTLIST.md). Those files retain
useful header and pagination notes; they are not the current verdicts.

## Scope and completion standard

Fourteen submitted URLs deduplicate to **11 queued repositories**. All 11 have
a disposition below. This pass also reviews six additional artifacts, the three
upgrade families, and exact-lineage community quants that the original wait
verdicts missed. Official and abliterated Flash-Next, and MLX versus GGUF K2
MoVA, are distinct artifacts. Compatibility does not transfer across formats.

[candidate-research-inventory.json](candidate-research-inventory.json) records
revisions, licenses/gating, complete API file counts, artifact bytes, and
publisher hashes. Numbered GGUF shard sets were checked for a complete
contiguous sequence. Repository totals include alternative quants and extras;
they are not download manifests. API hashes are not freshly verified file
content.

Research complete means identity, role, source evidence, format/backend route,
capacity bounds, decision, and next gate are recorded. It does not invent
missing licenses, kernels, or local quality. No weights, services, router
presets, or runtime builds were changed. Nothing here is admitted.

## Hardware and runtime boundary

Live read: headless RX 6900 XT 16,368 MiB, V620 30,704 MiB, display RX 6900 XT
16,368 MiB; free VRAM 16,341 / 30,324 / 14,448 MiB. RAM 101,109,370,880 bytes
(94.17 GiB) with 36,218,109,952 bytes (33.73 GiB) available. Swap is not model
capacity. Home filesystem had 2,232,408,604,672 free bytes. These are
observations, not idle-host certification.

The earlier reserved GPU pool of 56.45 GiB is a screening number. A weight
artifact larger than that pool can still be an authorized RAM-offload
candidate. Subtracting that pool from weights is an optimistic host-weight
lower bound, not a resident-memory guarantee. Do not import production
`mmap = 0` or 131K context into standalone experiments. Start text-only at
16K, then 32K, without MTP.

Installed binary: `0.4.0-dev`, build 10809, commit `5266f24da`. Matching
source lists `qwen35moe`, `qwen4exp`, `llama`, and `bailingmoe3`. `k2-horizon`
is absent from that binary. Source registration is necessary, not proof that
every operator, parser, projector, or ROCm path works. A 128 GB RAM upgrade is
a future baseline. ComfyUI/Diffusers do not inherit three-GPU sharding merely
because three cards are visible.

## Larger-model upgrade decisions

### Qwen3.5-122B-A10B: first larger reasoning/tool experiment

Use `unsloth/Qwen3.5-122B-A10B-GGUF` **Q4_K_M**, three shards,
76,536,964,608 bytes / 71.28 GiB. Apache-2.0 upstream; 122B total / 10B
activated plus vision and MTP in the source card. Decision: **isolated
evaluation**. Optimistic host-weight remainder versus the reserved GPU pool is
about 14.83 GiB, which is plausible on installed RAM when idle, not a live-fit
claim. Compare tools, repository work, long-document retrieval, and vision
separately against RVN / Coder-Next / Gemma. Publisher scores motivate a test;
they do not establish a local win. Q6_K (94.07 GiB) is a later fidelity
comparison. Q8_0 (120.95 GiB) is not the first download. Keep the faster
installed baseline even if Q4_K_M wins. Add the BF16 projector (912,263,904
bytes) only for the vision gate.

### Qwen3.5-35B-A3B: compact alternative

Use `unsloth/Qwen3.5-35B-A3B-GGUF` **Q6_K**, 28,852,861,568 bytes / 26.87 GiB.
Q8_0 is 36,903,139,968 bytes / 34.37 GiB. Apache-2.0, 35B total / 3B activated.
Decision: **isolated evaluation** for native tools and vision at lower compute.
Neither active-parameter count nor quant label proves better answers. A single
V620 fit is not guaranteed once cache and workspace are included. BF16
projector is 902,822,592 bytes, separate from text weights.

### Official Qwen3.8-Flash-Next: smaller quant selected

Use `unsloth/Qwen3.8-Flash-Next-GGUF` **UD-IQ4_XS**, three complete shards,
93,682,584,224 bytes / **87.25 GiB**, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`. Optimistic host-weight remainder
is about 30.80 GiB. Decision: **conditional isolated RAM-offload evaluation**
after the 122B baseline and only on an otherwise idle host. The 33.73 GiB
available-RAM snapshot does not demonstrate adequate overhead.

This is the official lineage, not Huihui. Upstream is experimental: 125B LM
parameters, 6B activated, plus 51B n-gram embeddings and 4B MTP. License is
Qwen Community 1.0. Publisher RAM/n-gram offload claims are not measurements
on these AMD cards. Text-only, no MTP first. BF16 projector is 907,542,944
bytes. MTP needs its own artifact and runtime proof. Do not choose 1-2-bit
quants merely to make capacity arithmetic work.

## The 11 queued additions

| Exact repository | Disposition | Decisive gate |
|---|---|---|
| `mradermacher/Omega_Sapphira_Joyous-L3.3-70B-v1.1-i1-GGUF` | Isolated specialist evaluation | Llama-3.3 runtime; creative prose first, not presumed reasoning upgrade |
| `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` | Conditional isolated RAM-offload | Distinct from official UD-IQ4_XS; retained capability vs refusal |
| `inclusionAI/Ling-3.0-flash-Fin` | Isolated specialist via exact-lineage GGUF | Finance documents only; no trading authority |
| `IFM/K2-Horizon-MoVA-36B-A4B-GGUF` | Isolated experimental runtime | Official BF16 plus community quants; `k2-horizon` needs IFM fork, not production binary |
| `IFM/K2-Horizon-375B-A23B` | Reject BF16; hold mixed-quant research | 706 GiB BF16 is impossible here; MQ87 is unproven fidelity |
| `inclusionAI/LLaDA-Image` | Isolated evaluation after source review | Custom Diffusers is reviewable, not forbidden; encoder may not fit one GPU |
| `Lightricks/LTX-2.5` | Hold access, then isolated evaluation | Hub gate plus license review; subset bytes are not residency |
| `GestaltLabs/Qwen3.8-27B-EXL3-11.5GB` | Hold exact EXL3 format | ExLlamaV3 lists ROCm as missing |
| `speach1sdef178/MiniMax-H3-Semantic-Bridge` | Hold adapter and legal base | 11 MiB adapter; H3 community license excludes US/EU/UK/Korea |
| `MohamedAhmedAE/llava-medical-3B-clip-vit-stage2` | Hold license and assembly | No clinical authority; size is not a quality reject |
| `Reallexi-llc/lexipix-models` | Hold opaque bundle | Per-file attribution exists; do not adopt the whole mobile tree |

### Omega Sapphira 70B

Revision `b482c8b48894`. Architecture `llama` in the GGUF header, so the pinned
runtime has the relevant table entry. Exact one-quant sizes:

- `i1-IQ4_XS` 37,902,664,032 bytes / 35.30 GiB
- `i1-Q4_K_M` 42,520,396,128 bytes / 39.60 GiB
- `i1-Q5_K_M` 49,949,819,232 bytes / 46.52 GiB
- `i1-Q6_K` 57,888,145,760 bytes / 53.91 GiB

Prefer **Q4_K_M or Q5_K_M at conservative context**, not maximum 131K. Roleplay
and unaligned tags are not coding or tool evidence. Isolated evaluation against
`heretic` on real work; no alias.

### Huihui Flash-Next abliterated

Revision `7e3bfc316b88`. Only published weight is **UD-Q4_K_XL**, four shards,
111,334,655,392 bytes / **103.69 GiB**, plus projector 907,542,592 bytes.
Header architecture is `qwen4exp`, which the pinned source lists. Shards are
not byte-identical to unsloth UD-Q4_K_XL.

This is **not an automatic capacity reject**. Optimistic host-weight remainder
versus 56.45 GiB GPU budget is about 47.24 GiB of 94.17 GiB RAM, before cache,
workspace, and desktop. Current available RAM (33.73 GiB) is the real current
blocker, not total RAM. Decision: **conditional isolated RAM-offload** after
official UD-IQ4_XS, idle host, retained-capability tests. Abliteration is not
proven by filename.

### Ling-3.0-flash-Fin

Official BF16 revision `4194bf7d6e4b` is 254,999,780,735 bytes / 237.49 GiB.
Architecture `BailingMoeV3ForCausalLM` maps to `bailingmoe3` in the pinned
source. Exact-lineage community GGUFs now exist. Preferred first artifact:
`bartowski/Ling-3.0-flash-Fin-GGUF` revision `a792b6bd51d2`, MIT,

- IQ4_XS two shards 68,739,201,632 bytes / 64.02 GiB
- Q4_K_M two shards 77,804,990,560 bytes / 72.46 GiB
- Q5_K_M three shards 90,661,271,776 bytes / 84.43 GiB

Card names llama.cpp `b10776` and this exact finance checkpoint. Decision:
**isolated evaluation** of Q4_K_M or IQ4_XS for document reconciliation and
calculation under review. No autonomous trading. 128 GB RAM is not a
prerequisite for the Q4 class. Do not convert the 237 GiB BF16 just to obtain
a GGUF that already exists.

### K2-Horizon MoVA 36B GGUF

Official IFM GGUF revision `c8dde8bc6afe` is BF16 74,924,627,296 bytes /
69.78 GiB, architecture `k2-horizon`. The production binary cannot load it.
That is a **runtime gap**, not a proof that no GGUF of this model can exist.

Exact-base community quant, `abenzerps/K2-Horizon-MoVA-36B-A4B-GGUF` revision
`a3f3d1537786`, Apache-2.0, source pin `05cab0a`:

- Q4_K_M 22,368,011,616 bytes / 20.83 GiB
- Q5_K_M 26,439,456,096 bytes / 24.62 GiB
- Q6_K 30,765,365,856 bytes / 28.65 GiB
- Q8_0 39,831,174,496 bytes / 37.10 GiB

The publisher says these need the MBZUAI-IFM `model/K2Horizon` fork until
upstream lands. Fork source is not a prebuilt AMD binary and not a production
replacement. Decision: **isolated experimental runtime** of Q6_K or Q8_0 in a
worktree off the production port. MLX 4-bit (19.63 GiB) remains a different
format. Advertised 512K context is not a local budget.

### K2-Horizon 375B

Official BF16 revision `d33e3ae45281`: 61 shards, 758,338,618,864 bytes /
**706.26 GiB**. Reject this raw artifact on current and planned RAM.

Community mixed quant `Baekpica/K2-Horizon-375B-A23B-Mixed-Quant-GGUF`
revision `4d8a99fed919`, MQ87 four shards 93,091,935,552 bytes / 86.70 GiB,
derived from that exact BF16 revision. Attention/shared experts at Q8_0 with
IQ1/IQ2 routed experts is an unproven capability-retention risk. Decision:
**reject BF16; hold MQ87 as lower-priority experimental research**, never as
the first larger-model experiment. NVIDIA demos are not gfx1030 proof.

### LLaDA-Image

Revision `e4e2703f410f`, Apache-2.0, 49,276,228,561 bytes / 45.89 GiB.
Components: text encoder 30.40 GiB, transformer 12.18 GiB, plus SigVQ /
queryformer / projection / VAE. Custom `LLaDAImagePipeline` uses
`trust_remote_code=True`. That is a **source-review gate**, not a permanent
ban. The encoder already exceeds one 16 GiB card; aggregate VRAM does not
rescue a single-device load recipe. Decision: **isolated evaluation after
pinning and reviewing GitHub plus repo-local encoder code**, isolated venv,
no credentials, sequential/offload placement, compare bilingual rendering and
edit fidelity against already-downloaded image stacks. Those stacks being
unproven affects scheduling, not intrinsic merit.

### LTX-2.5

Revision `5e6e71018ee1`, gated `auto`, 200,853,702,175 bytes total. Distilled
INT8 audio/video subset is about 36.06 GiB of files, not simultaneous
residency. Transformer 20.03 GiB already exceeds a 16 GiB card. Decision:
**operator must accept hub terms and review the LTX 2.x community license**
(including the revenue threshold in that agreement). Then a pinned low
resolution/frame-count isolated Comfy or pipeline run with staged loading.
Do not jump the still-unproven Z-Image / ACE-Step / Qwen Image Edit / FLUX.2
queue without a named video gap.

### Qwen3.8-27B EXL3

Revision `18028352ac66`, Apache-2.0, three EXL3 safetensors
11,473,119,236 bytes. ExLlamaV3 README currently lists **ROCm support** under
"What's missing?". Decision: **hold this exact format** until upstream ROCm
exists. Compact size is not a quality win over installed
`Qwen3.8-27B-OBLITERATED-Q6_K.gguf`. A different GGUF of the same family is
not execution of this EXL3 artifact.

### MiniMax-H3 Semantic Bridge

Adapter revision `8c2d9b0edb84`: 11,023,032-byte safetensors plus workflow,
not a standalone model. Base `MiniMaxAI/MiniMax-H3` revision `42ed227ee7df`
is 280 files, 498,474,749,480 bytes / 464.29 GiB. FL2VA transformer plus text
encoder plus VAEs is already about 134 GiB of safetensors before Ref2VA
duplicates. Community license **Applicable Territory excludes the United
States, EU, UK, and Republic of Korea**. Decision: **hold**. Do not download
H3 or the adapter for US deployment under that community license. Revisit
only with separately granted rights. Publisher cosine metrics are not video
quality.

### Medical LLaVA 3B stage-2

Revision `deef930c5967`, 2,604,016,419 bytes, **no top-level license**.
Training-state optimizer is 879,096,007 bytes and is not a serving artifact.
Assembly of LoRA plus `model.safetensors` plus `non_lora_trainables.bin` is
undocumented. Referenced text-base has Llama 3.2 terms; that does not finish
this checkpoint's rights. Decision: **hold license and reproducible
assembly**. Small size is not a quality reject. No clinical authority.

### LexiPix bundle

Revision `314776a9ab97`, 10,639,980,998 bytes. Top-level license undeclared,
but `manifest.json` labels per-artifact licenses and claimed lineages
(Qwen2.5-0.5B, Phi-4-mini, Gemma 2, Qwen2.5-1.5B). Those labels do not finish
inheritance. SD-Turbo GGUF is a **stable-diffusion.cpp** artifact, not an LLM
llama.cpp load. Decision: **hold the bundle**. Evaluate one attributable
artifact only if a named low-resource role appears. Core ML is irrelevant
here.

## Six additional artifacts

### BreezeBlue/Breeze-TTS-2

Revision `799624c0b4a1`. Main weights plus tokenizer 7,648,850,134 bytes /
7.12 GiB. Code Apache-2.0; weights and self-hosted outputs are
research/non-commercial. Official path is NVIDIA CUDA, about 7.7 GiB eager /
14.4 GiB fast-path on that stack. Decision: **hold production use;
conditional isolated eager ROCm probe** after license intent is clear.
Consensual or synthetic voices only. ASR round-trip plus listening, then
unload.

### openai-community/gpt2

Revision `607a30d783df`. `model.safetensors` 548,105,171 bytes / 0.51 GiB,
MIT, 124M pretrained LM, 1024-token training sequence. Decision: **no
product-upgrade download**. Tiny tokenizer fixture only if a test needs it.

### tencent/Hy4-preview

Revision `705d81ee5156`. 131 safetensors, 1,559,983,809,380 bytes /
**1,452.85 GiB**, Apache-2.0. Decision: **reject this BF16 locally**,
including after a 128 GB RAM upgrade. FP8 sibling is a different artifact and
is not substituted here.

### black-forest-labs/FLUX.2-klein-base-9b-fp8

Revision `9ecf2143d715`. Transformer 9,567,278,472 bytes / 8.91 GiB. Gated
non-commercial 9B base, not the downloaded Apache 4B. Decision: **hold
access/license, then role-matched A/B** against 4B / Qwen Image Edit.
Publisher ~29 GB NVIDIA figure is not a gfx1030 FP8 budget.

### dealignai/GLM-5.3-CYBERSECURITY-FP8

Revision `19ff6eef0a60`. 282 safetensors, 755,631,998,192 bytes / **703.74
GiB**. MIT metadata; publisher documents 8x H200. Decision: **reject this
exact raw artifact**. Not GLM-5.3-Flash. Refusal reduction is not
security-task correctness. No real credentials or autonomous security
actions.

### abenzerps/K2-Horizon-MoVA-36B-A4B-MLX-4bit

Revision `0c576733b69e`. 48 safetensors, 21,072,747,354 bytes / 19.63 GiB,
Apache-2.0, custom MLX implementation. Decision: **hold exact MLX artifact**
on this AMD production stack. Use the GGUF community quants plus IFM fork
path above if K2 is pursued.

## Evaluation order

1. Finish functional qualification of already downloaded workflows.
2. Qwen3.5-122B-A10B Q4_K_M.
3. Qwen3.5-35B-A3B Q6_K; Q8_0 only for a measured gain.
4. Omega Q4_K_M or Q5_K_M at conservative context.
5. Official Flash-Next UD-IQ4_XS, then Huihui UD-Q4_K_XL if headroom and
   official-lineage results justify a separate A/B.
6. Ling bartowski Q4_K_M as a finance-document specialist, no trading
   authority.
7. K2 MoVA community Q6_K/Q8_0 only in an isolated IFM-fork worktree.
8. Media/speech candidates after access, licensing, and backend gates.

This order is expected value, not a benchmark ranking. Stop conditions remain
in [MODEL-UPGRADE-EVALUATION.md](MODEL-UPGRADE-EVALUATION.md): healthy idle
host, serialized GPU ownership, bounded pressure, exact hashes,
load/inference/quality/unload, no production alias until admission. Recent
RCU-stall history still outranks any capacity estimate. Targeted throughput
is separate from functional proof.
