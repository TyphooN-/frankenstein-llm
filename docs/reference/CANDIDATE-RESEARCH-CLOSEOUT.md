# Candidate research closeout

Baseline research checked 2026-09-06; three explicit additions screened
2026-09-07 below. This is the current decision document. It supersedes
the adopt/wait/reject wording in
[CANDIDATE-ADDITIONS-INVESTIGATION.md](CANDIDATE-ADDITIONS-INVESTIGATION.md)
and the provisional ranking in
[MODEL-UPGRADE-SHORTLIST.md](MODEL-UPGRADE-SHORTLIST.md). Those files retain
useful header and pagination notes; they are not the current verdicts.

## Submitted candidate intake — 2026-09-08

The following intake records investigation requests, not download admission or
qualification. Existing entries are referenced rather than duplicated.

| Repository | Intake disposition | Next action |
|---|---|---|
| `medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic` | New; source pinned 2026-09-08, see below | No public GGUF exists and the pinned llama.cpp registers the architecture, so any use is an [ADR 0007](../decisions/0007-conditional-custom-quantization.md) conversion experiment. Blocked behind the current gate repairs. |
| `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` | Already in `candidate-research-inventory.json` | Retain existing research; reassess only from pinned evidence and resource feasibility. This request does not enqueue another download. |
| `wfakhri/OTel-2.0-LLM-31B-IT-GGUF` | Already screened below as a conditional telecom specialist | Retain source/license and workload prerequisites before admission; not a qualified general-driver upgrade. |

New submitted URL:
<https://huggingface.co/medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic>.
The OTel submission split `https` and `://` across lines; matching above is by
the exact repository identifier, not validation of the broken URL string.

### medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic — pinned intake

Source pinned 2026-09-08 from the repository itself, not from the card's prose.
`lastModified` 2026-09-07T18:23:54Z; metadata `license: apache-2.0`. Nothing here
is a download admission, a quality claim or a functional verdict.

**From `config.json` (primary source).** `architectures:
["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`, 64 hidden layers,
hidden size 5120, 24 attention heads over 4 KV heads, `head_dim` 256,
`intermediate_size` 17408, `vocab_size` 248320, `max_position_embeddings`
262144, `dtype` bfloat16, `rope_theta` 1e7 with `partial_rotary_factor` 0.25.
A vision encoder is present (`depth` 27, hidden size 1152) with image and video
token ids, so a projector is part of the model, not an optional extra.

**From the file listing.** 28 `model-*.safetensors` shards plus one
`model-extra-*` shard, `chat_template.jinja`, `abliteration_metadata.json`,
`hard_negative_residue.json`. **No `.gguf` file exists in the repository.** The
tags name `sglang`, `vllm`, `fp8` and `awq`: this is published for a
tensor-parallel Python serving stack, not for the llama.cpp router this host
runs. The `fp8` and `awq` branches do not change that — llama.cpp does not serve
either format.

**Runtime feasibility, checked locally rather than assumed.** The pinned
llama.cpp (`upstream/llama-cpp.lock.json`, v0.4.0, commit `5266f24`) *does* know
this architecture on both sides: `conversion/qwen3vl.py` registers
`Qwen3_5ForConditionalGeneration`, and `src/llama-arch.cpp` carries
`LLM_ARCH_QWEN35`. So a conversion path is plausible. It is not proven: nothing
here establishes that the hybrid linear/full attention alternation the card
describes converts correctly, produces coherent output, or runs on gfx1030 under
ROCm. Registration is not qualification.

**Consequence.** Using this candidate at all would require producing a GGUF
locally, which is precisely the "missing supported format" gap
[ADR 0007](../decisions/0007-conditional-custom-quantization.md) governs. It is
the first concrete instance of that ADR, and it inherits every precondition in
it. Memory fit is deliberately not estimated here: the repository's own method
is to read a real GGUF header with `scripts/gguf_header.py` and place it with
`scripts/gpu_placement.py`, and no such artifact exists yet. The card's "~60GB
VRAM" BF16 figure is a vLLM statement about a format this host does not serve.

**Card claims that do not survive contact with the repository.** The card states
a 248,044 vocabulary; `config.json` states 248320. The repository name says 27B
and the card says 28B. The benchmark scores (98/100, 96/100, 37/38) ship with no
harness, seed, prompt set or reproduction command, so they are unverified
marketing, not evidence. Treat every performance and capability statement on the
card as unverified until a local held-out gate says otherwise.

**Operating advice on the card that must not be followed.** It instructs
`--trust-remote-code` and a configuration with `redact_secrets: false` and
`tirith_enabled: false`. This workspace does not enable remote code execution to
admit a candidate and does not disable redaction to make one look better; ADR
0007 says in terms that security policy is never changed to make a candidate
pass. The abliterated lineage also puts this model under
[ADR 0004](../decisions/0004-prompt-corpus-admission.md) rather than outside it.

**Next action, unchanged by this intake:** none, until the mission's existing
failed gates are repaired. This is a research record.

## Submitted candidate intake — 2026-09-08, second batch

Five URLs were submitted for investigation. They dedupe to five distinct
repositories: **two are new identifiers, three already have a recorded
research verdict.** The machine-readable form of this table, including the
exact submitted URL strings, is `submitted_intake` in
[candidate-research-inventory.json](candidate-research-inventory.json).

**Queued is not investigated, and investigated is not qualified.** For this
batch no Hugging Face API call, revision pin, weight download or remote code
execution was performed, so nothing below asserts that a queued repository
exists or states its size, license or architecture. Nothing here admits a
download or authorizes model execution.

| Submitted repository | State | Basis |
|---|---|---|
| `ruvnet/ruos-foundry-swarm-qwen3-30b-a3b-e32` | **queued**, not investigated | First submission of this identifier here. No metadata collected. |
| `Jackrong/Qwopus3.8-27B-Flash` | **queued**, not investigated | First submission of this identifier here. No metadata collected. |
| `medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic` | investigated 2026-09-08, not qualified | Resubmission of the first-batch URL above; the pinned intake in this document already covers it. Not re-collected. |
| `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | investigated 2026-09-02, not qualified | Screened at revision `6821d6ad` in [CANDIDATE-MODEL-REVIEW-2026-09-02.md](../CANDIDATE-MODEL-REVIEW-2026-09-02.md): 304.65B parameters, ideal 4-bit weights alone ≈152.3 GB, verdict **SKIP** on this hardware. |
| `microsoft/VibeVoice-ASR-Streaming-1.5B` | investigated 2026-09-02, not qualified | Screened at revision `94efa5c0` in the same document: 1.5B name against 2.814B Hub metadata, custom VibeVoice code path, no verified llama.cpp or ComfyUI support, verdict **WATCH**. |

The two prior verdicts are recorded findings, not closed questions. A
resubmission on its own is not new evidence, so neither is reopened here; both
remain reversible on a pinned re-collection. The three already-investigated
entries are referenced rather than re-researched, which is what keeps this
intake a deduplicated queue rather than a growing pile of restatements.

**Next action:** none for any of the five. The two queued identifiers need a
primary-source collection through
`verification/local-coverage-foundation/research/collect_hf_metadata.py` before
anything can be said about them, and that is research, not admission. Every
candidate in this batch is behind the mission's existing failed gates.

## Additional submitted candidates — 2026-09-07 screening

These **three distinct repositories** are additional to the 11-repository
baseline below. Screening is not runtime qualification or download admission.
Pinned cards and complete paginated trees were inspected; no weights were
downloaded and no serving aliases were changed. None currently justifies
replacing RVN / `heretic` as the general Hermes driver.

### TobDeBer/M8 — hold for provenance

Submitted: <https://huggingface.co/TobDeBer/M8>.
Inspected revision: `5a22c6b30f9572cacdb7af691f1f142005cf98b3`.
The [pinned tree](https://huggingface.co/TobDeBer/M8/tree/5a22c6b30f9572cacdb7af691f1f142005cf98b3)
contains mixed Qwen3-0.6B and Gemma-4-12B artifacts, REAP-labelled variants,
and importance matrices; no readable model card was obtained. It does not
establish a single instruction-tuned assistant named M8. Importance matrices
are calibration artifacts, not standalone inference weights.

`gemma-4-12b-it-UD-IQ3_XXS_REAP80.gguf` is 4,848,827,424 bytes;
the `_down` sibling is 4,035,599,872 bytes. These API sizes do not establish
retained quality or safe residency. Decision: **conditional compression
research, hold for provenance**. Resolve source revision, license inheritance,
transformation method and GGUF tensor/runtime compatibility before transfer.
Only then compare against installed Gemma-4-12B on matching reader tasks;
do not grant executable tools because an artifact loads.

### TR-HASH pretraining — architecture watchlist, not a driver

Submitted:
<https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-100M-125B-Agentic-Pretraining>.
Inspected revision: `62d844123261ab58684030b99ef6bf352cd29d01`.
The [pinned card](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-100M-125B-Agentic-Pretraining/blob/62d844123261ab58684030b99ef6bf352cd29d01/README.md)
reports **100,366,720 parameters**, a 2,048-token context, and deterministic
token-ID multi-hash expert routing. **125B is training-token volume, not model
parameter count.** `final/model.safetensors` is 406,927,984 bytes; historical
optimizer/checkpoint packs are not additional inference shards.

The publisher explicitly says the checkpoint is not instruction tuned. Earlier
refinement/SFT descendants were withdrawn after an optimizer-update mismatch;
corrected refinement and replacement SFT are separate pending work. This does
not invalidate the pretraining checkpoint. Decision: **defer architecture
research; not a Hermes driver upgrade**. CC-BY-NC-4.0 and the custom framework
require license/source review; CUDA/Triton training is not gfx1030 inference
proof. Revisit a corrected instruction-tuned release with held-out evaluation,
not the entire training archive.

### OTel 2.0 GGUF — conditional telecom specialist

Submitted: <https://huggingface.co/wfakhri/OTel-2.0-LLM-31B-IT-GGUF>.
Inspected revision: `c476b7e66e21d438fba4248267d58781a7d7dcd5`.
The [pinned card](https://huggingface.co/wfakhri/OTel-2.0-LLM-31B-IT-GGUF/blob/c476b7e66e21d438fba4248267d58781a7d7dcd5/README.md)
identifies `farbodtavakkoli/OTel-2.0-LLM-31B-IT`, a Gemma-4-31B telecom OSFT
derivative, as upstream—not its separate QLoRA sibling. The upstream checkpoint
is advertised as frequently updated: pin its conversion source revision as
well as the quant repository before evaluation.

| Exact artifact | API-declared bytes |
|---|---:|
| `OTel-2.0-LLM-31B-IT-Q4_K_M.gguf` | 18,687,065,344 |
| `OTel-2.0-LLM-31B-IT-Q6_K.gguf` | 25,201,487,104 |
| `OTel-2.0-LLM-31B-IT-Q8_0.gguf` | 32,635,677,952 |
| `mmproj-OTel-2.0-LLM-31B-IT-f16.gguf` | 1,198,957,344 |

Decision: **conditional telecom specialist experiment**, not a general driver
replacement. Q6_K is a text-first candidate if a concrete standards/RAG workload
justifies it; budget KV/workspace and desktop reserves separately. Sizes are
not downloaded checksum or fit evidence. Metadata says Apache-2.0 but links
Gemma license terms: resolve the actual terms before admission. Verify installed
llama.cpp and exact gfx1030 execution independently; MI355X training does not
prove this backend. BF16 vision claims do not qualify this GGUF/projector pair.

Compare held-out telecom QA, grounded citations, abstention and multi-turn
native tools against official Gemma-4-31B and installed RVN, plus ordinary
coding regression. Avoid calibration/training examples in the held-out set.
The card explicitly excludes telecom-specific MCP/tool-call training despite
including general tool examples. Keep generated network configuration inert;
no real network changes are part of research.

## Scope and completion standard — baseline research

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

### Later 27B uncensored GGUFs (2026-09-07)

`mradermacher/Qwen3.8-27B-Abliterated-Uncensored-i1-GGUF` revision
`e236aa77635a`, Apache-2.0, architecture `qwen35` (pinned runtime lists it).
Imatrix quants of `Madras1/Qwen3.8-27B-Abliterated-Uncensored`. Useful sizes:
i1-Q4_K_M 16,547,401,312 bytes / 15.41 GiB; i1-Q6_K 22,082,530,912 bytes /
20.57 GiB.

`mradermacher/Qwen-3.8-27B-Uncensored-GGUF` revision `e0935c1167a9`,
Apache-2.0, static quants of `junafinity/Qwen-3.8-27B-Uncensored`, also
`qwen35`. Q6_K 22,431,000,768 bytes / 20.89 GiB; Q8_0 29,047,085,248 bytes /
27.05 GiB; optional mmproj-f16 927,607,392 bytes.

These are the same 27B uncensored/abliterated role as installed
`Qwen3.8-27B-OBLITERATED-Q6_K.gguf` (and `heretic`). Filename and imatrix
label are not quality or refusal-removal proof. Operator authorization:
*slot A/B on speed and usefulness* against those two occupants,
Q6_K only, after the functional mission releases the GPU. Queue:
[download-queue-slot-uncensored-27b.json](../../verification/local-coverage-foundation/download-queue-slot-uncensored-27b.json).
Do not add production aliases first. Disclose MTP/speculative differences;
heretic and obliterated use `draft-mtp`. Read GGUF headers on the challengers
before attributing tok/s to the weights.

## 2026-09-07 quantization comparisons

Two 27B GGUF candidates were investigated for possible Hermes driver upgrades.
Neither is admitted yet.

### ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF

GSQ-RCO IQ3_S 10.963 GiB. Small enough for a single 16 GiB card, but IQ3_S is
a low-precision quant and the repository is a research artifact, not a
production distribution. No quality comparison has been run against installed
`heretic` or `obliterated`. Decision: **conditional research, not a driver
upgrade**.

### unsloth/Qwen3.8-27B-GGUF

UD-IQ3_S 11.214 GiB. Same size class, same low-precision caveat, and the
template supports leading developer messages but has not been verified against
this repository's chat templates or tool policy. Decision: **conditional
research, not a driver upgrade**.

### wfakhri/OTel-2.0-LLM-31B-IT-GGUF

Telecom specialist, not a general driver. Q6_K 23.471 GiB; Q4_K_M 17.404 GiB
(25,201,487,104 and 18,687,065,344 bytes respectively; projector excluded).
See the OTel 2.0 section above for the full disposition.

These comparisons are research, not admission. A quantization label is not a
quality or execution guarantee.

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
