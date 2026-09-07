# Queued candidate additions: primary-source investigation

Fourteen URLs were queued for investigation; they dedupe to **eleven distinct
repositories** (`inclusionAI/LLaDA-Image`, `GestaltLabs/Qwen3.8-27B-EXL3-11.5GB`
and `inclusionAI/Ling-3.0-flash-Fin` were each submitted twice). Every one
resolved — there were **no lookup errors and no unavailable identifiers**, so
nothing here is a substitution for something that could not be found.

This document records what the publishers' own metadata says, what the pinned
runtime in `upstream/llama.cpp` can actually execute, and what each artifact
would cost on this host. **No weights were downloaded and nothing was loaded.**
Every "fits" below is arithmetic from [the placement
policy](GPU-EXECUTION-AND-MODEL-LOADING.md), not a load. No throughput figure
appears anywhere in this file, and none of these candidates is admitted,
downloaded, or GPU-qualified by this document. The gates in
[MODEL-UPGRADE-EVALUATION.md](MODEL-UPGRADE-EVALUATION.md) still apply in full.

## How the facts here were obtained

- Repository revision, file list and per-file LFS SHA-256 come from
  `verification/local-coverage-foundation/research/collect_hf_metadata.py`,
  which reads the Hugging Face metadata API and never fetches weights. LFS rows
  land in the tracked
  [`hf-lfs-index.tsv`](../../verification/local-coverage-foundation/research/hf-lfs-index.tsv).
- Architecture support was checked against the pinned submodule itself
  (v0.4.0, `5266f24`): the runtime table in `src/llama-arch.cpp` and the
  converter registrations under `conversion/`. Not against a memory of what
  llama.cpp supports.
- GGUF header fields — architecture, block count, KV head width, trained
  context — were read from the real artifacts through **bounded HTTP range
  requests** (4-16 MiB prefixes) parsed by
  [`scripts/gguf_header.py`](../../scripts/gguf_header.py). The header sits at
  the front of the file, so this is a metadata read, not a partial download of
  a model.
- Device budgets are the live ones from `scripts/gpu_vram.py` and
  `config/gpu-placement.json` on the three-GPU host, measured while writing
  this: ROCm0 14.98 GiB, ROCm1 (V620) 28.98 GiB, ROCm2 (display) 12.48 GiB,
  **56.45 GiB aggregate after reserves**. Host RAM is `MemTotal` 98,739,612 kB
  = **94.16 GiB**. The prospective 128 GB upgrade is not used in any figure
  here, and swap is not treated as capacity.

### A collector bug found and fixed while doing this

The metadata API caps a tree response at 50 entries and offers the remainder
through a `Link: rel="next"` header. The collector read the first page only.
Four of these eleven repositories are larger than the cap, and the truncation
was silent — the shortest listing looked like a complete one:

| Repository | Files before | after | Bytes before | after |
|---|---:|---:|---:|---:|
| `GestaltLabs/Qwen3.8-27B-EXL3-11.5GB` | 46 | 71 | 1,293,779 | 11,499,270,655 |
| `IFM/K2-Horizon-375B-A23B` | 49 | 76 | 480,687,540,244 | 758,372,571,816 |
| `inclusionAI/Ling-3.0-flash-Fin` | 49 | 77 | 177,712,976,620 | 254,999,780,735 |
| `speach1sdef178/MiniMax-H3-Semantic-Bridge` | 41 | 69 | 17,855,496 | 20,825,454 |

The first row is why this matters: the truncated listing contained no weight
shards at all, and would have supported a confident, wrong finding that the
advertised artifact was missing from its repository. Earlier saved listings
cannot establish their own completeness: directory entries also consume API
pages. Recollect an earlier inventory before relying on its completeness.
The collector now walks every page,
reports a mid-walk failure instead of returning a short list as complete, and
publishes through fsync/rename so a crash cannot leave a zero-byte document.

## Runtime support, decided against the pinned submodule

| Architecture the artifact declares | In v0.4.0 runtime? | In v0.4.0 converter? |
|---|---|---|
| `qwen4exp` (Qwen3.8-Flash-Next GGUF) | yes, `LLM_ARCH_QWEN4EXP` | n/a, already GGUF |
| `llama` (Omega Sapphira, SmolVLM, sd-turbo) | yes | n/a, already GGUF |
| `k2-horizon` (K2-Horizon GGUF) | **no** | **no** |
| `BailingMoeV3ForCausalLM` (Ling-3.0-flash-Fin) | yes, `bailingmoe3` | yes, `conversion/bailingmoe3.py` |
| `Qwen3_5ForConditionalGeneration` (EXL3 repo config) | yes, `qwen35` | yes — but the weights are not GGUF |
| `LLaDAImagePipeline` (LLaDA-Image) | not a llama.cpp model | — |
| LTX-2.5, MiniMax-H3 | not llama.cpp; ComfyUI path | — |

## Verdicts

### Isolated evaluation is justified

**`mradermacher/Omega_Sapphira_Joyous-L3.3-70B-v1.1-i1-GGUF`** — revision
`b482c8b48894faf06c487bdbe3bbb0f94b65bcf9`, 26 files, 753,210,169,565 bytes for
the whole quant family; a single quant is what would be fetched. License
`llama3.3`. Lineage per the card: imatrix ("weighted") quants of
`cactopus/Omega_Sapphira_Joyous-L3.3-70B-v1.1`, itself a mergekit SLERP merge,
tagged roleplay/storywriting/unaligned. Static (non-imatrix) quants live in a
sibling repository.

Header read from the real `i1-IQ4_XS` artifact: architecture `llama`, 80
blocks, embedding 8192, 64 heads with 8 KV heads, key and value width 128,
trained context 131,072, 724 tensors. That is an ordinary Llama-3.3-70B shape,
so the pinned source has the relevant architecture support. Actual binary and
ROCm execution remain untested. It is a candidate for a capability upgrade,
not a demonstrated improvement over the installed models.

At the repository's default context of 131,072 with `q4_0` K/V cache, the KV
cache alone is **11.25 GiB**, which dominates the fit:

| Quant | Weights | + KV @131,072 | Against the 56.45 GiB budget |
|---|---:|---:|---|
| `i1-IQ2_M` | 22.46 GiB | 33.71 GiB | fits |
| `i1-IQ3_M` | 29.74 GiB | 40.99 GiB | fits |
| `i1-IQ4_XS` | 35.30 GiB | 46.55 GiB | fits |
| `i1-Q4_K_M` | 39.60 GiB | 50.85 GiB | fits |
| `i1-Q4_1` | 41.27 GiB | 52.52 GiB | fits, with 3.93 GiB spare |
| `i1-Q5_K_S` | 45.32 GiB | 56.57 GiB | **does not fit** |
| `i1-Q6_K` | 53.91 GiB | 65.16 GiB | **does not fit** |

Role against the installed baseline: the general-reasoning alias `heretic` is
20.99 GiB of weights at the same 131,072 context. A 70B at `IQ4_XS` is a
different weight class in the same VRAM envelope, and the coverage plan already
names "high-capability slow specialist" as a slot with **no admitted model**.
Against that, the merge is undocumented in provenance, `unaligned` is a tag and
not evidence of retained capability, and the roleplay/storywriting orientation
is not the workload the slot is for.

**Decision: isolated evaluation, not adoption.** If it is fetched, fetch one
quant — `i1-IQ4_XS` or `i1-IQ3_M` — and gate it exactly like GLM-5.3-Flash: an
otherwise idle host, no alias, no router integration, and quality measured
against `heretic` on real work rather than against a refusal rate.

### Wait — real but blocked on capacity or access

**`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`** — revision
`7e3bfc316b880fefeb049596f11c49d6a18e05fb`, 8 files, 112,242,205,753 bytes.
License `other` / `qwen-community-1.0`. Base model `Qwen/Qwen3.8-Flash-Next`.
Header read from shard 1: architecture **`qwen4exp`**, which the pinned runtime
supports; 48 blocks, embedding 2560, 24 heads with 2 KV heads, key and value
width 256, 512 experts with 10 used, trained context 262,144.

The card says "GGUFs come from `unsloth/Qwen3.8-Flash-Next-GGUF`". They are not
byte-identical to it. Comparing the two repositories at revisions
`7e3bfc316b88` and `38bb39ee9782`, every `UD-Q4_K_XL` shard differs in both
size and SHA-256, by 128-160 bytes per shard, and the projector differs too
(`mmproj-model-bf16.gguf` 907,542,592 bytes here against unsloth's
`mmproj-BF16.gguf` 907,542,944). So these are distinct artifacts derived with
the same recipe, not a republication. Whether the tensors actually carry the
abliteration is **not** established by that comparison; only a load and a
behavioural test would show it, and neither was run.

Fit is the blocker. The four `UD-Q4_K_XL` shards are 103.68 GiB of weights
against 56.45 GiB of aggregate VRAM budget, leaving **47.23 GiB to spill into
host RAM** out of 94.16 GiB total — before the 3.38 GiB KV cache at 131,072,
the 0.85 GiB projector, and the desktop. `mmap = 0` in `llama-models.ini` means
that spill is resident, not paged. A smaller quant is the honest path, and this
repository does not publish one.

**Decision: wait.** It is the strongest lead the queue contains for the
Flash-Next line that the shortlist already lists as research-only, and it is
the only unrestricted candidate whose architecture the pinned runtime already
implements. Revisit if a smaller quant appears, or after the RAM upgrade, and
only as an authorized targeted RAM-spill comparison — never as a mission
default.

**`Lightricks/LTX-2.5`** — revision `5e6e71018ee1756ed329b697a7b4aedc934dfce9`,
17 files, 200,853,702,175 bytes total, license `other` /
`ltx-2.x-community-license-agreement`, and **`gated: auto`**: the terms must be
accepted on the hub before any file can be fetched. That is an access blocker,
not a technical one, and it is the operator's decision to clear, not this
session's.

The full repository is not the download. The distilled int8 audio/video subset —
transformer 20.03 GiB, `gemma4-12b-with-proj` text encoder 14.32 GiB, video VAE
1.37 GiB, audio VAE 0.34 GiB, duration head 0.004 GiB — is **36.06 GiB**, which
is inside the aggregate budget. The bf16 dev subset is 65.30 GiB and is not.
The local ComfyUI checkout (`3216c62`) carries an `LTXAV` model class, a
`duration_head.py` under `comfy/ldm/lightricks/`, and an `ltxav_gemma4_tokenizer`
built on a Gemma-12B-class encoder with separate video and audio projections —
shapes that match these filenames. That is a **structural** match inferred from
key names; nothing was loaded, so it is not a compatibility result.

**Decision: wait, then isolated evaluation.** The video capability is real and
would be new, but the repository's own gate must be cleared first, and
[CANDIDATE-STATUS-2026-09-03.md](../CANDIDATE-STATUS-2026-09-03.md) is explicit
that the already-downloaded ComfyUI stacks (Z-Image, ACE-Step, Qwen Image Edit,
FLUX.2) are still unproven as workflows. Another unproven media stack ahead of
those is not an upgrade.

### Reject for this host

**`GestaltLabs/Qwen3.8-27B-EXL3-11.5GB`** — revision
`18028352ac6686c45c170f823ed6d8476ac4b292`, 71 files, 11,499,270,655 bytes, of
which the weights are three safetensors shards totalling 11,473,119,236 bytes.
License `apache-2.0`; config declares `Qwen3_5ForConditionalGeneration` with a
3:1 linear/full attention interleave.

The format is EXL3, not GGUF: a safetensors container holding EXL3 codes, which
the card is careful to say "does not make EXL3 codes ordinary Transformers
parameters". Its stated runtime is ExLlamaV3 1.4.6, and the card's own
requirements section says to **"use a CUDA-capable NVIDIA system"**, with every
published measurement taken on a single NVIDIA RTX PRO 6000 Blackwell 96 GB
under `PyTorch 2.13.0+cu130`. This host is AMD gfx1030 under ROCm. There is no
gfx1030 path here that does not amount to porting someone else's inference
engine.

Role is also already filled: `obliterated`
(`Qwen3.8-27B-OBLITERATED-Q6_K.gguf`, 20.89 GiB) plus its BF16 projector is an
installed 27B Qwen3.8 with vision, running on the supported runtime.

**Decision: reject for this host.** Not a quality judgement — a backend one.

**`IFM/K2-Horizon-MoVA-36B-A4B-GGUF`** — revision
`c8dde8bc6afe5b28e75c7d78db4fbd65d5fcf679`, 5 files, 74,926,082,404 bytes, of
which `K2-Horizon-36B-BF16.gguf` is 74,924,627,296 bytes (69.78 GiB). License
`apache-2.0`. The card says the repository holds "GGUF versions … for use with
`llama.cpp`".

The pinned runtime cannot load it. The header declares
`general.architecture = "k2-horizon"`, and v0.4.0 has no such architecture in
`src/llama-arch.cpp` and no converter for `K2HorizonForCausalLM`. Being a GGUF
is not the same as being a loadable GGUF, and this is the case that separates
them. Three further facts from the header, read rather than assumed:
`general.name` is `Checkpoint_0002500`; `size_label` is `91x3.7B` with
`expert_count` 100 against a repository named 36B-A4B; the trained context is
524,288, whose KV cost at `q4_0` would be 27 GiB before any weights. **This is
the format-specific check the queue asked for: no conclusion about the MLX MoVA
artifact investigated earlier carries over, in either direction.**

**Decision: reject at the pinned runtime.** Only quantized artifacts under a
supported architecture would change this; a 69.78 GiB BF16 file is not a
desktop candidate regardless.

**`IFM/K2-Horizon-375B-A23B`** — revision
`d33e3ae45281865ebf9f044b12d3635b1d1e17fe`, 76 files, **758,372,571,816 bytes
(706.29 GiB)** across 61 bf16 shards. License `apache-2.0`, architecture
`K2HorizonForCausalLM` / `k2_horizon` with an `auto_map` to repository-local
`modeling_k2_horizon.py` — loading it means executing publisher code, which is
outside what this workspace permits. 192 experts, 8 active plus 1 shared, 61
layers, 524,288 max positions. Notably `mova_num_experts` is **0**: despite the
family name, this checkpoint is not the MoVA variant. No quantized release
exists in this repository.

**Decision: reject.** 706 GiB against 94.16 GiB of RAM is not a capacity
question worth refining, and 23B active parameters do not make it a 23B storage
footprint.

**`inclusionAI/Ling-3.0-flash-Fin`** — revision
`4194bf7d6e4bf31c37a3ab2edfe140a4ac50ba28`, 77 files, **254,999,780,735 bytes
(237.49 GiB)** across 64 bf16 shards. License `mit` — the most permissive here.
Base `inclusionAI/Ling-3.0-flash`; the card states 124B total and 5.1B activated
parameters with a 256K window, finance-tuned with financial institutions, and
that the release is BF16 for SGLang and vLLM.

The architecture is genuinely supported: `BailingMoeV3ForCausalLM` maps to
`conversion/bailingmoe3.py` and the runtime has `LLM_ARCH_BAILINGMOE3`. That
does not make it reachable — converting to GGUF requires the full 237.49 GiB
download first, and no third-party GGUF of *this* finance checkpoint was found.

**Decision: wait, with a standing constraint.** If a GGUF derivative appears at
a quant that fits, it is worth revisiting as a specialist. Even then, a
finance-tuned model gets no autonomous trading or advisory authority; it would
be a drafting and document-reconciliation tool under review, matching how the
repository treats the medical candidate below.

**`inclusionAI/LLaDA-Image`** — revision
`e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039`, 39 files, 49,276,228,561 bytes
(45.89 GiB), license `apache-2.0`, `diffusers` with
`_class_name: LLaDAImagePipeline`. Components: text encoder 30.40 GiB across 9
shards, transformer 12.18 GiB, sigvq 2.42 GiB, text projection 0.61 GiB, VAE
0.16 GiB, queryformer 0.09 GiB.

Two independent blockers. The pipeline is a custom class whose text encoder is
`modeling_llada2uni_moe.LLaDA2MoeModelLM` — repository-local code that
`diffusers` would import and execute, which this workspace does not do. And the
local ComfyUI checkout has no LLaDA support of any kind, so there is no
non-executing path either. FP8 and Turbo variants exist as separate
repositories and were not investigated; they are not in the queue.

**Decision: reject at present.** Image generation and editing are already
downloaded (Z-Image, Qwen Image Edit, FLUX.2) and still unproven as workflows.

**`MohamedAhmedAE/llava-medical-3B-clip-vit-stage2`** — revision
`deef930c59674c335e4501caa3d2ec6322d3ba46`, 22 files, 2,604,016,419 bytes.
**No license is declared on the repository.** `model_type: llava` over a Llama
3.2 3B text tower (`_name_or_path`
`MohamedAhmedAE/Llama-3.2-3B-Instruct-Medical-Finetuned-merged`) with a CLIP ViT
encoder, 28 layers, hidden 3072.

Half the repository is a training checkpoint, not a deployable model:
`training_state/` duplicates the adapter and base weights and adds
`optimizer.pt` (879,096,007 bytes), scheduler, scaler and RNG state. The
deployable part is a 389 MB LoRA adapter plus a 439 MB `model.safetensors` and
25 MB of `non_lora_trainables.bin` — with no merged artifact and no GGUF.

**Decision: reject.** An undeclared license on a Llama-derived medical model is
the blocker on its own; a 3B stage-2 training snapshot is also below the
installed vision baselines. Nothing here would be given clinical authority in
any case.

**`Reallexi-llc/lexipix-models`** — revision
`314776a9ab97cbd46535ff3b974afd4e78d798fe`, 13 files, 10,639,980,998 bytes
(9.91 GiB). **No license is declared.** Its own `manifest.json` describes it as
the artifact bundle for a mobile application ("LexiPix model manifest … the
single source of truth for what can be downloaded"), with `ios/` and `android/`
platform trees.

The contents are real and mostly llama.cpp-compatible — `sd_turbo-f16-q8_0.gguf`
(2.02 GB), `qwen2.5-1.5b-instruct-q4_k_m.gguf` (0.99 GB), three unattributed
`lexi-*` chat GGUFs, and SmolVLM 256M/500M with matching projectors, whose
header confirms architecture `llama`, 32 blocks, embedding 960, context 8192 —
plus a Core ML `.mlpackage.zip` that is iOS-only and irrelevant here. Every one
of them is far below the installed baselines for its role. The `lexi-*` files
carry no upstream attribution in the repository.

**Decision: reject.** Mobile-tier artifacts, undeclared license, no role this
stack lacks.

**`speach1sdef178/MiniMax-H3-Semantic-Bridge`** — revision
`8c2d9b0edb844d6002864a9addd61458dbe81c22`, 69 files, 20,825,454 bytes total.
License `other`. This is not a model: it is an **11,023,032-byte adapter**
(`MiniMaxH3_SemanticBridge_v1.safetensors`) plus a ComfyUI workflow JSON, two
before/after example video pairs, a research article, and 24 extraction and
comparison scripts. It modifies a MiniMax-H3 video model that is **not
installed here**, and the local ComfyUI does carry a `MiniMaxH3` model class,
so the base is at least supported in principle.

**Decision: reject as queued, revisit only if MiniMax-H3 itself is ever
adopted.** An adapter is not a capability without its base model, and the base
model is not in the download queue, is not in the shortlist, and would be a far
larger decision than the 11 MB adapter that prompted this entry.

## Summary

| Repository | Revision | Files | Bytes | License | Decision |
|---|---|---:|---:|---|---|
| `mradermacher/Omega_Sapphira_Joyous-L3.3-70B-v1.1-i1-GGUF` | `b482c8b48894` | 26 | 753,210,169,565 | llama3.3 | isolated evaluation, one quant |
| `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` | `7e3bfc316b88` | 8 | 112,242,205,753 | qwen-community-1.0 | wait, capacity |
| `Lightricks/LTX-2.5` | `5e6e71018ee1` | 17 | 200,853,702,175 | ltx-2.x-community | wait, gated |
| `inclusionAI/Ling-3.0-flash-Fin` | `4194bf7d6e4b` | 77 | 254,999,780,735 | mit | wait, no fitting GGUF |
| `GestaltLabs/Qwen3.8-27B-EXL3-11.5GB` | `18028352ac66` | 71 | 11,499,270,655 | apache-2.0 | reject, CUDA-only backend |
| `IFM/K2-Horizon-MoVA-36B-A4B-GGUF` | `c8dde8bc6afe` | 5 | 74,926,082,404 | apache-2.0 | reject, arch unsupported |
| `IFM/K2-Horizon-375B-A23B` | `d33e3ae45281` | 76 | 758,372,571,816 | apache-2.0 | reject, 706 GiB |
| `inclusionAI/LLaDA-Image` | `e4e2703f410f` | 39 | 49,276,228,561 | apache-2.0 | reject, remote code + no ComfyUI path |
| `MohamedAhmedAE/llava-medical-3B-clip-vit-stage2` | `deef930c5967` | 22 | 2,604,016,419 | **none declared** | reject |
| `Reallexi-llc/lexipix-models` | `314776a9ab97` | 13 | 10,639,980,998 | **none declared** | reject |
| `speach1sdef178/MiniMax-H3-Semantic-Bridge` | `8c2d9b0edb84` | 69 | 20,825,454 | other | reject, adapter without base |

Eleven of eleven investigated. One isolated-evaluation candidate, three waits,
seven rejects. **Nothing here is downloaded, admitted, or GPU-qualified**, and
the "do not download next" rule in
[CANDIDATE-STATUS-2026-09-03.md](../CANDIDATE-STATUS-2026-09-03.md) still
stands: the weights already on disk are unproven as product workflows, and that
outranks every candidate above.
