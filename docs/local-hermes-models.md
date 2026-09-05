# Local Hermes models - architecture, plan, and status

Last updated: 2026-09-04

This is the source-of-truth ledger for local LLMs on `frankenstein`. It records verified state, model-selection reasoning, operating commands, and unfinished work.

## Goal

Expose local GGUF models in Hermes CLI, TUI, and Desktop through the normal model picker. Models load on demand and only one large model stays in VRAM at a time.

## Hardware and runtime

- GPUs: `ROCm0` RX 6900 XT 16 GiB, `ROCm1` Radeon Pro V620 32 GiB, `ROCm2` RX 6900 XT 16 GiB.
- GPU split: default layer split with `--device ROCm0,ROCm1,ROCm2 --tensor-split 1,2,1`.
- RAM: 94 GiB.
- Home storage: 2.8 TiB free at the last check.
- llama.cpp: tracked submodule `upstream/llama.cpp`, pinned to v0.4.0 (`5266f24`), with the HIP/ROCm binary at `upstream/llama.cpp/build/bin/llama-server`.
- llama.cpp supports router mode (`--models-preset`) and native `draft-mtp`.
- Context policy: 131,072 tokens, one slot, Q4_0 K/V cache, Flash Attention.
- Endpoint: loopback only, `http://127.0.0.1:8080/v1`.

Do not use `--split-mode tensor`: Qwen3.8 MTP backend sampling is incompatible with that path. Build the pinned submodule only through `scripts/build-llama-cpp.sh`.

## Model roster

### Ridge - compact/fast baseline

- Source: https://huggingface.co/empero-ai/Qwen3.8-27B-Ridge-GGUF
- File: `/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-Ridge-3.7bpw.gguf`
- Size: 11.73 GiB.
- Alias: `ridge`.
- Native MTP verified.
- Role: fastest compact option and a useful baseline, not the maximum-quality daily driver on this 64-GiB aggregate-VRAM system.

### RVN Heretic - preferred uncensored daily driver

- Source: https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF
- Heretic/ARA method: https://github.com/p-e-w/heretic
- File: `/home/typhoon/git/frankenstein-llm/models/RVN-Q6_K-multilingual-mtp.gguf`
- Expected size: 22,533,850,592 bytes.
- Expected SHA-256: `1344d07425d73f0d1b8f36213910eae8fec21061b1a90835927e8485238f93a4`.
- Alias: `heretic`.
- Why selected: Q6 fidelity, Qwen multilingual/code/tool calibration, embedded MTP, official chat template, and the best published preservation/refusal tradeoff found in this sweep (publisher reports KL about 0.0085 and 0-1/100 refusals).
- Caveat: those behavior figures are publisher measurements, not independently reproduced here.

### OBLITERATUS V3 - maximum liberation option

- Source: https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED
- File: `/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-Q6_K.gguf`
- Expected size: 22,430,991,392 bytes.
- Expected SHA-256: `3535d4a15b75840fb391138ce04e6c73fde709320c6f5fe788cdbd586ee08e3a`.
- Alias: `obliterated`.
- Version: V3 with corrected baked template and restored MTP/vision tensors.
- Matching projector: `/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf` (931,145,888 bytes, SHA-256 `e484e3b7e907ed0e0644c0de56c3f5929c7ad5c9c6cc84d35a9d8dc08d461545`, repo revision `a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8`). Downloaded and publisher-hash verified 2026-09-01. Not yet wired into the production alias; vision gates remain pending.
- Role: use when avoiding soft deflections matters more than staying maximally close to stock.
- Published cost: MMLU 82.3% vs stock 84.5% (-2.1 percentage points); STEM was the most affected category. Recommended local agent sampling is temperature 0.2 with repetition penalty 1.15.

## Why both uncensored models

They optimize different objectives:

- `heretic`: best evidence-backed near-stock daily driver.
- `obliterated`: more aggressive removal of hard refusals and soft safety-lecture deflections, with a documented capability cost.
- `ridge`: compact speed/fit baseline.

### Fable Fusion 711 - intelligent unrestricted fantasy writer

- Source: https://huggingface.co/DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF
- File: `/home/typhoon/git/frankenstein-llm/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-Q6_K.gguf`.
- Expected SHA-256: `c394c7c0dcbd32e04f4695ad5d4dad8f2f5b7286b366d165815ac0cdc5ccf4aa`.
- Alias: `fable`.
- Role: Qwen3.6 27B Q6_K, Heretic/uncensored creative-writing and roleplay generalist with AMD-oriented MTP tensors. Use when plot logic, continuity, instruction following, and explicit fantasy prose all matter.
- Context preset: 65,536 tokens with native draft-MTP.

### Phr00tyMix v4 - prose-first unrestricted writer

- Source: https://huggingface.co/Phr00t/Phr00tyMix-v4-32B-GGUF
- File: `/home/typhoon/git/frankenstein-llm/models/Phr00tyMix-v4-32B-imat-Q6_K.gguf`.
- Expected SHA-256: `056fc85cb1f0873e90ddae4f6f747fbc4ce3ebf41fad15bfdc2a7e66e2af1700`.
- Alias: `phr00ty`.
- Role: 32B Q6_K prose/RP specialist combining creative, roleplay, and uncensored lineages. v4 supersedes v3 and specifically targets v3's coherence and instruction-following regression.
- Context/sampling preset: 65,536 tokens, temperature 1.5, min-p 0.1. The GGUF declares a native 131,072-token training context; 65,536 satisfies Hermes's minimum while retaining runtime headroom. It has no MTP tensors, so draft-MTP is intentionally not inherited by this preset.

The writer pair is complementary rather than redundant: `fable` is the intelligence/continuity-first unrestricted writer; `phr00ty` is the voice/prose/RP-first specialist.

Abliteration is not fine-tuning and does not add knowledge. It edits refusal behavior. Do not represent an abliterated model as safe or expose this loopback service publicly.

## Deferred/not selected

- `zai-org/GLM-5.3`: strong 753B/39B-active text flagship for hosted coding and authorized security comparison, but even aggressive local quants require hundreds of GiB. It is not a local candidate for this workstation.
- `zai-org/GLM-5.3-Flash`: promising 320B/18B-active multimodal hosted specialist. Its 93-120 GB extreme local quants technically approach the machine's combined RAM+VRAM envelope but retain only 71-82% in Unsloth's quant metric, require experimental `glm5next` llama.cpp support, and would rely on slow hybrid offload. The preferred approximately 200-GB 4-bit quant does not fit. Do not download yet.
- `orcarouter/GLM-5.3-Flash-Uncensored-FP8`: 306-GiB, 62-shard refusal-edited Flash checkpoint. Publisher refusal rates improved substantially, but standard capability retention is explicitly unverified, there was no exact GGUF derivative in the checked tree, and the current stable router lacks `glm5next`. Existing `heretic` remains the practical unrestricted local choice.
- `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF`: unusually good documentation and measured capability deltas, but its published point retains 12/100 refusals and reports higher KL than RVN. Keep as a reproducible comparison candidate.
- `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`: solid standard-quant/MTP release, but no demonstrated advantage over the chosen pair for this use.
- `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`: popular, but the chosen alternatives provide clearer behavior/provenance evidence.
- `Qwythos-27B-v1`: the original repo is Transformers/safetensors; do not download that 55-GB BF16 release merely to run llama.cpp. Re-evaluate its GGUF derivatives separately.
- `qwen38-mtp`: guidance/runtime flags, not another target chat model.
- Heretic/OBLITERATUS surgery tools: do not run locally; serve published artifacts only.
- Ornith 9B: no longer part of the immediate Desktop plan.

## Router architecture

Preset: `/home/typhoon/git/frankenstein-llm/llama-models.ini`

Service: `llama-router.service`

The router publishes `ridge`, `heretic`, `obliterated`, `fable`, and `phr00ty`, autoloads the requested model, and uses `--models-max 1` so switching evicts the previous model rather than exhausting VRAM. The first prompt after a switch waits for model loading; later prompts are immediate.

The old `llama-ridge.service` remains available as a rollback unit but must not run at the same time as `llama-router.service` because both bind port 8080 and own the same GPUs.

## Hermes and Desktop usage

All Hermes front ends share `~/.hermes/config.yaml`. Desktop does not need a separate model installation.

Select inside CLI/TUI/Desktop:

```text
/model heretic
/model obliterated
/model ridge
/model fable
/model phr00ty
```

The normal `/model` picker should also list all router-discovered models. Prefer a fresh session when changing model families; switching a long cloud-model conversation midstream gives the local model a history created under different hidden assumptions.

One-shot verification form:

```bash
hermes -z 'Reply with exactly pong.' --provider llamacpp-local -m heretic --reasoning none --ignore-rules
```

## Operations

```bash
git submodule update --init --recursive upstream/llama.cpp
scripts/build-llama-cpp.sh
systemctl --user status llama-router.service
systemctl --user restart llama-router.service
/home/typhoon/git/frankenstein-llm/scripts/local-model-status.sh
journalctl --user -u llama-router.service -f
```

Free all LLM VRAM for ComfyUI:

```bash
systemctl --user stop llama-router.service
```

Restore local models:

```bash
systemctl --user start llama-router.service
```

## Desktop installation

The first `hermes desktop` attempt reached npm deprecation warnings but failed in Electron 40.10.2's postinstall because the GitHub release asset timed out. The dependency code and native builds were not the cause. The canonical Electron mirror was reachable and the bounded build succeeded with:

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ hermes desktop --build-only
```

Verified artifact:

```text
/home/typhoon/.hermes/hermes-agent/apps/desktop/release/linux-unpacked/Hermes
```

Verified launcher entry:

```text
/home/typhoon/.local/share/applications/hermes.desktop
```

Launch with:

```bash
hermes desktop --skip-build
```

The npm `deprecated` lines shown in the screenshot are warnings, not by themselves a build failure. The final exit code and packaged artifact are authoritative.

## Crash and storage recovery

Two power interruptions occurred during the Heretic download. ZFS correctly rejected one full-size replacement with `Input/output error`; another full-size pre-crash copy failed the publisher SHA-256. Both invalid, replaceable downloads were removed. A fresh/resumed download was promoted only after the complete 22,533,850,592-byte file matched SHA-256 `1344d07425d73f0d1b8f36213910eae8fec21061b1a90835927e8485238f93a4`.

`zpool status zroot` initially retained four data errors after the outages while the single NVMe device counters remained `READ 0 / WRITE 0 / CKSUM 0`. The recovery scrub completed on Sat Aug 29 22:10:39 2026: `scrub repaired 0B in 01:14:31 with 0 errors`. Final pool/device counters were `READ 0 / WRITE 0 / CKSUM 0`, and ZFS reported `errors: No known data errors`.

If a future event reports affected paths, the privileged listing is:

```bash
sudo zpool status -v zroot
```

Do not run `zpool clear` until the scrub result and affected paths have been reviewed. The model artifacts are replaceable; unrelated user-data paths require restore/backup decisions rather than blind deletion.

## GLM-5.3-Flash local candidate status

Deferred for one clean idle-host retry on the current 96 GiB/three-GPU workstation. The 15-shard `6block/GLM-5.3-Flash-GGUF` IQ3_XXS artifact was downloaded and publisher-hash verified, and isolated llama.cpp PR #27752 support was built. Static placement exceeded a 16 GiB GPU. Auto-fit consumed roughly 66 GiB RSS and entered sustained reclaim pressure, but that attempt overlapped a 44-thread Linux kernel compile and is inconclusive.

No GLM alias was added. Retry 32K only after verifying no kernel compile, linker, downloader, or other heavy workload is active. Keep 64K, A/B, OrcaRouter-derivative, and router integration blocked until a clean 32K pass. Do not change ARC, swap, or other host policy to make this candidate load.

Full evidence and acceptance gates are in `docs/LOCAL-AI-MODEL-STRATEGY.md`.

## Current implementation status

| Item | Status |
|---|---|
| Ridge download/service/API/Hermes proof | Done |
| Router preset syntax and five-model discovery | Done; all five aliases visible through `/v1/models` and Desktop picker |
| Q6 OBLITERATUS V3 download/checksum | Done; publisher SHA-256 matched |
| Q6 RVN multilingual MTP download/checksum | Done; publisher SHA-256 matched after crash recovery |
| Router service file and unit verification | Active, enabled, loopback-only; Ridge rollback unit disabled/inactive |
| Hermes `llamacpp-local` provider and five aliases | Configured; cloud default remains `openai-codex / gpt-5.6-sol` |
| Direct API test for Ridge and OBLITERATUS | Done; deterministic completion passed and one-model eviction observed |
| Direct API test for Heretic | Done; deterministic completion passed after scrub |
| Hermes one-shot test for Ridge and OBLITERATUS | Done; aliases resolved, completion passed, zero estimated API cost |
| Hermes one-shot test for Heretic | Done; alias resolved, exact completion passed, one API call |
| Desktop packaged build | Done: Electron 40.10.2 Linux unpacked app and launcher entry verified |
| Desktop launch | Done: packaged Wayland window mapped in Hyprland and rendered Skills Hub without fatal overlay |
| Desktop picker visual confirmation | Done; packaged Desktop visibly listed all five aliases under `LOCAL LLAMA.CPP ROUTER` |
| Fable/Phr00ty download and checksum | Done; both complete files matched their publisher SHA-256 values before atomic promotion |
| Fable/Phr00ty direct API and Hermes acceptance | Done; direct deterministic completion passed and each Hermes alias completed one API call through the custom provider |
| ZFS post-outage scrub | Done; 0B repaired, 0 errors, zero device counters, no known data errors |

## Files

- `docs/local-hermes-models.md`: this source-of-truth ledger.
- `docs/HERMES-DESKTOP-LOCAL-MODELS.md`: short user operating guide.
- `llama-models.ini`: router presets.
- `scripts/download-uncensored-models.sh`: resumable downloads plus SHA-256 verification.
- `scripts/download-heretic-clean.sh`: crash-recovery download with range resume, full SHA-256 gate, and atomic promotion.
- `scripts/download-writing-models.sh`: resumable Fable/Phr00ty downloads with publisher SHA-256 gates and atomic promotion.
- `scripts/verify-router-models.py`: deterministic direct-router acceptance suite.
- `scripts/verify-hermes-models.sh`: deterministic Hermes alias acceptance suite.
- `scripts/local-model-status.sh`: health and model-state inspection.
- `scripts/serve-ridge.sh`: legacy single-Ridge rollback path.
- `hermes-llamacpp-ridge.yaml`: historical Ridge-only configuration example; superseded by the router.
