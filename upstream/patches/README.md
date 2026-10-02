# Local convolution-weight compatibility patches

Current checkout and live runtime: `bed0a856606ee4a24a164066f73d2379447033f5` with the SHA256-declared new-base patch. The older patch below is retained as historical reproduction evidence.

## Historical original patch

Base: llama.cpp `9f31776c3773cf03f98535c19b7e6d394af374b4`.

`0001-convolution-kernel-types.patch` is a private local patch, not an upstream submission. It applies only to the historical base. The current checkout uses the new-base patch below. No upstream patch submission, push, or PR has been made.

## Behavior

A shared graph helper casts non-F32 SSM/convolution kernels to F32 before `ggml_ssm_conv`. F32 kernels stay unchanged. Eleven call sites in ten model source files use the helper. Backend assertions remain intact. No model, quantization, security, or GPU kernel settings are changed.

The existing architecture test gains `--conv-kernel-types`. It compares F16/BF16 kernels against F32 kernels containing exactly the same rounded values, including CPU and available GPU execution. It checks that at least one actual convolution tensor was exercised. This matters: `llama_model_size()` is not a reliable indicator for in-memory synthetic fixtures and initially caused false skips.

## Verification performed

A separate CPU build lives at `upstream/llama.cpp/build-hermes-ssm-cpu`. The live `build/bin` libraries were not overwritten.

Before applying the graph fix, the qwen4exp mixed-kernel regression aborted with exit 134 at the float-stride assertion. After applying it, qwen4exp F16 and BF16 cases passed with NMSE 0.0. Focused runs for qwen35, qwen35moe, qwen3next, mamba, mamba2, lfm2, and plamo2 exited 0. The existing CPU SSM_CONV backend test also exited 0. `git diff --check` and reverse patch applicability checks passed. A patched CPU `llama-server` was built and its `--version` exercised.

Reproduction after applying the patch to a clean pinned checkout:

```sh
cd upstream/llama.cpp
git apply ../patches/0001-convolution-kernel-types.patch
cmake -S . -B build-hermes-ssm-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build-hermes-ssm-cpu --target test-llama-archs test-backend-ops llama-server -j 6
OMP_NUM_THREADS=2 build-hermes-ssm-cpu/bin/test-llama-archs --conv-kernel-types -a qwen4exp -s 42
OMP_NUM_THREADS=2 build-hermes-ssm-cpu/bin/test-backend-ops test -o SSM_CONV -b CPU
```

The mixed-type test deliberately aborts a fresh executable child running the invalid raw F16 operation. Its assertion message is expected; the parent must exit 0 and report actual architecture comparisons. The parent uses `posix_spawnp` rather than a bare `fork`: `common_init()` has already started the asynchronous logger, so a forked child would inherit threaded-process locks. The negative case is dispatched before logger initialization, disables core dumps locally, and reports initialization failures separately from the expected SIGABRT. The parent bounds its wait to 30 seconds and reaps only its own child.

## Provisional ROCm and real-model proof

Saved numerical results cover qwen4exp, qwen35, qwen35moe, qwen3next, mamba, mamba2, lfm2, and plamo2: F16 and BF16 kernels on each of the three AMD GPUs, with 48 successful comparisons against the rounded CPU reference. This is numerical compatibility proof, not a performance benchmark or a certificate of filesystem integrity.

The real CYBER-FROST GGUF completed load, warmup, four-token generation, exact `PONG`, and schema-constrained `{"status":"ready","count":3}` checks in an isolated same-pin runtime. The private server and test harness exited 0. Placement used context 256, 36 GPU layers, `per_layer_token_embd=CPU`, and a 1:2:1 layer split. This runtime combines the patched graph library with the unchanged same-revision HIP backend; it is not a full fresh HIP rebuild, live-router qualification, or full-context/memory-recovery acceptance.

A user-approved Hermes/GPT source review found no blocking F16/BF16 CPU/HIP runtime-code defect, but identified the threaded-fork hazard in the negative test. The fresh-process change above fixes that finding. Its CPU regression was verified red/green, and the registered `test-conv-kernel-types` CTest passed. Opus review did not complete before its included-usage session limit; the alternate review must not be misrepresented as a completed Opus review.

## Not completed / activation constraints

- The patched `9f31776c` runtime was deployed through a dedicated release and owned systemd drop-in after explicit authorization. Saved live CYBER-FROST canaries passed exact `PONG` and schema JSON at context 32768 after removing the publisher template's forced thinking assignment. This is functional proof, not strict resource or filesystem acceptance.
- Host Btrfs/artifact integrity remains unresolved; provisional successful inference does not erase checksum-corruption evidence. Final scrub status and independent trusted backup/stability evidence are still required before production acceptance.
- Repeated host reboots interrupted execution and left some lane-created objects/logs empty. Evidence is private and fsynced where possible, but copies on the same affected filesystem are not independent backups.
- Automatic candidate qualification starts at login. Its completed 14-candidate batch recorded nine passes and five failures under a kernel-build conflict waiver. Four response-capable models failed RAM recovery; an unwaived retest refused admission during the active kernel build. No thresholds were relaxed and no user workload was stopped.
- The fresh new-base HIP build, selected runtime artifact checks, authorized cutover, and targeted strict recovery retests are now complete as described below. Whole-host integrity and untested backend/model coverage remain separate. No paid overage was enabled.
- `scripts/build-llama-cpp.sh` now invokes `scripts/verify_llama_patches.py`: an undeclared dirty submodule still fails, and a declared patch is accepted only when its SHA256 and complete resulting tracked source tree match. The verifier leaves the real index and source unchanged. The primary pin and gitlink now name the exact new upstream base, with the declared new-base patch applied and verified.
- Existing parent-repository changes were preserved and are not part of this patch.

Hermes model visibility was independently checked against the installed pipx runtime: live `/v1/models`, `fetch_api_models`, and the actual provider-picker assembly each returned all 29 aliases including `cyber-frost-38`. No Hermes source patch was necessary for that currently observed state. This does not verify that every alias loads or is appropriate for chat (some are embedding/reranker presets).

## Latest-upstream runtime (activated)

The separately preserved `0001-convolution-kernel-types-bed0a8566.patch` targets official `master` revision `bed0a856606ee4a24a164066f73d2379447033f5`, observed on 2026-10-02. Do not apply the older patch above to this base.

This port retains the upstream architecture test's regex filter, backend selection, and seed/stdev interfaces, including correctly typed callback userdata. All twelve graph call sites in eleven model files use the shared conversion helper. The new GLM5-next site was verified red (SIGABRT at the same float-stride assertion), then green (F16/BF16 CPU NMSE 0.0). The complete registered convolution CTest and CPU SSM_CONV backend test exited 0 on the new base.

A fresh matching HIP/server build completed all six requested targets. Upstream ggml is now 0.25.3; never mix its graph/server libraries with the older 0.24.0 HIP backend. CPU and three-GPU numerical regressions passed. User-approved independent GPT-6.1 review found one patch-verifier bypass involving index flags; it was reproduced, fixed, and independently closed. The new runtime was copied into a dedicated release, cold-probed for exact 29-alias parity and actual library origins, and activated with a new owned override while retaining the prior release/override for rollback. Live CYBER-FROST exact PONG and schema JSON passed at its unchanged 32768-context profile. The primary submodule now uses the same upstream revision and declared patch; the original source changes were preserved in a named Git stash before migration.

The patch SHA256 and candidate lock are checked with a temporary Git index: reconstruct the declared tree from the exact base, compare actual tracked source, and reject missing patches, extra tracked edits, changed patch bytes, wrong bases, and escaping/symlinked manifest paths. This writes disposable Git objects but does not stage into the real index, reset source, or apply changes to the real checkout. Ignored/generated build outputs are not certified by that check.

## CYBER-FROST template asset

`config/chat-templates/cyber-frost-3.8.jinja` is the publisher's embedded GGUF template with only the first `{%- set enable_thinking = true %}` assignment removed. The remaining 8,971 bytes are unchanged, including chat, vision, and tool formatting. The preset references this repository asset relative to the repository working directory rather than tying it to one installed runtime release. The full 32768-context, `13,27,10` split, CPU per-layer embedding, and F16 KV profile remain unchanged.

Provenance hashes:

- Publisher embedded template: `ba1946683f7615254fb246f0c0a652fd3aa02066ef8328ed4b8af08219749395`.
- Request-controlled asset: `666b82b29f5801f4f546e5724b45bf5f14be7d20b66149df44164626b072ce6d`.

Strict targeted retests freshly passed coherence, JSON, unloading, VRAM/RAM recovery for seven aliases: `distill-35b-a3b-abliterated`, `mindmeld-27b`, `penjing-27b`, `qwen25-coder-32b-abliterated`, `cyber-frost-38`, `genesis-27b`, and `qwopus`. No conflict waiver or cached qualification was used; the original 2 GiB RAM tolerance stayed unchanged. Qwopus also passed its required tool-call check. This is not qualification of all 29 catalog entries.

To reproduce the current patched build from an initialized clean submodule:

```sh
git -C upstream/llama.cpp apply ../patches/0001-convolution-kernel-types-bed0a8566.patch
python3 scripts/verify_llama_patches.py
scripts/build-llama-cpp.sh  # host-detected backend(s); no source guard bypass
```

The verifier refuses missing patches and visibility-suppressing index flags; it does not automatically erase user edits or clear flags.

The offline upstream-pin/build/template/patch-policy suite currently contains 53 passing tests. Backend orchestration uses documented fake compiler fixtures; patch enforcement uses real disposable Git trees. They do not certify host filesystem integrity or untested backend/model coverage.
