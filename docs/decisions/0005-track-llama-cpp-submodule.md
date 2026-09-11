# ADR 0005: Track llama.cpp master with reproducible revision records

- Status: Accepted
- Date: 2026-09-04

## Context

The local model router previously depended on an untracked external llama.cpp checkout and user-local symlinks into its build directory. Removing that external checkout left the router process alive but made every future model spawn and service restart fail.

llama.cpp is a source dependency with its own Git history. Copying its source into this repository would obscure provenance and make upstream updates difficult to review. Keeping another untracked checkout outside the canonical repository would repeat the failure.

Release v0.4.0 is required for this stack. Relevant changes include Gemma-4 vision and assistant fixes, Qwen3.8-Flash-Next support, lower model-load RAM peaks, sparse flash attention, per-slot context limits, data-URL media input, and stricter server/tool-call handling.

## Decision

Track `https://github.com/ggml-org/llama.cpp.git` as a Git submodule at:

`upstream/llama.cpp`

The update policy is current upstream `master`, not release tags. Resolve master
at the start of each update and retain its exact SHA in the outer gitlink and
`upstream/llama-cpp.lock.json`. Exact revision records preserve reproducibility;
they are not a policy to wait for tagged releases. The September 10 master build
is recorded below separately from full runtime qualification.
A mismatch between the lock and the submodule worktree fails the build before
CMake runs. The gitlink is checked separately by
`verification/upstream-pin/test_llama_cpp_pin.py`.

Build in the submodule's ignored `build/` directory with:

- ROCm/HIP backend;
- `gfx1030`, matching all three installed GPUs;
- Ninja and `Release` configuration;
- automatic logical-CPU discovery through `nproc`;
- `llama-server`, `llama-cli`, `llama-quantize`, and `llama-gguf` targets.

Use `scripts/build-llama-cpp.sh` as the canonical build entry point. Router, sidecar, converter, and rollback scripts execute binaries directly from `upstream/llama.cpp/build/bin/`. Do not create required runtime links under `~/.local/bin`, and do not recreate `/home/typhoon/src`.

The router remains loopback-only and limited to one resident model. A version update is not accepted until the binary identity, required CLI flags, router startup, model discovery, deterministic text/structured/tool behavior, relevant Gemma-4 vision behavior, and unload paths pass. Performance measurements remain out of scope until separately authorized.

## Updating

1. Fetch upstream `master`, resolve it to an exact commit, and inspect the delta
   from the running binary, including ROCm, model-loading, and flag changes.
2. Compile that exact master revision, not the latest release tag. Keep the
   working runtime intact until the candidate build succeeds; never update
   sources or binaries underneath an active qualification or benchmark.
3. Record branch `master` and the exact compiled commit in
   `upstream/llama-cpp.lock.json`; remove the obsolete release-tag selector when
   the new build is integrated.
4. Stage the submodule and confirm `git ls-files -s upstream/llama.cpp` reports that same commit. Staging picks up whatever the worktree is on, so a fetch left checked out on `master` pins `master` here while every other record still reads the release.
5. Run `scripts/build-llama-cpp.sh` only when no other optimized build is active.
6. Verify binary identity, CLI/preset compatibility, and functional behavior.
   Controlled Radeon/ROCm benchmarking is now separately authorized. Re-baseline
   MTP-off/on measurements after each runtime, quant, or topology change; do not
   carry an old optimum into the new build without measurement.
7. Commit the outer gitlink, lock, service paths, documentation, and verification together.

A checkout is restored with:

    git submodule update --init --recursive upstream/llama.cpp

## Further performance work

Status: Planned; no unmeasured speedup or optimal configuration is claimed.
The requested compiler direction is **full LTO**, not fat LTO objects. Fat
objects package native code alongside intermediate representation and are not
the performance objective. Preserve correctness and qualified capabilities;
do not trade them for a headline token rate.

### Recorded build baseline (2026-09-10)

- Source: `df03399b885831b2a1603b3abb0d8c156808e363`, build 10902.
- ROCm/HIP `gfx1030`, Release `-O3`, native CPU tuning, HIP graphs,
  Flash Attention and CPU weight repacking.
- `GGML_LTO=ON` produced GCC `-flto=auto` compile/link commands for GGML
  CPU/core targets. This establishes full GCC LTO in that scope, not
  whole-program LTO across the server, model code, shared libraries and HIP
  device kernels.
- Build exit 0; all three Radeon devices enumerated; a live embedding request
  returned 4096 finite values. Repository tests passed with generated scratch
  fixtures excluded: 1040 tests and 917 subtests. These checks do not replace
  the complete text/tool/vision/unload qualification gates.
- Local logs, binary checksum, rollback binaries and smoke evidence are in
  ignored `proofs/llama-master-update/`. Compilation overlapped user-authorized
  system stress; no controlled performance conclusion follows from it.

### Investigation backlog

- [ ] Extend and verify full host LTO coverage over model, common and server
  targets. Inspect actual compile/link commands and object provenance; a CMake
  switch alone is not proof. Compare shared-library and broader-link scope
  without sacrificing backend compatibility. Investigate HIP device LTO
  separately against the installed ROCm toolchain rather than assuming host
  LTO enables it or mixing incompatible GCC/LLVM intermediate representations.
- [ ] Evaluate profile-guided optimization with representative prefill, decode,
  tool, long-context and CPU-offload workloads. Keep profile training separate
  from held-out evaluation; bind profiles to exact source and compiler versions.
- [ ] Compare applicable ROCm kernel selection, graph execution and allocation
  paths. Test forced MMQ versus automatic dispatch only where supported;
  exclude CUDA-only and CDNA-only assumptions from Radeon gfx1030 decisions.
- [ ] Sweep batch/ubatch sizes, CPU thread counts, context/cache settings and
  supported GPU split modes/placements per artifact. Include desktop reserve,
  sustained V620 thermals and the two 6900 XTs; capacity ratios alone are not
  performance-optimal placement evidence. Treat cache precision as a quality
  variable, not a free speed optimization.
- [ ] Jointly sweep embedded-head MTP draft depth and confidence, including
  MTP-off and ungated controls, for each exact model/quant/placement. Start from
  existing presets but do not accept depth 2 as an optimum. Refine or expand
  around measured winners after every runtime or topology change. The
  [qwen38-mtp reference](https://github.com/sudoingX/qwen38-mtp) supplies
  hypotheses and methodology, not transferable NVIDIA performance results.

### Acceptance and promotion

Run repeated, interleaved comparisons on a quiet host with matched prompts,
sampling, context, cache state and exact artifacts. Separate cold loading,
prefill, time to first token, decode and end-to-end latency; record variability,
RAM/VRAM, thermal behavior and correctness. Optimize useful completed work, not
MTP acceptance rate alone. Stress-run functional evidence must be labeled
confounded for comparative timing.

Retain the baseline and rollback until applicable functional gates pass and
reproducible measurements justify promotion. Avoid blanket fast-math, unsafe
floating-point assumptions or unsupported ISA flags. Store raw measurements,
effective build/runtime arguments, toolchain/source/binary identities and a
decision report under `proofs/`; record selected settings in maintained config
only after evidence supports them. This backlog does not itself launch tuning
or modify serving defaults.

## Rollback

Check out the prior outer repository commit and run `git submodule update --init --recursive`. Rebuild the pinned prior gitlink, then reinstall the tracked user-service units. Do not retain an untracked second production checkout as a rollback mechanism.

## Consequences

Source, provenance, build output, services, models, and verification now live under `/home/typhoon/git/frankenstein-llm`. The upstream source remains independently reviewable, while the outer repository records exactly which revision the stack uses. Fresh clones require explicit submodule initialization and a local ROCm build; generated build products remain untracked.
