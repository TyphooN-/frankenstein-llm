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
they are not a policy to wait for tagged releases. The existing v0.4.0 revision
is the last recorded build until a master update is compiled and verified.
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

## Rollback

Check out the prior outer repository commit and run `git submodule update --init --recursive`. Rebuild the pinned prior gitlink, then reinstall the tracked user-service units. Do not retain an untracked second production checkout as a rollback mechanism.

## Consequences

Source, provenance, build output, services, models, and verification now live under `/home/typhoon/git/frankenstein-llm`. The upstream source remains independently reviewable, while the outer repository records exactly which revision the stack uses. Fresh clones require explicit submodule initialization and a local ROCm build; generated build products remain untracked.
