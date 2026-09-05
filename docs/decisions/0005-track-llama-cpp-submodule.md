# ADR 0005: Track llama.cpp as a pinned in-repository submodule

- Status: Accepted
- Date: 2026-09-04

## Context

The local model router previously depended on an untracked external llama.cpp checkout and user-local symlinks into its build directory. Removing that external checkout left the router process alive but made every future model spawn and service restart fail.

llama.cpp is a source dependency with its own Git history. Copying its source into this repository would obscure provenance and make upstream updates difficult to review. Keeping another untracked checkout outside the canonical repository would repeat the failure.

Release v0.4.0 is required for this stack. Relevant changes include Gemma-4 vision and assistant fixes, Qwen3.8-Flash-Next support, lower model-load RAM peaks, sparse flash attention, per-slot context limits, data-URL media input, and stricter server/tool-call handling.

## Decision

Track `https://github.com/ggml-org/llama.cpp.git` as a Git submodule at:

`upstream/llama.cpp`

The outer repository gitlink and `upstream/llama-cpp.lock.json` both pin tag `v0.4.0` at commit `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`. A mismatch between the lock and the submodule worktree fails the build before CMake runs. The gitlink is checked separately by `verification/upstream-pin/test_llama_cpp_pin.py`, because the build script reads the worktree and cannot see a stale index.

Build in the submodule's ignored `build/` directory with:

- ROCm/HIP backend;
- `gfx1030`, matching all three installed GPUs;
- Ninja and `Release` configuration;
- automatic logical-CPU discovery through `nproc`;
- `llama-server`, `llama-cli`, `llama-quantize`, and `llama-gguf` targets.

Use `scripts/build-llama-cpp.sh` as the canonical build entry point. Router, sidecar, converter, and rollback scripts execute binaries directly from `upstream/llama.cpp/build/bin/`. Do not create required runtime links under `~/.local/bin`, and do not recreate `/home/typhoon/src`.

The router remains loopback-only and limited to one resident model. A version update is not accepted until the binary identity, required CLI flags, router startup, model discovery, deterministic text/structured/tool behavior, relevant Gemma-4 vision behavior, and unload paths pass. Performance measurements remain out of scope until separately authorized.

## Updating

1. Review the upstream release and resolve its tag to an exact commit.
2. In the submodule, fetch tags and check out that commit detached.
3. Update `upstream/llama-cpp.lock.json` to the same tag and commit.
4. Stage the submodule and confirm `git ls-files -s upstream/llama.cpp` reports that same commit. Staging picks up whatever the worktree is on, so a fetch left checked out on `master` pins `master` here while every other record still reads the release.
5. Run `scripts/build-llama-cpp.sh` only when no other optimized build is active.
6. Run functional qualification without token-rate or latency measurements.
7. Commit the outer gitlink, lock, service paths, documentation, and verification together.

A checkout is restored with:

    git submodule update --init --recursive upstream/llama.cpp

## Rollback

Check out the prior outer repository commit and run `git submodule update --init --recursive`. Rebuild the pinned prior gitlink, then reinstall the tracked user-service units. Do not retain an untracked second production checkout as a rollback mechanism.

## Consequences

Source, provenance, build output, services, models, and verification now live under `/home/typhoon/git/frankenstein-llm`. The upstream source remains independently reviewable, while the outer repository records exactly which revision the stack uses. Fresh clones require explicit submodule initialization and a local ROCm build; generated build products remain untracked.
