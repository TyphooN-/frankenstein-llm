# GLM-5.3-Flash Isolated Build and Functional Runbook

Production source: `/home/typhoon/git/frankenstein-llm/upstream/llama.cpp`

Production revision: v0.4.0 at `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`.

The former external PR #27773 worktree no longer exists. v0.4.0 has no `glm5next` architecture identifier, so GLM qualification is blocked until a still-required experimental revision is reviewed and recreated as an isolated worktree below `upstream/llama.cpp/.worktrees/`. It must not modify the production submodule gitlink or build directory.

Model: `/home/typhoon/git/frankenstein-llm/models/glm53flash-regular-iq3xxs/GLM-5.3-Flash-IQ3_XXS-00001-of-00015.gguf`

Expected artifact set: 15 shards, exactly 120,994,791,264 bytes, each verified against `manifest.tsv`.

## Preconditions

1. Downloader has exited with code 0 or its tmux pane is dead with status 0.
2. All 15 final shard filenames exist; no `.partial` remains.
3. Re-run exact size and SHA-256 checks over every manifest row.
4. A reviewed experimental worktree exists below the tracked submodule, its HEAD equals the required revision, and it is clean.
5. Production submodule HEAD remains v0.4.0 and production router health remains OK.
6. No unrelated optimized build, Cargo/rustc, CMake/Ninja, package build, or GPU-heavy generation is active.
7. Record baseline `MemAvailable` and per-GPU VRAM.

## Build

Use an ignored evidence build directory inside the experimental worktree. Substitute the reviewed worktree path for `$GLM_SRC`:

```text
GLM_SRC=/home/typhoon/git/frankenstein-llm/upstream/llama.cpp/.worktrees/glm53flash-local
cmake -S "$GLM_SRC" \
  -B "$GLM_SRC/build-rocm" \
  -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1030 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$GLM_SRC/build-rocm" \
  --target llama-server llama-cli --parallel "$(nproc)"
```

Build only when machine-level optimized-build exclusivity is satisfied. Capture exact configure/build exits and logs.

`gate_glm32.py` reads the same two values from the environment -- `GLM_SRC` for the worktree and `GLM_BINARY` to override the binary directly -- and defaults to the paths above, so the runbook and the gate cannot drift. It refuses to start when that binary is absent and records the worktree's resolved `HEAD` as the run's `runtime_revision` instead of asserting a literal. The superseded `build-pr27752-c9ddd682/` and `build-pr27773/` directories here were produced from the deleted `/home/typhoon/src` worktrees; nothing on disk can still establish which revision they contain, so they are not a valid runtime for a new verdict.

## Binary proof

1. Run the experimental `llama-server --version` and `llama-server --help`.
2. Confirm `ldd` resolves ROCm/HIP libraries from the expected installation.
3. Confirm the binary exposes required context, cache type, device, GPU-layer/offload, flash-attention, host, port, Jinja, and model flags.
4. Do not infer supported flags from the production v0.4.0 binary.

## 32K standalone load

- Bind only to `127.0.0.1` on an unused non-production port.
- Use the experimental binary and first model shard.
- Use `--ctx-size 32768`, `--cache-type-k q4_0`, and `--cache-type-v q4_0`.
- Enable the model's built-in Jinja template.
- Discover safe hybrid placement with this binary's current fit/offload options. Do not blindly copy production `tensor-split = 1,2,1`.
- Preserve at least a conservative OS/runtime reserve. Abort cleanly rather than driving swap/OOM.
- Leave production router configuration and aliases untouched.

Record:

- exact command and environment
- model metadata/architecture recognition
- load completion and load time
- baseline and peak `MemAvailable`
- per-GPU baseline and peak VRAM
- server health and `/v1/models`
- one deterministic direct chat completion
- one coding/reasoning completion
- one structured JSON completion with programmatic parsing
- malformed input/error behavior
- normal shutdown, no descendants, and returned RAM/VRAM

Do not record or compare tokens/second.

## 64K conditional gate

Attempt 64K only if 32K completes and measured headroom safely covers the additional KV/runtime allocation plus an OS reserve. Repeat the same functional and clean-unload checks. If headroom is marginal, record 64K as deliberately skipped rather than forcing it.

## Quality A/B

Compare GLM against production alias `heretic` using identical prompts and correctness validators:

1. dependency/scheduling reasoning
2. small repository-style bug fix in a disposable fixture
3. strict JSON schema output
4. tool selection and arguments
5. authorized local-lab security review
6. long-input fact retention within the successfully served context

Judge correctness, coherence, completeness, hallucination, loop behavior, and instruction adherence. Do not use throughput as an admission criterion.

## Admission

Add a production router alias only if:

- 32K load/generation and clean unload pass
- memory headroom is operationally safe
- no output corruption or unsupported architecture behavior appears
- GLM wins at least one named workflow over `heretic`
- the isolated PR runtime can be maintained without destabilizing production

Otherwise keep the verified model and evidence as an experimental specialist and make no production change.