# Pinned llama.cpp convolution compatibility overlay

The authoritative source/build contract is
[`upstream/llama-cpp.lock.json`](../llama-cpp.lock.json).
It selects the official `ggml-org/llama.cpp` repository, branch `master`, at
`ef6bc6f367082338006c8379bc7bcab3e1584766`.
This is a fixed source snapshot, not a claim to track the moving upstream tip.

## Declared overlay

The lock declares exactly:

- `upstream/patches/0001-convolution-kernel-types-ef6bc6f367.patch`
- SHA256 `78c9300d0af6bb9c02c0159ea338dbe0b60db3dcef2612ddc2a8c73fb6cbd279`

This overlay keeps convolution-kernel compatibility changes and their typed
regression coverage bound to that exact upstream revision. Do not apply a patch
named for another revision to the selected source.

Other patch files in this directory are inactive historical provenance. Their
presence does not select them for application or establish that an old patch is
compatible with the current pin. Use the lock's explicit `local_patches` list,
not a directory glob, to identify the declared overlay.

## Source verification

From the repository root, the maintained verifier checks the pinned source and
declared patch contract without compiling or launching a model runtime:

```sh
python3 scripts/verify_llama_patches.py
```

A successful source check is not native numerical, ABI, model, resource, or
performance qualification. Source and build configuration do not establish which
binaries or libraries a service currently has loaded. Compilation, execution,
model acquisition, fit, capability qualification, and service activation retain
their separate checks and authorization.

Build parallelism is declared as `nproc` in the lock. CPU, HIP, CUDA, and Vulkan
build-directory declarations are build intent, not evidence that every backend
has been built or exercised on a particular machine.
