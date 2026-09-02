# 0003. Hardware allocation and memory policy

- Status: accepted
- Date: 2026-09-02

## Context

The current host is a Xeon E5-2696 v4 with 96 GiB RAM, a 1600 W PSU, and three AMD GPUs. HugeTLB pages are reserved for XMRig.

Live compressed swap is zram, not disk-backed zswap:

- `/dev/zram0` algorithm `zstd`, disksize 16G, used as SWAP at priority 100
- udev initializes it with `ATTR{comp_algorithm}="zstd"` and `ATTR{disksize}="16G"`
- `/sys/module/zswap/parameters/enabled` is `N` on the running kernel

Treat zram as the compressed-swap path. Do not document zswap as the active policy.

## Decision

- Prefer GPU0 (RX 6900 XT 16 GiB, headless) and GPU1 (Radeon Pro V620 32 GiB, headless) for compute.
- Minimize GPU2 (RX 6900 XT 16 GiB, display) compute.
- Keep llama.cpp on-demand with a single resident model.
- Preserve three 1 GiB HugeTLB pages for RandomX.
- Use `/dev/zram0` zstd 16G as compressed swap.
- Never hard-code linux-tkg `_version`; leave it empty for current-latest selection.
- Do not manage kernel compilation from this workspace.

## Consequences

- Media and large single-GPU loads prefer the 32 GiB V620.
- Router presets split tensors across ROCm0/ROCm1/ROCm2 as `3,6,2` unless a gate overrides that for a specific model.
- Benchmarking remains blocked until explicit post-reboot confirmation.
