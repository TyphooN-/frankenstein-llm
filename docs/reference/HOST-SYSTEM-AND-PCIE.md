# Host System and PCIe (Benchmark Reference)

This host is the reference machine for every benchmark in this repo. Record the
platform facts here so future runs can be compared against a known baseline, and
so the upgrade plan is not lost.

## Platform

| Item | Value |
| --- | --- |
| CPU | Intel Xeon E5-2696 v4 (22C/44T, Broadwell-EP), base 2.20 GHz |
| Board | MSI X99A GODLIKE (MS-7883) — C610/X99 chipset |
| BIOS | Dok-T/MS-7883 mod (PCIe bifurcation + Above-4G/ResizeBar + Xeon M-Flash fix) — https://github.com/Dok-T/MS-7883 |
| BCLK | Overclocked to 104.09 MHz (stock 100 MHz) — ~+4% CPU, DRAM, PCIe, DMI, and QPI clocking |
| RAM | 96 GB G.Skill Trident Z B-Die (UDIMM), 8 DIMMs, 2 DIMMs/channel (2DPC), quad-channel; mixed rank — 4×16 GB (2R) + 4×8 GB (1R), tuned CL12 @ 2498 MHz |
| NUMA | Single node (0-43) |
| Kernel | 7.2.5-3-omarchy since the 2026-09-29 Omarchy reinstall (previously 7.2.4-273-tkg-eevdf-llvm: linux-tkg, eevdf + LLVM + full LTO) |
| Scheduler | eevdf (kernel default), performance governor, tickless idle |

## GPU / PCIe

All three GPUs are Navi 21 (RDNA2). The board's PCIe lanes are split across the
three x16 slots; each GPU gets an x8 link. This is the normal three-GPU layout on
this platform (see the BIOS mod's "PCIe Hardware Limitations" section) and is not
a degraded link.

| GPU | BDF | DRM card (2026-09-29) | Slot | Negotiated | Capable | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 6900 XT (ASRock Phantom) | 03:00.0 | card1 | 2 | PCIe 3.0 (8.0 GT/s) x8 = 63 Gb/s | x16 4.0 | Headless / ROCm work |
| Radeon Pro V620 | 07:00.0 | card2 | 6 | PCIe 2.0 (5.0 GT/s) x8 = 32 Gb/s | x16 4.0 | V620 root port is PCIe 2.0 |
| 6900 XT (Liquid Devil) | 0a:00.0 | card3 | 4 | PCIe 3.0 (8.0 GT/s) x8 = 63 Gb/s | x16 4.0 | ROCm work |

The V620's PCIe 2.0 x8 link is ~2x lower per-link bandwidth than the 6900 XTs'
PCIe 3.0 x8, but for LLM inference (model resident in VRAM, bottleneck is HBM/VRAM
bandwidth + compute, not PCIe) this is irrelevant. No action needed.

All links train clean at boot — no retrain, no "degraded below," no AER
uncorrectable errors. The only GPU noise in dmesg is normal page-fault / GFXHUB
protection-fault messages during model loading (VRAM mapping), not hardware faults.

## BCLK overclock — benchmark caveat

The 104.09 MHz BCLK (+4% over 100 MHz) raises CPU, DRAM, PCIe, DMI, and QPI
clocks proportionally. When comparing token/s or latency numbers across hosts or
across BIOS updates, note the BCLK. A stock-100-MHz host will run ~4% slower on
CPU/DRAM-bound paths (KV-cache movement, prompt prefill on CPU fallback, page
faults) even with identical GPU config. This host's numbers are the baseline;
flag them as "BCLK 104.09" in any cross-host comparison.

## Fan control and GPU tuning

DRM card numbers are not stable across installs: the 2026-09-29 reinstall has no
`card0` and the GPUs are `card1`..`card3`. Everything below is keyed by PCI address.
Gate evidence keeps the logical labels `card0`..`card2` = ROCm0..ROCm2, resolved by
`scripts/gpu_vram.py:logical_card_paths()`.

- ASRock 6900 XT (03:00.0): on 7.2.5-3-omarchy manual fan control works
  (`pwm1_enable=1`, `pwm1=255`; the old linux-tkg kernel returned `EINVAL`). It is
  held at 100% (~2984 RPM measured) by the operator's root unit
  `asrock-fan-100.service` (`~/asrock-fan-100.sh`), not by this repository.
- V620 (07:00.0): `pwm1` is read-only; no manual override.
- `amdgpu-clocks.service` applies `/etc/default/amdgpu-custom-state.pci:<BDF>` for all
  three cards; the repo copies in `config/hardware/amdgpu/` map card0 → 03:00.0,
  card1 → 07:00.0, card2 → 0a:00.0. Overdrive needs `amdgpu.ppfeaturemask=0xfff7ffff`
  on the kernel command line. The driver may round an offset (ASRock reads back
  −40 mV for a −45 mV profile).
- All amdgpu devices are kept out of PCI runtime suspend (udev
  `ATTR{power/control}="on"`); a suspended card hid its sensors from ROCm SMI.

## Host configuration after the 2026-09-29 reinstall

- Swap is zram only: zstd 16G, priority 100 (ADR 0003). Omarchy's hibernation
  swapfile, resume hook and `resume=` parameters were removed; hibernate targets are
  masked. 3 × 1 GiB HugeTLB pages are reserved on the command line.
- ROCm 7.2.4 from Arch `extra`; its clang guesses `/opt/rocm/lib` as the HIP root,
  which `scripts/build-llama-cpp.sh` handles by exporting `HIP_PATH`.
- Host Python provides ROCm PyTorch 2.13 and torchvision 0.28; every lane under
  `venvs/` reuses them (`scripts/setup-python-runtimes.sh`).

## Stability record: 2026-09-29

Under combined CPU, disk and GPU DMA load this configuration (BCLK 104.09, tuned
mixed-rank RAM, GPU undervolts) produced repeated memory corruption: `Bad page state`
reports (one a single-bit flip, refcount `0x40000000`), general protection faults on
pointers one bit from canonical, kernel oopses that took the host down, btrfs
checksum failures on data at rest (lifetime `corruption_errs` 3 → 13), and
previously verified model files that failed SHA-256 on re-read. memtest86+ passed
and NVMe SMART is clean (0 media errors), so the fault needs the full-system load
memtest does not generate. The kernel taints the boot `[S]=CPU_OUT_OF_SPEC`.
Qualification results recorded on this configuration are provisional until a pass
completes with no kernel memory errors and no SHA-256 quarantines.

## Upgrade plan (noted for documentation / future runs)

Pending hardware upgrades to this host. Record the new config here when applied
and re-baseline any benchmark that is CPU/DRAM/PCIe-sensitive.

1. **RAM: 96 GB → 128 GB.** The current 96 GB is 4×16 GB (2R) + 4×8 GB (1R) at
   2DPC quad-channel — a mixed-rank population that the IMC has to reconcile on
   every access. The upgrade replaces the 4×8 GB (1R) DIMMs with 4×16 GB (2R)
   DIMMs so every slot is 16 GB, giving 4×32 GB = 128 GB at 2DPC, all 2R. The
   rank mismatch is removed. Verify the CL12 @ 2498 MHz tuning still holds with
   the all-2R population after the swap; the IMC load profile changes when rank
   count per channel goes from mixed (2R + 1R) to uniform 2R. Larger RAM headroom
   helps KV-cache offload, concurrent model residency, and the RAM-offload path
   for bigger models (see ADR 0003).

2. **GPU: add a 4th Navi 21 / Pro card (32 GB).** Add a Radeon W6800 Pro (32 GB)
   or a second V620 (32 GB) as the fourth GPU. Verify the board's fourth x16
   slot's root port width after install — the C610 chipset's lane split may give
   the new card x4 or x8, and the M.2 NVMe port can downgrade if the last slot
   is populated (see the BIOS mod's PCIe limitations). Re-check `lspci -vv` and
   `dmesg` after install and update the table above.

Do not change the per-GPU VRAM accounting in ADR 0003 until the new card is
installed and its VRAM is measured with `nvidia-smi`-equivalent (`rocm-smi` /
`/sys/class/drm/card*/device/mem_info_vram_used`).

## Measurement notes for benchmarks on this host

- Record the BCLK (currently 104.09 MHz) with every benchmark result.
- Record the GPU that ran the workload (headless 6900 XT vs V620 vs 6900 XT
  Liquid Devil) — VRAM, HBM/VRAM bandwidth, and PCIe generation differ.
- Record RAM total (currently 96 GB) and the DRAM tuning (G.Skill B-Die, CL12
  @ 2498 MHz, 2DPC quad-channel, mixed rank 2R + 1R) — affects KV-cache offload
  and any CPU-fallback path that is DRAM-bandwidth-bound.
- The V620's PCIe 2.0 x8 link is the only per-card bandwidth asymmetry; it does
  not affect inference throughput but does affect model-load time from NVMe→VRAM.
