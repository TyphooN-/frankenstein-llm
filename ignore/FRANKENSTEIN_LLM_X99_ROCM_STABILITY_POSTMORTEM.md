# Frankenstein-LLM Lore: X99 / Broadwell-EP / ROCm Stability Campaign Post-Mortem

**Status:** living post-mortem / campaign lore  
**Compiled:** 2026-09-11  
**Campaign duration:** roughly two months, across many tuning sessions and chat branches  
**Machine:** MSI X99A GODLIKE GAMING + Xeon E5-2696 v4 + 96 GiB-class DDR4 + three AMD GPUs  
**Purpose:** preserve the hard-won stability knowledge behind the `frankenstein-llm` machine so future tuning, qualification, and automation do not repeat the same mistakes.

> This is not a single-crash incident report. It is the accumulated record of a long stability campaign in which several independent margins were present at the same time: host memory/uncore/input-rail instability, GPU undervolt instability, ROCm/HSA failures, and misleading or contaminated test results.
>
> Some exact values came from different branches of the campaign. Where a value is not known with confidence from the preserved cross-branch context, this document says so rather than inventing one.

---

## 1. Executive summary

The machine is intentionally operated far outside ordinary X99 conditions:

- Intel Xeon E5-2696 v4, 22 cores / 44 threads.
- BCLK around **104.09 MHz**.
- Roughly **98,304 MB** of DDR4 running around **DDR4-2498**.
- Extremely tight memory timings, including **1T / CL12**.
- High VCCIN experimentation, sometimes above Intel's documented electrical limits.
- Three AMD Navi21-family GPUs used for local LLM / ROCm / HSA workloads.
- Aggressive per-GPU undervolts and high board power limits.
- Arch Linux, custom LLVM-built kernel, ZFS, ROCm, llama.cpp / `llama-server`, Geekbench, Memtest86+.
- Extreme custom water cooling: Heatkiller IV Pro, two D5 pumps, and approximately **10 × 120 mm** of radiator area, plus active VRM airflow.

The central lesson is that **there was never one wall**.

At least two independent fault domains were eventually separated:

1. **Host/platform instability**  
   Manifested as Linux page allocator and memory-management corruption: `Bad page state`, `list_del corruption`, `lruvec_stat_mod_folio` GPFs, memcg corruption, impossible page metadata, poisoned/dead pointers, allocator crashes, RCU stalls, and hard locks.

2. **GPU-specific instability**  
   Manifested as AMDGPU GFXHUB/UTCL2/TCP page faults on a specific PCIe device, or an extraordinarily repeatable HSA userspace crash at exactly `libhsa-runtime64.so +0x55033`.

A major source of confusion early in the campaign was allowing one failure domain to contaminate interpretation of another. The rule that emerged is:

> **The first genuine failure wins. Once host MM corruption occurs, the boot is poisoned. Once a GPUVM/HSA path has faulted, subsequent GPU results from that unrecovered state are not independent stability datapoints.**

The best host behavior was obtained only after treating VCCIN, VCCIO, VDIMM, LLC, switching frequency, memory timings, BIOS training state, workload order, ambient temperature, and GPU voltage as interacting variables rather than independent monotonic knobs.

---

## 2. Cross-branch provenance

This post-mortem is compiled from:

- approximately two months of tuning discussion across multiple chat branches,
- uploaded Linux kernel logs,
- Memtest86+ observations,
- BIOS screenshots,
- `sensors` snapshots,
- ROCm / AMDGPU fault logs,
- `rocm-smi` bus mapping,
- GPU overclock/undervolt state dumps,
- Geekbench 4/5/7 runs,
- ZFS activity,
- llama/Hermes/local-model workloads,
- and repository robustness work that grew out of these failures.

Not every intermediate 5 mV rung survived into the current context. The important stable checkpoints, repeated signatures, and lessons are preserved here.

---

## 3. Hardware and platform

### CPU

- **Intel Xeon E5-2696 v4**
- Broadwell-EP
- **22 cores / 44 threads**
- 55 MiB L3
- Base clock reported by Linux/benchmark tooling around 2.2 GHz nominal for this SKU.
- Operated around **104.09 MHz BCLK** in the primary aggressive profile.

### Motherboard

- **MSI X99A GODLIKE GAMING (MS-7883)**
- Linux DMI logs have reported:
  - BIOS `1.A1_0.4.3`
  - date `07/26/2025`
- Click BIOS screenshots identify:
  - `E7883IMS.1A1`
  - BIOS build date `06/13/2019`

That date/string discrepancy is preserved as observed rather than resolved here.

### Cooling

The CPU/VRM cooling is not ordinary ambient air cooling:

- **Heatkiller IV Pro** CPU block
- **2 × D5 pumps**
- approximately **10 × 120 mm radiator space**
- active VRM fan / airflow

CPU temperatures during many BIOS and Linux tests were only in the 20s–30s °C. This gives large thermal headroom, but it does **not** remove electrical stress, FIVR/input-rail stress, or transient overshoot risk at very high VCCIN.

### Memory

- Installed capacity reported as **98,304 MB** (~96 GiB nominal).
- Main aggressive operating frequency: **~DDR4-2498**
- BCLK: **~104.09 MHz**
- Memory Fast Boot generally disabled during training-sensitive work.

### GPUs

Three AMD GPUs:

| Linux/ROCm index | PCI BDF | Identity | Current known tuning |
|---|---|---|---|
| card0 / GPU0 | `0000:03:00.0` | **ASRock Phantom Gaming**, Navi21 / 6900 XT-class | 2710 MHz max SCLK state, 323 W cap, **-20 mV** current VDD GFX offset |
| card1 / GPU1 | `0000:07:00.0` | **AMD Radeon Pro V620 Azure** | 250 W cap, **-65 mV** current offset, auto performance level |
| card2 / GPU2 | `0000:0A:00.0` | **Liquid Devil**, Navi21 / 6900 XT-class | 2710 MHz max SCLK state, 346 W cap, **-80 mV** current offset |

The ASRock identity for `03:00.0` was confirmed during the GPUVM fault campaign.

### Operating system / software

- Arch Linux
- Custom kernel:
  - `7.2.4-273-tkg-eevdf-llvm`
  - PREEMPT(full)
  - LLVM/clang/lld-oriented build environment
- ZFS + SPL out-of-tree modules
- AMDGPU + ROCm/HSA
- llama.cpp / `llama-server`
- Python local-model orchestration
- Hermes / local coding-model workflows
- Geekbench 4.4.4, Geekbench 5, Geekbench 7
- Memtest86+ 7.20

---

## 4. Primary memory profile

The long-running aggressive profile converged around:

```text
BCLK        ~104.09 MHz
DDR         ~2498 MT/s
Capacity    98,304 MB

Command Rate   1T
tCL            12
tRCD           17
tRP            14
tRAS           36

tRFC           340 primary/current branch
               330 also tested and occasionally behaved differently
tREFI          65535 intended baseline
tWR            19
tWTR           3
tWTR_L         8
tRRD           7
tRRD_L         7
tRTP           8
tFAW           28
tCKE           2
tCCD           4
tCCD_L         6
tCCD_WR        4
tCCD_WR_L      6
```

Observed tertiary settings included:

```text
tRDRDDR/DD/DS      1 / 1 / 1
tRDRDDS_L          2
tWRWRDR/DD/DS      2 / 2 / 2
tWRWRDS_L          5
tRDWRSR/DR/DD      5 / 5 / 5
tRDWRDS            1
tWRRDDR/DD/DS      1 / 1 / 1
```

RTL/IOL were left on Auto and trained. One observed channel was approximately RTL 47 / IOL 8.

### DRAM voltage and power settings

Long-time baseline:

- **VDIMM 1.430 V set**, ~1.424 V readback.
- DRAM switching frequency: **625 kHz**
- DRAM OCP: Enhanced

Later combined stability branch:

- **VDIMM 1.440 V**

The 1.44 V branch had been tried earlier without obvious improvement, but later became interesting when combined with a different VCCIN/VCCIO state. That is a recurring theme: the machine is strongly interaction-dependent.

### CPU power delivery settings

Important later configuration:

- CPU switching frequency: **2000 kHz**
- CPU LLC: **Ultra / +75%**
- CPU phase: optimized
- CPU OCP: **160%**

The move from the older ~50% LLC setting to **75% LLC**, together with 2000 kHz switching, materially changed the VCCIN/VCCIO behavior and improved some lower-VCCIN branches.

---

## 5. Memtest86+ phase: the original boss wall

Before Linux became the primary arbiter, Memtest86+ 7.20 exposed a very repeatable wall.

### Canonical Test 9 wall

Typical behavior:

- Test 9
- around **26–27 GiB**
- approximately **4:17–4:33**
- hard lock or first error
- repeated familiar address:
  - `0006c08afcc8`

Other recurring pressure points:

- Test 6 around **27–29 GiB**
- Test 6 around **97–98 GiB**
- Test 8 around **77 GiB**
- Test 8 around **97–98 GiB**

This signature was useful for finding gross instability, but later Linux proved much more discriminating for subtle host corruption.

### Timing conclusions from the Memtest phase

Several timing changes did **not** move the dominant wall:

- CL13 did not help.
- `tRTP 8` was best; 9 gave no benefit, 10 regressed.
- `tCCD_WR_L 6`; 7 gave no benefit.
- `tFAW 28` beat 30.
- Restoring looser tertiary timings did not materially move the wall.
- Timing chasing eventually produced diminishing returns.

#### tRFC 330 vs 340

`tRFC 330` and `340` behaved non-monotonically.

- `340` became the cleaner baseline for remapping.
- `330` sometimes improved an otherwise marginal state.
- `330` could also expose refresh-related instability.

One intended `tRFC 340` experiment was accidentally run at `330`, contaminating a whole VCCIO branch. The lesson was simple: **verify timing state after every BIOS save**.

#### tREFI contamination incident

At one point a save unexpectedly changed:

```text
tREFI 65535 -> 8320
```

That run was considered contaminated. Later screenshots confirmed `65535` restored.

---

## 6. Linux changed the definition of "stable"

The turning point in the campaign was recognizing that a Memtest pass or a pretty BIOS state did not mean the machine was stable.

### Decisive host/platform failures

These are treated as **hard host failures**:

- `BUG: Bad page state`
- `list_del corruption`
- `lruvec_stat_mod_folio` GPF
- `kfree`, `kmem_cache_free`, `__kmalloc_node_noprof` faults
- `warn_free_bad_obj`
- impossible page/folio reference counts
- impossible `mapping` or `memcg` pointers
- non-canonical pointers
- allocator freelist corruption
- RCU stalls after corruption
- hard locks with no recoverable log

Common corrupted pointer patterns included:

```text
de...
dead000000000100
dead000000000122
ff00000000000000
00ff...
efad...
fdfd...
```

These are not "Geekbench bugs" or "ZFS bugs." They are evidence that the host's memory-management state has become corrupted.

### ZFS is usually the detector, not the cause

ZFS often appeared in stacks because ARC/ABD allocation/free/eviction activity is excellent at exercising page lifecycle paths.

Examples included:

- ARC eviction
- ABD free
- `zio_done`
- SPL kmem cache free

The working conclusion is:

> **ZFS is a high-quality consumer/detector of corrupted memory-management state. Its presence in a stack is not evidence that ZFS caused the underlying corruption.**

### Soft host failures

These fail a tuning rung but do not automatically poison the whole kernel state:

- Geekbench userspace segfault
- Geekbench userspace GPF
- Geekbench assertion failure
- `std::runtime_error: ml failure`
- unrelated userspace applications such as `rofi` segfaulting

A particularly useful recurring Geekbench x86_32 address was:

```text
IP 0x080d0537
```

It appeared multiple times across different boots/rungs, making it much more meaningful than a random application crash.

---

## 7. First-failure rule and contamination model

This became one of the most important diagnostic rules of the campaign.

### Host corruption first

If a boot first shows:

```text
Bad page state
list_del
lruvec
allocator corruption
memcg corruption
non-canonical kernel pointer
```

then:

- the boot is **poisoned**,
- later GPU/HSA failures are not clean GPU datapoints,
- later application crashes are not independent,
- reboot before drawing a new conclusion.

### GPU fault first

If an otherwise host-clean boot first shows an AMDGPU GPUVM fault on one PCI BDF:

- treat that card as the primary suspect,
- do not add VCCIN merely because the fault is printed by the kernel,
- reset/reboot before treating a subsequent GPU fault as an independent timing result.

### HSA userspace fault first

The repeated HSA signature discussed later can be a strong GPU-specific heat-soak marker if no host-MM failure preceded it.

---

## 8. BIOS rail telemetry: useful, but easy to over-interpret

The board's Auto rail readbacks repeatedly clustered around:

```text
Vcore low    ~0.872 V
Vcore high   ~0.880 V

Ring low-ish ~1.020 V
Ring high    ~1.032 V

SA low       ~0.960 V
SA high      ~0.968 V
```

At different points the campaign used increasingly strict BIOS "gates":

- Ring solid + Vcore solid.
- Later, all visible rails solid.
- Later still, deliberately searching for an **all-high** state.

These gates were useful for **mapping**, but Linux repeatedly proved that a perfect-looking BIOS state could still hard-lock or corrupt memory within minutes.

### Dithering

The common transitions are exactly ~8 mV:

```text
0.872 <-> 0.880
0.960 <-> 0.968
```

Later research showed that the monitoring path is quantized enough that these transitions may represent adjacent ADC/reporting bins rather than a literal regulator "mode switch."

Therefore:

> **Rail dithering is telemetry, not proof of instability. Rail solidity is a useful branch gate when deliberately chosen, but not a sufficient stability criterion.**

### Sensor `in14`

Linux `sensors` repeatedly showed:

```text
in14: ~1.88 V
```

It was initially assumed to be loaded VCCIN because no other obvious board rail fit the magnitude.

Later review made the conclusion more cautious:

- it is a plausible VCCIN candidate,
- but the Nuvoton channel mapping/scaling is board-dependent,
- software readings are quantized,
- `in14` staying near 1.88 V while BIOS VCCIN moved substantially means it should **not** be used as an absolute calibrated VCCIN measurement without external verification.

The alarms shown by `sensors` were mostly meaningless because the min/max thresholds were zero/unconfigured.

---

## 9. BCLK boundary

The machine's practical reference-clock edge was repeatedly around:

```text
104.09 MHz  -> main operating point
104.15 MHz  -> observed hard no-boot edge in an earlier branch
```

Do not assume a 4% BCLK increase is trivial on this platform. The machine has repeatedly shown that this is near its own usable reference-clock boundary.

---

## 10. VCCIN / VCCIO campaign

The relationship was strongly **non-monotonic**. A VCCIO level that looked terrible at one VCCIN could become useful at another.

### 0.70 V VCCIO branch

Low VCCIO moved BIOS state thresholds dramatically.

One notable state around ~1.72 V VCCIN showed:

```text
Vcore 0.880 high solid
Ring  1.032 high solid
SA    0.960 low solid
```

A historically promising branch:

```text
VCCIO  0.70 V
VCCIN  ~1.725 V
tRFC   330
```

later failed with `list_del corruption` at:

```text
~3145.13 s = ~52m25s
```

So it was not actually stable.

Much later, 0.70 VCCIO was revisited at high VCCIN and sometimes failed very quickly with hard locks or Bad Page / ARC-evict behavior. At another combined VCCIN/VDIMM branch it appeared to behave better long enough to run broader tests, demonstrating again that VCCIO cannot be judged in isolation.

### 1.03 V VCCIO branch

Historically the best host runtime before the later high-voltage work:

```text
VCCIO  1.030 V
VCCIN  1.830 V
tRFC   340
VDIMM  1.43 V
LLC    75%
Switch 2000 kHz
```

First hard host failure:

```text
~20,352 s = ~5h39m
BUG: Bad page state
```

Important contamination detail:

- first 3–4 hours were without GPU AI load,
- Python/ROCm GPU work began later,
- host Bad Page appeared after the GPU workload started.

This raised the possibility that a marginal GPU/DMA/HSA path could contribute to some corruptions, but the event itself was still genuine host page-state corruption. It was never reclassified as "definitely GPU."

At later high-VCCIN points, 1.03 still produced host memcg/cgroup corruption in the ~10-minute region.

### 1.05–1.08 region

This region looked promising during one mapping pass, especially around SA transition behavior.

That experiment was later discovered to have been contaminated because `tRFC 330` had accidentally been saved when `340` was intended.

The test was restarted.

### 1.08 VCCIO / tRFC340

Around:

```text
VCCIN ~1.755 V
VCCIO 1.08 V
```

all BIOS rails could appear solid, yet Linux produced:

```text
warn_free_bad_obj
corrupted deff... pointer
SPL/ZFS free path
~83.8 s
```

Decisive host fail.

### 1.07 VCCIO / tRFC340

A heavily mapped branch.

Examples:

```text
~1.840 V VCCIN:
  Vcore high-solid
  Ring high-solid
  SA dithering
  Geekbench DOM/assertion weirdness
  later Geekbench segfault around ~726 s

~1.850 V:
  Vcore solid
  Linux hard lock

~1.855 V:
  panic/hard lock during boot

~1.860 V:
  Bad page around ~204 s
  page/folio mismatch
  ZFS ARC/ABD path

~1.870 V:
  Bad page around ~254 s
  absurd refcount / metadata corruption
```

At still higher VCCIN, the branch was deliberately mapped until all rails became solid.

Two notable BIOS states:

```text
~1.940 V VCCIN:
  Vcore 0.880 high-solid
  Ring  1.032 high-solid
  SA    0.968 high-solid

~1.950 V VCCIN:
  Vcore 0.880 solid
  Ring  1.032 solid
  SA    0.960 solid-low
```

The non-monotonic SA transition was real as observed.

### 1.10 VCCIO branch

At higher VCCIN, 1.10 VCCIO produced a dramatically different rail map.

One early all-solid point:

```text
VCCIO 1.10 V
VCCIN ~2.010 V set
Vcore 0.880 solid
Ring  1.032 solid
SA    0.968 solid-high
```

but Linux hard-locked during boot. This established:

> **BIOS-state floor != runtime-stability floor.**

A later apparent non-dither point around 2.050 VCCIN proved not to be fully settled; Vcore later resumed dithering and Linux produced an immediate Bad Page.

A more settled point around:

```text
VCCIN ~2.070 V
VCCIO 1.10 V
Vcore 0.880 solid
Ring  1.032 solid
SA    0.960 low-solid
```

still failed in Linux with hard memcg/MM corruption.

Other 1.10-VCCIO rungs showed:

- Geekbench 4 passing while Geekbench 5 became the new wall.
- `page_counter_cancel` / memcg corruption.
- `mod_memcg_state` corruption.
- `list_del` / `__rmqueue_pcplist` corruption.
- hard locks.

This branch was useful because it showed **real movement of the failure wall**, even though it did not immediately solve the platform.

### 1.11–1.14 VCCIO exploration

At high VCCIN, 1.11 looked promising while 1.12 and 1.13 could return SA to the low state.

A later interesting fixed-rail region showed that VCCIO could be moved all the way through roughly 1.14 V without visibly changing:

```text
Vcore 0.880
Ring  1.032
SA    0.960
```

That created a valuable control condition: VCCIO could be tested while the visible Auto rails stayed fixed.

### 1.13 VCCIO branch

1.13 VCCIO was eventually chosen as a dedicated branch because:

- prior Memtest history suggested it could be a useful island,
- 0.70 VCCIO became clearly bad in some high-VCCIN states,
- bouncing VCCIO constantly was making VCCIN interpretation impossible.

The intended methodology became:

```text
VCCIO = 1.13 V fixed
solve VCCIN first
optimize VCCIO only afterward
```

At high VCCIN this branch still produced:

- blue visual-corruption/hard-lock phenotypes,
- Bad Page,
- Geekbench GPFs,
- then eventually materially longer clean runs.

### High-VCCIN range

Setpoints were experimentally pushed through the low 2.0s and into the ~2.2 V region.

Examples observed in BIOS included:

```text
2.005 V set -> ~1.968 V BIOS value in one state
2.160 V set -> ~2.128 V readback
2.175 V set -> ~2.144 V readback
2.205 V set -> ~2.176 V readback
```

An "all-high" state could be obtained around the 2.2-V configured region, but even that could hard-lock in Linux or later show SA dithering after returning to BIOS.

The key lesson:

> **Do not confuse "all rails high and solid" with actual runtime stability.**

---

## 11. High-VCCIN research and risk boundary

During the campaign, overclocking sources were researched because X99/Haswell-E/Broadwell-E extreme overclockers do in fact discuss very high input voltages.

Observed community/vendor-overclocking guidance included roughly:

- ~2.0 V input used in ambient-cooled X99 overclocking examples.
- ~2.2 V input discussed for LN2 Broadwell-E.
- Intel overclocking presentation material describing X99 VCCIN control capability extending to `2.3V+` and higher static controls.

However, Intel's Xeon E5 v4 electrical datasheet was also checked.

Documented VCCIN figures for the E5 v4 family included approximately:

```text
Nominal target           1.82 V
Maximum operating target 1.85 V
Absolute maximum         1.98 V
```

Therefore:

- A BIOS allowing 2.2–2.3 V does not make that a manufacturer-supported continuous operating range.
- Extreme water cooling reduces temperature but does not erase electrical or transient stress.
- The campaign intentionally entered territory beyond Intel's documented limits.
- The presence of successful forum/LN2 examples is evidence that such settings are used in overclocking, not evidence of long-term safety for this CPU.

This is lore, not a safety endorsement.

---

## 12. Ambient temperature mattered

Room temperature moved from roughly:

```text
~68°F -> ~78°F
```

during part of the campaign.

That ~5.6°C change was enough to expose failures that had not appeared under cooler conditions.

Both host DDR/uncore margin and GPU heat-soak margin are sensitive to ambient. A tuning result is not complete until it survives warmer-room testing.

---

## 13. False positives and misleading signals

### LibreWolf blue/striped canvas event

One striking full-screen striping/corruption screenshot was eventually identified by the user as **LibreWolf canvas-detection/fingerprinting behavior**, not GPU/display corruption.

That event must be discarded as a hardware-stability signal.

Later "blue" hard-lock phenotypes were treated more cautiously:

- useful as a repeatable failure phenotype,
- not automatically proof of host MM corruption unless supported by dmesg or system behavior.

### `cfg80211: failed to load regulatory.db`

Observed:

```text
cfg80211: failed to load regulatory.db
```

This was not part of the memory/GPU stability problem.

---

## 14. GPU fault campaign

The GPU work became much easier once faults were tied to **PCI BDF**, not just a transient GPU index.

### PCI mapping

```text
GPU[0] / card0 -> 0000:03:00.0 -> ASRock Phantom Gaming
GPU[1] / card1 -> 0000:07:00.0 -> Radeon Pro V620
GPU[2] / card2 -> 0000:0A:00.0 -> Liquid Devil
```

Future automation should store and compare **BDF + hardware identity**, not blindly trust `card0`, `card1`, etc. across arbitrary system changes.

---

## 15. Radeon Pro V620: HSA heat-soak wall

The V620 is Navi21-family with 72 CUs versus 80 CUs on a full 6900 XT. In actual workloads its deficit was only around 5–6% in some comparisons.

The firmware/driver accepts high clock ceilings, but the card's smaller board-power budget means raw 6900 XT-style overclock behavior cannot be assumed.

### Early undervolt exploration

Short/light tests suggested aggressive undervolts were possible:

- around `-90 mV` initially seemed stable.
- `-130 mV` could appear okay.
- `-145 mV` appeared fine in lighter testing.
- `-150 mV` produced visible Vulkan artifacts.
- around `-160 mV` was unstable.

Long-duration HSA compute completely changed the conclusion.

### Exact HSA signature

Repeated Python crashes occurred in:

```text
libhsa-runtime64.so.1.18.0
```

at exactly:

```text
+0x55033
```

across ASLR-randomized library bases.

Examples included first/repeated faults around:

```text
~2200 s
~2269 s
~2510 s
~2829 s
~17,319 s
~17,468 s
~1607 s on a later cleaner boot
```

The exact instruction offset repeating across boots was one of the strongest signatures in the whole campaign.

Interpretation:

- If the host was clean before the crash, `libhsa +0x55033` was treated as a **V620/HSA heat-soak marker**.
- If Geekbench or host MM had already failed earlier in the same boot, the later HSA event was considered contaminated.

At one point increasing actual V620 voltage moved the first HSA failure from the ~40–47 minute region to nearly **4h48m uptime**, strongly suggesting a genuine GPU voltage/heat-soak margin.

### Direct V620 GPUVM fault

`07:00.0` also produced direct AMDGPU GFXHUB faults under more aggressive undervolt:

```text
[gfxhub] page fault
client UTCL2
Faulty UTCL2 client ID: TCP
PERMISSION_FAULTS: 0x3
MAPPING_ERROR: 0
```

Those are strong GPU-specific events and should not be scored as VCCIN failures.

### Current V620 state

Current confirmed configuration:

```text
card1 / 07:00.0
VDD GFX offset: -65 mV
Power cap:      250 W
Performance:    auto
```

---

## 16. ASRock Phantom Gaming / `03:00.0`: long-soak GPUVM wall

The ASRock card has become the dominant current GPU-specific failure source.

Signature:

```text
amdgpu 0000:03:00.0: [gfxhub] page fault
Process llama-server
client UTCL2
Faulty UTCL2 client ID: TCP
PERMISSION_FAULTS: 0x3
MAPPING_ERROR: 0
```

Representative events occurred around:

```text
5480.8 s
5756.1 s
8652.2 s
9032.6 s
10930.8 s
11246.8 s
```

At 8652/9032 the status was sometimes `0x00801031` with `MORE_FAULTS: 1`; at other times `0x00801030`.

### Voltage progression

The ASRock card had historically been around `-45 mV`.

It was progressively moved toward more actual voltage:

```text
-45 mV
-40 mV
...
-30 mV
-25 mV
-20 mV   <-- current confirmed state
```

Important contamination rule:

A fault a few minutes after changing the undervolt **without a full GPU/runtime reset after the prior GPUVM fault** is not a clean independent stability result.

Current confirmed ASRock state:

```text
card0 / 03:00.0
SCLK state 1: 2710 MHz
MCLK state 1: 1000 MHz
VDD GFX offset: -20 mV
Power cap: 323 W
Performance: manual
```

The host voltages should not be changed merely because this card produces a direct UTCL2/TCP fault.

---

## 17. Liquid Devil / `0A:00.0`

Current confirmed state:

```text
card2 / 0A:00.0
SCLK state 1: 2710 MHz
MCLK state 1: 1000 MHz
VDD GFX offset: -80 mV
Power cap: 346 W
Performance: manual
```

Earlier lore referenced the Liquid Devil as tolerating roughly `-90 mV` in some usage. Current setting is less aggressive at `-80 mV`.

No equally distinctive repeated fault signature has yet made this card the primary suspect in the preserved campaign state.

---

## 18. Current confirmed GPU configuration

From the most recent `amdgpu-custom-state` application:

```text
card0 / ASRock / 03:00.0
  SCLK 500 -> 2710 MHz
  MCLK 97  -> 1000 MHz
  VDD GFX Offset: -20 mV
  SCLK max: 3000 MHz
  MCLK max: 1075 MHz
  Power cap: 323 W
  Performance: manual

card1 / V620 / 07:00.0
  VDD GFX Offset: -65 mV
  Power cap: 250 W
  Performance: auto

card2 / Liquid Devil / 0A:00.0
  SCLK 500 -> 2710 MHz
  MCLK 97  -> 1000 MHz
  VDD GFX Offset: -80 mV
  SCLK max: 3000 MHz
  MCLK max: 1075 MHz
  Power cap: 346 W
  Performance: manual
```

---

## 19. Current host state: what is known and what is not

The later campaign reached a combined branch around:

```text
VCCIN  2.185 V explicitly stated at one key transition
VDIMM  1.44 V
VCCIO  tested through 0.70 / 1.03 / 1.07 / 1.13 V
```

This branch was notable because the BIOS rails stabilized and the host then survived long enough that **GPU-specific failures became the dominant visible errors**.

After that, VCCIN was incremented further in some branches. Because several 5 mV steps occurred across chat branches and not every exact setpoint survived into the current context, this document does **not** assign a false exact value to the later clean host run.

What *is* known:

- VCCIN had reached the low-to-mid 2.1/2.2 V configured region during the high-voltage campaign.
- VDIMM 1.44 V was deliberately introduced in a later combined branch.
- VCCIO was repeatedly tested at 0.70, 1.03, 1.07, 1.10, 1.11, 1.13, 1.14 and nearby points.
- The host eventually ran for hours without a host-MM signature while `03:00.0` GPUVM faults became the active problem.
- That is the strongest evidence so far that the host and GPU problems can be separated.

Before this lore is used as an executable tuning recipe, the exact current BIOS values should be captured directly from the machine and appended here.

---

## 20. Notable host-failure checkpoints

Selected checkpoints worth preserving:

| Branch | Approximate result |
|---|---|
| tRFC330 / VCCIO 0.70 / VCCIN ~1.725 | `list_del corruption` at ~3145 s (~52m25s) |
| tRFC340 / VCCIO 1.03 / VCCIN ~1.830 | longest early host run; `Bad page state` at ~20,352 s (~5h39m) |
| VCCIO 1.08 / VCCIN ~1.755 | `warn_free_bad_obj` / SPL-ZFS free path at ~84 s |
| VCCIO 1.07 / VCCIN ~1.840 | Geekbench assertion/segfault around ~727 s |
| VCCIO 1.07 / VCCIN ~1.850 | hard lock |
| VCCIO 1.07 / VCCIN ~1.860 | Bad Page around ~204 s |
| VCCIO 1.07 / VCCIN ~1.870 | Bad Page around ~254 s |
| VCCIO 1.07 / VCCIN ~1.940 | first observed all-high BIOS gate |
| VCCIO 1.07 / VCCIN ~1.950 | distinct all-solid but SA-low state |
| VCCIO 1.10 / VCCIN ~2.010 | BIOS all-solid/high-SA, Linux boot hard lock |
| VCCIO 1.10 / VCCIN ~2.07 | settled low-SA gate, later hard memcg/MM corruption |
| VCCIO 1.10 / high VCCIN | GB4 could pass while GB5 exposed hard MM corruption |
| VCCIO 1.03 / ~2.205 set | `css_rstat_flush` / memcg-style GPF around ~647 s before Geekbench start |
| VCCIO 1.13 / high VCCIN | blue hard-lock window, Bad Pages, then later materially longer clean runs |

This table is intentionally representative, not exhaustive.

---

## 21. Geekbench as a host gate

Geekbench turned out to be a useful staging tool because it could expose host instability quickly.

### Typical interpretation

```text
Geekbench userspace segfault / GPF:
  soft host fail

Geekbench finishes:
  useful milestone, not final proof

Kernel Bad Page/list_del/lruvec while Geekbench runs:
  hard host fail; boot poisoned
```

At one stage:

- Geekbench 4 became passable.
- Geekbench 5 became the new wall.
- Later Geekbench 7 completed on a much healthier branch.

Representative Geekbench 7 scores observed:

```text
~982 single / 11795 multi
~979 single / 11830 multi
```

These scores were less important than the fact that the test completed without host MM corruption.

---

## 22. Why "more voltage" was sometimes the wrong mental model

The campaign repeatedly demonstrated:

- more VCCIN could help,
- then hurt,
- then restore a BIOS rail state,
- then still fail Linux,
- while a VCCIO change could move the apparent VCCIN floor by a large amount.

Examples:

- 1.08 VCCIO could appear bad at one VCCIN and useful at another.
- 1.10 VCCIO moved the BIOS all-solid threshold down dramatically but did not guarantee runtime stability.
- SA could go high at one VCCIN and low again at a higher VCCIN.
- Vcore could become solid, resume dithering, then become solid again.
- VDIMM 1.44 looked useless in one era but became interesting in a later combined branch.

The correct model is a **multidimensional stability surface**, not a monotonic ladder.

---

## 23. Working fault taxonomy for automation

This should eventually be encoded directly into `frankenstein-llm` health/qualification tooling.

### HOST_HARD_MM

Match examples:

```text
BUG: Bad page state
list_del corruption
lruvec_stat_mod_folio
page does not match folio
warn_free_bad_obj
__rmqueue_pcplist
page_counter_cancel
mod_memcg_state
css_rstat_flush
non-canonical address
dead0000000001..
invalid mapping
```

Action:

```text
FAIL
mark boot contaminated
stop qualification
reboot required
```

### HOST_SOFT

Examples:

```text
Geekbench segfault
Geekbench GPF
Geekbench assertion
ml failure
unrelated userspace segfault during host stress
```

Action:

```text
FAIL rung
capture evidence
kernel not automatically considered poisoned
```

### GPUVM_CARD0

Match:

```text
amdgpu 0000:03:00.0
[gfxhub] page fault
UTCL2
TCP
PERMISSION_FAULTS: 0x3
```

Interpretation:

```text
ASRock GPU-specific until contrary evidence
do not score as VCCIN failure
```

### GPUVM_V620

Match:

```text
amdgpu 0000:07:00.0
[gfxhub] page fault
UTCL2
TCP
```

Interpretation:

```text
V620-specific GPU path
```

### HSA_V620_HEATSOAK

Match:

```text
python ... libhsa-runtime64.so.1.18.0
fault offset +0x55033
```

Interpretation:

```text
strong V620/HSA heat-soak marker IF host was clean first
```

### CONTAMINATED

Conditions:

```text
host hard fault already occurred
GPUVM fault already occurred without reset
run changed multiple voltage axes unintentionally
wrong tRFC/tREFI loaded
```

Action:

```text
do not compare time-to-failure as a clean independent datapoint
```

---

## 24. Lessons that should become `frankenstein-llm` behavior

The hardware campaign directly motivates software architecture.

### 24.1 First-failure timestamp must be first-class evidence

The most useful information is often:

```text
what failed first?
at what uptime?
how long after workload start?
on which PCI BDF?
```

A qualification report should store those explicitly.

### 24.2 Missing evidence must never become a pass

A clean-looking terminal is not proof.

A model or host should only be marked qualified when the intended checks completed and produced recorded evidence.

```text
missing != pass
stale != pass
interrupted != pass
unknown != pass
```

### 24.3 A failed boot/run becomes a contaminated state

After a host MM failure, stop collecting "independent" GPU evidence.

After a GPUVM fault, do not claim that a voltage change made the next five-minute re-fault an independent long-soak result unless the card/runtime was reset.

### 24.4 Hardware identity should use stable identifiers

Store:

- PCI BDF
- device identity
- board nickname
- power cap
- voltage offset
- clock state

Do not rely only on transient `card0` / `GPU0` labels.

### 24.5 Capture tuning state with every result

A useful evidence record should include at least:

```text
timestamp
boot ID
kernel
BIOS
BCLK
memory frequency
timings hash/snapshot
VCCIN set + readback
VCCIO set + readback
VDIMM set + readback
LLC
switching frequency
ambient temp if available
GPU BDF -> identity map
GPU clocks
GPU power cap
GPU voltage offset
workload command/model
workload start monotonic time
first fault monotonic time
dmesg excerpt
qualification result
contamination state
```

### 24.6 One-variable experiments should be machine-enforced where possible

The campaign repeatedly lost time when:

- tRFC was accidentally 330 instead of 340,
- tREFI changed unexpectedly,
- multiple rails were moved together,
- a GPU offset was changed on a boot already contaminated by a previous GPU fault.

Automation should compare the intended state to the actual state before declaring a run valid.

---

## 25. Current `frankenstein-llm` robustness direction

The repository is primarily Python + Bash + systemd tooling around local inference and qualification, not a Rust project.

The robustness work should fit that architecture.

Already-discussed hardening direction:

- bounded message sizes,
- exact input schemas,
- key/type allowlists,
- canonical UUID validation,
- duplicate JSON-key rejection,
- poison-message tests,
- fail-closed evidence handling,
- subprocess process-group management,
- timeout + SIGTERM -> grace -> SIGKILL escalation,
- cleanup in `finally`,
- atomic state/evidence writes,
- stale/truncated state recovery,
- Bash `set -Eeuo pipefail` where appropriate,
- signal/exit-code preservation,
- systemd restart semantics that distinguish operator stop from failure.

The stability campaign explains **why** that rigor matters. Real hardware faults create partial state, interrupted jobs, poisoned boots, stale artifacts, and misleading downstream symptoms.

---

## 26. Current unresolved questions

### Host

- What exact VCCIN/VCCIO/VDIMM combination is the true long-duration host floor?
- Is the final host margin primarily VCCIN, VCCIO, VDIMM, refresh behavior, or an interaction among them?
- Does tRFC 330 have a reproducible useful island at the later high-input-voltage state?
- How much of the BIOS 8 mV "dithering" is real electrical movement versus ADC/reporting quantization?
- What is `in14` physically connected to and how is it scaled?

### ASRock GPU0

- Does `-20 mV` finally clear the long-soak UTCL2/TCP wall?
- If not, what is the first clean offset after a full reset/reboot?
- Is 2710 MHz itself too aggressive for the desired undervolt floor?

### V620

- Is `-65 mV` fully clean under multi-hour HSA heat soak?
- Does the exact `libhsa +0x55033` signature disappear at the current offset?
- Is the limit voltage-only, or voltage/frequency/TDP interaction?

### Liquid Devil

- Is `-80 mV` genuinely long-soak clean under the same llama workload?
- It has not yet produced the same repeated distinctive fault pattern in the preserved record.

---

## 27. Next experimental discipline

Recommended campaign discipline from this point:

1. **Capture exact BIOS state before every new long run.**
2. **Record boot ID and monotonic workload start time.**
3. **Freeze host settings while tuning a GPU.**
4. **Freeze GPU settings while mapping host voltage.**
5. **Reboot/reset after any hard host fault.**
6. **Reset GPU/runtime after a GPUVM fault before calling a new voltage point independent.**
7. **Use PCI BDF in every GPU fault record.**
8. **Run GB4/GB5/GB7 as staged host gates, then long mixed workload.**
9. **Do not call a setting stable until it survives heat soak and warmer ambient.**
10. **Treat the first fault as the causal lead; later cascades are evidence of contamination, not extra votes.**

---

## 28. Signature crib sheet

### Host hard-MM

```text
Bad page state
list_del corruption
lruvec_stat_mod_folio
__rmqueue_pcplist
page_counter_cancel
mod_memcg_state
css_rstat_flush
invalid mapping
page still charged to cgroup
dead000000000100
dead000000000122
00ff...
deff...
```

### V620 HSA

```text
libhsa-runtime64.so.1.18.0
+0x55033
```

### AMD GPUVM

```text
[gfxhub] page fault
UTCL2
TCP
PERMISSION_FAULTS: 0x3
MAPPING_ERROR: 0
```

### ASRock

```text
0000:03:00.0
```

### V620

```text
0000:07:00.0
```

### Liquid Devil

```text
0000:0A:00.0
```

---

## 29. Things we learned the hard way

- A beautiful BIOS state can crash Linux immediately.
- A terrible-looking rail state can sometimes run longer than expected.
- VCCIO is not monotonic.
- VCCIN is not monotonic.
- SA high is not automatically better than SA low.
- Dithering is not automatically instability.
- Memtest passing does not guarantee Linux allocator stability.
- Geekbench is useful because it is disposable and repeatable, not because Geekbench is the root cause.
- ZFS is often where corruption gets caught, not where it begins.
- A GPUVM fault is not a VCCIN metric.
- A userspace HSA segfault can be an excellent GPU marker when it repeats at the exact same library offset.
- A GPU test after host MM corruption is contaminated.
- A GPU voltage test after a prior GPUVM fault without reset is contaminated.
- Short undervolt tests lied about the V620 heat-soak floor.
- Warmer ambient can turn a "stable" configuration into a failure.
- Changing two knobs at once feels fast and often costs more time later.
- BIOS save contamination (`tRFC`, `tREFI`) can invalidate hours of mapping.
- The machine is a system, not a set of independent sliders.

---

## 30. Where the campaign stands now

The most encouraging recent development is that host-MM corruption stopped being the first visible failure for long enough that **specific GPU faults became dominant**.

That is progress.

Current confirmed GPU tuning:

```text
ASRock / 03:00.0      -20 mV, 2710 MHz, 323 W
V620 / 07:00.0        -65 mV, 250 W
Liquid Devil / 0A:00  -80 mV, 2710 MHz, 346 W
```

The ASRock is the current active GPU target because it continues to show long-soak `UTCL2/TCP/PERMISSION_FAULTS` behavior as voltage is walked toward zero.

The host is **not declared solved**, but recent hours-long operation without `Bad page`, `list_del`, `lruvec`, or memcg corruption while GPU-specific faults appear is the strongest separation of fault domains achieved so far.

---

## 31. Final post-mortem conclusion

The initial mental model was:

> "There is one unstable overclock. Find the voltage or timing that fixes it."

The evidence forced a different model:

> "There are multiple independent margins, and every failure must be classified by domain, ordering, hardware identity, and contamination state."

The campaign became successful only when:

- Linux MM failures were treated as authoritative host evidence,
- ZFS was recognized as a detector rather than automatically blamed,
- GPU faults were tied to PCI BDF,
- HSA's exact `+0x55033` signature was separated from random host corruption,
- rail states were treated as telemetry rather than proof,
- VCCIN/VCCIO were mapped as a multidimensional surface,
- and first-failure provenance became more important than the raw number of errors.

That philosophy now belongs in `frankenstein-llm` itself.

**The machine is called Frankenstein for a reason: it works by respecting every subsystem's scars.**

---

## Appendix A: external electrical/overclocking research notes

Research performed during the campaign found both of the following to be true:

1. X99/Haswell-E/Broadwell-E overclocking material genuinely discusses VCCIN/VRIN in the ~2.0–2.3 V region, especially in extreme overclocking.
2. Intel's Xeon E5 v4 electrical documentation gives substantially lower supported/absolute values.

These are not contradictory. The BIOS and enthusiast tooling permit settings outside Intel's guaranteed envelope.

For future maintainers: do not quote an overclocking control range as if it were a long-term safe-operating specification.

---

## Appendix B: lore update checklist

When this campaign changes materially, append:

```text
[ ] exact BIOS screenshot / values
[ ] exact VCCIN set/readback
[ ] exact VCCIO set/readback
[ ] exact VDIMM set/readback
[ ] BCLK and DDR rate
[ ] tRFC / tREFI verification
[ ] boot ID
[ ] ambient temperature
[ ] GPU BDF map
[ ] GPU offsets / clocks / power caps
[ ] workload start time
[ ] first failure time
[ ] first failure classification
[ ] whether run was contaminated
[ ] whether reboot/GPU reset occurred
[ ] longest clean duration
```

This prevents the next two months from becoming archaeology.
