# ADR 0009: Portable host profiles (AMD ROCm and NVIDIA CUDA)

- Status: Accepted (phased; phase 1 in progress)
- Date: 2026-09-30
- Amends: [ADR 0003](0003-hardware-allocation-and-memory-policy.md),
  [ADR 0005](0005-track-llama-cpp-submodule.md),
  [ADR 0006](0006-placement-policy-prefers-the-rx-6900-xts.md)

## Context

The repository was written for one host: a Xeon E5-2696 v4 with three Navi 21
GPUs (`gfx1030`) under ROCm. That host is baked in at several layers:

- The llama.cpp lock and `scripts/build-llama-cpp.sh` accept exactly one
  backend (`ROCm/HIP`) and one target (`gfx1030`), and a test pins both.
- `scripts/gpu_vram.py` enumerates devices from the KFD topology only, so an
  NVIDIA card is invisible to placement, telemetry and the gates.
- Presets in `llama-models.ini` carry `tensor-split` values sized for this
  host's three cards, although `scripts/gpu_placement.py` can already compute
  them from measured VRAM.
- systemd units and many scripts hard-code `/home/typhoon/git/frankenstein-llm`.

The operator wants the repository usable by anyone with similar hardware, and
has NVIDIA hardware of their own to support: an RTX 3060 12 GB desktop card and
a laptop with a 6 GB Quadro.

## Decision

1. **Target hardware (not a qualification claim).** Linux hosts with one or more GPUs from:
   - AMD, through ROCm/HIP. `gfx10.3` (RDNA2) is the reference; other ROCm
     targets are built for when detected, but are unqualified until someone
     runs the gates on them.
   - NVIDIA, through CUDA, subject to the installed toolkit and pinned llama.cpp
     supporting the detected compute capability. Older cards may need an older
     compatible toolkit; detection alone does not establish compiler support.
   - Vulkan, for GPUs that have neither runtime, and CPU-only as a last resort.

   The intended mixed-vendor design builds one llama.cpp per backend and
   serving each vendor's devices from its own server. A single model is not
   split across vendors.
2. **The build follows the host, and the lock follows the source.** The lock
   pins the llama.cpp revision and lists the backends the repository knows how
   to build. Which backends and GPU targets are built is detected from the host,
   with an explicit override. An explicitly requested GPU backend with nothing
   detected refuses rather than guessing; automatic selection falls back to CPU
   when neither HIP nor CUDA reports a device. Vulkan is explicit, not auto-selected.
   The reference host's `gfx1030` stays recorded as the
   reference target.
3. **Devices are vendor-neutral rows.** Device enumeration reports every GPU
   with its backend, its runtime index within that backend, identity (PCI
   address, UUID), total and used memory, and whether it drives a display.
   Placement, telemetry and the gates consume those rows instead of reading
   KFD directly.
4. **Placement is computed per host.** `tensor-split` and device selection come
   from `scripts/gpu_placement.py` against the local device rows and a local
   policy file. The reference host's measured policy remains as its profile;
   a new host starts from a generic default, not from it.
5. **Paths are relative to the checkout.** Units are installed from templates
   with the checkout path substituted; nothing assumes a user name or home
   directory.
6. **Evidence stays honest.** Qualification receipts already record the kernel
   and command line; they will also record the host profile (backend, device
   names, VRAM). A verdict earned on one host is never reused on another.

## Phases

1. Build: backend detection and per-backend build directories (`build` for
   HIP, keeping existing paths; `build-cuda`, `build-vulkan`, `build-cpu`).
   NVIDIA device enumeration beside KFD.
2. Placement and presets: vendor-neutral device rows through
   `gpu_placement.py`; generated per-host `tensor-split`; model selection by
   fit, including small-VRAM tiers (6 GB and 12 GB) and 1-bit `Q1_0` models.
3. Install: templated units, an installer that detects the host and writes the
   local profile, and removal of hard-coded home paths from scripts.
4. Evidence: host profile in receipts; gates and telemetry on NVIDIA.

## Consequences

### Implementation checkpoint

- Build detection, per-backend directories and offline orchestration tests are
  implemented. Tests exercise mock host/toolchain fixtures; they do not establish
  a successful CUDA or Vulkan compilation or hardware qualification.
- Physical KFD/NVIDIA VRAM inventory is implemented as the additive
  `physical_inventory()` API and `--physical-inventory` CLI mode. NVIDIA rows
  retain raw identifiers beside validated join keys; malformed, duplicate or
  unreadable observations report errors rather than fabricated values. Legacy
  ROCm APIs and default CLI behavior are unchanged.
- Physical inventory never establishes a runtime index: all its rows carry
  `runtime_index=None`, a mapping error and `placement_usable=false`. Visibility
  and ordering variables are recorded, not applied. NVIDIA memory is physical
  parent memory, not a MIG-instance placement budget. Independent CUDA runtime
  mapping, MIG-instance enumeration and NVIDIA placement remain pending;
  `nvidia-smi` indices must not be assumed to equal CUDA runtime indices.
- Inventory proof is offline fixtures and reference-host read-only observations,
  not NVIDIA hardware, CUDA runtime or MIG qualification.
- Phases 2–4 remain pending; existing services and model presets are reference-host
  configuration, not portable installation output. No NVIDIA model fit or Bonsai
  runtime compatibility verdict has been earned by these build changes.

- The reference host's hard pins become defaults plus detection. Tests that
  asserted `gfx1030` now assert that the reference target is recorded and that
  detection refuses an empty result.
- Nothing here is qualified on NVIDIA hardware until an operator runs the gates
  on it; the 3060 is not installed in the reference host as of this date.
- Per-host policy and measurements live in a local, untracked profile so one
  operator's measurements are never applied to another's cards.
