# Phase-four candidate qualification

This directory holds the policy and non-inference admission checks for the six
researched candidates downloaded by phase four. Download completion proves only
that pinned bytes arrived intact; it is not a functional verdict.

`gate_candidate_policy.py` runs before model loading. It:

- checks every phase-four file by expected size without re-hashing model weights;
- proves the phase-one Qwen3-ASR snapshot is the revision phase four deduplicated;
- keeps WeMM's 2048-dimensional multimodal vectors in a distinct database and
  alias from the 4096-dimensional text embedding index;
- verifies the reviewed WeMM executable files against their pinned digests and
  rejects any new executable file;
- enforces low privilege for Gemma-4 Heretic and requires a positive grounding
  verdict before UI-Mate can receive desktop-control authority.

Current live qualification routing:

- Qwen3-Coder-Next is added to the repository-agent A/B while Qwen2.5 Coder keeps
  the separate FIM lane.
- Gemma-4 Heretic text and vision presets are reader-only and are never sent a
  tools payload.
- FLUX.2-klein-4B is discoverable through the local ComfyUI path configuration;
  a successful real workflow is still required before promotion.
- Qwen3-ASR remains exercised by the TTS-to-ASR round trip.

Not yet claimed:

- UI-Mate has no functional verdict and must be tested against the same sandboxed
  grounding/action fixtures as UI-TARS before any bounded control integration.
- WeMM has a text-only functional verdict (2048-D, L2, paraphrase>related>noise)
  on the V620. Image/video retrieval is not claimed: AutoProcessor needs
  torchvision, which is not in the ROCm candidate runtime. No vectors were
  written into the Qwen3 text index.
- No candidate is considered admitted merely because this static gate passes.
- No throughput, latency, token-rate, or comparative benchmark is performed here.

Run lightweight checks:

    python3 -m unittest discover -s verification/candidate-qualification -p 'test*.py'
    python3 verification/candidate-qualification/gate_candidate_policy.py

The second command publishes ignored runtime evidence under `evidence/`.

UI-Mate and WeMM use Qwen3.5 classes unavailable in the older UI-TARS runtime.
`requirements.in` and the hash-locked `requirements.lock` define their isolated
`venvs/candidates` environment. The lock is resolved with uv's CPU Torch backend
but omits Torch itself: `setup_runtime.sh` reuses the host ROCm Torch through
`--system-site-packages` and refuses a non-ROCm build. This avoids allowing a
resolver to install NVIDIA CUDA or Triton packages alongside the AMD runtime.
