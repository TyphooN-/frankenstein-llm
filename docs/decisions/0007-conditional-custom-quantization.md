# ADR 0007: Qualify public artifacts first; custom quantization is conditional

- Status: Proposed; no model conversion, download or benchmark authorized by this document
- Date: 2026-09-08
- Builds on: [ADR 0002](0002-serialized-functional-qualification.md),
  [ADR 0005](0005-track-llama-cpp-submodule.md), and
  [ADR 0006](0006-placement-policy-prefers-the-rx-6900-xts.md)

## Context

The goal is the best useful model quality and performance on this host, not the
largest parameter count or highest synthetic tokens/second. The heterogeneous
RX 6900 XT pair and Radeon Pro V620 require identity-safe placement, display
headroom and measured thermal behavior. ADR 0006 supplies a placement policy,
not proof that a model or quantization performs well on the current boot.

[Candidate research](../reference/CANDIDATE-RESEARCH-CLOSEOUT.md) records
conditional public quantization comparisons. Screening, download completion,
functional qualification and optimization are separate states. Failed or
interrupted workflow gates must not be bypassed to promote another artifact.

Producing our own GGUF is possible only when the exact source architecture and
conversion/quantization path are supported. It does not require training a new
foundation model. Neither a custom quant nor a runtime fork inherently improves
quality or throughput. A smaller file may use slower kernels or lose useful
behavior; a higher-fidelity artifact may force expensive RAM offload.

## Proposed decision

Prefer a pinned, qualified public artifact when it meets the workload. Attempt a
custom quant only for a recorded gap: missing supported format, a useful memory
fit boundary, or a reproducible quality deficit at a target size. Keep separate
winners for general assistance, repository work and specialist workflows when
no single candidate dominates. Do not silently replace the general driver.

### 1. Establish the baseline and experiment contract

Finish or diagnose existing functional gates first. Record source revision,
artifact digests, tokenizer/template, runtime commit/build flags, context, KV
cache types, placement and hardware/boot identity. Select the actual user
workloads and minimum acceptable correctness before comparing candidates.

Compare one candidate with the incumbent and a public matched-size control.
Where feasible, add a higher-fidelity reference from the same source revision.
Do not attribute differences between instruction tuning, chat templates or
source weights to quantization. Test shipped templates first; a controlled
same-template comparison is a separately labeled experiment. Test MTP or other
speculative decoding separately from the non-speculative baseline.

### 2. Admission and resource preflight

Pin the source revision, inherited license, conversion tool revision and every
required component, including projectors. Reject unclear provenance or terms.
Verify checksums against trusted manifests where available; record locally
computed hashes separately rather than describing them as provenance proof.

Measure available disk and RAM before transfer or conversion. Budget source
weights, converted high-precision GGUF, calibration data, intermediate files,
output quants and temporary space. Read the exact tool's current help and
supported architectures; do not invent command flags or assume streaming memory
usage. Pilot the conversion path before committing to a large build. No package
or driver upgrades, GPU allocation changes, paid compute or public publication
are implied by admission.

Use original supported BF16/FP16 weights, or a documented high-precision source
conversion. Do not requantize a low-bit GGUF and call it equivalent to a fresh
quantization. If the high-precision source cannot fit the supported conversion
path within measured resources, stop and retain public controls.

### 3. Reproducible local build

Use the repository-pinned llama.cpp toolchain when it supports the model. Keep
weights and build artifacts outside Git, consistent with
[ADR 0001](0001-git-tracked-workspace-without-weights.md). Store a small recipe
and manifest containing exact commands, versions, hashes, timings, resource
peaks and output metadata; no secrets or private prompts.

For importance-matrix quantization, use a licensed, representative calibration
corpus disjoint from held-out evaluation tasks. Pin its transformation and hash;
record seed and sample/token budget. Calibration must cover intended languages,
code and native tool structures without leaking evaluation answers. First
produce one target quant chosen to cross a measured useful fit boundary, not an
unbounded matrix of every quantization type.

A runtime fork is a separate decision: only patch a demonstrated conversion or
kernel defect after a minimal reproducer and upstream compatibility review.
Keep patches small and pinned; never change cryptographic or security policy to
make a candidate pass. Fine-tuning or weight surgery requires its own data,
license and quality plan and is not part of this quantization experiment.

### 4. Correctness and performance qualification

Request explicit benchmark authorization and use the existing serialized
admission path on a healthy idle host. No competing qualification, interactive model,
compiler or media workload; never stop user processes merely to admit a test.
An unstable boot or changed GPU tuning invalidates comparison evidence.

Run the same held-out functional gates on all candidates: coherent output,
structured output, native multi-turn tools, repository read/edit/test behavior,
negative paths and relevant language or specialist tasks. Perplexity alone is
not admission. Preserve failures and raw bounded evidence, including HTTP
status, timeout and unload results; do not raise tolerances to produce a pass.

Measure prompt processing, generation, time to first token, end-to-end task
latency, task success and peak RAM/VRAM at the intended contexts. Include thermal
steady state and clean unload. Repeat comparable runs in alternating order and
report spread, not just the best run. Compare the preferred XT placement with
other policy-admissible placements only when they fit measured headroom. RAM
offload is allowed if its quality gain justifies measured latency. A fit estimate
is not a measured peak or safety guarantee.

### 5. Promotion, rollback and stop conditions

Promote only after mandatory correctness and safety gates pass and the candidate
shows a reproducible useful tradeoff against the incumbent at the agreed target.
Record quality regressions explicitly; no automatic promotion from a speed
headline. Keep the prior artifact and preset available for rollback, update
candidate/placement records and documentation, then requalify the exact deployed
preset. Do not publish derived weights without separate approval and license
review.

Stop after the initial target fails admission or lacks a useful measured gain.
Allow one diagnosis-driven corrective variant, not blind parameter sweeps.
Missing source rights, unsupported conversion, excessive resource requirements,
unstable host, silent numerical errors or worse workload correctness are stop
conditions. Retaining the incumbent is a valid outcome.

## Consequences and current end state

This plan makes custom quantization an auditable experiment, not a promise of
free performance. It adds local storage and verification costs, and may conclude
that existing public artifacts are better. No model is declared best by this
ADR; no quantization recipe is executable until source, exact backend commands,
resource budget, held-out tasks and success thresholds have been selected.

The next decision is a bounded experiment proposal naming those inputs and the
specific fit or quality gap. Existing qualification failures remain a separate repair
backlog and must not be relabeled as completed by this document.
