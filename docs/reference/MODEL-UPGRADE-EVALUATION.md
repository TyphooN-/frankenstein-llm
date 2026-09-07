# Local model upgrade evaluation strategy

Status: evaluation policy and rationale, not measured model recommendations.
Scope: frankenstein-llm local stack. Targeted upgrade benchmarks require explicit execution and the healthy, uncontended-host gates below; this document does not enable services.

## Objective

Choose models that improve an existing workflow enough to justify their resource cost. Filling memory is not the objective. Compare every candidate against the installed model serving the same role: general assistant/tool use, repository coding, reasoning, or multimodal work. Bigger weights or a newer release are not proof of a capability upgrade.

When two or more artifacts share a role (for example the 27B uncensored/abliterated chat slot), evaluate them against each other on **speed and usefulness** before adding another download or alias. Usefulness is task-level quality on the role's real work, not refusal rate or a filename. Speed is llama-bench prompt/decode under identical settings, with MTP/speculative paths disclosed rather than silently credited to the weights. Keep the current occupant until a challenger wins both a useful quality delta and an acceptable resource/latency tradeoff.

Consider both a stronger architecture at a useful quantization and a higher-fidelity quantization of an existing family. Preserve exact lineage and artifact identity; results from a base model do not automatically transfer to a quantized or abliterated derivative.

## Two resource tiers

1. Near-VRAM-resident: seek useful candidates approaching the aggregate GPU capacity without consuming the memory needed for context, work buffers, projectors, the driver, and the desktop. Nominal 64 GB across three GPUs is not a 64 GB weight budget or a single allocation pool. Per-device limits and placement determine feasibility.
2. RAM-offloaded: consider larger candidates using host RAM when a meaningful quality improvement could justify slower interaction. Use measured installed and available RAM for current acceptance. Treat 128 GB as a separate prospective scenario unless live inspection confirms it is installed. Do not count swap as acceptable inference capacity.

Report weight bytes, measured resident memory, and estimates separately. Sum all shards programmatically and distinguish decimal GB from binary GiB. Include host-side load transients and context growth, not just steady-state weight residency.

## Heterogeneous GPU placement

Evaluate equal three-GPU placement where supported and feasible, then placements favoring the two RX 6900 XTs with the remainder on the V620. Reserve measured desktop headroom on GPU2. Resolve runtime device indices through actual hardware identity, not assumed DRM enumeration order.

The user reports well-cooled, undervolted 6900 XTs at 2710 MHz and a thermally constrained V620. These motivate tests; they do not establish a speed ranking. Sustained clocks, memory bandwidth, thermal behavior, kernel support, and transfers can change the result.

A split of 6,6,6 is an equal ratio, not 6 GiB per card. Layer granularity, cache ownership, projectors, and runtime buffers affect actual residency. Compare preset configuration, router-advertised arguments, loaded child arguments, and measured residency before saying an edit is live.

Layer placement across three cards is not automatically simultaneous execution or a speedup for one request. Evaluate only split modes supported by the exact architecture, quantization, backend, and speculative-decoding configuration. Do not infer optimal performance from utilization percentages.

## Dense and MoE offload require separate evaluation

Dense inference repeatedly accesses a large portion of the model. Moving layers to CPU execution introduces host compute and memory-bandwidth costs; moving tensors between host and GPU can introduce link-transfer costs. Identify which mechanism the runtime actually uses rather than treating every offload path as equivalent.

MoE models activate a subset of experts per token, but total parameters still determine weight storage. Expert placement, routing locality, caching, and backend implementation determine whether reduced active computation translates into useful latency. Active-parameter count alone does not establish fit or throughput.

Expect diminishing speed returns as offload grows, potentially abrupt degradation under pressure rather than a smooth curve. Do not invent a tokens/sec estimate from parameter count or nominal bandwidth. A slower model remains worthwhile only if measured task outcomes justify the tradeoff.

## Admission sequence

1. Research exact downloadable artifacts, revision, license, lineage, quantization quality, total versus active parameters, and actual Linux AMD gfx1030 runtime support. Separate model-card claims from local evidence. No speculative giant downloads or unreviewed remote model code.
2. Estimate safe fit for current hardware and separately for prospective RAM upgrades. Reject configurations that rely on exhausting physical memory or swap.
3. Run isolated load, coherence, task correctness, context-stability, and unload checks before production admission. Keep an existing usable baseline available.
4. Benchmark promising candidates against that baseline using identical prompts, context targets, generation settings, and clearly recorded runtime features. If templates or speculative paths differ, disclose them; do not silently attribute their effects to weight quality.
5. Promote only after reproducible functional proof and an explicit quality/resource tradeoff. Loading successfully alone is insufficient.

## Benchmark boundary and method

The user's later request explicitly permits targeted benchmarks of larger upgrade candidates, including RAM-offloaded candidates. A further operator request authorizes a same-slot speed and usefulness A/B of the two mradermacher 27B Q6_K GGUFs against installed `heretic` and `obliterated`. It does not blanket-enable every existing throughput mission. Preserve the distinction in execution controls and require explicit benchmark execution rather than a permissive config default. Do not run that A/B while the functional mission lock is held.

Run timing comparisons only on a healthy, uncontended host with verified kernel/backend provenance, no competing build or GPU mission, and one inference owner. Do not change voltage, clocks, security settings, or the production router to manufacture a favorable result.

Measure separately:
- Task-level quality: code correctness, native tool calls, reasoning and retrieval accuracy, and relevant multimodal outcomes.
- Prompt processing and time to first token, separating cold loading from warm inference.
- Decode throughput and end-to-end latency at representative output lengths.
- Context scaling and the associated memory/quality effects.
- Per-device peak VRAM, host memory, memory/IO pressure, and swap deltas.
- Sustained thermal/clock behavior, repeated trials, and clean unload.

Use fixed workloads, controlled warmup and cache state, repeat runs, and report variability. Tiny smoke-test responses are functional evidence, not performance benchmarks. Report tradeoffs rather than collapsing unrelated quality and speed measurements into an unexplained score.

Abort or defer on unsafe memory pressure, kernel/allocator errors, missing GPUs, or workload collisions. Preserve evidence; do not reinterpret a confounded run as a model defect or a successful fit. Do not alter shared mission state or interrupt an active router without coordinating ownership.

## Deliverable and decisions

Maintain a role-specific shortlist with exact artifact sizes, sources, runtime requirements, current/prospective fit, expected benefit, uncertainty, and next experiment. Classify candidates as adopt after qualification, evaluate isolated, wait for compatible quant/runtime, or reject for this stack.

Keep recommendations evidence-based: no maximum-performance claims without measurements, no claim that 128 GB exists without inspection, and no assumption that a larger model is a better daily driver. Preserve a fast baseline even when a slower specialist earns a place in the stack.
