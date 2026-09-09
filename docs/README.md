# Documentation

## Start here

- [Scripted model operation](MODEL-RUNS.md): serving, qualification, benchmarks and editable configuration.

- [User guide](USER-GUIDE.md): setup, model choice, daily workflows and safety boundaries.
- [Architecture](reference/ARCHITECTURE.md): implementation layers, data flows, gates, supervision and evidence.
- [Configuration](reference/CONFIGURATION.md): presets, manifests, environment variables, ports and schemas.
- [Operations](reference/OPERATIONS.md): installation, downloads, services, qualification, updates and rollback.
- [GPU execution and model loading](reference/GPU-EXECUTION-AND-MODEL-LOADING.md): which ROCm index is which card, how each preset's layer split was computed, why three cards add capacity rather than speed, and when a model is actually loaded.
- [Placement, measured](reference/PLACEMENT-MEASUREMENTS.md): the live three-GPU placement A/B ADR 0006 was decided without. What including the V620 costs, what splitting a single-card model costs, and which presets a context change could move.
- [Larger-model shortlist](reference/MODEL-UPGRADE-SHORTLIST.md): provisional candidates, exact publisher sizes, and admission caveats.
- [Model upgrade evaluation](reference/MODEL-UPGRADE-EVALUATION.md): quality versus resource cost, VRAM/RAM tiers, and controlled candidate benchmarks.
- [Candidate research closeout](reference/CANDIDATE-RESEARCH-CLOSEOUT.md): current dispositions for the eleven queued additions, upgrade families, and extra artifacts. Research is not admission.
- [Queued candidate additions](reference/CANDIDATE-ADDITIONS-INVESTIGATION.md): earlier primary-source notes for the eleven additions; verdicts there are superseded by the closeout.
- [Troubleshooting](reference/TROUBLESHOOTING.md): symptoms, diagnostic checks and recovery.
- [Coverage plan](LOCAL-HERMES-CAPABILITY-COVERAGE-PLAN.md): living capability roadmap against current downloads and ledger states.
- [Candidate status](CANDIDATE-STATUS-2026-09-06.md): current researched / downloaded / qualified reading.
- [Capability matrix](reference/CAPABILITY-MATRIX.md): what exists, what is admitted, and how to obtain current proof.
- [Developer guide](reference/DEVELOPER-GUIDE.md): tests, extending capabilities and evidence contracts.
- [Coverage map](reference/COVERAGE-MAP.md): outer-repository file inventory and reference ownership.
- [Repository checks](REPOSITORY-CHECKS.md): verification commands and host-safety caveats.
- [Native throughput artifacts](benchmarks/README.md): measured `llama-bench` prompt/generation rates per weight file, with the placement, kernel and MTP caveats each figure was measured under. Each artifact is a dated measurement of one run, not a live reading and not a quality ranking.
- [Decisions](decisions/README.md): durable architectural rationale.
- [SALINE LOOP](AI-METAL-BAND.md): an operator creative brief for a war metal / blackened death / deathgrind project, mapped to the local weights that would produce each asset, including what character-consistent music video this stack can and cannot do. A concept note, not a capability claim.

- [Handoff 2026-09-07](HANDOFF-2026-09-07.md): state at the 2026-09-07 reboot — what is
  proven, what the boot-time qualification run is expected to prove, the authorized 27B
  download that was not started, and what a reboot invalidates.

## Reading status correctly

Dated research and status files are historical observations, not current host state.
Use the capability ledger and runtime status commands in the guide for a current
reading. Policy admission, downloaded artifacts and functional qualification are
separate states. A fixture test is not a live GPU qualification.

Performance testing requires explicit user authorization after confirmation of the
intended kernel. A reboot alone does not grant authorization. Documentation does
not change that boundary or authorize service changes.

The references cover the outer repository implementation, not every upstream
llama.cpp internal. Upstream source is pinned by the tracked submodule and lock.
