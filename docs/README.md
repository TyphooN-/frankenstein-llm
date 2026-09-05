# Documentation

## Start here

- [User guide](USER-GUIDE.md): setup, model choice, daily workflows and safety boundaries.
- [Architecture](reference/ARCHITECTURE.md): implementation layers, data flows, gates, supervision and evidence.
- [Configuration](reference/CONFIGURATION.md): presets, manifests, environment variables, ports and schemas.
- [Operations](reference/OPERATIONS.md): installation, downloads, services, qualification, updates and rollback.
- [Troubleshooting](reference/TROUBLESHOOTING.md): symptoms, diagnostic checks and recovery.
- [Capability matrix](reference/CAPABILITY-MATRIX.md): what exists, what is admitted, and how to obtain current proof.
- [Developer guide](reference/DEVELOPER-GUIDE.md): tests, extending capabilities and evidence contracts.
- [Coverage map](reference/COVERAGE-MAP.md): outer-repository file inventory and reference ownership.
- [Repository checks](REPOSITORY-CHECKS.md): verification commands and host-safety caveats.
- [Decisions](decisions/README.md): durable architectural rationale.

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
