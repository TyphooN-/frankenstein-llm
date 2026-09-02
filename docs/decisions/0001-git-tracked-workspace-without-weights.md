# 0001. Git-tracked workspace without model weights

- Status: accepted
- Date: 2026-09-02

## Context

The local AI workspace originally lived at `/home/typhoon/frankenstein-llm` as an untracked directory holding both source and large GGUF/safetensors artifacts. A private GitHub repository now exists at `git@github.com:TyphooN-/frankenstein-llm.git`, and the canonical checkout is `/home/typhoon/git/frankenstein-llm`.

Committing weights would bloat Git, leak unpublished conversions, and make ordinary clones unusable.

## Decision

- Track source, configuration, scripts, systemd templates, verification code, and documentation.
- Ignore the entire `/models/` tree, local virtualenvs, nested upstream clones, logs, transfer partials, and ad-hoc probe dumps.
- Keep named fixtures under `verification/*/fixtures/` when they are small and required by tests.
- Do not recreate Git metadata under `/home/typhoon/frankenstein-llm`. That path is retired.

## Consequences

- Clones are small and reviewable.
- Weights remain host-local and must be downloaded or copied separately.
- Runtime units and `llama-models.ini` use `/home/typhoon/git/frankenstein-llm` as the root.
