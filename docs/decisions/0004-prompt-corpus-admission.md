# ADR 0004: Admit prompt corpora as inert, pinned evidence

- Status: Accepted
- Date: 2026-09-04

## Context

The local stack needs complementary evaluation for harmful compliance, false refusal, benign utility, indirect prompt injection, and tool authorization. Public prompt collections vary sharply in provenance and safety. Some flatten upstream benchmarks and discard model, judge, category, split, and query metadata. Others mix prompts, executable files, credential-like material, or copied production system prompts under a repository-level license that does not establish row-level provenance.

A downloaded prompt collection is therefore not automatically executable test input. Treating corpus text as commands, tool arguments, URLs, code, or agent memory would turn evaluation data into an authority channel.

## Decision

Maintain a machine-readable source catalog under `verification/prompt-corpus-admission/` with immutable revisions, explicit licensing status, disposition, suite assignment, execution policy, and rationale.

Only sources marked `candidate`, with verified licensing and `inert-text-only` policy, may produce admitted local artifacts. Every artifact must be normalized to bounded UTF-8 JSONL or CSV and have a separate manifest binding the catalog source and revision to exact bytes, SHA-256, row count, and field schema. Admission rejects symlinks, path traversal, executable permissions, binary/NUL content, credential markers, unknown fields, schema drift, and any source marked reference-only or rejected. Corpus bytes remain ignored by Git; admission evidence is runtime state rather than source.

Corpus content has no authority. Evaluators may present strings to a model or synthetic mock environment, but must never execute supplied commands, code, URLs, tools, side effects, credentials, payments, messages, filesystem operations, network requests, or persistent-memory writes.

Keep the following score families separate:

1. harmful-request refusal and substantive harmful compliance;
2. false refusal and benign utility;
3. indirect prompt injection and tool-integrity behavior;
4. future MCP and computer-use safety in disposable synthetic environments.

A passing score in one family cannot qualify another.

## Source disposition

- Prefer provenance-preserving canonical sources such as PHTest, JailbreakBench behaviors, AgentDojo, and InjecAgent after exact artifact manifests are created and reviewed.
- Keep evaluator implementations and larger frameworks such as FalseRefusal, HarmBench, StrongREJECT, MCP Security Bench, and OS-Harm reference-only until their dependencies, constituent licenses, and execution environments are separately admitted.
- Use `offensive-ai-compilation` and Trail of Bits `awesome-ml-security` as research catalogs only.
- Reject `Dobliuw/Prompts` as an unlicensed aggregation that strips important upstream metadata.
- Reject wholesale ingestion of TheBigPromptLibrary because mixed provenance, executable material, and credential-like content make repository-level licensing insufficient for safe corpus admission.

## Consequences

The initial slice adds policy and validation but does not download any external prompt rows and does not claim a model score. Curated adapters must preserve upstream IDs, splits, categories, labels, and provenance while producing normalized inert artifacts. Live evaluation remains serialized with model qualification and cannot operate real tools or external systems.

This adds an explicit preparation step, but it prevents benchmark contamination, accidental secret ingestion, arbitrary tool execution, and misleading cross-suite safety claims.
