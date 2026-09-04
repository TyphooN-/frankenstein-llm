# Local capability status — 2026-09-02 static recovery

**Historical.** This pass froze phases 1–2 after an outage. Current operational
status is `docs/CANDIDATE-STATUS-2026-09-03.md`. The ignored
`../evidence/capability-ledger.json` snapshot is also 2026-09-02 and still
omits phases 3–4.

Machine-readable source of truth *for this snapshot*: `../evidence/capability-ledger.json`
(`schema: hermes-local-capability-ledger/1`). This document is the prose reading
of that file. Where the two disagree, the ledger is correct for 2026-09-02 only.

Scope of this pass: static and control-plane only. No model was loaded, no GPU
was used, no service was started or stopped, and no download was started.

## Why this pass exists

A power outage ended the previous lane before handoff (boot ID changed to
`42471379-317f-4fcb-b185-96ca96f1c971`). The `/tmp` evidence was lost, but the
workspace scaffolds survived on disk. This pass re-derived state from durable
files rather than from the lost transcript.

## The distinction that governs every row below

**Downloaded is not ready. Ready is not qualified.**

A capability is only `functionally_passed` when a gate artifact on disk records
`pass: true` from a run that actually happened. Static readiness — the code
compiles, the unit tests pass, the weights are the right size — is evidence that
the gate is *worth running*, and nothing more.

## Status

### Functionally qualified (8)

Each has a gate artifact recording an observed pass.

| Capability | Evidence | Recorded |
|---|---|---|
| embeddings | `gate-embeddings.json` | 2026-09-01 10:07 |
| native-tool-use | `gate-native-tool-use.json` | 2026-09-01 10:13 |
| rag | `gate-rag-structural/behavioural/live.json` | 2026-09-01 09:56–12:25 |
| vision-grounding | `gate-vision-grounding.json` | 2026-09-01 10:47 |
| ocr | `gate-ocr.json` | 2026-09-01 10:48 |
| reranking | `gate-reranker.json` | 2026-09-01 11:30 |
| asr | `gate-asr.json` | 2026-09-01 12:57 |
| fim | `gate-fim.json` | 2026-09-01 13:08 |

`rag` and `native-tool-use` have no artifact of their own; they are composed from
services and router presets that are already admitted.

### Failed / interrupted (1)

**computer-use-grounding.** The UI-TARS-1.5-7B gate ran 22:02–22:09 on the
previous boot and was killed by SIGTERM during the `grounding` phase (exit 4).
Two sections passed before the kill (`gpu_execution`, `placement`); eight never
ran. The unload check failed only because the process died holding VRAM.

This is an unfinished run, not a quality finding about the model. The weights are
downloaded and integrity-checked. The gate must be re-run to completion before
anything is concluded about UI-TARS.

### Blocked pending a safe boot (4)

Static readiness is complete for all four; each needs GPU/model execution that
the current host state forbids.

- **tts** — Qwen3-TTS-12Hz-1.7B-Base, 12/12 files. Gate needs GPU synthesis plus
  a second model load for the ASR round trip.
- **image-generation** — Z-Image Turbo BF16, size-exact. ComfyUI has never been
  started for this checkpoint; no image has been produced.
- **music-generation** — ACE-Step 1.5 turbo all-in-one, size-exact. No song has
  been generated.
- **repository-agent** — no artifact of its own; drives the existing `heretic`
  router preset over loopback. Qualifying it means loading a 27B model.

The media preflight on this boot correctly reports
`functional_gate_ready_now: false` with the active kernel build as the blocker.

### Structurally ready, no functional run expected (1)

**security-agent-scaffold.** The Strix pilot builds and validates run
specifications and never launches a container, so there is no functional run to
have. 31/31 hostile mutations refused, 12/12 scope destinations classified as
expected, 44 unit tests passing.

### Genuinely missing (2)

See `GAP-DECISIONS-2026-09-02.md`. Neither justifies a download today.

- **image-editing** — distinct from image-generation.
- **computer-use-end-to-end** — distinct from computer-use-grounding.

## Download and integrity state

All 9 pinned artifacts are complete across both phases. The post-outage
re-verification finished at 07:03/07:07 on this boot and re-checked sizes; the
SHA-256 reconciliation is the earlier 2026-09-01 16:57 pass, which recorded
0 mismatches and 0 problems over 105,318,725,786 declared bytes.

The reconciliation also lists 34,441,799,184 reclaimable bytes in quarantined
`.bad-*` files and stale partials. Nothing was deleted in this pass.

## Control-plane repairs made in this pass

Four holes were found by audit and closed. Each is pinned by a new negative test.

1. **TTS gate could pass without its central check.** `--skip-asr` omitted the
   intelligibility section entirely, so the section loop never saw a failure and
   the gate could report `pass: true` having never transcribed anything. A
   missing section is now treated exactly like a failing one.
2. **Repository-agent could rewrite its own oracle.** `test_calculator.py` was in
   the writable allowlist, so "make the tests pass" was satisfiable by deleting
   the assertions. The suite is now read-only and `run_tests` reports whether it
   is still byte-identical; a tampered run never counts.
3. **Strix scaffold never validated the container network.** Scope is enforced on
   the network, outside the agent — an argument that collapses under
   `--network=host` or the default bridge. A `network.policed` control now
   refuses host, bridge, default, empty, `container:*`, and any network outside
   the dedicated `strix-pilot*` prefix.
4. **Media preflight could read as covering image editing.** Coverage claims are
   now data (`WORKFLOW_CLAIMS`), separating "the weight is on disk" from "a gate
   observed it working". Every workflow currently reports
   `functionally_proven: false`.
