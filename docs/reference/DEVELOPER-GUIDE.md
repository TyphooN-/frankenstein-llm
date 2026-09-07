# Developer guide

How to change this repository without breaking the properties it exists to
protect. Read [architecture](ARCHITECTURE.md) first; this file assumes it.

Related: [configuration](CONFIGURATION.md) · [operations](OPERATIONS.md) ·
[coverage map](COVERAGE-MAP.md)

## Contents

- [Conventions](#conventions)
- [Running the tests](#running-the-tests)
- [Test inventory](#test-inventory)
- [Adding a router preset](#adding-a-router-preset)
- [Adding a sidecar](#adding-a-sidecar)
- [Adding a capability](#adding-a-capability)
- [Writing a gate](#writing-a-gate)
- [Writing an evidence artifact](#writing-an-evidence-artifact)
- [Adding a download queue](#adding-a-download-queue)
- [Admitting a prompt corpus](#admitting-a-prompt-corpus)
- [Documentation rules](#documentation-rules)
- [Writing an ADR](#writing-an-adr)

## Conventions

**Fail closed, and prove it from the failing side.** A control counts as present
only when something fails without it. Every policy constant in this repository
has a negative test; a control with only positive tests is decoration.

**One writer per fact.** If a number appears twice, something must reconcile the
copies. See [configuration → ownership](CONFIGURATION.md#ownership). When you
cannot avoid a copy, add a cross-file test — `test_queue_manifests.py` is the
model for that.

**Absence is not a pass.** Missing artifact, unreadable JSON, missing timestamp,
interrupted run, unknown model status, unknown router alias, unknown corpus
source: all refuse. Never write `.get(x, True)` for a permission.

**Durable publication.** Anything a later run or a reboot reads is written
`.tmp`/`.staged` → `fsync` → `os.replace` → parent-directory `fsync`. Tolerate a
filesystem that refuses directory fsync for state; do not tolerate it for model
promotion.

**Single writer.** Long-running writers take an exclusive `flock` and exit 75
rather than racing.

**Never read `/proc/<pid>/cmdline` in a supervisor.** Linux serves it through
`access_remote_vm`, so a task holding its own mmap write lock can wedge the
reader. Use `comm`, the `cwd` symlink and `cgroup`, tolerate the task exiting,
and remember `comm` is truncated to 15 characters — match the truncated form.
(`gate_router_models.py` and `post_reboot_gate.py` do read cmdlines; both are
short-lived foreground checks, not unattended supervisors.)

**Untrusted data stays data.** Screen text, retrieved documents and corpus rows
are never instructions. If you add a path that feeds external text to a model,
carry the envelope and add an injection case.

**No throughput.** Do not add a timing measurement to a gate. Every artifact
carries `throughput_measured: false`, and the ledger reports an artifact claiming
otherwise as a problem. See
[why](CAPABILITY-MATRIX.md#no-numeric-model-characterization-harness).

**Comment the load-bearing part.** The prevailing style here explains *why* a
check exists and what breaks without it, usually with the incident that motivated
it. Match that density; do not narrate what the code plainly says.

## Running the tests

```bash
python3 -m pytest                                # everything tracked
python3 -m pytest --ignore=verification/repository-agent   # safe on a busy host
python3 -m pytest verification/upstream-pin/ -v
python3 -m unittest discover -s verification/candidate-qualification -p 'test*.py'
```

`pytest.ini` scopes collection to `verification/` and excludes `.git`, `logs`,
`models`, `tools`, `venvs`, `verification/**/evidence` and
`verification/repository-agent/fixture/`. That last exclusion matters: the
fixture's three tests are the *oracle* for the repository-agent gate and are
defective on purpose. Reported as project failures they would train everyone to
ignore a red suite. The gate copies the fixture to a temporary directory and runs
it with `unittest` inside Bubblewrap, so it never sees this configuration.

The live repository-agent gate creates private Linux namespaces; unit tests are not equivalent to live isolation proof. Some broad tests invoke host probes that can hang under kernel pressure. Run the full suite
only after builds and other host pressure have drained.

## Test inventory

| Suite | Covers |
|---|---|
| `verification/upstream-pin/test_llama_cpp_pin.py` | lock fields, `.gitmodules`, staged gitlink vs lock, build-script/lock agreement, stale runtime paths in tracked files, every preset key against `llama-server --help` |
| `verification/mission-supervisor/test_mission_supervisor.py` | quiet-timeout parsing, bounded quiet wait and its blocker reporting, `/proc` classification against fixtures, durable state, operator-stop vs step-failure classification |
| `verification/local-coverage-foundation/test_download_queue_cli.py` | `--help` exits before any lock/log/state side effect; unexpected arguments exit 2 |
| `.../test_download_parallel.py` | per-file ownership after `realpath`, concurrency, per-artifact state accounting |
| `.../test_download_resume.py` | short partial resumes, full-size-correct partial promotes, full-size-wrong partial quarantines, transport selection |
| `.../test_queue_manifests.py` | byte totals agree across queue, phase runner and mission supervisor |
| `.../test_capability_ledger.py` | absent, unreadable, interrupted, failing and stale evidence; gate-identity binding; tracked mapping |
| `.../validators/test_gatelib.py` | clean-unload fails when it could not measure; cleanup runs and never raises out of `finally` |
| `.../rag/test_rag_store.py` | extraction, ingest safety, sync patterns |
| `verification/candidate-qualification/test_candidate_policy.py` | privilege tiers, unknown-alias refusal, abliteration rule, control-without-verdict refusal, index separation, dedup claim |
| `.../test_wemm_gate.py` | WeMM gate contract |
| `verification/repository-agent/test_repo_agent.py` | write outside the allowlist, symlinked target, escaped path, rewritten oracle, sandbox that will not start, green run invalidated by a later write, evidence lifecycle, model selection |
| `verification/computer-use-grounding/test_gate_logic.py` | coordinate mapping, action parsing, injection judging, scoring, telemetry aggregation, atomic artifacts, single-instance lock, signal semantics |
| `.../test_grounding_contract.py` | the two checkpoints' geometry, action-space and injection-judging differences |
| `verification/operator-runs/test_serve_model.py` | preset/catalog agreement, weight-filename listings, fail-closed catalog validation |
| `verification/router-functional/test_gate_router_models.py` | per-preset check lists, privilege profile, model coverage, release accounting, contended-host detection without reading `cmdline` |
| `.../test_local_model_status.py` | status classification, error handling against fake responses, alias-to-weight-file resolution; the `--host-sharing` report — absent/corrupt/unknown-schema mission state never read as idle, `current_step` excluded from the verdict in both directions, every supervisor status classified exactly once, blocking vs advisory conflicts, an unreadable process table downgrading the verdict, and exit 0 for every verdict |
| `verification/generative-media/test_media_policy.py` | GPU policy, build detection from `comm`/`cwd`, fail-closed vs routine `/proc` errors, build-name agreement with the supervisor, workflow claims, graph reference resolution |
| `.../test_functional_gate.py` | workflow graph construction |
| `verification/tts-local/test_tts_gate.py` | round-trip scoring, admission rules, text normalisation |
| `verification/qualitative-characterization/test_characterize.py` | corpus scope and tripwires, inert non-executable output, loopback-only transport, mechanical checks, truncation treated as inconclusive, refusal signal quoted rather than asserted |
| `verification/prompt-corpus-admission/test_admit_corpus.py` | every rejection path in admission |
| `verification/security-agents/strix-scaffold/test_strix_scaffold.py` | one negative test per policy control, plus offline scope denial |
| `verification/glm53flash-local/test_gate_glm32.py` | VRAM telemetry contracts |
| `verification/docs/test_documentation.py` | every tracked file appears in the coverage map; local links and anchors resolve; model-choice docs name the weight file each alias loads; the CONFIGURATION alias table matches `llama-models.ini`; the documentation audit covers every tracked documentation file exactly once and its declared count is the real one |

## Adding a router preset

1. Add the section to `llama-models.ini`. Keys are `llama-server` long options
   without `--`; an unknown key is fatal for the whole router, not just that
   alias.
2. Give it a privilege tier. Either add it to
   `candidate_policy.INCUMBENT_PRESETS`, or add it to a candidate's `presets`
   tuple so it inherits that candidate's tier. If you skip this,
   `uncovered_router_presets()` becomes non-empty and the router gate fails —
   which is the intended outcome, because an alias with no privilege decision
   must not receive tools by falling through.
3. Decide whether it belongs in `gate_router_models.CHAT_MODELS` or
   `VISION_MODELS`. The gate derives its check list from the privilege tier, so a
   `low` preset is checked for coherence and structured output and is never sent
   a tools payload.
4. If it is a tool-using tier, confirm the weights can actually express a tool
   call before trusting the alias. Load it and read
   `chat_template_caps.supports_tools` from
   `http://127.0.0.1:8080/props?model=<alias>`. Community merges and abliterations
   regularly ship a chat template with the tools branch stripped, which produces
   an empty `tool_calls` array that looks exactly like a model declining. Point
   `chat-template-file` at a template extracted from weights of the same
   architecture rather than writing one; see
   [chat templates](CONFIGURATION.md#chat-templates).
5. Add the key to `scripts/serve-model.py`'s `VALUE_FLAGS` or `BOOL_FLAGS` if the
   preset uses one that tool does not yet know. It refuses unknown keys, so a
   standalone run would otherwise either fail or serve a different model than the
   router does under the same alias.
6. Run the pin suite so the new keys are checked against the installed build.
7. Restart the router.

## Adding a sidecar

1. Write `services/sidecar-<name>.env` with the seven `SIDECAR_*` variables.
   Explain any non-obvious flag in a comment — the existing four all do.
2. Pick an unused loopback port and record it in
   [configuration → ports](CONFIGURATION.md#ports).
3. `systemctl --user start llama-sidecar@<name>.service`.
4. Write a gate that asserts a *content* property, not a 200. The reranker is the
   cautionary case: a GGUF converted without the classifier head loads fine,
   answers `/v1/rerank`, and returns near-identical scores for everything.

## Adding a capability

The full path, in order. Skipping a step does not fail fast; it produces a claim
nothing supports.

1. **Research and pin.** Record the decision in a dated review document with the
   exact repository and revision. `ADOPT` means promoted to the backlog, not
   installed.
2. **Collect metadata.** `research/collect_hf_metadata.py` reads the Hugging Face
   API into `research/hf/*.json` and `hf-lfs-index.tsv`. Never hand-type a size
   or a digest.
3. **Build the queue entry** with `research/build_download_queue.py` or
   `build_candidate_queue.py`. Choose a `capability` string; it is the join key
   the ledger groups by.
4. **Update every copy of the byte total** — the queue's `total_bytes`, the next
   phase runner's expectation, the mission supervisor's `UPSTREAM` tuple — then
   run `test_queue_manifests.py`.
5. **Download** through the queue unit.
6. **Serve it**: a router preset, a sidecar env file, a ComfyUI path category, or
   an isolated venv.
7. **Give it a privilege tier** in `candidate_policy`, plus prerequisites if it
   must not be used before another gate passes.
8. **Write the gate.** See below.
9. **Declare the evidence** in `build_capability_ledger.CAPABILITY_EVIDENCE` and
   the gate identity in `EXPECTED_EVIDENCE_GATES`. A capability absent from the
   map can never be qualified, and that is the correct default for anything newly
   downloaded — but leaving it absent forever is how a gap turns invisible, so
   record the reason in [the matrix](CAPABILITY-MATRIX.md#the-matrix).
10. **Add the step** to `run_functional_mission.STEPS` in dependency order.
11. **Update the docs and the [coverage map](COVERAGE-MAP.md).**

## Writing a gate

A gate is a program that observes behaviour and publishes one JSON verdict.

- Assert a property of the returned **content**. "It answered" is not evidence.
- If the component holds GPU memory, check it gives it back. Use
  `gatelib.unload_gate` / `ensure_unloaded`; a clean-unload check that could not
  measure must fail, not pass.
- Publish an in-progress record *before* the first model call, marked
  `interrupted: true`, so a run that dies mid-flight leaves an artifact the
  ledger refuses rather than the pass it superseded.
- If the gate has sections, treat a missing section as a failing section.
- If the gate executes candidate-written code, sandbox it, and make the sandbox
  mandatory: no unisolated fallback, and a boundary failure aborts rather than
  being handed back to the model as a retryable error.
- If the gate hands out tools, refuse any preset the privilege model does not put
  in the tool-using tier, and refuse unknown presets separately so the reported
  reason stays accurate when an allowlist is later widened.
- Take a single-instance lock if two copies would collide over a GPU.
- Record no timing.

## Writing an evidence artifact

The ledger reads exactly these fields. Anything else is for humans.

```jsonc
{
  "gate": "local-repository-agent",     // must equal EXPECTED_EVIDENCE_GATES[path]
  "pass": true,                          // absent or non-true is not a pass
  "interrupted": false,                  // true ⇒ evidence-interrupted, whatever else says
  "recorded_at": "2026-09-04T20:31:32-04:00",   // or finished_at; must parse with
                                                //  datetime.fromisoformat
  "sections_missing": [],                // non-empty ⇒ forced to fail
  "throughput_measured": false,          // true ⇒ reported as a ledger problem
  "benchmarking_performed": false
}
```

`recorded_at` is compared against the newest mtime of the weights the capability
owns. Evidence older than its weights is `evidence-stale` — reported as "this
evidence did not judge these bytes", not as a finding about the model.

## Adding a download queue

Naming convention, so the mixture already in the tree is not mistaken for
drift: **prose describes a queue by what it installs** ("the researched-candidate
queue", "the image-editing queue"), while **on-disk identifiers keep the
historical `phaseN` spelling**. Queue files, state files, completion stamps, unit
names and the `SystemExit` strings that quote them all record what was actually
downloaded and when; renaming them would rewrite evidence rather than clarify it.
`docs/reference/TROUBLESHOOTING.md` therefore still quotes `phase-one stamp
mismatch` verbatim, because that is the literal text
`run_download_phase2.py` raises.

1. Generate `download-queue-phaseN.json` from collected metadata, where `N` is
   the next unused number.
2. Add `run_download_phaseN.py`, modelled on the image-editing or
   researched-candidate runner: it waits on every earlier queue's state **and**
   stamp, then `execve`s `download_queue.py` with the five `HERMES_DOWNLOAD_*`
   paths set.
3. Add `local-ai-model-downloads-phaseN.service` with `After=` the previous
   queue, `ProtectSystem=strict`, `ProtectHome=read-only` and `ReadWritePaths`
   limited to `models/` and the foundation directory.
4. Add the queue to `run_functional_mission.UPSTREAM` with its exact byte total.
5. Extend `build_capability_ledger.QUEUES`.
6. Extend `test_queue_manifests.py` so the new copies are reconciled.
7. Describe it in the operator docs by its contents, not by its number.

## Admitting a prompt corpus

1. The source must already be `disposition: candidate` with a verified licence
   and `execution_policy: inert-text-only` in `source-catalog.json`. If it is
   not, that is a review decision and belongs in an ADR amendment, not in a
   manifest.
2. Normalise the rows to bounded UTF-8 JSONL or CSV under the ignored `corpora/`,
   preserving upstream IDs, splits, categories, labels and provenance.
3. Write a `hermes-prompt-corpus-artifact/1` manifest binding source ID, exact
   40-character revision, one approved suite, relative path, format, exact bytes,
   SHA-256, row count, field schema and `inert_text_only: true`.
4. Run admission:
   ```bash
   python3 verification/prompt-corpus-admission/admit_corpus.py MANIFEST.json \
     --evidence verification/prompt-corpus-admission/evidence/NAME.json
   ```
5. Keep the score families separate. A refusal score never establishes tool
   authorization integrity, and a low harmful-compliance number bought by
   refusing benign requests is a regression, not a win.

Corpus content has no authority: never execute a supplied command, code, URL,
tool call, credential, payment, message, filesystem operation, network request or
memory write.

## Documentation rules

- **A dated document stays dated.** When a snapshot is superseded, add a banner
  at the top naming it as historical and pointing at the current reading. Do not
  rewrite its body into the present tense — the pins and the reasoning in it are
  the reason it is kept.
- **Undated references describe behaviour, not host state.** Anything that
  changes when a service restarts belongs behind a command the reader runs, not
  in a sentence they read.
- **Publisher numbers are labelled as such**, every time.
- **A new tracked file needs a coverage-map row.** `verification/docs/` fails
  otherwise.
- **Local links are checked**, including anchors. Run
  `python3 -m pytest verification/docs/` after editing documentation.
- Prose may name a forbidden path in order to forbid it; fenced command blocks
  may not. The pin suite enforces exactly that split.

## Writing an ADR

Numbered sequentially under `docs/decisions/`, self-contained: context, decision,
consequences, status. Do not duplicate model weights or live qualification
evidence into one. Add the row to
[`docs/decisions/README.md`](../decisions/README.md) in the same commit.
