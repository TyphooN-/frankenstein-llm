# Frankenstein local-worker dispatch

This directory owns the reproducible Hermes integration, not a second model manager. It contains the native `local_worker` plugin **and the required Hermes runtime backport**. The upstream supervisor chooses bounded summarization, extraction, comparison and grounded drafting automatically; it retains its selected model, verifies untrusted results and keeps code edits, commands and sensitive decisions.

The worker receives provided text only: no tools, filesystem, shell, credentials or full conversation history. Provider, replaceable model alias and endpoint come from `auxiliary.local_helper`, through Frankenstein's existing router/presets. No global `delegate_task` rerouting, implicit provider discovery or cloud fallback is added.

## Reproducible runtime package

`runtime/manifest.json` pins Hermes base commit `d923a5eb487a17c0a68801bba9ef0f56c81040a8`, the patch SHA-256, and every target's preimage/postimage SHA-256 and Git executable mode. `runtime/hermes-local-helper.patch` carries the transport, observer/config/consumer integration, helper CLI, regression tests and runtime documentation. The upstream MIT notice is retained in `runtime/LICENSE.hermes`.

The source-only installer defaults to a **read-only check**. It requires the exact pinned checkout, validates the complete source set, rejects tampering, dirty/staged target files, symlinks and partial application, and recognizes byte-identical already-applied installs. It never resets files, stages changes, changes profiles/credentials/permissions/services, enables billing or manages models. Unrelated source/index changes are preserved. The operator must be the sole writer of the target paths while applying. If I/O fails, preserve the checkpoint for inspection rather than automatically rolling anything back.

An installed Hermes source checkout can remain Git-dirty because this backport is applied. Its complete recoverable implementation is versioned **here**; a separate Hermes commit is not required. A future Hermes base needs an explicitly rebased/reviewed package, not a weakened version check.

## Install

Use an existing source checkout at the pinned commit, or obtain that commit from the manifest's upstream repository. Do not switch/reset a dirty live checkout merely to satisfy the pin. Use Hermes' maintained PM setup for that source; do not mutate its environment with raw pip/uv.

From the Frankenstein repository root, set `HERMES_SOURCE` to that checkout:

```sh
python3 integrations/hermes-local-worker/install_runtime.py --hermes-source "$HERMES_SOURCE"
python3 integrations/hermes-local-worker/install_runtime.py --hermes-source "$HERMES_SOURCE" --apply
```

The first command reports `ready` or `already_applied`. The second applies only a complete verified preimage and verifies all postimages; a repeat is a no-op. No profile settings are copied from the operator's machine.

Configure only the intended profile using native Hermes commands. Choose an existing alias managed by Frankenstein; `heretic` is the current incumbent, not a dependency built into the plugin:

```sh
LOCAL_MODEL=heretic
hermes config set auxiliary.local_helper.provider llama.cpp
hermes config set auxiliary.local_helper.model "$LOCAL_MODEL"
hermes config set auxiliary.local_helper.base_url http://127.0.0.1:8080/v1
hermes config set auxiliary.local_helper.fallback_enabled false
```

The runtime provides bounded defaults: 15-second deadline, 512 output tokens, finite input/response caps and one local-helper permit. No credential or paid fallback is needed. Leave the main model, global delegation and unrelated auxiliary tasks alone. Session titling/profile-description are separate optional consumers; see the bundled runtime documentation for their independent opt-in policy.

Place/link this plugin directory at the active profile's `$HERMES_HOME/plugins/frankenstein-local-worker`, without overwriting an existing unrelated path. Validate and enable it:

```sh
hermes plugins doctor --ci integrations/hermes-local-worker
hermes plugins enable frankenstein-local-worker --no-allow-tool-override
hermes plugins list --plain --no-bundled
```

Supported gateways can reload plugin code live; read the activation result. A standalone process that has not reloaded needs relaunching. **Start a new conversation** for the tool and session-frozen policy. Existing conversations are not silently retrofitted and their cached prefixes must not be mutated. Native Tool Search may expose search/describe/call instead of a direct `local_worker` schema; this is supported and does not require changing global deferral settings.

## Verify

From the Frankenstein root, run the offline plugin/installer contracts:

```sh
python3 -B -m unittest discover -s integrations/hermes-local-worker/tests -v
```

The installer tests use disposable real Git repositories and exercise no-write checks, exact/idempotent application, preservation of unrelated staging, tampered/undeclared payloads, wrong pins, dirty/hidden/staged edits, unsafe paths, symlinks, malformed/oversized manifests and executable modes. Transport calls in plugin unit tests are explicitly mocked.

For runtime proof, apply the package to a **separate clean checkout at the pinned base, with both its source and Git metadata outside the real Hermes home**, then use that checkout's canonical `scripts/run_tests.sh` with a PM-built test interpreter. A linked worktree can still put recovery markers under the live checkout's `.git/worktrees`; use a standalone local clone when the live repository is inside the guarded home. Allow sufficient per-file time for the parametrized policy matrix (558 collected cases); a file timeout is not a passing subset. Run the nine bundled `tests/agent/test_local_helper_*.py` / `test_qualified_local_helper.py` and `tests/scripts/test_local_helper.py` files, plus appropriate existing auxiliary/config/oneshot regressions. Keep its fixture home and scratch outside the real operator's Hermes home; preserve the real filesystem guards. Do not run a source-install test against the live checkout or weaken guards to obtain green.

Actual automatic-dispatch acceptance is separate: use a fresh upstream supervisor with `tool_choice: auto`, observe native discovery plus one real configured local-helper request, inspect the tool-free wire shape, compare worker and final results to supplied facts, and confirm the supervisor/provider and no-fallback policy stayed unchanged. First-use discovery has overhead; this is not evidence of net token or latency savings.

Functional dispatch does not establish every operation's model quality, strict model-file provenance or host stability. Privileged attestors and production/consensus/funds operations are outside this integration.
