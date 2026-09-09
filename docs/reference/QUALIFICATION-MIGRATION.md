# Qualification naming migration

The supervisor, status command, service, runtime directory, Python identifiers,
environment overrides, and operating documentation now use **qualification**.
A qualification measures acceptance; a gate is a check set; an attempt is one
execution. Admission and permission are unrelated words and are unchanged.

## New entry points

- `./scripts/qualification-status.sh` (also accepts `--json`)
- `python3 scripts/qualification_status.py`
- `python3 verification/qualification-supervisor/run_qualification.py --plan`
- `systemctl --user status local-ai-qualification.service`
- `HERMES_QUALIFICATION_QUIET_TIMEOUT` replaces the previous timeout override.
- State/logs: `verification/qualification-supervisor/qualification-state.json`
  and `qualification.log`; individual gate log names remain unchanged.

## Existing installations

Stop the existing supervisor and confirm its MainPID and cgroup have no live
processes before updating a live checkout. Stop manually launched runners too.
Do not change source files out from under an active gate.

The one-time state migration has completed on the maintained host. Its helper,
tests, and recovery evidence are archived outside the repository under the
operator's qualification-closeout checkpoint. There is no supported alternate
status command or compatibility fallback.

When transferring runtime state from another installation, hold both supervisor
locks, preflight conflicting targets, and preserve the original evidence outside
the checkout. Copy receipt bytes, keys, verdicts, and timestamps exactly. Never
promote failure, inconclusive, interrupted, or running records to passes.
Divergent state requires explicit reconciliation, not an overwrite. A stopped
attempt without a completion receipt is interrupted, not running or passed.

Install the new tracked unit with the same security settings and disable the old
unit. Preserve whether it was enabled; do not automatically restart qualification
as part of the naming migration. Audit user overrides separately, including any
old environment-variable name or hard-coded state path. Reload the user manager,
then verify the new unit's FragmentPath and ExecStart and that the old unit is
inactive. Start `local-ai-qualification.service` only after host admission permits
it and inspection of `--plan` shows the intended remaining work.

## Receipts versus current eligibility

Copying a receipt is not a new qualification. Exact-key matching still determines
reuse. This release also changes gate and cache behavior, including unload proof
and postcheck ordering: receipts measured with those older criteria can legitimately
miss. Changed source hashes (including renamed source paths) are not silently
waived or automatically rekeyed. Existing evidence is retained, and `--plan`
explains eligibility; do not use `--force-requalify` merely for the rename.
