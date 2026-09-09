# Qualification naming migration

The supervisor, status command, service, runtime directory, Python identifiers,
environment overrides, and operating documentation now use **qualification**.
A qualification measures acceptance; a gate is a check set; an attempt is one
execution. Admission and permission are unrelated words and are unchanged.

## New entry points

- `python3 scripts/qualification_status.py`
- `python3 verification/qualification-supervisor/run_qualification.py --plan`
- `systemctl --user status local-ai-qualification.service`
- `HERMES_QUALIFICATION_QUIET_TIMEOUT` replaces the previous timeout override.
- State/logs: `verification/qualification-supervisor/qualification-state.json`
  and `qualification.log`; individual gate log names remain unchanged.

## Existing installations

Do not run old and new supervisors concurrently. Before updating a live checkout,
stop `local-ai-functional-mission.service` and confirm its MainPID and cgroup have
no live processes. Stop any manually launched `run_functional_mission.py` too.
Do not change source files out from under an active gate.

After updating the checkout, inspect the offline migration plan:

```bash
python3 scripts/migrate_qualification_state.py
python3 scripts/migrate_qualification_state.py --apply
```

The migration acquires both supervisor locks, refuses visible old/new runners,
preflights conflicting targets, and copies only known runtime files. It keeps the
old `verification/mission-supervisor` directory as a backup. It translates paths
in the state snapshot but preserves receipt bytes, keys, verdicts, and timestamps
exactly. It never promotes historical failure/inconclusive/running records to
passes. Divergent new state must be reconciled explicitly, not overwritten.

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

The legacy names in this migration document and migration tests/tool are
intentional compatibility references, not supported parallel public interfaces.
