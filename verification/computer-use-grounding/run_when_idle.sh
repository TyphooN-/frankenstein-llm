#!/usr/bin/env bash
# Serialize the UI-TARS three-GPU admission gate against the local router and
# report a control-plane result that cannot be more optimistic than the evidence.
#
# The name is historical. This runner no longer waits for an idle host: the gate
# it drives measures function (placement, grounding, typed action selection,
# injection resistance, clean unload) and never measures throughput, so a
# concurrent kernel build is context to record, not a reason to refuse. Host load
# is still captured into the artifact by the gate's own preflight.
#
# Two properties matter more than anything else here:
#   1. Signals reach the Python child and this script waits for its real status.
#      A trap that exits while Python is still running -- or that reports the
#      trap's own status instead of the child's -- turns a killed run into a
#      "success", which is precisely how a run can disappear without a trace.
#   2. Success is never reported without computer-use-grounding.json. If Python
#      exits 0 but the artifact is missing, stale or unreadable, this script
#      fails closed.
set -uo pipefail

ROOT=/home/typhoon/git/frankenstein-llm
GATE=$ROOT/verification/computer-use-grounding/gate_computer_use.py
PY=$ROOT/venvs/computer-use/bin/python
EVIDENCE=$ROOT/verification/computer-use-grounding/evidence
LOG=$EVIDENCE/computer-use-runner.log
GATE_LOG=$EVIDENCE/computer-use-gate-run.log
ARTIFACT=$EVIDENCE/computer-use-grounding.json
RESULT=$EVIDENCE/computer-use-runner-result.json
ROUTER=llama-router.service

# Exit codes above the gate's own (0 pass, 1 fail, 3 already running,
# 4 interrupted) so the control plane can tell them apart.
EXIT_NO_ARTIFACT=5

mkdir -p "$EVIDENCE"
# Keep the inherited stderr (the journal, under systemd) on fd 3 so lifecycle
# lines land in both places. The incident this hardening responds to left a
# journal containing start lines and nothing else.
exec 3>&2
exec >>"$LOG" 2>&1
export PYTHONUNBUFFERED=1

journal() { printf '%s\n' "$*" >&3 2>/dev/null || true; }

# ExecStopPost hook. The manager sets SERVICE_RESULT/EXIT_CODE/EXIT_STATUS here,
# so this is the one place that can always record how the unit ended -- including
# the endings the payload never gets to describe, such as SIGKILL after the stop
# timeout. The transient run this replaces produced start lines and nothing else.
if [ "${1:-}" = "--stop-post" ]; then
  if [ -s "$ARTIFACT" ]; then artifact_note=present; else artifact_note=MISSING; fi
  stop_line=$(printf 'unit stop result=%s exit_code=%s exit_status=%s artifact=%s invocation=%s' \
    "${SERVICE_RESULT:-unknown}" "${EXIT_CODE:-none}" "${EXIT_STATUS:-none}" \
    "$artifact_note" "${INVOCATION_ID:-none}")
  printf '%s %s\n' "$(date -Is)" "$stop_line"
  journal "$stop_line"
  "$PY" - "$RESULT" "$ARTIFACT" <<'STOPPOST' || true
import json, os, sys, tempfile
path, artifact = sys.argv[1], sys.argv[2]
try:
    payload = json.loads(open(path).read())
except Exception:  # noqa: BLE001 - a stop record must exist even with no prior result
    payload = {"runner": "run_when_idle.sh", "note": "no runner result preceded this stop"}
present = os.path.exists(artifact) and os.path.getsize(artifact) > 0
payload["unit_stop"] = {
    "service_result": os.environ.get("SERVICE_RESULT"),
    "exit_code": os.environ.get("EXIT_CODE"),
    "exit_status": os.environ.get("EXIT_STATUS"),
    "invocation_id": os.environ.get("INVOCATION_ID"),
    "artifact_present": present,
}
payload["control_plane_success"] = bool(payload.get("control_plane_success") and present)
handle = tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False)
with handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(handle.name, path)
STOPPOST
  exit 0
fi

STARTED_AT=$(date -Is)
STARTED_EPOCH=$(date +%s)
child=0
signal_seen=""
router_was_active=0
router_restored=0
python_rc=""

log() { printf '%s runner %s\n' "$(date -Is)" "$*"; journal "runner $*"; }

log "start invocation=${INVOCATION_ID:-none} pid=$$ args=[$*]"

restore_router() {
  [ "$router_restored" -eq 0 ] || return 0
  router_restored=1
  if [ "$router_was_active" -eq 1 ]; then
    log "restoring $ROUTER (was active before this run)"
    systemctl --user start "$ROUTER" || log "WARNING: could not restart $ROUTER"
  else
    log "leaving $ROUTER stopped (it was not active before this run)"
  fi
}

# Machine-readable control-plane record, written even when the run is cut short.
write_result() {
  local status=$1 artifact_state=$2 artifact_pass=$3
  "$PY" - "$RESULT" "$status" "$artifact_state" "$artifact_pass" <<'PY' || true
import json, os, sys, tempfile
path, status, artifact_state, artifact_pass = sys.argv[1:5]
payload = {
    "runner": "run_when_idle.sh",
    "exit_code": int(status),
    "python_exit_code": os.environ.get("RUNNER_PYTHON_RC") or None,
    "signal_seen": os.environ.get("RUNNER_SIGNAL") or None,
    "artifact_path": os.environ.get("RUNNER_ARTIFACT"),
    "artifact_state": artifact_state,
    "artifact_pass": {"true": True, "false": False}.get(artifact_pass),
    "control_plane_success": int(status) == 0 and artifact_pass == "true",
    "router_was_active": os.environ.get("RUNNER_ROUTER_WAS_ACTIVE") == "1",
    "router_restore_ran": os.environ.get("RUNNER_ROUTER_RESTORED") == "1",
    # Only a router that was stopped by this run can be "restored" by it.
    "router_restored": (os.environ.get("RUNNER_ROUTER_WAS_ACTIVE") == "1"
                        and os.environ.get("RUNNER_ROUTER_RESTORED") == "1"),
    "started_at": os.environ.get("RUNNER_STARTED_AT"),
    "finished_at": os.environ.get("RUNNER_FINISHED_AT"),
    "invocation_id": os.environ.get("INVOCATION_ID"),
    "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(),
}
directory = os.path.dirname(path)
handle = tempfile.NamedTemporaryFile("w", dir=directory, delete=False)
with handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(handle.name, path)
PY
}

on_exit() {
  local status=$?
  restore_router
  RUNNER_PYTHON_RC="$python_rc" RUNNER_SIGNAL="$signal_seen" \
    RUNNER_ARTIFACT="$ARTIFACT" RUNNER_ROUTER_WAS_ACTIVE="$router_was_active" \
    RUNNER_ROUTER_RESTORED="$router_restored" RUNNER_STARTED_AT="$STARTED_AT" \
    RUNNER_FINISHED_AT="$(date -Is)" \
    write_result "$status" "${artifact_state:-unknown}" "${artifact_pass:-unknown}"
  log "end rc=$status python_rc=${python_rc:-none} signal=${signal_seen:-none} artifact=${artifact_state:-unknown}"
  exit "$status"
}
trap on_exit EXIT

# Forward the signal to Python and then keep waiting. Do not exit here: the
# child owns model unload, telemetry shutdown and the artifact write, and this
# script's status must be the child's real status, not the trap's.
forward() {
  local sig=$1
  signal_seen=$sig
  log "received SIG$sig; forwarding to gate pid=${child:-none} and waiting"
  if [ "$child" -ne 0 ] && kill -0 "$child" 2>/dev/null; then
    kill -s "$sig" "$child" 2>/dev/null || true
  fi
}
trap 'forward TERM' TERM
trap 'forward HUP' HUP
trap 'forward INT' INT

if systemctl --user is-active --quiet "$ROUTER"; then
  router_was_active=1
  log "stopping $ROUTER for exclusive GPU access"
  systemctl --user stop "$ROUTER" || log "WARNING: could not stop $ROUTER"
fi

{
  printf '\n===== gate run %s (runner pid %s) =====\n' "$STARTED_AT" "$$"
} >>"$GATE_LOG"

"$PY" "$GATE" "$@" >>"$GATE_LOG" 2>&1 &
child=$!
log "gate started pid=$child log=$GATE_LOG"

# `wait` returns 128+signum when a trapped signal interrupts it, which does not
# mean the child is gone. Re-wait until the child is genuinely reaped so the
# status reported is the child's own.
rc=0
while :; do
  wait "$child"
  rc=$?
  [ "$rc" -ge 128 ] || break
  kill -0 "$child" 2>/dev/null || break
done
python_rc=$rc
log "gate exited rc=$rc"

restore_router

# Fail closed: a zero from Python is not a pass unless the artifact says so and
# was written by this run.
artifact_state=missing
artifact_pass=unknown
if [ -s "$ARTIFACT" ]; then
  artifact_check=$("$PY" - "$ARTIFACT" "$STARTED_EPOCH" <<'PY'
import json, os, sys
path, started = sys.argv[1], int(sys.argv[2])
try:
    summary = json.loads(open(path).read())
except Exception as error:  # noqa: BLE001
    print(f"unreadable {type(error).__name__}")
    raise SystemExit(0)
fresh = os.stat(path).st_mtime >= started - 1
print(f"{'fresh' if fresh else 'stale'} {str(bool(summary.get('pass'))).lower()}")
PY
)
  artifact_state=${artifact_check%% *}
  case "$artifact_check" in *" true") artifact_pass=true ;; *" false") artifact_pass=false ;; esac
fi
log "artifact state=$artifact_state pass=$artifact_pass path=$ARTIFACT"

if [ "$rc" -eq 0 ] && { [ "$artifact_state" != "fresh" ] || [ "$artifact_pass" != "true" ]; }; then
  log "FAIL-CLOSED: gate exited 0 but artifact is $artifact_state/pass=$artifact_pass"
  rc=$EXIT_NO_ARTIFACT
fi

exit "$rc"
