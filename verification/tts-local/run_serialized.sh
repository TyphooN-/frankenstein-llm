#!/usr/bin/env bash
# Serialize the local TTS + ASR round-trip gate after UI-TARS finishes.
set -uo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
GATE=$ROOT/verification/tts-local/gate_tts.py
PY=$ROOT/venvs/tts/bin/python
EVIDENCE=$ROOT/verification/tts-local/evidence
LOG=$EVIDENCE/tts-runner.log
mkdir -p "$EVIDENCE"
exec >>"$LOG" 2>&1
printf 'runner start %s\n' "$(date -Is)"

router_was_active=0
# Enqueue the router restart; never wait for it.
#
# This runs from the EXIT trap, and the trap is reached while systemd is
# executing this unit's own stop job. local-ai-qualification.service is ordered
# After=llama-router.service, so systemd puts the qualification stop ahead of
# any llama-router start in the same transaction: a blocking `systemctl start`
# here waits for a job that is waiting for this trap to return. On 2026-09-09
# stopping the legacy supervisor unit wedged exactly that way -- the trap sat in
# `systemctl --user start llama-router.service` (pid 116389) while
# `systemctl --user list-jobs` showed the router start queued behind the
# qualification stop -- until TimeoutStopSec expired at 90s and SIGKILL ended it.
#
# --no-block returns as soon as the job is enqueued; systemd runs it once this
# unit's stop completes. The router is therefore *requested*, not observed
# ready, and nothing here may claim otherwise. Callers that need a live router
# start it themselves and poll for it (see repository-agent/run_serialized.sh).
restore_router() {
  if [ "$router_was_active" -eq 1 ]; then
    router_was_active=0
    if systemctl --user --no-block start llama-router.service; then
      printf 'router restore queued; readiness not verified %s\n' "$(date -Is)"
    else
      printf 'WARNING: router restore could not be queued %s\n' "$(date -Is)"
    fi
  fi
}
on_exit() {
  # Capture first: the runner's status is the gate's status, never the trap's.
  local status=$?
  restore_router
  exit "$status"
}
interrupted() {
  local signal=$1 status=$2
  printf 'runner interrupted signal=%s status=%s %s\n' "$signal" "$status" "$(date -Is)"
  exit "$status"
}
trap on_exit EXIT
trap 'interrupted HUP 129' HUP
trap 'interrupted INT 130' INT
trap 'interrupted TERM 143' TERM

# UI-TARS already established host idleness. Re-check immediately and refuse
# rather than overlap a newly started compiler or other heavyweight workload.
if ! "$ROOT/venvs/computer-use/bin/python" \
    "$ROOT/verification/computer-use-grounding/gate_computer_use.py" \
    --preflight-only --max-load 6.0; then
  printf 'TTS refused: host no longer idle %s\n' "$(date -Is)"
  exit 75
fi

if systemctl --user is-active --quiet llama-router.service; then
  router_was_active=1
  systemctl --user stop llama-router.service
fi
"$PY" "$GATE"
rc=$?
printf 'runner end rc=%s %s\n' "$rc" "$(date -Is)"
exit "$rc"
