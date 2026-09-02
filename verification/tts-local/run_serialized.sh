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
restore_router() {
  if [ "$router_was_active" -eq 1 ]; then
    systemctl --user start llama-router.service || true
  fi
}
interrupted() {
  local signal=$1 status=$2
  printf 'runner interrupted signal=%s status=%s %s\n' "$signal" "$status" "$(date -Is)"
  exit "$status"
}
trap restore_router EXIT
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
