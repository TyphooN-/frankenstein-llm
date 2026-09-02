#!/usr/bin/env bash
# Run the bounded local repository-agent gate after GPU-heavy TTS completes.
set -uo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
GATE=$ROOT/verification/repository-agent/gate_repo_agent.py
LOG=$ROOT/verification/repository-agent/evidence/repo-agent-runner.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
printf 'runner start %s\n' "$(date -Is)"

# Do not trigger router model loading if a new heavy workload appeared.
if ! "$ROOT/venvs/computer-use/bin/python" \
    "$ROOT/verification/computer-use-grounding/gate_computer_use.py" \
    --preflight-only --max-load 6.0; then
  printf 'repository agent refused: host no longer idle %s\n' "$(date -Is)"
  exit 75
fi

systemctl --user start llama-router.service
for _ in $(seq 1 60); do
  if curl -fsS --max-time 5 http://127.0.0.1:8080/v1/models >/dev/null; then
    python3 "$GATE"
    rc=$?
    printf 'runner end rc=%s %s\n' "$rc" "$(date -Is)"
    exit "$rc"
  fi
  sleep 2
done
printf 'router unavailable %s\n' "$(date -Is)"
exit 1
