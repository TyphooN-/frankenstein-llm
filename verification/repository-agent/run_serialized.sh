#!/usr/bin/env bash
# Run the bounded local repository-agent gate after GPU-heavy TTS completes.
#
# The lane is an A/B: the incumbent preset and the reviewed Qwen3-Coder-Next
# candidate answer the same fixture under the same read-only oracle. They run one
# after another because the router keeps a single model resident, and both run
# even if the first fails -- a failing incumbent must not hide the candidate's
# result, which is the whole point of qualifying the candidate.
set -uo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
GATE=$ROOT/verification/repository-agent/gate_repo_agent.py
LOG=$ROOT/verification/repository-agent/evidence/repo-agent-runner.log
MODELS=(heretic qwen3-coder-next)
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
    worst=0
    for model in "${MODELS[@]}"; do
      printf 'repository agent model=%s start %s\n' "$model" "$(date -Is)"
      python3 "$GATE" --model "$model"
      rc=$?
      printf 'repository agent model=%s rc=%s %s\n' "$model" "$rc" "$(date -Is)"
      if [ "$rc" -eq 75 ]; then
        # No new model attempt after admission is lost. Preserve any genuine
        # earlier failure rather than allowing a later refusal to hide it.
        if [ "$worst" -eq 1 ]; then exit 1; fi
        exit 75
      elif [ "$rc" -ne 0 ] && [ "$rc" -ne 76 ]; then
        worst=1
      elif [ "$rc" -eq 76 ] && [ "$worst" -eq 0 ]; then
        worst=76
      fi
    done
    printf 'runner end rc=%s %s\n' "$worst" "$(date -Is)"
    exit "$worst"
  fi
  sleep 2
done
printf 'router unavailable %s\n' "$(date -Is)"
exit 1
