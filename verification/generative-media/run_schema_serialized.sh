#!/usr/bin/env bash
# Start a loopback-only ComfyUI instance, validate models/nodes, then unload.
set -uo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
DIR=$ROOT/verification/generative-media
LOG=$DIR/evidence/comfy-server.log
RUNLOG=$DIR/evidence/media-schema-runner.log
mkdir -p "$DIR/evidence"
exec >>"$RUNLOG" 2>&1
printf 'runner start %s\n' "$(date -Is)"

if ! "$ROOT/venvs/computer-use/bin/python" \
    "$ROOT/verification/computer-use-grounding/gate_computer_use.py" \
    --preflight-only --max-load 6.0; then
  printf 'media schema refused: host not idle %s\n' "$(date -Is)"
  exit 75
fi

baseline0=$(cat /sys/class/drm/card0/device/mem_info_vram_used)
baseline1=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
baseline2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
pid=
cleanup() {
  [ -n "${pid:-}" ] && kill -TERM "$pid" 2>/dev/null || true
  [ -n "${pid:-}" ] && wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
unset DISPLAY
"$ROOT/venvs/comfy/bin/python" "$ROOT/tools/ComfyUI/main.py" \
  --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --extra-model-paths-config "$DIR/extra_model_paths.yaml" \
  --cuda-device 1 >"$LOG" 2>&1 &
pid=$!

ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 5 http://127.0.0.1:8188/system_stats >/dev/null; then
    ready=1
    break
  fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 2
done
[ "$ready" -eq 1 ] || exit 1
python3 "$DIR/live_schema_gate.py"
rc=$?
cleanup
pid=
sleep 8
printf 'vram baseline=%s,%s,%s after=%s,%s,%s\n' \
  "$baseline0" "$baseline1" "$baseline2" \
  "$(cat /sys/class/drm/card0/device/mem_info_vram_used)" \
  "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" \
  "$(cat /sys/class/drm/card2/device/mem_info_vram_used)"
printf 'runner end rc=%s %s\n' "$rc" "$(date -Is)"
exit "$rc"
