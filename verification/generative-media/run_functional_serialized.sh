#!/usr/bin/env bash
# Run real image, music, and image-edit workflows on the headless V620.
set -uo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
DIR=$ROOT/verification/generative-media
EVIDENCE=$DIR/evidence
LOG=$EVIDENCE/comfy-functional-server.log
RUNLOG=$EVIDENCE/media-functional-runner.log
RESULT=$EVIDENCE/media-functional.json
mkdir -p "$EVIDENCE/inputs" "$EVIDENCE/outputs" "$EVIDENCE/temp"
exec >>"$RUNLOG" 2>&1
printf 'runner start %s\n' "$(date -Is)"

# Refuse overlap with a compiler, download, or other heavy model process.
if ! "$ROOT/venvs/computer-use/bin/python" \
    "$ROOT/verification/computer-use-grounding/gate_computer_use.py" \
    --preflight-only --max-load 6.0; then
  printf 'functional media refused: host not idle %s\n' "$(date -Is)"
  exit 75
fi

baseline0=$(cat /sys/class/drm/card0/device/mem_info_vram_used)
baseline1=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
baseline2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
pid=
router_was_active=0
signal_seen=

# Enqueue the router restart; never wait for it.
#
# The restore runs from the EXIT trap, which is reached while systemd is
# executing this unit's own stop job. local-ai-qualification.service is ordered
# After=llama-router.service, so systemd puts the qualification stop ahead of
# any llama-router start in the same transaction: a blocking `systemctl start`
# here waits for a job that is waiting for this trap to return, and the unit
# only gets out of it when TimeoutStopSec fires. That deadlock was observed on
# 2026-09-09 in the sibling TTS runner. --no-block returns as soon as the job is
# enqueued, so the router is *requested* here, never observed ready.
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
# Stopping ComfyUI is separate from restoring the router on purpose: the VRAM
# residue check below has to read the cards with no other GPU consumer being
# introduced, so the router stays stopped until this script is done measuring.
stop_comfy() {
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  pid=
}
on_exit() {
  # Capture first: the runner's status is the gate's status, never the trap's.
  local status=$?
  stop_comfy
  restore_router
  exit "$status"
}
forward() {
  signal_seen=$1
  printf 'runner signal=%s %s\n' "$1" "$(date -Is)"
  [ -n "${pid:-}" ] && kill -s "$1" "$pid" 2>/dev/null || true
}
trap on_exit EXIT
trap 'forward HUP' HUP
trap 'forward INT' INT
trap 'forward TERM' TERM

if systemctl --user is-active --quiet llama-router.service; then
  router_was_active=1
  systemctl --user stop llama-router.service
fi

# Use one stable device identity. --cuda-device rewrites HIP visibility and
# double-remaps ordinals; do not combine it with ROCR filtering.
unset DISPLAY HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=GPU-a21e268c0b0a73d7
"$ROOT/venvs/comfy/bin/python" "$DIR/comfy_runtime.py" \
  "$ROOT/tools/ComfyUI/main.py" \
  --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --extra-model-paths-config "$DIR/extra_model_paths.yaml" \
  --input-directory "$EVIDENCE/inputs" \
  --output-directory "$EVIDENCE/outputs" \
  --temp-directory "$EVIDENCE/temp" \
  --fp32-vae --reserve-vram 1 >"$LOG" 2>&1 &
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
if [ "$ready" -ne 1 ]; then
  printf 'ComfyUI unavailable %s\n' "$(date -Is)"
  exit 1
fi

python3 "$DIR/live_schema_gate.py" || exit $?
"$ROOT/venvs/comfy/bin/python" "$DIR/functional_gate.py"
rc=$?
stop_comfy
sleep 8

after0=$(cat /sys/class/drm/card0/device/mem_info_vram_used)
after1=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
after2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
printf 'vram baseline=%s,%s,%s after=%s,%s,%s\n' \
  "$baseline0" "$baseline1" "$baseline2" "$after0" "$after1" "$after2"
# The 32 GiB card must release model residency; tolerate allocator/desktop noise.
if [ $((after1 - baseline1)) -gt $((768 * 1024 * 1024)) ]; then
  printf 'V620 retained excess VRAM after unload\n'
  rc=1
fi
if [ ! -s "$RESULT" ]; then
  printf 'functional result missing\n'
  rc=1
fi
printf 'runner end rc=%s signal=%s %s\n' "$rc" "${signal_seen:-none}" "$(date -Is)"
exit "$rc"
