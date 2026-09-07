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

restore_router() {
  if [ "$router_was_active" -eq 1 ]; then
    systemctl --user start llama-router.service || true
    router_was_active=0
  fi
}
cleanup() {
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  pid=
  restore_router
}
forward() {
  signal_seen=$1
  printf 'runner signal=%s %s\n' "$1" "$(date -Is)"
  [ -n "${pid:-}" ] && kill -s "$1" "$pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'forward HUP' HUP
trap 'forward INT' INT
trap 'forward TERM' TERM

if systemctl --user is-active --quiet llama-router.service; then
  router_was_active=1
  systemctl --user stop llama-router.service
fi

# ComfyUI is a single-device workflow runtime. Expose only the two headless GPUs
# and select remapped device 1, the physical 32 GiB V620; GPU2 stays excluded.
export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
# Route matmuls through hipBLAS rather than hipBLASLt. These cards are gfx1030
# (recorded by ComfyUI itself at startup) and torch 2.13's own hipBLASLt support
# list is gfx9 only -- its "Attempting to use hipBLASLt on an unsupported
# architecture" override did not fire here. On 2026-09-07 the image-edit
# workflow died in node 14 (KSampler) with HIPBLAS_STATUS_INVALID_VALUE out of
# hipblasLtMatmulAlgoGetHeuristic, which is hipBLASLt reporting that it has no
# algorithm for that problem shape, while image generation and music generation
# completed in the same process. This selects a backend that covers the shape;
# it is not a tolerance change and the workflows are still scored on their
# actual output.
export TORCH_BLAS_PREFER_HIPBLASLT=0
unset DISPLAY
"$ROOT/venvs/comfy/bin/python" "$ROOT/tools/ComfyUI/main.py" \
  --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --extra-model-paths-config "$DIR/extra_model_paths.yaml" \
  --input-directory "$EVIDENCE/inputs" \
  --output-directory "$EVIDENCE/outputs" \
  --temp-directory "$EVIDENCE/temp" \
  --cuda-device 1 --reserve-vram 1 >"$LOG" 2>&1 &
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
cleanup
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
