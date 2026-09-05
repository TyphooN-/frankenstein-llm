#!/usr/bin/env bash
# Start a STANDALONE llama-server for functional probing.
#
# Never touches llama-models.ini, llama-router.service or port 8080. This is a
# throwaway process the caller stops when done, used so a component can be tested
# without altering production router configuration.
#
# Env:
#   PROBE_PORT      listen port                    (default 8091)
#   PROBE_MODEL     GGUF path                      (required)
#   PROBE_ALIAS     served model name              (default probe)
#   PROBE_MMPROJ    optional projector path
#   PROBE_DEVICE    llama.cpp device list          (default ROCm1,ROCm2)
#   PROBE_SPLIT     tensor split                   (default 2,1)
#   PROBE_CTX       context size                   (default 16384)
#   PROBE_FLASH_ATTN  on|off|auto                  (default off)
#
# PROBE_FLASH_ATTN defaults to off deliberately: the installed HIP build aborts
# with "invalid device function" in launch_fattn for head-size-256 models on
# gfx1030. See EXECUTION-LEDGER.md, router flash-attention finding.
set -euo pipefail

PORT="${PROBE_PORT:-8091}"
MODEL="${PROBE_MODEL:?PROBE_MODEL is required}"
ALIAS="${PROBE_ALIAS:-probe}"
DEVICE="${PROBE_DEVICE:-ROCm1,ROCm2}"
SPLIT="${PROBE_SPLIT:-2,1}"
CTX="${PROBE_CTX:-16384}"
FLASH="${PROBE_FLASH_ATTN:-off}"
PIDFILE="${PROBE_PIDFILE:-/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/probe-${ALIAS}.pid}"
LOG="${PROBE_LOG:-/home/typhoon/git/frankenstein-llm/logs/probe-${ALIAS}.log}"

[ -f "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 2; }
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "probe '$ALIAS' already running as PID $(cat "$PIDFILE")" >&2
    exit 3
fi

args=(
    --host 127.0.0.1 --port "$PORT"
    --alias "$ALIAS"
    --model "$MODEL"
    --device "$DEVICE"
    --tensor-split "$SPLIT"
    --n-gpu-layers all
    --ctx-size "$CTX"
    --flash-attn "$FLASH"
    --parallel 1
    --load-mode none
    --jinja
    --temp 0
)
# llama.cpp refuses a quantized V cache unless flash attention is on, so the KV
# quantization has to follow the flash-attention setting rather than be fixed.
if [ "$FLASH" = "off" ]; then
    args+=(--cache-type-k q8_0)
else
    args+=(--cache-type-k q8_0 --cache-type-v q8_0)
fi
[ -n "${PROBE_MMPROJ:-}" ] && args+=(--mmproj "$PROBE_MMPROJ")

mkdir -p "$(dirname "$LOG")"
setsid /home/typhoon/git/frankenstein-llm/upstream/llama.cpp/build/bin/llama-server "${args[@]}" >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "probe '$ALIAS' started pid=$(cat "$PIDFILE") port=$PORT flash_attn=$FLASH device=$DEVICE log=$LOG"
