#!/usr/bin/env bash
# Start a STANDALONE llama-server for the vision/grounding probe.
#
# This deliberately does not use llama-models.ini, llama-router.service, or port
# 8080. The production router keeps its own config and binary untouched; this is
# a throwaway process on 127.0.0.1:8090 that the gate stops when it finishes.
#
# Requires the router to hold no resident model, because the probe wants the same
# GPUs. Check with: curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[].status.value'
set -euo pipefail

PORT="${VISION_PROBE_PORT:-8090}"
MODEL="${VISION_PROBE_MODEL:-/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-Q6_K.gguf}"
MMPROJ="${VISION_PROBE_MMPROJ:-/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-OBLITERATED-mmproj-bf16.gguf}"
PIDFILE="${VISION_PROBE_PIDFILE:-/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/vision-probe.pid}"
LOG="${VISION_PROBE_LOG:-/home/typhoon/git/frankenstein-llm/logs/vision-probe.log}"

for path in "$MODEL" "$MMPROJ"; do
    [ -f "$path" ] || { echo "missing artifact: $path" >&2; exit 2; }
done

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "vision probe already running as PID $(cat "$PIDFILE")" >&2
    exit 3
fi

mkdir -p "$(dirname "$LOG")"
setsid /home/typhoon/.local/bin/llama-server \
    --host 127.0.0.1 --port "$PORT" \
    --alias vision-probe \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --device ROCm0,ROCm1,ROCm2 \
    --tensor-split 1,2,1 \
    --n-gpu-layers all \
    --ctx-size 16384 \
    --cache-type-k q8_0 --cache-type-v f16 \
    --flash-attn off \
    --parallel 1 \
    --no-mmap \
    --jinja \
    --temp 0 \
    >>"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "vision probe started pid=$(cat "$PIDFILE") port=$PORT log=$LOG"
