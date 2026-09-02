#!/usr/bin/env bash
# Ridge 27B on all three ROCm devices with native MTP.
# Layer split + tensor proportions 1,2,1 == 16GB : 32GB : 16GB.
set -euo pipefail

MODEL="${RIDGE_MODEL:-/home/typhoon/git/frankenstein-llm/models/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8080}"
CTX="${LLAMA_CTX:-131072}"
NMAX="${SPEC_DRAFT_N_MAX:-2}"
BIN="${LLAMA_SERVER_BIN:-/home/typhoon/.local/bin/llama-server}"
LOG="${LLAMA_LOG:-/home/typhoon/git/frankenstein-llm/logs/ridge-server.log}"

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  echo "run /home/typhoon/git/frankenstein-llm/scripts/download-ridge.sh first" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"

exec "$BIN" \
  -m "$MODEL" \
  --alias Qwen3.8-27B-Ridge \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CTX" \
  -ngl 999 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max "$NMAX" \
  --parallel 1 \
  --device ROCm0,ROCm1,ROCm2 \
  --tensor-split 1,2,1 \
  --jinja \
  --no-mmap \
  "$@"
