#!/usr/bin/env bash
# Download Ridge GGUF (text+MTP). Vision mmproj is optional and off by default.
set -euo pipefail

DEST="${LLM_MODELS_DIR:-/home/typhoon/git/frankenstein-llm/models}"
REPO="empero-ai/Qwen3.8-27B-Ridge-GGUF"
FILE="${RIDGE_FILE:-Qwen3.8-27B-Ridge-3.7bpw.gguf}"
BASE="https://huggingface.co/${REPO}/resolve/main"
mkdir -p "$DEST"

download() {
  local name="$1"
  local out="$DEST/$name"
  if [[ -f "$out" ]]; then
    echo "already present: $out"
    return 0
  fi
  echo "downloading $name -> $out"
  curl -fL --retry 5 --retry-all-errors -C - \
    -o "$out.partial" \
    "${BASE}/${name}?download=true"
  mv "$out.partial" "$out"
}

download "$FILE"
if [[ "${RIDGE_MMPROJ:-0}" == "1" ]]; then
  download mmproj-Qwen3.8-27B-BF16.gguf
fi

ls -lh "$DEST/$FILE"
echo "ok"
