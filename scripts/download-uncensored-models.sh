#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=/home/typhoon/git/frankenstein-llm/models
mkdir -p "$MODEL_DIR"

fetch() {
  local url=$1 final=$2 sha=$3
  local partial="${final}.partial"
  if [[ -f "$final" ]]; then
    printf 'already present: %s\n' "$final"
  else
    curl -fL --retry 8 --retry-all-errors --retry-delay 3 -C - \
      --output "$partial" "$url"
    printf '%s  %s\n' "$sha" "$partial" | sha256sum -c -
    mv "$partial" "$final"
  fi
  printf '%s  %s\n' "$sha" "$final" | sha256sum -c -
}

fetch \
  'https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/resolve/main/Qwen3.8-27B-OBLITERATED-Q6_K.gguf' \
  "$MODEL_DIR/Qwen3.8-27B-OBLITERATED-Q6_K.gguf" \
  '3535d4a15b75840fb391138ce04e6c73fde709320c6f5fe788cdbd586ee08e3a'

fetch \
  'https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/resolve/main/RVN-Q6_K-multilingual-mtp.gguf' \
  "$MODEL_DIR/RVN-Q6_K-multilingual-mtp.gguf" \
  '1344d07425d73f0d1b8f36213910eae8fec21061b1a90835927e8485238f93a4'
