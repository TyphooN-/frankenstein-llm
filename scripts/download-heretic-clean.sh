#!/usr/bin/env bash
set -euo pipefail
URL='https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF/resolve/main/RVN-Q6_K-multilingual-mtp.gguf'
DIR=/home/typhoon/git/frankenstein-llm/models
FINAL="$DIR/RVN-Q6_K-multilingual-mtp.gguf"
TMP="$FINAL.fresh"
SHA='1344d07425d73f0d1b8f36213910eae8fec21061b1a90835927e8485238f93a4'
curl -fL --retry 8 --retry-all-errors --retry-delay 3 -C - --output "$TMP" "$URL"
printf '%s  %s\n' "$SHA" "$TMP" | sha256sum -c -
mv -f "$TMP" "$FINAL"
printf 'Installed verified model: %s\n' "$FINAL"
