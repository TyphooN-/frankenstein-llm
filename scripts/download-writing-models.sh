#!/usr/bin/env bash
set -euo pipefail

DIR=/home/typhoon/git/frankenstein-llm/models
mkdir -p "$DIR"

fetch() {
  local repo=$1 file=$2 sha=$3
  local final="$DIR/$file"
  local partial="$final.partial"
  exec 9>"$partial.lock"
  flock 9
  if [[ -f "$final" ]]; then
    printf '%s  %s\n' "$sha" "$final" | sha256sum -c -
    return
  fi
  curl -fL --retry 8 --retry-all-errors --retry-delay 3 -C - \
    --output "$partial" \
    "https://huggingface.co/$repo/resolve/main/$file"
  printf '%s  %s\n' "$sha" "$partial" | sha256sum -c -
  mv -f "$partial" "$final"
  printf 'Installed verified model: %s\n' "$final"
}

download_fable() {
  fetch \
    'DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF' \
    'Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-Q6_K.gguf' \
    'c394c7c0dcbd32e04f4695ad5d4dad8f2f5b7286b366d165815ac0cdc5ccf4aa'
}

download_phr00ty() {
  fetch \
    'Phr00t/Phr00tyMix-v4-32B-GGUF' \
    'Phr00tyMix-v4-32B-imat-Q6_K.gguf' \
    '056fc85cb1f0873e90ddae4f6f747fbc4ce3ebf41fad15bfdc2a7e66e2af1700'
}

case "${1:-all}" in
  fable) download_fable ;;
  phr00ty) download_phr00ty ;;
  all) download_fable; download_phr00ty ;;
  *) printf 'usage: %s [all|fable|phr00ty]\n' "$0" >&2; exit 2 ;;
esac
