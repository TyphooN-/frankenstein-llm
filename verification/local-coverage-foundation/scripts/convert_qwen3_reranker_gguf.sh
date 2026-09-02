#!/usr/bin/env bash
# Convert the publisher Qwen3-Reranker-8B safetensors snapshot into a llama.cpp
# reranker GGUF, then quantize it.
#
# Why convert instead of downloading a community GGUF: llama.cpp's own
# convert_hf_to_gguf.py recognises Qwen3-Reranker, preserves the cls.output.weight
# classifier head and writes pooling_type=RANK. Conversions that miss the head
# still load and still answer /v1/rerank -- with meaningless, near-constant
# scores. Converting from publisher weights whose SHA-256 we already verified
# keeps the provenance chain intact end to end.
#
# CPU only; no GPU is touched. Needs a Python venv, so this is a post-reboot
# execution step rather than a foundation step.
set -euo pipefail

SRC="${RERANK_SRC:-/home/typhoon/git/frankenstein-llm/models/reranker-src/Qwen3-Reranker-8B}"
OUT_DIR="${RERANK_OUT:-/home/typhoon/git/frankenstein-llm/models/reranker}"
LLAMA_SRC="${LLAMA_SRC:-/home/typhoon/src/llama.cpp}"
VENV="${RERANK_VENV:-/home/typhoon/git/frankenstein-llm/venvs/convert}"
QUANT="${RERANK_QUANT:-Q6_K}"
F16="$OUT_DIR/Qwen3-Reranker-8B-f16.gguf"
FINAL="$OUT_DIR/Qwen3-Reranker-8B-${QUANT}.gguf"

[ -d "$SRC" ] || { echo "missing source snapshot: $SRC" >&2; exit 2; }
[ -f "$LLAMA_SRC/convert_hf_to_gguf.py" ] || { echo "missing converter in $LLAMA_SRC" >&2; exit 2; }
mkdir -p "$OUT_DIR"

if [ ! -x "$VENV/bin/python" ]; then
    echo "creating conversion venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    # CPU torch is enough: the converter only reads safetensors shards.
    "$VENV/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install --quiet -r "$LLAMA_SRC/requirements/requirements-convert_hf_to_gguf.txt"
fi

if [ ! -f "$F16" ]; then
    echo "converting $SRC -> $F16"
    "$VENV/bin/python" "$LLAMA_SRC/convert_hf_to_gguf.py" "$SRC" --outfile "$F16.partial" --outtype f16
    mv -f "$F16.partial" "$F16"
fi

if [ ! -f "$FINAL" ]; then
    echo "quantizing -> $FINAL"
    /home/typhoon/.local/bin/llama-quantize "$F16" "$FINAL.partial" "$QUANT"
    mv -f "$FINAL.partial" "$FINAL"
fi

echo "--- structural check of the produced GGUF ---"
python3 /home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/scripts/inspect_gguf.py "$FINAL"

echo "conversion complete: $FINAL"
echo "next: systemctl --user start llama-sidecar@reranker.service"
echo "then: python3 /home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/validators/gate_reranker.py"
