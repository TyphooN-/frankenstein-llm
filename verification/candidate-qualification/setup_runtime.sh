#!/usr/bin/env bash
# Recreate the isolated Qwen3.5 candidate runtime without replacing host ROCm Torch.
set -euo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
DIR=$ROOT/verification/candidate-qualification
VENV=$ROOT/venvs/candidates

command -v uv >/dev/null || { printf 'uv is required\n' >&2; exit 1; }
[ -s "$DIR/requirements.lock" ] || { printf 'missing requirements.lock\n' >&2; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --system-site-packages "$VENV"
fi
uv pip sync --python "$VENV/bin/python" "$DIR/requirements.lock"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
expected = {
    "transformers": "5.5.4",
    "accelerate": "1.14.0",
    "qwen-vl-utils": "0.0.14",
    "pillow": "12.3.0",
    "safetensors": "0.8.0",
}
for package, version in expected.items():
    observed = metadata.version(package)
    if observed != version:
        raise SystemExit(f"{package}: expected {version}, got {observed}")
if not torch.version.hip:
    raise SystemExit("host torch is not a ROCm build")
print(f"candidate runtime ready: transformers={transformers.__version__} torch={torch.__version__} hip={torch.version.hip}")
PY
