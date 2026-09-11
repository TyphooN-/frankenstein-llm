#!/usr/bin/env bash
# Local ComfyUI runtime for the image/music generative-media gates.
# Local tool install only: no service is deployed, published, or exposed off loopback.
set -uo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
TOOLS=$ROOT/tools
VENV=$ROOT/venvs/comfy
LOG=$ROOT/proofs/logs/comfy-setup.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date -Is) ==="

if [ ! -d "$TOOLS/ComfyUI/.git" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$TOOLS/ComfyUI" || exit 10
fi
cd "$TOOLS/ComfyUI" && echo "comfy commit: $(git rev-parse HEAD)"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV" || exit 11
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip || true

# Install ComfyUI deps EXCEPT torch (system ROCm torch 2.13.0 is reused via --system-site-packages).
grep -viE '^(torch|torchvision|torchaudio)([=<>~!].*)?$' requirements.txt > /tmp/comfy-reqs.txt
echo "--- requirements (torch excluded) ---"; cat /tmp/comfy-reqs.txt
"$VENV/bin/python" -m pip install -r /tmp/comfy-reqs.txt
rc=$?
echo "pip rc=$rc"
echo "--- import check ---"
"$VENV/bin/python" - <<'PY'
import importlib.util as u
mods=['torch','torchsde','einops','transformers','safetensors','aiohttp','yaml','PIL','scipy','tqdm','psutil','kornia','spandrel','av','pydantic','alembic','sqlalchemy','soundfile']
miss=[m for m in mods if not u.find_spec(m)]
print('missing:',miss)
import torch
print('torch',torch.__version__,'hip',torch.version.hip,'devices',torch.cuda.device_count())
PY
echo "=== setup end rc=$rc $(date -Is) ==="
