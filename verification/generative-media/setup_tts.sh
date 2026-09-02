#!/usr/bin/env bash
# Local Qwen3-TTS runtime. Reuses the system ROCm torch; never upgrades it.
set -uo pipefail
VENV=/home/typhoon/git/frankenstein-llm/venvs/tts
LOG=/tmp/hermes-verify-coverage-continuation-20260901/logs/tts-setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== tts setup start $(date -Is) ==="
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || exit 11
"$VENV/bin/python" -m pip install --quiet --upgrade pip || true
# --no-deps on torch-adjacent packages would break qwen-tts, so install normally but
# pin torch out of the resolver's reach: the system ROCm build must win.
"$VENV/bin/python" -m pip install qwen-tts soundfile librosa 2>&1 | tail -20
rc=$?
echo "--- verifying system ROCm torch survived ---"
"$VENV/bin/python" - <<'PY'
import torch
print('torch', torch.__version__, 'hip', getattr(torch.version,'hip',None),
      'path', torch.__file__)
print('devices', torch.cuda.device_count())
try:
    from qwen_tts import Qwen3TTSModel
    print('qwen_tts import OK')
except Exception as e:
    print('qwen_tts import FAILED:', type(e).__name__, e)
PY
echo "=== tts setup end rc=$rc $(date -Is) ==="
