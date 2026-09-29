#!/usr/bin/env bash
# Local Qwen3-TTS runtime. Reuses the system ROCm torch; never upgrades it.
set -uo pipefail
VENV=/home/typhoon/git/frankenstein-llm/venvs/tts
LOG=/home/typhoon/git/frankenstein-llm/proofs/logs/tts-setup.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== tts setup start $(date -Is) ==="
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || exit 11
"$VENV/bin/python" -m pip install --quiet --upgrade pip || true
# --no-deps on torch-adjacent packages would break qwen-tts, so install normally but
# pin torch out of the resolver's reach: the system ROCm build must win.
"$VENV/bin/python" -m pip install qwen-tts soundfile librosa 2>&1 | tail -20
rc=$?
# qwen-tts pulls torchaudio from PyPI, whose newest release (2.11, frozen upstream)
# ships C++ ops compiled against torch 2.11; against the host torch 2.13 they fail
# to load and `import torchaudio` raises. qwen_tts only uses the pure-Python
# torchaudio.compliance.kaldi, and torchaudio runs without its extension when the
# library files are absent (only forced alignment needs them), so drop them.
if ! "$VENV/bin/python" -c 'import torchaudio' 2>/dev/null; then
  TA_LIB=$("$VENV/bin/python" -c 'import importlib.util as u, pathlib; print(pathlib.Path(u.find_spec("torchaudio").origin).parent / "lib")')
  echo "--- torchaudio C++ ops incompatible with host torch; removing $TA_LIB/*.so ---"
  rm -f "$TA_LIB"/*.so
fi
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
