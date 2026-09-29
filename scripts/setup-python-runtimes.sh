#!/usr/bin/env bash
# (Re)create every per-lane Python runtime under venvs/ on a fresh host.
#
#   scripts/setup-python-runtimes.sh [lane ...]      # default: all lanes
#
# Lanes: candidates asr computer-use tts comfy. Every venv reuses the host ROCm
# Torch through --system-site-packages; nothing here may resolve or replace it.
# Prerequisites (pacman): python-pytorch-opt-rocm python-numpy python-pillow
# python-scipy python-soundfile python-jinja python-safetensors python-psutil, uv.
#
# candidates, tts and comfy use their existing lane scripts. asr and
# computer-use sync hash-locked files from verification/python-runtimes/, which
# are compiled with --no-emit-package torch so a CUDA wheel can never land beside
# the AMD runtime. Idempotent; ends by checking each lane's torch is the host's.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANES=("$@")
[[ ${#LANES[@]} -gt 0 ]] || LANES=(candidates asr computer-use tts comfy)

python3 -c 'import torch; assert torch.version.hip, "host torch is not a ROCm build"' \
    || { echo "install python-pytorch-opt-rocm first" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }

locked_lane() {  # <lane>: uv venv + uv pip sync from verification/python-runtimes/<lane>
    local lane=$1 venv="$ROOT/venvs/$1" lock="$ROOT/verification/python-runtimes/$1/requirements.lock"
    [[ -s $lock ]] || { echo "missing $lock" >&2; exit 1; }
    [[ -x $venv/bin/python ]] || uv venv --quiet --python 3.14 --system-site-packages "$venv"
    uv pip sync --quiet --python "$venv/bin/python" "$lock"
}

for lane in "${LANES[@]}"; do
    echo "== $lane"
    case $lane in
        candidates)    bash "$ROOT/verification/candidate-qualification/setup_runtime.sh" ;;
        asr|computer-use) locked_lane "$lane" ;;
        tts)           bash "$ROOT/verification/generative-media/setup_tts.sh" ;;
        comfy)         bash "$ROOT/verification/generative-media/setup_comfy.sh" ;;
        *) echo "unknown lane: $lane" >&2; exit 2 ;;
    esac
done

echo "== torch provenance"
status=0
for lane in "${LANES[@]}"; do
    py="$ROOT/venvs/$lane/bin/python"
    if ! "$py" - "$lane" <<'PY'
import sys, torch
lane = sys.argv[1]
ok = bool(torch.version.hip) and "/venvs/" not in torch.__file__
print(f"{lane:13s} torch {torch.__version__} hip={torch.version.hip} gpus={torch.cuda.device_count()} "
      f"{'host' if '/venvs/' not in torch.__file__ else 'VENV-LOCAL'} {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
    then status=1; fi
done
exit $status
