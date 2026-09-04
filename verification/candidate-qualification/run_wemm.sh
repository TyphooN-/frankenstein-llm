#!/usr/bin/env bash
# Run the WeMM text gate through the isolated ROCm candidate runtime.
set -euo pipefail
ROOT=/home/typhoon/git/frankenstein-llm
PY=$ROOT/venvs/candidates/bin/python
if [ ! -x "$PY" ]; then
  printf 'missing candidate runtime at %s; run setup_runtime.sh\n' "$PY" >&2
  exit 1
fi
exec "$PY" "$ROOT/verification/candidate-qualification/gate_wemm.py"
