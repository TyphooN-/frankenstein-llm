#!/usr/bin/env bash
# Run only after other builds have drained. This builds; it never benchmarks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/build-llama-cpp.sh"
exec cmake --build "$ROOT/upstream/llama.cpp/build" --target llama-bench --parallel "$(nproc)"
