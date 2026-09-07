#!/usr/bin/env bash
# Publishes markdown from already-recorded native benchmark JSON. Never runs a
# model, never touches a GPU, and never starts or stops a service.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/bench_report.py" write "$@"
