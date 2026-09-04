#!/usr/bin/env bash
set -euo pipefail
exec python3 /home/typhoon/git/frankenstein-llm/scripts/local_model_status.py "$@"
