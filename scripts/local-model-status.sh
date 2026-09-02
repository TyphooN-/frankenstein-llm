#!/usr/bin/env bash
set -euo pipefail
base=http://127.0.0.1:8080
printf 'Health: '
curl -fsS "$base/health"
printf '\nAvailable models:\n'
curl -fsS "$base/v1/models" | python3 -m json.tool
printf '\nRouter status:\n'
curl -fsS "$base/models" | python3 -m json.tool
