#!/usr/bin/env bash
# Run one hand-started download queue as a transient user unit.
#
#   scripts/start-download-queue.sh <name>
#
# <name> selects verification/local-coverage-foundation/download-queue-<name>.json
# and the matching state/lock/stamp/log files. The unit is download-<name>.service;
# follow it with `journalctl --user -u download-<name> -f`. WORKERS and CONN
# override the per-queue file workers and aria2 connection budget.
#
# For queues outside local-ai-model-downloads.service only (slot, restore and
# candidate queues). The downloader resumes from .partial files, so re-running
# after an interruption or reboot is safe.
set -euo pipefail

[[ $# -eq 1 && $1 =~ ^[a-z0-9.-]+$ ]] || { echo "usage: $0 <queue-name>" >&2; exit 2; }
N=$1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
F="$ROOT/verification/local-coverage-foundation"
[[ -f "$F/download-queue-$N.json" ]] || { echo "no queue: $F/download-queue-$N.json" >&2; exit 66; }

systemd-run --user --unit="download-$N" --collect -p Nice=10 -p IOSchedulingClass=best-effort \
    -E PYTHONUNBUFFERED=1 \
    -E HERMES_DOWNLOAD_WORKERS="${WORKERS:-4}" \
    -E HERMES_DOWNLOAD_CONNECTION_BUDGET="${CONN:-16}" \
    -E HERMES_DOWNLOAD_QUEUE="$F/download-queue-$N.json" \
    -E HERMES_DOWNLOAD_STATE="$F/download-state-$N.json" \
    -E HERMES_DOWNLOAD_LOCK="$F/download-queue-$N.lock" \
    -E HERMES_DOWNLOAD_STAMP="$F/downloads-$N-complete.ok" \
    -E HERMES_DOWNLOAD_LOG="$F/downloads-$N.log" \
    --working-directory="$ROOT" \
    python3 "$F/download_queue.py"
