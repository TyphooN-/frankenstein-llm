#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 0 || ( $# -eq 1 && "$1" == --start ) ]]; then
    systemctl --user start --no-block local-ai-qualification.service
    printf '%s\n' 'Qualification start queued; admission and gate success are not yet verified.' \
        'Watch: ./scripts/qualification-status.sh --watch 5'
    exit 0
fi
if [[ "$1" == --plan ]]; then
    shift
    exec python3 "$ROOT/verification/qualification-supervisor/run_qualification.py" --plan "$@"
fi
if [[ "$1" == --help || "$1" == -h ]]; then
    printf '%s\n' 'Usage: qualify-models.sh [--start | --plan | --execute [options]]' \
        'No arguments / --start: queue the installed qualification service in the background.' \
        '--plan: inspect receipt eligibility without starting qualification.' \
        '--execute: advanced foreground runner; do not overlap the service.' \
        'Foreground options: python3 scripts/model-runs.py qualify --help'
    exit 0
fi
exec python3 "$ROOT/scripts/model-runs.py" qualify "$@"
