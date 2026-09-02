#!/usr/bin/env bash
set -euo pipefail
OUT=/home/typhoon/git/frankenstein-llm/verification
mkdir -p "$OUT"
for model in "$@"; do
  hermes -z 'Reply with exactly the single word pong.' \
    -m "$model" --reasoning none --ignore-rules \
    --usage-file "$OUT/hermes-$model-usage.json" \
    >"$OUT/hermes-$model-output.txt"
  test -s "$OUT/hermes-$model-output.txt"
  python3 - "$OUT/hermes-$model-usage.json" "$model" <<'PY'
import json, sys
p, expected = sys.argv[1:]
d = json.load(open(p))
assert d.get('completed') is True and d.get('failed') is False, d
assert d.get('model') == expected, d
assert d.get('provider') == 'custom', d
assert d.get('api_calls') == 1, d
assert d.get('output_tokens', 0) > 0, d
print(f"{expected}: Hermes alias transport OK (one completed API call)", flush=True)
PY
done
