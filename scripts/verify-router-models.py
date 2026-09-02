#!/usr/bin/env python3
import json
import pathlib
import sys
import urllib.request

BASE = "http://127.0.0.1:8080"
OUT = pathlib.Path("/home/typhoon/git/frankenstein-llm/verification")
OUT.mkdir(parents=True, exist_ok=True)
models = sys.argv[1:] or ["ridge", "obliterated", "heretic"]
for model in models:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly the single word pong."}],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=360) as r:
        response = json.load(r)
    (OUT / f"direct-{model}.json").write_text(json.dumps(response, indent=2) + "\n")
    content = response["choices"][0]["message"].get("content", "").strip()
    if content != "pong":
        raise SystemExit(f"{model}: unexpected content {content!r}")
    print(f"{model}: direct API OK ({content})", flush=True)
with urllib.request.urlopen(BASE + "/models", timeout=30) as r:
    state = json.load(r)
(OUT / "router-state-after-direct.json").write_text(json.dumps(state, indent=2) + "\n")
print("direct API suite complete")
