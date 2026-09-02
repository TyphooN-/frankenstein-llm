#!/usr/bin/env python3
"""Gate: the local stack must emit native tool calls, not XML inside content.

Hermes and every downstream agent read message.tool_calls[]. A model that writes
<tool_call>{...}</tool_call> into message.content technically "called a tool" and
is still unusable, because the transport never parses it. This gate fails that
case explicitly, and also checks the second half of the loop: after a tool result
is appended, the model must produce a final answer instead of calling again.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/validators")
from gatelib import GateFailure, check, post_json, record  # noqa: E402

# Defaults to the production router; override to test a standalone probe
# instance without changing router configuration.
BASE = os.environ.get("HERMES_CHAT_URL", "http://127.0.0.1:8080")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_pool_status",
            "description": "Return the health of one ZFS pool by name.",
            "parameters": {
                "type": "object",
                "properties": {"pool": {"type": "string"}},
                "required": ["pool"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_loaded_models",
            "description": "List models currently resident in the local router.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

XML_SHAPED = re.compile(r"<\s*(tool_call|function_call|tool)\b", re.I)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="heretic")
    args = parser.parse_args()

    summary = {"gate": "native-tool-use", "model": args.model, "base_url": BASE}
    try:
        first = post_json(f"{BASE}/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Check the health of the zroot pool."}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        message = first["choices"][0]["message"]
        content = message.get("content") or ""
        calls = message.get("tool_calls") or []
        summary["first_turn"] = {
            "finish_reason": first["choices"][0].get("finish_reason"),
            "tool_calls": calls,
            "content": content[:400],
        }

        check(
            not XML_SHAPED.search(content),
            "model emitted an XML-shaped tool call inside message.content; Hermes will not parse it",
        )
        check(bool(calls), "message.tool_calls was empty; no native tool call was produced")
        check(len(calls) == 1, f"expected exactly one tool call, got {len(calls)}")

        function = calls[0].get("function", {})
        check(function.get("name") == "get_pool_status", f"wrong tool selected: {function.get('name')}")
        arguments = json.loads(function.get("arguments") or "{}")
        summary["arguments"] = arguments
        check(arguments.get("pool") == "zroot", f"argument extraction wrong: {arguments}")
        check(bool(calls[0].get("id")), "tool call has no id; the result cannot be correlated")

        second = post_json(f"{BASE}/v1/chat/completions", {
            "model": args.model,
            "messages": [
                {"role": "user", "content": "Check the health of the zroot pool."},
                message,
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "name": "get_pool_status",
                    "content": json.dumps({"pool": "zroot", "state": "ONLINE", "errors": 1}),
                },
            ],
            "tools": TOOLS,
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        final = second["choices"][0]["message"]
        final_text = final.get("content") or ""
        summary["second_turn"] = {"content": final_text[:400], "tool_calls": final.get("tool_calls")}
        check(not (final.get("tool_calls") or []), "model re-called the tool instead of answering from the result")
        check(
            "online" in final_text.lower() or "1" in final_text,
            "final answer did not incorporate the tool result",
        )
        summary["pass"] = True
    except (GateFailure, Exception) as error:  # noqa: BLE001
        summary["pass"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
    return record("gate-native-tool-use", summary)


if __name__ == "__main__":
    raise SystemExit(main())
