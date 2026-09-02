#!/usr/bin/env python3
import json
from pathlib import Path
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "obliterated"
OUT = Path("/home/typhoon/git/frankenstein-llm/verification/qwen38-obliterated-functional")
OUT.mkdir(parents=True, exist_ok=True)


def post(payload: dict, name: str) -> dict:
    request = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def message(result: dict) -> dict:
    return result["choices"][0]["message"]


def main() -> int:
    summary = {}

    basic = post(
        {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": (
                    "Explain why an order-book delta should be applied to candidate state and CRC-validated "
                    "before commit. Reply with exactly three numbered points, no preamble."
                ),
            }],
            "max_tokens": 240,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "basic-coherence",
    )
    basic_text = message(basic).get("content", "").strip()
    basic_lines = [line for line in basic_text.splitlines() if line.strip()]
    summary["basic_coherence"] = {
        "pass": len(basic_lines) == 3 and all(line.startswith(f"{i}.") for i, line in enumerate(basic_lines, 1)),
        "content": basic_text,
        "draft_n": basic.get("timings", {}).get("draft_n"),
        "draft_n_accepted": basic.get("timings", {}).get("draft_n_accepted"),
    }

    schema = {
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "systems": {"type": "array", "items": {"type": "string"}},
            "confirmed": {"type": "boolean"},
        },
        "required": ["severity", "systems", "confirmed"],
        "additionalProperties": False,
    }
    structured = post(
        {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": "Extract this incident: Confirmed high-severity failure affected api-1 and db-2.",
            }],
            "max_tokens": 160,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "incident", "strict": True, "schema": schema},
            },
        },
        "structured-json",
    )
    structured_text = message(structured).get("content", "").strip()
    try:
        parsed = json.loads(structured_text)
        structured_ok = (
            parsed == {"severity": "high", "systems": ["api-1", "db-2"], "confirmed": True}
            or parsed == {"severity": "high", "systems": ["db-2", "api-1"], "confirmed": True}
        )
    except Exception:
        parsed = None
        structured_ok = False
    summary["structured_json"] = {
        "pass": structured_ok,
        "content": structured_text,
        "parsed": parsed,
        "draft_n": structured.get("timings", {}).get("draft_n"),
        "draft_n_accepted": structured.get("timings", {}).get("draft_n_accepted"),
    }

    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_ticket",
            "description": "Look up one support ticket by its integer ticket ID.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "integer"}},
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    }, {
        "type": "function",
        "function": {
            "name": "lookup_service",
            "description": "Look up service status by service name.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
        },
    }]
    tool_result = post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Look up ticket 4172."}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 160,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "tool-call",
    )
    calls = message(tool_result).get("tool_calls") or []
    tool_ok = False
    if len(calls) == 1:
        function = calls[0].get("function", {})
        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except Exception:
            arguments = None
        tool_ok = function.get("name") == "lookup_ticket" and arguments == {"ticket_id": 4172}
    summary["tool_call"] = {
        "pass": tool_ok,
        "tool_calls": calls,
        "draft_n": tool_result.get("timings", {}).get("draft_n"),
        "draft_n_accepted": tool_result.get("timings", {}).get("draft_n_accepted"),
    }

    summary["all_pass"] = all(item.get("pass", False) for item in summary.values())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
