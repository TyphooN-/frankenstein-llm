#!/usr/bin/env python3
"""Functional router admission for local chat presets; deliberately no benchmarks."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/candidate-qualification")
sys.path.insert(0, "/home/typhoon/git/frankenstein-llm/verification/mission-supervisor")
from candidate_policy import tool_grant_allowed_for_preset  # noqa: E402
from run_functional_mission import (  # noqa: E402
    MISSION_BLOCKING_REASONS,
    conflict_reason,
    process_metadata,
)

BASE = "http://127.0.0.1:8080"
CHAT_MODELS = ["ridge", "heretic", "obliterated", "fable", "phr00ty",
               "qwen3-coder-next", "gemma4-heretic"]
VISION_MODELS = ["obliterated-vision", "gemma4-heretic-vision"]
VISION_FIXTURE = Path("/home/typhoon/git/frankenstein-llm/verification/computer-use-grounding/fixtures/screen-app.png")

# Not every preset is judged on the same contract. A preset the candidate policy
# refuses tools to must not be sent a tools payload at all: demanding a tool call
# from a reader either fails a model that is behaving correctly, or normalises
# handing executable functions to weights that policy says never receive them.
# The set is derived from the policy rather than restated, so a privilege change
# cannot leave a stale check list behind.
CORE_CHECKS = ("coherence", "structured_output", "tool_call")
READER_CHECKS = ("coherence", "structured_output")
OUT = Path(__file__).resolve().parent / "evidence"
ARTIFACT = OUT / "router-functional.json"
VRAM_TOLERANCE = 768 << 20
RAM_TOLERANCE = 2 << 30
# Functional checks are not throughput claims. cargo/rustc/makepkg outside the
# kernel tree and desktop load do not confound a PONG / tool-call / unload
# verdict. Refuse only what the mission itself waits out, plus an in-flight
# download queue that can replace the weights under the router.
DOWNLOAD_UNIT_PREFIX = "local-ai-model-downloads"


def checks_for(model: str) -> tuple[str, ...]:
    """The checks a preset must satisfy, and therefore whether it is offered tools."""
    if not tool_grant_allowed_for_preset(model):
        return READER_CHECKS
    return CORE_CHECKS


def http_json(path: str, payload: dict | None = None, timeout: int = 900) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.strip().split()[0]) * 1024
    return values


def vram_used() -> dict[str, int]:
    values: dict[str, int] = {}
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        path = card / "device" / "mem_info_vram_used"
        try:
            values[card.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return values


def blocked_workloads(proc_root: Path = Path("/proc")) -> list[dict[str, object]]:
    """Find workloads that would confound a functional router qualification.

    Reads only kernel metadata (``comm``, ``cwd``, ``cgroup``). Never opens
    ``cmdline``. Classification is the mission supervisor's ``conflict_reason``;
    this gate then refuses the same classes the mission waits for, plus an
    in-flight download queue.
    """
    blocked: list[dict[str, object]] = []
    self_pid = os.getpid()
    pids = sorted(int(entry.name) for entry in proc_root.glob("[0-9]*")
                  if entry.name.isdigit())
    for pid in pids:
        if pid == self_pid:
            continue
        command, cwd, cgroup = process_metadata(proc_root / str(pid))
        if not command:
            continue
        reason = conflict_reason(command, cwd, cgroup)
        if reason in MISSION_BLOCKING_REASONS:
            pass
        elif DOWNLOAD_UNIT_PREFIX in cgroup:
            reason = "download-queue"
        else:
            continue
        blocked.append({"pid": pid, "reason": reason, "command": command[:500],
                        "cwd": cwd[:300]})
    return blocked


def pressure() -> dict[str, str]:
    values = {}
    for kind in ("memory", "io"):
        try:
            values[kind] = Path(f"/proc/pressure/{kind}").read_text().strip()
        except OSError:
            values[kind] = "unavailable"
    return values


def sample(label: str) -> dict:
    mem = meminfo()
    return {
        "label": label,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mem_available_bytes": mem.get("MemAvailable", -1),
        "swap_used_bytes": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
        "vram_used_bytes": vram_used(),
        "pressure": pressure(),
    }


def model_states() -> dict[str, str]:
    data = http_json("/models").get("data", [])
    return {item["id"]: item.get("status", {}).get("value", "unknown") for item in data}


def model_metadata(model: str) -> dict:
    """Return one model from the router's supported collection endpoint."""
    for item in http_json("/models").get("data", []):
        if item.get("id") == model:
            return item
    raise KeyError(f"router did not list model {model!r}")


def wait_state(model: str, wanted: set[str], timeout: int = 300) -> str:
    deadline = time.monotonic() + timeout
    last = "missing"
    while time.monotonic() < deadline:
        last = model_states().get(model, "missing")
        if last in wanted:
            return last
        time.sleep(1)
    raise TimeoutError(f"{model} remained {last}; wanted {sorted(wanted)}")


def chat(model: str, prompt: str | list[dict], *, schema: dict | None = None, tools: list | None = None) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 160,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    response = http_json("/v1/chat/completions", payload)
    # Never persist timing/token-rate fields: this gate is functional only.
    return {"message": response["choices"][0]["message"]}


def unload(model: str) -> None:
    try:
        http_json("/models/unload", {"model": model}, timeout=300)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        if "not running" not in body:
            raise
    wait_state(model, {"unloaded"}, timeout=300)


def record_release(result: dict, before: dict) -> None:
    """Compare post-unload host state against the pre-load baseline.

    ``vram_used`` omits a card whose sysfs node did not answer. Substituting the
    baseline for a missing reading gave a zero delta, so a card that stopped
    reporting scored exactly like a card that released its memory -- the eviction
    check passed with no evidence behind it. Missing readings, and a baseline
    that saw no card at all, are recorded as problems instead.
    """
    after = sample("after_unload")
    result["after_unload"] = after
    problems = result["problems"]
    if not before["vram_used_bytes"]:
        problems.append("no GPU reported VRAM before the load; eviction is unproven")
    for card, base in before["vram_used_bytes"].items():
        if card not in after["vram_used_bytes"]:
            problems.append(f"{card} stopped reporting VRAM after unload; eviction is unproven")
            continue
        delta = after["vram_used_bytes"][card] - base
        if delta > VRAM_TOLERANCE:
            problems.append(f"{card} retained {delta} bytes after unload")
    ram_delta = before["mem_available_bytes"] - after["mem_available_bytes"]
    if ram_delta > RAM_TOLERANCE:
        problems.append(f"available RAM remained {ram_delta} bytes below baseline")
    swap_growth = after["swap_used_bytes"] - before["swap_used_bytes"]
    if swap_growth > RAM_TOLERANCE:
        problems.append(f"swap grew by {swap_growth} bytes")


def check_model(model: str) -> dict:
    required = checks_for(model)
    before = sample("before")
    result: dict = {"model": model, "before": before, "checks": {}, "problems": [],
                    "required_checks": list(required),
                    "tools_offered": "tool_call" in required}
    try:
        conflicts = blocked_workloads()
        if conflicts:
            raise RuntimeError(f"refusing confounded qualification; active workloads: {conflicts}")
        pong = chat(model, "Reply with exactly the single uppercase word PONG.")
        text = (pong["message"].get("content") or "").strip()
        result["checks"]["coherence"] = {"pass": text == "PONG", "content": text}

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ready", "blocked"]},
                "count": {"type": "integer"},
            },
            "required": ["status", "count"],
            "additionalProperties": False,
        }
        structured = chat(model, "Return status ready and count 3.", schema=schema)
        raw = (structured["message"].get("content") or "").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        result["checks"]["structured_output"] = {
            "pass": parsed == {"status": "ready", "count": 3},
            "content": raw,
            "parsed": parsed,
        }

        if "tool_call" in required:
            tools = [{
                "type": "function",
                "function": {
                    "name": "lookup_ticket",
                    "description": "Look up one ticket by integer ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticket_id": {"type": "integer"}},
                        "required": ["ticket_id"],
                        "additionalProperties": False,
                    },
                },
            }]
            tool = chat(model, "Look up ticket 4172. Do not answer without using the tool.", tools=tools)
            calls = tool["message"].get("tool_calls") or []
            valid_call = False
            if len(calls) == 1:
                fn = calls[0].get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = None
                valid_call = fn.get("name") == "lookup_ticket" and args == {"ticket_id": 4172}
            result["checks"]["tool_call"] = {"pass": valid_call, "tool_calls": calls}
        else:
            result["tool_policy"] = (
                "low-privilege candidate: no tools payload was sent, so no tool-call "
                "capability is claimed for this preset")
        result["metadata"] = model_metadata(model)
        result["loaded"] = sample("loaded")
    except Exception as error:  # noqa: BLE001 - every model must leave evidence
        result["problems"].append(f"functional error: {type(error).__name__}: {error}"[:500])
    finally:
        try:
            unload(model)
        except Exception as error:  # noqa: BLE001
            result["problems"].append(f"unload error: {type(error).__name__}: {error}"[:500])
        record_release(result, before)
    # A check that never ran is not a check that passed: the required set has to
    # match exactly, so a section skipped by an early exception fails the model.
    result["pass"] = not result["problems"] and all(
        check.get("pass") is True for check in result["checks"].values()
    ) and set(result["checks"]) == set(required)
    return result


def check_vision_model(model: str) -> dict:
    before = sample("before")
    result: dict = {"model": model, "before": before, "checks": {}, "problems": []}
    try:
        conflicts = blocked_workloads()
        if conflicts:
            raise RuntimeError(f"refusing confounded qualification; active workloads: {conflicts}")
        encoded = base64.b64encode(VISION_FIXTURE.read_bytes()).decode()
        response = chat(
            model,
            [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                {
                    "type": "text",
                    "text": "What is the exact label of the prominent blue button? Reply with exactly that label.",
                },
            ],
        )
        text = (response["message"].get("content") or "").strip()
        result["checks"]["vision_grounding"] = {"pass": text == "Run Gate", "content": text}
        result["metadata"] = model_metadata(model)
        result["loaded"] = sample("loaded")
    except Exception as error:  # noqa: BLE001 - every model must leave evidence
        result["problems"].append(f"functional error: {type(error).__name__}: {error}"[:500])
    finally:
        try:
            unload(model)
        except Exception as error:  # noqa: BLE001
            result["problems"].append(f"unload error: {type(error).__name__}: {error}"[:500])
        record_release(result, before)
    result["pass"] = (
        not result["problems"]
        and result["checks"].get("vision_grounding", {}).get("pass") is True
    )
    return result


def write_atomic(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    temp = ARTIFACT.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, ARTIFACT)


def main() -> int:
    summary = {
        "gate": "router-functional",
        "benchmarking_performed": False,
        "throughput_measured": False,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "kernel_release": os.uname().release,
        "kernel_build_signature": Path("/proc/version").read_text().strip(),
        "kernel_command_line": Path("/proc/cmdline").read_text().strip(),
        "models": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for model in CHAT_MODELS:
        item = check_model(model)
        summary["models"].append(item)
        write_atomic(summary)
        print(f"{model}: {'PASS' if item['pass'] else 'FAIL'}", flush=True)
    for model in VISION_MODELS:
        vision_item = check_vision_model(model)
        summary["models"].append(vision_item)
        write_atomic(summary)
        print(f"{model}: {'PASS' if vision_item['pass'] else 'FAIL'}", flush=True)
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["pass"] = all(item["pass"] for item in summary["models"])
    write_atomic(summary)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
