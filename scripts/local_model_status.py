#!/usr/bin/env python3
"""Bounded, loopback-only status summary for the local llama.cpp router."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from model_catalog import ALIAS_LABEL, describe_alias, local_registry
import socket
import sys
from typing import Any
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 5.0
MAX_TIMEOUT = 30.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

# llama.cpp v0.4.0 reports one of six model states (tools/server/server-models.h):
# downloading, downloaded, unloaded, loading, loaded, sleeping. Only a state that
# still owns a child process occupies the router's single resident slot, which is
# upstream's own is_running(): loaded, loading, or sleeping. "sleeping" is a live
# child that idled, so it is resident; "downloaded" only means weights reached
# local disk. Classifying by "anything except unloaded" -- correct against the
# three-state runtime this stack ran before v0.4.0 -- therefore reports a cached
# download as a resident model. Enumerate the vocabulary instead, and fail closed
# on a state this pin does not define rather than guess which side it belongs on.
RESIDENT_STATUSES = frozenset({"loading", "loaded", "sleeping"})
VACANT_STATUSES = frozenset({"downloading", "downloaded", "unloaded"})
KNOWN_STATUSES = RESIDENT_STATUSES | VACANT_STATUSES


class StatusError(RuntimeError):
    pass


def fetch_json(path: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    if not 0 < timeout <= MAX_TIMEOUT:
        raise StatusError(f"timeout must be within (0, {MAX_TIMEOUT}] seconds")
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise StatusError(f"{path}: response exceeds {MAX_RESPONSE_BYTES} bytes")
        payload = json.loads(raw)
    except StatusError:
        raise
    except (OSError, ValueError, urllib.error.URLError, socket.timeout) as error:
        raise StatusError(f"{path}: {type(error).__name__}: {error}") from error
    if not isinstance(payload, dict):
        raise StatusError(f"{path}: expected a JSON object")
    return payload


def collect_status(timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    health = fetch_json("/health", timeout)
    if health.get("status") != "ok":
        raise StatusError(f"/health: unhealthy response {health!r}")

    payload = fetch_json("/models", timeout)
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise StatusError("/models: missing data array")

    # A router model id is an alias, and an alias alone does not say which
    # weights it resolves to. Resolve it against this checkout's presets so the
    # summary names the artifact, not just the label used to request it.
    registry, catalog = local_registry()
    models = []
    for raw in raw_models:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise StatusError("/models: malformed model entry")
        status = raw.get("status")
        value = status.get("value") if isinstance(status, dict) else None
        if not isinstance(value, str):
            raise StatusError(f"/models: {raw['id']!r} has no status value")
        if value not in KNOWN_STATUSES:
            raise StatusError(f"/models: {raw['id']!r} has unknown status {value!r}")
        architecture = raw.get("architecture")
        architecture = architecture if isinstance(architecture, dict) else {}
        inputs = architecture.get("input_modalities", [])
        outputs = architecture.get("output_modalities", [])
        models.append({
            "id": raw["id"],
            "status": value,
            "description": describe_alias(raw["id"], registry, catalog),
            "input_modalities": inputs if isinstance(inputs, list) else [],
            "output_modalities": outputs if isinstance(outputs, list) else [],
        })

    models.sort(key=lambda model: model["id"])
    counts = Counter(model["status"] for model in models)
    resident = [model for model in models if model["status"] in RESIDENT_STATUSES]
    return {
        "endpoint": BASE_URL,
        "health": "ok",
        "model_count": len(models),
        "status_counts": dict(sorted(counts.items())),
        "loaded_models": [model["id"] for model in resident],
        "loaded_descriptions": [
            f"{model['description']}  [{ALIAS_LABEL}: {model['id']}]" for model in resident],
        "models": models,
        "loads_models": False,
    }


def render_text(summary: dict[str, Any]) -> str:
    counts = ", ".join(f"{key}={value}" for key, value in summary["status_counts"].items())
    lines = [
        f"Router: {summary['health']} ({summary['endpoint']})",
        f"Models: {summary['model_count']} ({counts or 'none'})",
        f"Loaded: {'; '.join(summary.get('loaded_descriptions') or summary['loaded_models']) or 'none'}",
    ]
    for model in summary["models"]:
        inputs = ",".join(model["input_modalities"]) or "unknown"
        outputs = ",".join(model["output_modalities"]) or "unknown"
        # The artifact leads. The router keys everything by alias, so the alias
        # still has to appear -- but as the compatibility handle it is, on the
        # detail line, not as the name of the thing.
        lines.append(f"  {model.get('description') or model['id']}")
        lines.append(f"    {model['status']} [{inputs} -> {outputs}]"
                     f"  [{ALIAS_LABEL}: {model['id']}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit the bounded summary as JSON")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)
    try:
        summary = collect_status(args.timeout)
    except StatusError as error:
        print(f"local router unavailable: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
