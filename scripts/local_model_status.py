#!/usr/bin/env python3
"""Bounded, loopback-only status summary for the local llama.cpp router.

``--host-sharing`` extends the same observer with the second question an
operator actually asks before typing at a local model: *is the qualification
mission about to take the GPU out from under me?* That report is advisory. It
reads the router over loopback, the mission's own durable state file, and the
process table through the supervisor's own classifier -- and it starts, stops
and loads nothing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from model_catalog import ALIAS_LABEL, describe_alias, local_registry
from pathlib import Path
import socket
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
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


MISSION_STATE = ROOT / "verification/mission-supervisor/mission-state.json"
MISSION_SUPERVISOR = ROOT / "verification/mission-supervisor/run_functional_mission.py"
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_CONFLICT_SAMPLES = 8

# run_functional_mission writes one of these into "status". A step only holds the
# GPU under the two running values; the pending ones mean the supervisor is alive
# and working towards the next step, and the terminal ones mean it has stopped.
#
# `current_step` is deliberately absent from this classification. It is set after
# the quiet-host wait already succeeded, it is never cleared when an individual
# step ends -- only at the three whole-mission exits -- and it does not exist at
# all in a freshly initialised state. So it is null exactly when the supervisor is
# starting up and about to claim the GPU, and non-null for minutes after the step
# it names has finished. Reading it as "no step is running" inverts the answer in
# the dangerous direction.
RUNNING_MISSION_STATUSES = frozenset({"running", "running-with-failures"})
PENDING_MISSION_STATUSES = frozenset({"starting", "waiting-artifacts", "waiting-safe-host"})
TERMINAL_MISSION_STATUSES = frozenset({
    "blocked-policy", "failed", "functional-foundation-complete",
    "functional-foundation-incomplete", "interrupted",
})

VERDICT_STEP_RUNNING = "mission-step-running"
VERDICT_MAY_TAKE_GPU = "mission-may-take-the-gpu"
VERDICT_IDLE_LAST_WRITE = "mission-idle-per-last-write"


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


def load_supervisor(path: Path = MISSION_SUPERVISOR):
    """Import the mission supervisor so its host classifier is reused, not copied.

    The point of importing it is that this report cannot drift from the decision
    it describes: the blocking reasons, the memory floor and the ``/proc`` walk
    are the supervisor's own. Nothing at that module's import time starts a
    mission -- the signal handlers are installed under ``__main__`` -- but it does
    parse ``HERMES_MISSION_QUIET_TIMEOUT`` eagerly and raises on a bad value, so
    every failure to load is caught and reported rather than crashing a read-only
    status command.
    """
    spec = importlib.util.spec_from_file_location("_mission_supervisor", path)
    if spec is None or spec.loader is None:
        raise StatusError(f"{path}: not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_mission_state(path: Path = MISSION_STATE) -> dict[str, Any]:
    """Classify the mission's durable state without ever reading absence as safety.

    The file is ignored runtime state, written by the supervisor and by nothing
    else. Missing, truncated, unparseable, or carrying a status this reader does
    not know are all reported as ``unknown`` -- which the verdict treats exactly
    like an active mission, because a reader that cannot tell has not been told
    the mission is idle.
    """
    report: dict[str, Any] = {
        "state_path": str(path),
        "readable": False,
        "problem": None,
        "status": None,
        "current_step": None,
        "disposition": "unknown",
        # Stated in the payload so a consumer that only reads JSON still gets the
        # correction, not just a reader of the prose.
        "current_step_is_not_a_liveness_signal": True,
    }
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_STATE_BYTES + 1)
    except OSError as error:
        report["problem"] = f"{type(error).__name__}: {error}"
        return report
    if len(raw) > MAX_STATE_BYTES:
        report["problem"] = f"state exceeds {MAX_STATE_BYTES} bytes"
        return report
    try:
        document = json.loads(raw)
    except ValueError as error:
        report["problem"] = f"{type(error).__name__}: {error}"
        return report
    if not isinstance(document, dict):
        report["problem"] = "state is not a JSON object"
        return report
    report["readable"] = True
    schema = document.get("schema")
    if schema != "frankenstein-functional-mission/1":
        report["problem"] = f"unknown schema {schema!r}"
        return report
    status = document.get("status")
    report["status"] = status if isinstance(status, str) else None
    current = document.get("current_step")
    report["current_step"] = current if isinstance(current, str) else None
    if report["status"] in RUNNING_MISSION_STATUSES:
        report["disposition"] = "running"
    elif report["status"] in PENDING_MISSION_STATUSES:
        report["disposition"] = "pending"
    elif report["status"] in TERMINAL_MISSION_STATUSES:
        report["disposition"] = "terminal"
    else:
        report["problem"] = f"unknown status {report['status']!r}"
    return report


def scan_host(supervisor: Any) -> dict[str, Any]:
    """Report the host through the supervisor's own quiet-host predicates.

    ``conflicts()`` reads only ``comm``, the ``cwd`` symlink and ``cgroup`` for
    each task, never ``/proc/<pid>/cmdline``; it raises rather than returning an
    empty list when the process table cannot be read. Both properties are the
    reason it is called here instead of a second implementation.
    """
    report: dict[str, Any] = {
        "scan": "unavailable",
        "problem": None,
        "blocking_conflicts": [],
        "advisory_conflicts": [],
        "blocking_reasons": sorted(supervisor.MISSION_BLOCKING_REASONS),
        "mem_available_bytes": None,
        "mem_floor_bytes": int(supervisor.MIN_AVAILABLE),
        "meets_mission_memory_floor": None,
    }
    try:
        found = supervisor.conflicts()
        available = supervisor.mem_available()
    except Exception as error:  # noqa: BLE001 - an unreadable host is reported, not raised
        report["problem"] = f"{type(error).__name__}: {error}"
        return report
    blocking = supervisor.MISSION_BLOCKING_REASONS
    blockers = [item for item in found if item.get("reason") in blocking]
    advisory = [item for item in found if item.get("reason") not in blocking]
    report.update({
        "scan": "ok",
        "blocking_conflicts": blockers[:MAX_CONFLICT_SAMPLES],
        "advisory_conflicts": advisory[:MAX_CONFLICT_SAMPLES],
        "blocking_count": len(blockers),
        "advisory_count": len(advisory),
        "blocking_counts_by_reason": dict(Counter(str(item["reason"]) for item in blockers)),
        "conflict_sample_limit": MAX_CONFLICT_SAMPLES,
        "mem_available_bytes": available,
        "meets_mission_memory_floor": available >= supervisor.MIN_AVAILABLE,
    })
    return report


def collect_host_sharing(timeout: float = DEFAULT_TIMEOUT,
                         state_path: Path | None = None) -> dict[str, Any]:
    """Advisory host-sharing report. It observes; it never admits, waits or stops.

    Three verdicts, and none of them is "safe". A terminal mission status earns
    only ``mission-idle-per-last-write``, because the state file records what the
    supervisor last wrote, not a lease: the unit is ``WantedBy=default.target``
    and can be started, or reach the head of its own quiet-host wait, in the gap
    between this read and the next request typed at the router.
    """
    # Resolved here rather than as a default argument so the constant is read at
    # call time, which is what makes the path substitutable in a test.
    state_path = MISSION_STATE if state_path is None else state_path
    reasons: list[str] = []
    try:
        router: dict[str, Any] = {"reachable": True, "problem": None, **collect_status(timeout)}
    except StatusError as error:
        # The router being down is a fact about host sharing, not a failure of the
        # report: the mission restarts it itself as a step.
        router = {"reachable": False, "problem": str(error), "endpoint": BASE_URL,
                  "loaded_models": [], "loads_models": False}

    mission = read_mission_state(state_path)
    try:
        host = scan_host(load_supervisor())
    except Exception as error:  # noqa: BLE001 - reported as an unavailable scan
        host = {"scan": "unavailable", "problem": f"{type(error).__name__}: {error}",
                "blocking_conflicts": [], "advisory_conflicts": [],
                "blocking_reasons": [], "mem_available_bytes": None,
                "mem_floor_bytes": None, "meets_mission_memory_floor": None}

    if mission["disposition"] == "running":
        verdict = VERDICT_STEP_RUNNING
        reasons.append(f"mission status {mission['status']!r} last recorded an executing step; "
                       "this is not live process proof")
    elif mission["disposition"] == "pending":
        verdict = VERDICT_MAY_TAKE_GPU
        reasons.append(
            f"mission status {mission['status']!r} last recorded the supervisor "
            "working towards its next step; this is not live process proof")
    elif mission["disposition"] == "terminal":
        verdict = VERDICT_IDLE_LAST_WRITE
        reasons.append(f"mission status {mission['status']!r} was its last durable write")
    else:
        verdict = VERDICT_MAY_TAKE_GPU
        reasons.append(
            "mission state could not be classified"
            + (f" ({mission['problem']})" if mission["problem"] else "")
            + "; an unreadable or absent state is not evidence of an idle mission")

    if host["scan"] != "ok":
        verdict = VERDICT_MAY_TAKE_GPU
        reasons.append(f"host scan unavailable ({host['problem']}); "
                       "a host that cannot be read is not a quiet one")
    else:
        if host["blocking_conflicts"]:
            reasons.append(
                "processes the mission would treat as blockers are running: "
                + ", ".join(sorted(host["blocking_counts_by_reason"])))
        if host["meets_mission_memory_floor"] is False:
            reasons.append(
                f"MemAvailable {host['mem_available_bytes']} is below the mission's "
                f"{host['mem_floor_bytes']} floor")

    return {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "advisory": True,
        # The single most important field: nothing here reserves the GPU, and the
        # mission never consults it.
        "atomic_admission": False,
        "loads_models": False,
        "verdict": verdict,
        "reasons": reasons,
        "mission": mission,
        "host": host,
        "router": router,
    }


def render_host_sharing(report: dict[str, Any]) -> str:
    mission, host, router = report["mission"], report["host"], report["router"]
    lines = [
        f"Verdict: {report['verdict']} (advisory; not an admission guarantee)",
        f"Mission: status={mission['status'] or 'unknown'} "
        f"disposition={mission['disposition']}"
        + (f" problem={mission['problem']}" if mission["problem"] else ""),
        f"  current_step={mission['current_step'] or 'none'}"
        " (recorded value; not a liveness signal -- see the reference)",
    ]
    if host["scan"] == "ok":
        lines.append(
            f"Host: blocking={host['blocking_count']} "
            f"advisory={host['advisory_count']} "
            f"MemAvailable={host['mem_available_bytes']} "
            f"floor={host['mem_floor_bytes']}")
        lines.append(f"  Process details limited to {MAX_CONFLICT_SAMPLES} per category.")
        for item in host["blocking_conflicts"]:
            lines.append(f"  blocker pid={item['pid']} {item['command']} ({item['reason']})")
    else:
        lines.append(f"Host: scan unavailable ({host['problem']})")
    if router["reachable"]:
        lines.append(f"Router: ok ({router['endpoint']}) "
                     f"loaded={'; '.join(router['loaded_models']) or 'none'}")
    else:
        lines.append(f"Router: unreachable ({router['problem']})")
    for reason in report["reasons"]:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


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
    parser.add_argument(
        "--host-sharing", action="store_true",
        help="advisory router + mission + host report; reads only, and its exit "
             "code reports whether the report was produced, never whether the "
             "host is free")
    args = parser.parse_args(argv)
    if args.host_sharing:
        report = collect_host_sharing(args.timeout)
        print(json.dumps(report, indent=2, sort_keys=True) if args.json
              else render_host_sharing(report))
        # Zero means only that the report was produced. Never use this command
        # as an admission predicate in `&& start-a-model` shell chains.
        return 0
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
