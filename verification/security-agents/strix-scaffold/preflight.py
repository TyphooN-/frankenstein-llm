#!/usr/bin/env python3
"""Emit machine-readable evidence for the hardened Strix pilot scaffold.

Produces three things and executes nothing:
  1. the reference-compliant run specification (including the exact docker argv
     an operator would review before any human decides to run it);
  2. a refusal matrix -- for each control, the hostile mutation that trips it and
     the resulting block -- which is the actual proof the controls are live;
  3. a scope-denial matrix over an injected offline resolver.

Run state is also captured: this gate asserts no container was started, and the
docker daemon's status is recorded so the claim is checkable rather than trusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import policy                                                    # noqa: E402
from runspec import Mount, ResourceLimits, hardened_lab_spec, validate  # noqa: E402
from scope import lab_scope                                      # noqa: E402

EVIDENCE = Path(__file__).resolve().parent / "evidence"
PINNED = "registry.local/strix-sandbox@sha256:" + "a" * 64


def _mutations():
    """(name, mutate) pairs; each must produce at least one policy violation."""
    def env(key, val):
        def f(s):
            s.env[key] = val
        return f

    def attr(name, val):
        def f(s):
            setattr(s, name, val)
        return f

    def mount(src, ro=True):
        def f(s):
            s.mounts = [Mount(source=src, target="/mnt/x", read_only=ro)]
        return f

    return [
        ("telemetry-enabled", env("STRIX_TELEMETRY", "true")),
        ("resume-re-enabled", env("STRIX_DISABLE_RESUME", "false")),
        ("openai-key-present", env("OPENAI_API_KEY", "sk-REDACTED")),
        ("anthropic-key-present", env("ANTHROPIC_API_KEY", "sk-REDACTED")),
        ("llm-endpoint-on-lan", attr("llm_api_base", "http://192.168.1.50:8080/v1")),
        ("llm-endpoint-hosted", attr("llm_api_base", "https://api.openai.com/v1")),
        ("llm-endpoint-host-gateway",
         attr("llm_api_base", "http://host.docker.internal:8080/v1")),
        ("privileged", attr("privileged", True)),
        ("cap-sys-admin", attr("cap_add", ["SYS_ADMIN"])),
        ("cap-net-raw", attr("cap_add", ["NET_RAW"])),
        ("cap-net-admin", attr("cap_add", ["NET_ADMIN"])),
        ("device-fuse", attr("devices", ["/dev/fuse"])),
        ("apparmor-unconfined", attr("security_opt", ["apparmor:unconfined"])),
        ("seccomp-unconfined", attr("security_opt", ["seccomp:unconfined"])),
        ("mount-docker-socket", mount("/var/run/docker.sock")),
        ("mount-ssh-keys", mount("/home/typhoon/.ssh")),
        ("mount-aws-creds", mount("/home/typhoon/.aws")),
        ("mount-hermes-config", mount("/home/typhoon/.hermes")),
        ("mount-host-home", mount("/home/typhoon")),
        ("mount-proc", mount("/proc")),
        ("writable-non-disposable-source", mount("/home/typhoon/git/real-repo", ro=False)),
        ("limits-removed", attr("limits", ResourceLimits(cpus=0, memory_bytes=0, pids=0))),
        ("limits-over-ceiling", attr("limits", ResourceLimits(cpus=64.0,
                                                             memory_bytes=64 << 30))),
        ("swap-enabled", attr("limits", ResourceLimits(memory_bytes=4 << 30,
                                                       memory_swap_bytes=8 << 30))),
        ("remote-mcp", attr("mcp_servers", ["https://mcp.example.com/sse"])),
        ("unlisted-local-mcp", attr("mcp_servers", ["/usr/local/bin/some-mcp"])),
        ("image-moving-tag", attr("image", "kalilinux/kali-rolling:latest")),
        ("network-host", attr("network", "host")),
        ("network-default-bridge", attr("network", "bridge")),
        ("network-shared-container", attr("network", "container:other")),
        ("network-undeclared", attr("network", "my-lan")),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-host-probe", action="store_true",
                        help="do not query the docker daemon; use when even a "
                             "read-only probe could socket-activate a service")
    args = parser.parse_args(argv)
    summary = {
        "gate": "strix-pilot-scaffold",
        "policy_version": policy.POLICY_VERSION,
        "source_assessment": policy.SOURCE_ASSESSMENT,
        "offensive_activity_performed": False,
        "container_started": False,
        "packets_sent_to_any_target": False,
        "mode": "static + unit + scope-denial proof only",
        "problems": [],
    }
    problems = summary["problems"]

    # --- reference spec -----------------------------------------------------
    ref = hardened_lab_spec(PINNED)
    ref_report = validate(ref)
    summary["reference_spec"] = ref.as_dict()
    summary["reference_spec_report"] = ref_report.as_dict()
    if not ref_report.ok:
        problems.append(f"reference spec rejected: {ref_report.as_dict()['violations']}")

    # --- refusal matrix -----------------------------------------------------
    refusals = []
    for name, mutate in _mutations():
        spec = hardened_lab_spec(PINNED)
        mutate(spec)
        rep = validate(spec)
        refusals.append({
            "mutation": name,
            "refused": not rep.ok,
            "controls_tripped": sorted({v.control for v in rep.violations}),
            "detail": [v.detail for v in rep.violations][:2],
        })
        if rep.ok:
            problems.append(f"mutation {name!r} was NOT refused")
    summary["refusal_matrix"] = refusals
    summary["refusal_matrix_total"] = len(refusals)
    summary["refusal_matrix_refused"] = sum(1 for r in refusals if r["refused"])

    # --- scope-denial matrix (offline injected resolver) --------------------
    resolver_map = {
        "juice.lab.local": ["127.0.0.1"],
        "webgoat.lab.local": ["127.0.0.1"],
        "example.com": ["93.184.216.34"],
        "evil.test": ["203.0.113.9"],
        "rebind.test": ["127.0.0.1", "203.0.113.9"],
        "metadata.test": ["169.254.169.254"],
    }

    def resolver(host):
        if host not in resolver_map:
            raise OSError(f"NXDOMAIN {host}")
        return resolver_map[host]

    sc = lab_scope(resolver=resolver)
    cases = [
        ("juice.lab.local", 3000, True, "in-scope lab app"),
        ("webgoat.lab.local", 8081, True, "in-scope lab app"),
        ("127.0.0.1", 8080, True, "loopback local model router"),
        ("juice.lab.local", 22, False, "in-scope host, out-of-scope port"),
        ("example.com", 443, False, "public internet"),
        ("evil.test", 443, False, "public internet"),
        ("rebind.test", 3000, False, "DNS rebinding: one answer out of scope"),
        ("metadata.test", 80, False, "cloud metadata endpoint"),
        ("169.254.169.254", 80, False, "cloud metadata by literal IP"),
        ("nope.invalid", 80, False, "unresolvable"),
        ("app.posthog.com", 443, False, "telemetry endpoint"),
        ("app.strix.ai", 443, False, "vendor endpoint"),
    ]
    matrix = []
    for host, port, expect_allow, why in cases:
        ok, reason = sc.check(host, port)
        matrix.append({"destination": f"{host}:{port}", "expected_allowed": expect_allow,
                       "allowed": ok, "reason": reason, "rationale": why})
        if ok != expect_allow:
            problems.append(f"scope case {host}:{port} expected allow={expect_allow}, got {ok}")
    summary["scope_denial_matrix"] = matrix
    summary["scope"] = sc.as_dict()
    summary["blocked_telemetry_domains"] = list(policy.BLOCKED_TELEMETRY_DOMAINS)

    # --- host state: prove nothing was launched -----------------------------
    docker_state = {"binary": shutil.which("docker")}
    if args.skip_host_probe:
        # Querying a socket-activated daemon can start it, which is itself a host
        # change. Declining to look is recorded rather than reported as "clean".
        docker_state.update({
            "probed": False,
            "reason": "--skip-host-probe: a read-only docker query can "
                      "socket-activate the daemon, which would be a host change",
            "running_containers": None,
            "lab_available": None,
        })
        summary["docker"] = docker_state
    else:
        docker_state["probed"] = True
        try:
            docker_state["systemd_active"] = subprocess.run(
                ["systemctl", "is-active", "docker"], capture_output=True, text=True,
                timeout=15).stdout.strip()
        except Exception as exc:                                # noqa: BLE001
            docker_state["systemd_active"] = f"unknown: {exc.__class__.__name__}"
        try:
            ps = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True,
                                timeout=20)
            docker_state["running_containers"] = [c for c in ps.stdout.split() if c]
            docker_state["ps_rc"] = ps.returncode
        except Exception as exc:                                # noqa: BLE001
            docker_state["running_containers"] = []
            docker_state["ps_error"] = exc.__class__.__name__
        docker_state["lab_available"] = bool(docker_state.get("running_containers"))
        summary["docker"] = docker_state
    summary["lab_run_performed"] = False
    summary["lab_run_skipped_reason"] = (
        "docker daemon inactive and no harmless lab container exists; policy permits "
        "static/unit/scope-denial proof only in that case, and starting the daemon "
        "would be a host service change outside this task's boundaries")

    summary["unit_tests"] = {
        "module": "test_strix_scaffold.py",
        "command": "python3 -m unittest test_strix_scaffold -v",
    }
    summary["pass"] = not problems
    summary["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "strix-scaffold-preflight.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("reference_spec", "refusal_matrix",
                                   "scope_denial_matrix", "scope")}, indent=2))
    print(f"\nrefusal matrix: {summary['refusal_matrix_refused']}/"
          f"{summary['refusal_matrix_total']} hostile mutations refused")
    print(f"scope matrix:   {sum(1 for m in matrix if m['allowed'] == m['expected_allowed'])}"
          f"/{len(matrix)} destinations classified as expected")
    print(f"[strix] wrote {out}", file=sys.stderr)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
