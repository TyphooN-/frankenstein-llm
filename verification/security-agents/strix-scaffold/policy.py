#!/usr/bin/env python3
"""Frozen containment policy for the Strix authorized-testing pilot.

This module is the single source of truth for what the pilot is *allowed* to be.
It encodes, as data, the "Required pilot profile" from
``SECURITY-AGENT-STACK-ASSESSMENT-2026-09-01.md`` so that a run configuration can
be mechanically refused instead of reviewed by eye.

Design rule, same as the capability gates: a control counts as present only when
something fails without it. Every constant here is exercised by a negative test
in ``test_strix_scaffold.py`` -- a policy nobody can trip is not a control.

Nothing in this package launches Strix, starts a container, or emits a packet.
It builds and validates specifications; execution is deliberately out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field

POLICY_VERSION = "strix-pilot-hardening/1"
SOURCE_ASSESSMENT = (
    "/home/typhoon/git/frankenstein-llm/verification/security-agents/"
    "SECURITY-AGENT-STACK-ASSESSMENT-2026-09-01.md"
)

# --- 1. Telemetry -----------------------------------------------------------
# Strix defaults TelemetrySettings.enabled = True. The pilot forces it off and
# additionally blocks the endpoints at the network layer, because an env var is
# a request and a firewall rule is a control.
REQUIRED_ENV: dict[str, str] = {
    "STRIX_TELEMETRY": "false",
    "STRIX_DISABLE_RESUME": "true",      # open mount-guard bypass issue
    "OPENAI_AGENTS_DISABLE_TRACING": "1",
    "DO_NOT_TRACK": "1",
    "SCARF_NO_ANALYTICS": "true",
}
BLOCKED_TELEMETRY_DOMAINS: tuple[str, ...] = (
    "app.posthog.com",
    "us.i.posthog.com",
    "eu.i.posthog.com",
    "posthog.com",
    "scarf.sh",
    "static.scarf.sh",
    "app.strix.ai",
    "strix.ai",
)

# --- 2. Credentials ---------------------------------------------------------
# No hosted provider may be reachable even by accident: the pilot is local-router
# only, so any of these being set is a configuration error, not a preference.
FORBIDDEN_ENV_KEYS: tuple[str, ...] = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
    "OPENROUTER_API_KEY", "PERPLEXITY_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "STRIX_API_KEY",
)

# --- 3. Model endpoint ------------------------------------------------------
# LLM_API_BASE must terminate on this host's loopback router. A LAN address is
# refused: "local" means loopback here, not "somewhere on my network".
ALLOWED_LLM_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1", "[::1]")
LOCAL_ROUTER_PORT = 8080

# --- 4. Container hardening -------------------------------------------------
FORBIDDEN_MOUNT_SOURCES: tuple[str, ...] = (
    "/var/run/docker.sock", "/run/docker.sock",   # container escape
    "/home/typhoon/.ssh", "/root/.ssh",
    "/home/typhoon/.aws", "/home/typhoon/.config/gh",
    "/home/typhoon/.gnupg", "/home/typhoon/.docker",
    "/home/typhoon/.hermes", "/home/typhoon/.claude",
    "/home/typhoon/.mozilla", "/home/typhoon/.config/google-chrome",
    "/run", "/proc", "/sys", "/dev",
    "/home/typhoon",                               # bare home mount
    "/",
)
FORBIDDEN_CAPABILITIES: tuple[str, ...] = (
    "SYS_ADMIN", "SYS_PTRACE", "SYS_MODULE", "SYS_RAWIO",
    "SYS_BOOT", "MKNOD", "AUDIT_CONTROL", "ALL",
)
# NET_ADMIN/NET_RAW are what upstream grants for scanning. They stay refused in
# this pilot: egress is policed outside the container, so the sandbox has no
# legitimate need to reconfigure its own networking.
FORBIDDEN_CAPABILITIES_PILOT: tuple[str, ...] = ("NET_ADMIN", "NET_RAW")
FORBIDDEN_DEVICES: tuple[str, ...] = ("/dev/fuse", "/dev/kvm", "/dev/dri")
FORBIDDEN_SECURITY_OPT: tuple[str, ...] = (
    "apparmor:unconfined", "apparmor=unconfined",
    "seccomp:unconfined", "seccomp=unconfined",
    "label:disable", "systempaths=unconfined",
)

# --- 5. Mandatory resource limits ------------------------------------------
# Upstream makes these opt-in; the pilot makes them mandatory and bounded.
@dataclass(frozen=True)
class ResourceLimits:
    cpus: float = 2.0
    memory_bytes: int = 4 << 30          # 4 GiB
    memory_swap_bytes: int = 4 << 30     # equal to memory => swap disabled
    pids: int = 512
    log_max_size: str = "16m"
    log_max_file: int = 3
    wall_clock_seconds: int = 1800       # 30 min hard deadline
    ulimit_nofile: int = 4096

    def as_docker_args(self) -> list[str]:
        return [
            f"--cpus={self.cpus}",
            f"--memory={self.memory_bytes}",
            f"--memory-swap={self.memory_swap_bytes}",
            f"--pids-limit={self.pids}",
            f"--log-opt=max-size={self.log_max_size}",
            f"--log-opt=max-file={self.log_max_file}",
            f"--ulimit=nofile={self.ulimit_nofile}:{self.ulimit_nofile}",
        ]


MAX_LIMITS = ResourceLimits(
    cpus=8.0, memory_bytes=16 << 30, memory_swap_bytes=16 << 30,
    pids=4096, wall_clock_seconds=7200,
)

# --- 6. MCP -----------------------------------------------------------------
ALLOW_REMOTE_MCP = False
ALLOWED_LOCAL_MCP: tuple[str, ...] = ()   # empty: none allowlisted for the pilot

# --- 7. Network namespace ---------------------------------------------------
# scope.py only means anything because egress is policed outside the agent, on a
# network the agent cannot reconfigure. That argument collapses if the container
# shares somebody else's namespace: --network=host puts the agent directly on the
# host's stack, and the default bridge has unrestricted outbound internet. Both
# are refused, so the pilot must run on a dedicated, policed network.
FORBIDDEN_NETWORK_MODES: tuple[str, ...] = ("host", "bridge", "default", "")
NETWORK_NAMESPACE_SHARING_PREFIX = "container:"
REQUIRED_NETWORK_PREFIX = "strix-pilot"
ALLOWED_ISOLATED_NETWORKS: tuple[str, ...] = ("none",)   # no egress at all is safe

# --- 8. Image pinning -------------------------------------------------------
# kali-rolling:latest and go tools @latest are moving targets; the pilot requires
# a digest-pinned image so a rerun means the same thing tomorrow.
REQUIRE_DIGEST_PINNED_IMAGE = True
FORBIDDEN_IMAGE_TAGS: tuple[str, ...] = ("latest", "rolling", "edge", "nightly", "dev")


class PolicyViolation(Exception):
    """Raised when a proposed run configuration breaks the frozen policy."""

    def __init__(self, control: str, detail: str) -> None:
        super().__init__(f"[{control}] {detail}")
        self.control = control
        self.detail = detail


@dataclass
class Finding:
    control: str
    detail: str
    severity: str = "block"

    def as_dict(self) -> dict:
        return {"control": self.control, "detail": self.detail, "severity": self.severity}


@dataclass
class PolicyReport:
    violations: list[Finding] = field(default_factory=list)
    satisfied: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def block(self, control: str, detail: str) -> None:
        self.violations.append(Finding(control, detail))

    def pass_(self, control: str) -> None:
        self.satisfied.append(control)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "policy_version": POLICY_VERSION,
            "violations": [v.as_dict() for v in self.violations],
            "controls_satisfied": sorted(set(self.satisfied)),
        }
