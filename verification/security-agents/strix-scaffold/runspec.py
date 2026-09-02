#!/usr/bin/env python3
"""Build and validate a hardened Strix run specification.

``RunSpec`` is a *proposal*; ``validate`` is the gate. The builder emits a spec
that already satisfies the policy, but validation is written to work on any spec
-- including hostile ones -- because the point of the exercise is that an
operator (or a model that talked an operator into it) cannot widen the profile
without the check failing.

This module never executes docker. ``docker_argv`` returns a list for review and
for evidence; running it is a separate, deliberate, human step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from urllib.parse import urlparse

import policy
from policy import (
    ALLOWED_ISOLATED_NETWORKS, ALLOWED_LLM_HOSTS, BLOCKED_TELEMETRY_DOMAINS,
    FORBIDDEN_CAPABILITIES, FORBIDDEN_CAPABILITIES_PILOT, FORBIDDEN_DEVICES,
    FORBIDDEN_ENV_KEYS, FORBIDDEN_IMAGE_TAGS, FORBIDDEN_MOUNT_SOURCES,
    FORBIDDEN_NETWORK_MODES, FORBIDDEN_SECURITY_OPT, MAX_LIMITS,
    NETWORK_NAMESPACE_SHARING_PREFIX, REQUIRED_NETWORK_PREFIX, PolicyReport,
    REQUIRED_ENV, ResourceLimits,
)
from scope import Scope, lab_scope

DIGEST_RE = re.compile(r"^[\w./-]+@sha256:[0-9a-f]{64}$")


@dataclass
class Mount:
    source: str
    target: str
    read_only: bool = True

    def as_docker_arg(self) -> str:
        mode = "ro" if self.read_only else "rw"
        return f"--volume={self.source}:{self.target}:{mode}"


@dataclass
class RunSpec:
    image: str
    scope: Scope
    mounts: list[Mount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    cap_add: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    security_opt: list[str] = field(default_factory=list)
    privileged: bool = False
    network: str = "strix-pilot-egress"
    mcp_servers: list[str] = field(default_factory=list)
    llm_api_base: str = f"http://127.0.0.1:{policy.LOCAL_ROUTER_PORT}/v1"
    strix_llm: str = "openai/heretic"
    source_dir: str | None = None
    source_is_disposable_copy: bool = False

    # -- docker argv (for review/evidence only; nothing is executed here) ----
    def docker_argv(self) -> list[str]:
        argv = ["docker", "run", "--rm", "--init",
                f"--network={self.network}",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
                f"--stop-timeout={min(60, self.limits.wall_clock_seconds)}"]
        argv += self.limits.as_docker_args()
        for cap in self.cap_add:
            argv.append(f"--cap-add={cap}")
        for dev in self.devices:
            argv.append(f"--device={dev}")
        for opt in self.security_opt:
            argv.append(f"--security-opt={opt}")
        if self.privileged:
            argv.append("--privileged")
        for m in self.mounts:
            argv.append(m.as_docker_arg())
        for k, v in sorted({**REQUIRED_ENV, **self.env}.items()):
            argv.append(f"--env={k}={v}")
        argv.append(f"--env=LLM_API_BASE={self.llm_api_base}")
        argv.append(f"--env=STRIX_LLM={self.strix_llm}")
        argv.append(self.image)
        return argv

    def as_dict(self) -> dict:
        return {
            "image": self.image,
            "network": self.network,
            "privileged": self.privileged,
            "cap_add": list(self.cap_add),
            "devices": list(self.devices),
            "security_opt": list(self.security_opt),
            "mounts": [{"source": m.source, "target": m.target,
                        "read_only": m.read_only} for m in self.mounts],
            "env_keys": sorted({**REQUIRED_ENV, **self.env}),
            "llm_api_base": self.llm_api_base,
            "strix_llm": self.strix_llm,
            "mcp_servers": list(self.mcp_servers),
            "limits": {
                "cpus": self.limits.cpus,
                "memory_bytes": self.limits.memory_bytes,
                "memory_swap_bytes": self.limits.memory_swap_bytes,
                "pids": self.limits.pids,
                "log_max_size": self.limits.log_max_size,
                "log_max_file": self.limits.log_max_file,
                "wall_clock_seconds": self.limits.wall_clock_seconds,
            },
            "scope": self.scope.as_dict(),
            "source_dir": self.source_dir,
            "source_is_disposable_copy": self.source_is_disposable_copy,
            "docker_argv": self.docker_argv(),
        }


def _under(path: str, parent: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except (ValueError, OSError):
        return False


def validate(spec: RunSpec) -> PolicyReport:
    """Check a proposed spec against the frozen policy. Never raises."""
    rep = PolicyReport()

    # 1. image pinning
    if policy.REQUIRE_DIGEST_PINNED_IMAGE and not DIGEST_RE.match(spec.image):
        rep.block("image.digest-pinned",
                  f"image {spec.image!r} is not pinned to a sha256 digest")
    else:
        rep.pass_("image.digest-pinned")
    tag = spec.image.split("@")[0].rsplit(":", 1)
    if len(tag) == 2 and tag[1].lower() in FORBIDDEN_IMAGE_TAGS:
        rep.block("image.moving-tag", f"image uses moving tag {tag[1]!r}")
    else:
        rep.pass_("image.moving-tag")

    # 2. telemetry + required env
    merged = {**REQUIRED_ENV, **spec.env}
    for key, want in REQUIRED_ENV.items():
        if str(merged.get(key, "")).lower() != want.lower():
            rep.block("telemetry.disabled",
                      f"{key} must be {want!r}, got {merged.get(key)!r}")
    if all(str(merged.get(k, "")).lower() == v.lower() for k, v in REQUIRED_ENV.items()):
        rep.pass_("telemetry.disabled")

    # 3. no hosted-provider credentials
    leaked = sorted(k for k in merged if k in FORBIDDEN_ENV_KEYS)
    if leaked:
        rep.block("credentials.none", f"forbidden credential env present: {leaked}")
    else:
        rep.pass_("credentials.none")

    # 4. loopback-only model endpoint
    parsed = urlparse(spec.llm_api_base)
    if parsed.scheme not in ("http", "https"):
        rep.block("llm.loopback", f"LLM_API_BASE scheme {parsed.scheme!r} unsupported")
    elif (parsed.hostname or "") not in ALLOWED_LLM_HOSTS:
        rep.block("llm.loopback",
                  f"LLM_API_BASE host {parsed.hostname!r} is not loopback")
    else:
        rep.pass_("llm.loopback")

    # 5. container privilege surface
    if spec.privileged:
        rep.block("container.no-privileged", "--privileged is prohibited")
    else:
        rep.pass_("container.no-privileged")

    banned_caps = set(FORBIDDEN_CAPABILITIES) | set(FORBIDDEN_CAPABILITIES_PILOT)
    bad_caps = sorted({c.upper().removeprefix("CAP_") for c in spec.cap_add} & banned_caps)
    if bad_caps:
        rep.block("container.capabilities", f"prohibited capabilities: {bad_caps}")
    else:
        rep.pass_("container.capabilities")

    bad_dev = sorted(d for d in spec.devices
                     if any(d.startswith(f) for f in FORBIDDEN_DEVICES))
    if bad_dev:
        rep.block("container.devices", f"prohibited devices: {bad_dev}")
    else:
        rep.pass_("container.devices")

    bad_opt = sorted(o for o in spec.security_opt
                     if o.replace("=", ":").lower() in
                     {f.replace("=", ":").lower() for f in FORBIDDEN_SECURITY_OPT})
    if bad_opt:
        rep.block("container.security-opt", f"prohibited security-opt: {bad_opt}")
    else:
        rep.pass_("container.security-opt")

    # 6. mounts: no socket, no credentials, no host namespaces, RO source
    bad_mounts = []
    for m in spec.mounts:
        src = m.source
        for forbidden in FORBIDDEN_MOUNT_SOURCES:
            if src == forbidden or _under(src, forbidden) and forbidden != "/":
                bad_mounts.append(f"{src} (matches {forbidden})")
                break
            if forbidden == "/" and Path(src).resolve() == Path("/"):
                bad_mounts.append(f"{src} (root filesystem)")
                break
    if bad_mounts:
        rep.block("mounts.forbidden-source", f"prohibited mounts: {bad_mounts}")
    else:
        rep.pass_("mounts.forbidden-source")

    writable = [m.source for m in spec.mounts if not m.read_only]
    if writable and not spec.source_is_disposable_copy:
        rep.block("mounts.read-only-source",
                  f"writable mounts {writable} require a disposable copy")
    else:
        rep.pass_("mounts.read-only-source")

    # 7. resource limits present and bounded
    lim, mx = spec.limits, MAX_LIMITS
    over = []
    if lim.cpus <= 0 or lim.cpus > mx.cpus:
        over.append(f"cpus={lim.cpus}")
    if lim.memory_bytes <= 0 or lim.memory_bytes > mx.memory_bytes:
        over.append(f"memory={lim.memory_bytes}")
    if lim.memory_swap_bytes > lim.memory_bytes:
        over.append(f"memory_swap={lim.memory_swap_bytes} exceeds memory (swap enabled)")
    if lim.pids <= 0 or lim.pids > mx.pids:
        over.append(f"pids={lim.pids}")
    if lim.wall_clock_seconds <= 0 or lim.wall_clock_seconds > mx.wall_clock_seconds:
        over.append(f"wall_clock={lim.wall_clock_seconds}")
    if not lim.log_max_size:
        over.append("log_max_size unset")
    if over:
        rep.block("limits.bounded", f"limits missing or out of range: {over}")
    else:
        rep.pass_("limits.bounded")

    # 8. MCP
    if spec.mcp_servers:
        remote = [s for s in spec.mcp_servers
                  if s.startswith(("http://", "https://", "ws://", "wss://", "sse://"))]
        if remote and not policy.ALLOW_REMOTE_MCP:
            rep.block("mcp.no-remote", f"remote MCP servers prohibited: {remote}")
        not_allowed = [s for s in spec.mcp_servers if s not in policy.ALLOWED_LOCAL_MCP]
        if not_allowed:
            rep.block("mcp.allowlisted", f"MCP servers not allowlisted: {not_allowed}")
    else:
        rep.pass_("mcp.no-remote")
        rep.pass_("mcp.allowlisted")

    # 9. scope is written and deny-by-default
    if not spec.scope.rules:
        rep.block("scope.written", "engagement scope is empty")
    else:
        rep.pass_("scope.written")

    # 10. network namespace: the scope rules above are enforced on the network,
    # so a shared or unpoliced namespace silently voids every one of them.
    net = (spec.network or "").strip()
    lowered = net.lower()
    if lowered in FORBIDDEN_NETWORK_MODES:
        rep.block("network.policed",
                  f"network {net!r} is an unpoliced or shared namespace, "
                  "which voids the engagement scope")
    elif lowered.startswith(NETWORK_NAMESPACE_SHARING_PREFIX):
        rep.block("network.policed",
                  f"network {net!r} shares another container's namespace")
    elif lowered in ALLOWED_ISOLATED_NETWORKS:
        rep.pass_("network.policed")
    elif not net.startswith(REQUIRED_NETWORK_PREFIX):
        rep.block("network.policed",
                  f"network {net!r} is not a dedicated {REQUIRED_NETWORK_PREFIX}* network")
    else:
        rep.pass_("network.policed")

    return rep


def hardened_lab_spec(image: str, source_dir: str | None = None,
                      disposable_copy: bool = False) -> RunSpec:
    """The reference-compliant spec: this is what the pilot is allowed to run."""
    mounts: list[Mount] = []
    if source_dir:
        mounts.append(Mount(source=source_dir, target="/workspace/source",
                            read_only=not disposable_copy))
    return RunSpec(
        image=image,
        scope=lab_scope(),
        mounts=mounts,
        limits=ResourceLimits(),
        source_dir=source_dir,
        source_is_disposable_copy=disposable_copy,
    )
