#!/usr/bin/env python3
"""Unit and scope-denial proof for the hardened Strix pilot scaffold.

Every control in ``policy.py`` gets a negative test: a spec that trips it must be
refused. A control with only positive tests is decoration, so the compliant spec
is asserted clean *and* each hostile mutation of it is asserted blocked.

No container is started and no packet is sent; ``Scope`` is driven with an
injected resolver so DNS behaviour (including rebinding) is deterministic and
entirely offline.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import policy                                                    # noqa: E402
from policy import PolicyViolation, ResourceLimits               # noqa: E402
from runspec import Mount, RunSpec, hardened_lab_spec, validate  # noqa: E402
from scope import Scope, ScopeRule, lab_scope                    # noqa: E402

PINNED = ("registry.local/strix-sandbox@sha256:"
          + "a" * 64)


def fake_resolver(mapping: dict[str, list[str]]):
    def _resolve(host: str) -> list[str]:
        if host not in mapping:
            raise OSError(f"NXDOMAIN {host}")
        return mapping[host]
    return _resolve


class TestCompliantSpec(unittest.TestCase):
    """The reference spec must pass, or every negative test below is vacuous."""

    def test_reference_spec_is_clean(self):
        rep = validate(hardened_lab_spec(PINNED))
        self.assertTrue(rep.ok, f"reference spec rejected: {rep.as_dict()['violations']}")

    def test_reference_spec_satisfies_every_control(self):
        rep = validate(hardened_lab_spec(PINNED))
        for control in ("telemetry.disabled", "credentials.none", "llm.loopback",
                        "container.no-privileged", "container.capabilities",
                        "container.devices", "container.security-opt",
                        "mounts.forbidden-source", "limits.bounded",
                        "mcp.no-remote", "scope.written", "image.digest-pinned",
                        "network.policed"):
            self.assertIn(control, rep.satisfied)

    def test_docker_argv_drops_caps_and_is_read_only(self):
        argv = hardened_lab_spec(PINNED).docker_argv()
        self.assertIn("--cap-drop=ALL", argv)
        self.assertIn("--read-only", argv)
        self.assertIn("--security-opt=no-new-privileges", argv)
        self.assertNotIn("--privileged", argv)
        self.assertTrue(any(a.startswith("--pids-limit=") for a in argv))
        self.assertTrue(any(a.startswith("--memory=") for a in argv))
        self.assertTrue(any(a.startswith("--cpus=") for a in argv))

    def test_docker_argv_never_mounts_docker_socket(self):
        argv = hardened_lab_spec(PINNED).docker_argv()
        self.assertFalse([a for a in argv if "docker.sock" in a])


class TestTelemetryAndCredentials(unittest.TestCase):
    def test_telemetry_enabled_is_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.env["STRIX_TELEMETRY"] = "true"
        rep = validate(spec)
        self.assertFalse(rep.ok)
        self.assertIn("telemetry.disabled", [v.control for v in rep.violations])

    def test_resume_must_stay_disabled(self):
        spec = hardened_lab_spec(PINNED)
        spec.env["STRIX_DISABLE_RESUME"] = "false"
        self.assertFalse(validate(spec).ok)

    def test_blocked_domains_are_declared(self):
        for host in ("app.posthog.com", "scarf.sh", "app.strix.ai"):
            self.assertIn(host, policy.BLOCKED_TELEMETRY_DOMAINS)

    def test_hosted_provider_credentials_are_blocked(self):
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY"):
            spec = hardened_lab_spec(PINNED)
            spec.env[key] = "sk-should-never-be-here"
            rep = validate(spec)
            self.assertFalse(rep.ok, f"{key} was not refused")
            self.assertIn("credentials.none", [v.control for v in rep.violations])


class TestLoopbackOnlyRouter(unittest.TestCase):
    def test_lan_endpoint_is_blocked(self):
        for base in ("http://192.168.1.50:8080/v1", "http://10.0.0.5:8080/v1",
                     "https://api.openai.com/v1", "http://host.docker.internal:8080/v1"):
            spec = hardened_lab_spec(PINNED)
            spec.llm_api_base = base
            rep = validate(spec)
            self.assertFalse(rep.ok, f"{base} was not refused")
            self.assertIn("llm.loopback", [v.control for v in rep.violations])

    def test_loopback_endpoint_is_allowed(self):
        for base in ("http://127.0.0.1:8080/v1", "http://localhost:8080/v1"):
            spec = hardened_lab_spec(PINNED)
            spec.llm_api_base = base
            self.assertTrue(validate(spec).ok, base)


class TestContainerHardening(unittest.TestCase):
    def test_privileged_is_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.privileged = True
        self.assertIn("container.no-privileged",
                      [v.control for v in validate(spec).violations])

    def test_sys_admin_and_net_raw_are_blocked(self):
        for cap in ("SYS_ADMIN", "CAP_SYS_ADMIN", "NET_ADMIN", "NET_RAW", "ALL"):
            spec = hardened_lab_spec(PINNED)
            spec.cap_add = [cap]
            rep = validate(spec)
            self.assertFalse(rep.ok, f"{cap} was not refused")
            self.assertIn("container.capabilities", [v.control for v in rep.violations])

    def test_fuse_device_is_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.devices = ["/dev/fuse"]
        self.assertIn("container.devices",
                      [v.control for v in validate(spec).violations])

    def test_apparmor_unconfined_is_blocked(self):
        for opt in ("apparmor:unconfined", "apparmor=unconfined", "seccomp:unconfined"):
            spec = hardened_lab_spec(PINNED)
            spec.security_opt = [opt]
            rep = validate(spec)
            self.assertFalse(rep.ok, f"{opt} was not refused")
            self.assertIn("container.security-opt", [v.control for v in rep.violations])

    def test_docker_socket_mount_is_blocked(self):
        for src in ("/var/run/docker.sock", "/run/docker.sock"):
            spec = hardened_lab_spec(PINNED)
            spec.mounts = [Mount(source=src, target="/var/run/docker.sock")]
            rep = validate(spec)
            self.assertFalse(rep.ok, f"{src} was not refused")
            self.assertIn("mounts.forbidden-source", [v.control for v in rep.violations])

    def test_credential_and_home_mounts_are_blocked(self):
        for src in ("/home/typhoon/.ssh", "/home/typhoon/.aws", "/home/typhoon/.claude",
                    "/home/typhoon/.hermes", "/home/typhoon", "/proc", "/"):
            spec = hardened_lab_spec(PINNED)
            spec.mounts = [Mount(source=src, target="/mnt/x")]
            rep = validate(spec)
            self.assertFalse(rep.ok, f"{src} was not refused")

    def test_writable_source_requires_disposable_copy(self):
        spec = hardened_lab_spec(PINNED, source_dir="/tmp/strix-src", disposable_copy=False)
        spec.mounts = [Mount(source="/tmp/strix-src", target="/workspace/source",
                             read_only=False)]
        self.assertIn("mounts.read-only-source",
                      [v.control for v in validate(spec).violations])

    def test_disposable_copy_may_be_writable(self):
        spec = hardened_lab_spec(PINNED, source_dir="/tmp/strix-src-copy",
                                 disposable_copy=True)
        self.assertTrue(validate(spec).ok)


class TestNetworkNamespace(unittest.TestCase):
    """Scope is enforced on the network, so the namespace is part of the scope."""

    def test_host_network_is_blocked(self):
        for network in ("host", "HOST", "bridge", "default", ""):
            spec = hardened_lab_spec(PINNED)
            spec.network = network
            rep = validate(spec)
            self.assertFalse(rep.ok, f"network {network!r} was not refused")
            self.assertIn("network.policed", [v.control for v in rep.violations])

    def test_shared_container_namespace_is_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.network = "container:some-other-agent"
        rep = validate(spec)
        self.assertFalse(rep.ok)
        self.assertIn("network.policed", [v.control for v in rep.violations])

    def test_undeclared_network_is_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.network = "my-lan"
        self.assertIn("network.policed",
                      [v.control for v in validate(spec).violations])

    def test_dedicated_and_isolated_networks_are_allowed(self):
        for network in ("strix-pilot-egress", "strix-pilot-lab", "none"):
            spec = hardened_lab_spec(PINNED)
            spec.network = network
            self.assertTrue(validate(spec).ok, network)

    def test_docker_argv_carries_the_validated_network(self):
        argv = hardened_lab_spec(PINNED).docker_argv()
        self.assertIn("--network=strix-pilot-egress", argv)
        self.assertFalse([a for a in argv if a == "--network=host"])


class TestResourceLimits(unittest.TestCase):
    def test_unbounded_limits_are_blocked(self):
        for lim in (ResourceLimits(cpus=0), ResourceLimits(memory_bytes=0),
                    ResourceLimits(pids=0), ResourceLimits(wall_clock_seconds=0),
                    ResourceLimits(log_max_size="")):
            spec = hardened_lab_spec(PINNED)
            spec.limits = lim
            self.assertIn("limits.bounded",
                          [v.control for v in validate(spec).violations])

    def test_limits_above_ceiling_are_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.limits = ResourceLimits(cpus=64.0, memory_bytes=64 << 30)
        self.assertFalse(validate(spec).ok)

    def test_swap_must_not_exceed_memory(self):
        spec = hardened_lab_spec(PINNED)
        spec.limits = ResourceLimits(memory_bytes=4 << 30, memory_swap_bytes=8 << 30)
        self.assertIn("limits.bounded",
                      [v.control for v in validate(spec).violations])


class TestImagePinning(unittest.TestCase):
    def test_moving_tag_is_blocked(self):
        for image in ("kalilinux/kali-rolling:latest", "strix-sandbox:latest",
                      "strix-sandbox:nightly"):
            spec = hardened_lab_spec(image)
            rep = validate(spec)
            self.assertFalse(rep.ok, f"{image} was not refused")

    def test_digest_pinned_image_is_accepted(self):
        self.assertTrue(validate(hardened_lab_spec(PINNED)).ok)


class TestMcp(unittest.TestCase):
    def test_remote_mcp_is_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.mcp_servers = ["https://mcp.example.com/sse"]
        rep = validate(spec)
        self.assertFalse(rep.ok)
        self.assertIn("mcp.no-remote", [v.control for v in rep.violations])

    def test_non_allowlisted_local_mcp_is_blocked(self):
        spec = hardened_lab_spec(PINNED)
        spec.mcp_servers = ["/usr/local/bin/some-mcp"]
        self.assertIn("mcp.allowlisted",
                      [v.control for v in validate(spec).violations])


class TestScopeDenial(unittest.TestCase):
    """The scope-denial proof: out-of-scope destinations must be refused."""

    def setUp(self):
        self.resolver = fake_resolver({
            "juice.lab.local": ["127.0.0.1"],
            "example.com": ["93.184.216.34"],
            "evil.test": ["203.0.113.9"],
            "rebind.test": ["127.0.0.1", "203.0.113.9"],   # DNS rebinding
            "metadata.test": ["169.254.169.254"],
            "cname.test": ["198.51.100.7"],
        })
        self.scope = lab_scope(resolver=self.resolver)

    def test_in_scope_lab_host_is_allowed(self):
        ok, reason = self.scope.check("juice.lab.local", 3000)
        self.assertTrue(ok, reason)

    def test_in_scope_host_on_out_of_scope_port_is_denied(self):
        ok, _ = self.scope.check("juice.lab.local", 22)
        self.assertFalse(ok)

    def test_public_internet_host_is_denied(self):
        for host in ("example.com", "evil.test", "cname.test"):
            ok, reason = self.scope.check(host, 443)
            self.assertFalse(ok, f"{host} was allowed: {reason}")

    def test_dns_rebinding_is_denied(self):
        """One out-of-scope answer poisons the whole destination."""
        ok, reason = self.scope.check("rebind.test", 3000)
        self.assertFalse(ok, reason)
        self.assertIn("out-of-scope", reason)

    def test_cloud_metadata_is_never_reachable(self):
        ok, _ = self.scope.check("metadata.test", 80)
        self.assertFalse(ok)
        ok, _ = self.scope.check("169.254.169.254", 80)
        self.assertFalse(ok)

    def test_unresolvable_host_is_denied(self):
        ok, reason = self.scope.check("nope.invalid", 80)
        self.assertFalse(ok)
        self.assertIn("resolution failed", reason)

    def test_empty_scope_denies_everything(self):
        empty = Scope(resolver=self.resolver)
        ok, reason = empty.check("juice.lab.local", 3000)
        self.assertFalse(ok)
        self.assertIn("deny-by-default", reason)

    def test_assert_allowed_raises_on_out_of_scope(self):
        with self.assertRaises(PolicyViolation):
            self.scope.assert_allowed("example.com", 443)

    def test_scope_rule_cannot_admit_link_local(self):
        with self.assertRaises(PolicyViolation):
            ScopeRule.parse("169.254.0.0/16", [80])

    def test_scope_rule_cannot_admit_metadata_host(self):
        with self.assertRaises(PolicyViolation):
            ScopeRule.parse("169.254.169.254/32", [80])

    def test_excluded_route_beats_include(self):
        sc = lab_scope(resolver=self.resolver)
        sc.excluded_routes = ["juice"]
        ok, reason = sc.check("juice.lab.local", 3000)
        self.assertFalse(ok)
        self.assertIn("excluded route", reason)

    def test_scope_check_never_raises_for_hostile_input(self):
        for host in ("", "..", "a" * 300, "127.0.0.1\n", "; rm -rf /"):
            ok, _ = self.scope.check(host, 80)
            self.assertFalse(ok)


class TestNoExecution(unittest.TestCase):
    """The scaffold must remain inert: nothing here may launch anything."""

    def test_no_subprocess_or_socket_calls_in_scaffold(self):
        here = Path(__file__).resolve().parent
        for name in ("policy.py", "runspec.py"):
            src = (here / name).read_text()
            for banned in ("subprocess", "os.system", "os.exec", "popen"):
                self.assertNotIn(banned, src, f"{name} references {banned}")

    def test_scope_module_only_resolves_never_connects(self):
        src = (Path(__file__).resolve().parent / "scope.py").read_text()
        self.assertNotIn("socket.create_connection", src)
        self.assertNotIn(".connect(", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
