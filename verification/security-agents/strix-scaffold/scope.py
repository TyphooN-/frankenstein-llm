#!/usr/bin/env python3
"""Deny-by-default destination allowlist for the Strix pilot.

The assessment's finding 5 is the reason this file exists: Caido scope rules live
*inside* the agent, and an agent that can run ``exec_command`` can simply curl
past them. So scope is enforced here, outside the agent, on the only thing the
agent cannot rewrite -- the set of destinations its network namespace can reach.

Everything is denied unless a rule admits it. A hostname is only in scope if it
resolves entirely into admitted IP space, which is what stops a DNS rebind or a
CNAME onto a third party from quietly widening the engagement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import socket

from policy import PolicyViolation

# Destinations that are never in scope even if an operator lists them: cloud
# instance metadata and link-local space are the classic pivot from "we tested a
# web app" to "we exfiltrated the host's credentials".
NEVER_ALLOWED_NETWORKS: tuple[str, ...] = (
    "169.254.0.0/16",     # link-local + IMDS 169.254.169.254
    "fe80::/10",
    "100.64.0.0/10",      # CGNAT / tailnet
    "224.0.0.0/4",        # multicast
    "255.255.255.255/32",
)


@dataclass(frozen=True)
class Destination:
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class ScopeRule:
    """One admitted destination: a CIDR (or single IP) plus allowed ports."""
    network: ipaddress.IPv4Network | ipaddress.IPv6Network
    ports: frozenset[int]
    note: str = ""

    @classmethod
    def parse(cls, cidr: str, ports: list[int], note: str = "") -> "ScopeRule":
        net = ipaddress.ip_network(cidr, strict=False)
        for banned in NEVER_ALLOWED_NETWORKS:
            b = ipaddress.ip_network(banned)
            if net.version == b.version and net.overlaps(b):
                raise PolicyViolation(
                    "scope.never-allowed",
                    f"{cidr} overlaps permanently excluded network {banned}")
        return cls(network=net, ports=frozenset(ports), note=note)


@dataclass
class Scope:
    """A written engagement scope. Deny-by-default; excludes beat includes."""
    rules: list[ScopeRule] = field(default_factory=list)
    excluded_routes: list[str] = field(default_factory=list)
    resolver: object = None      # injectable for tests; defaults to real DNS

    def _resolve(self, host: str) -> list[str]:
        try:
            ipaddress.ip_address(host)
            return [host]
        except ValueError:
            pass
        if self.resolver is not None:
            return list(self.resolver(host))
        infos = socket.getaddrinfo(host, None)
        return sorted({i[4][0] for i in infos})

    def _ip_allowed(self, ip: str, port: int) -> bool:
        addr = ipaddress.ip_address(ip)
        for banned in NEVER_ALLOWED_NETWORKS:
            b = ipaddress.ip_network(banned)
            if addr.version == b.version and addr in b:
                return False
        return any(addr in r.network and port in r.ports for r in self.rules)

    def check(self, host: str, port: int) -> tuple[bool, str]:
        """Return (allowed, reason). Never raises on an out-of-scope target."""
        if not self.rules:
            return False, "empty scope: deny-by-default"
        for route in self.excluded_routes:
            if route and route in host:
                return False, f"host matches excluded route {route!r}"
        try:
            addrs = self._resolve(host)
        except Exception as exc:                       # noqa: BLE001
            return False, f"resolution failed: {exc.__class__.__name__}"
        if not addrs:
            return False, "resolution returned no addresses"
        # Every resolved address must be admitted. One stray answer is enough to
        # make the destination out of scope.
        bad = [a for a in addrs if not self._ip_allowed(a, port)]
        if bad:
            return False, f"resolves to out-of-scope address(es) {bad}"
        return True, f"in scope via {addrs}"

    def assert_allowed(self, host: str, port: int) -> None:
        ok, reason = self.check(host, port)
        if not ok:
            raise PolicyViolation("scope.denied", f"{host}:{port} -> {reason}")

    def as_dict(self) -> dict:
        return {
            "rules": [
                {"network": str(r.network), "ports": sorted(r.ports), "note": r.note}
                for r in self.rules
            ],
            "excluded_routes": list(self.excluded_routes),
            "never_allowed_networks": list(NEVER_ALLOWED_NETWORKS),
            "default": "deny",
        }


def lab_scope(resolver=None) -> Scope:
    """The only scope the pilot may use until a written engagement replaces it:
    loopback OWASP lab ports plus the loopback local model router."""
    return Scope(
        rules=[
            ScopeRule.parse("127.0.0.1/32", [3000, 8081, 8082],
                            note="OWASP Juice Shop / WebGoat / DVWA on loopback"),
            ScopeRule.parse("127.0.0.1/32", [8080], note="local model router"),
        ],
        excluded_routes=[],
        resolver=resolver,
    )
