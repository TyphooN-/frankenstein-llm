# Authorized Security-Agent Stack Assessment

Date: 2026-09-01
Scope: PurpleAILAB/Decepticon, Armur-Ai/Pentest-Swarm-AI, usestrix/strix
Policy: authorized testing, HackerOne/bug-bounty, CTF/lab, and owned systems only

## Executive decision

| Candidate | Decision | Place in the stack |
|---|---|---|
| Strix | ADMIT AS A HARDENED, ISOLATED PILOT | Primary application-security/white-box/black-box autonomous testing lane, after local-model and containment gates pass |
| Pentest-Swarm-AI | PILOT ISOLATED, LAB ONLY; DO NOT ADMIT | Evaluate its sequential runner, stigmergic swarm, and bug-bounty scope import in a disposable lab only; it does not justify a second operational pentest stack yet |
| Decepticon | PILOT ISOLATED ONLY IF ITS UNIQUE BREADTH IS NEEDED; DO NOT INSTALL ON THE MAIN WORKSTATION | Its bundled Hermes MCP bridge is the correct integration seam, but only from a dedicated VM/host appliance after privacy and isolation blockers are resolved |

Do not run two autonomous offensive frameworks against the same target concurrently. They will duplicate traffic, defeat rate/deconfliction assumptions, and make audit evidence ambiguous.

## Evidence snapshot

- Decepticon: commit `31e1c8e786c83bb20f5c3d9ebc482cf9fb8ffa06`; Apache-2.0; latest release v1.1.40; active; current head had 30 reported checks, with security scanning and core CI successful.
- Pentest-Swarm-AI: commit `e57c300addf7b8d9d33386268cae3b27705af9c8`; AGPL-3.0; latest release v0.1.0; README labels swarm alpha. `go test ./...` passed locally on 2026-09-01. Current GitHub head showed successful Test/Build/Docker checks but failed Lint and build-and-push checks.
- Strix: commit `608ef4a37b710dbeb1a4f724b180cf75b1d78fc9`; Apache-2.0; latest release v1.5.3; very active. The repository has a release workflow but no general test/lint workflow under `.github/workflows` at this commit.
- No candidate was installed. Source was shallow-cloned to `/tmp/hermes-verify-security-agents/` for review.

## 1. Strix

### What it adds

Strix is the best match for a practical, bounded workflow that Hermes does not already provide as a cohesive service:

- source-aware application security review plus dynamic validation;
- web/API testing using Caido, browser automation, terminal, Python PoCs, Nuclei and related tooling;
- multi-agent orchestration and evidence-backed findings;
- local run artifacts, Markdown/JSON/SARIF output, headless CI mode, diff-scoped review;
- local OpenAI-compatible endpoints through LiteLLM using `LLM_API_BASE`;
- MCP extension points and a local viewer.

This is narrower and more operationally useful than deploying a full red-team/C2 platform. It complements Hermes repository automation rather than replacing it.

### Local-model fit

Technically compatible with the local router:

- `STRIX_LLM` selects a LiteLLM model identifier.
- `LLM_API_BASE` binds a custom OpenAI-compatible endpoint.
- Streaming can be disabled for gateways that mishandle structured deltas.
- The framework caps tool calls per turn and repairs duplicate tool-call IDs.

Compatibility is not admission. The chosen local model must still pass a Strix-specific lab gate for multi-turn native tool calls, result consumption, context compaction, coherent target scoping, and completion of an OWASP lab task. The existing `heretic` tool-use lane is the first candidate; GLM is not eligible unless its own admission gates pass.

Upstream's local-provider guidance warns that most sub-70B models struggle with this workload and sub-30B models frequently produce malformed tool calls. That is guidance, not proof; exact local-model acceptance testing remains mandatory.

### Security and privacy findings

Strengths:

- Commands execute in a per-session Docker sandbox rather than directly on the host.
- Exposed container ports bind to 127.0.0.1 by default.
- Container logs are bounded by default.
- Source mounts and run artifacts are explicit.
- OpenAI Agents tracing is disabled.

Blockers/defaults requiring override:

1. Telemetry defaults **on** (`TelemetrySettings.enabled = True`) and sends PostHog/Scarf events. It does not send raw findings in the reviewed event schema, but private-stack policy requires `STRIX_TELEMETRY=false` plus outbound verification.
2. Sandbox memory, CPU and PID limits are opt-in. They must be mandatory.
3. The sandbox receives `NET_ADMIN` and `NET_RAW`, and `host.docker.internal` maps to the host gateway.
4. Local source trees are bind-mounted read-write by default.
5. Docker containers do not provide a hard destination allowlist. Caido has scope rules, but arbitrary `exec_command` with curl/httpx can bypass proxy scope. Network enforcement must therefore live outside the agent/container.
6. Some manifests may add `SYS_ADMIN`, `/dev/fuse`, and `apparmor:unconfined`; those modes are prohibited in the pilot.
7. A custom MCP server can expose arbitrary capabilities. The pilot must use no MCP servers unless individually allowlisted.
8. The sandbox's internal `pentester` account has passwordless sudo, and an open issue reports a resume-path mount-guard bypass. Resume must remain disabled until that path is fixed or independently contained.
9. The published sandbox build uses moving inputs including `kali-rolling:latest` and several Go tools at `@latest`; use a rebuilt, internally pinned image rather than trusting a floating upstream rebuild.

### Required pilot profile

- Run only against OWASP Juice Shop/WebGoat/DVWA or an explicitly authorized staging target.
- Dedicated Docker network and a separate egress firewall/proxy that permits only resolved in-scope IPs/ports plus the loopback local LLM endpoint.
- `STRIX_TELEMETRY=false`; block PostHog, Scarf and `app.strix.ai` at the network layer.
- No cloud LLM/search/provider credentials. Local router only.
- Mandatory cgroup caps: memory, CPUs, PIDs, logs and wall-clock deadline.
- Read-only source mount for review; use a disposable copy if patch generation needs writes.
- No Docker socket inside the sandbox; no host home, credential stores, SSH agent, browser profile, `/run`, `/proc` host namespace, or Hermes config mounts.
- Refuse any manifest requiring SYS_ADMIN/FUSE/AppArmor disablement.
- No remote MCP; local MCP tools must be explicitly allowlisted.
- Disable resume until the mount-guard issue is closed and verified.
- Written target allowlist, excluded routes, rate limits, test accounts, destructive-action prohibitions and stop conditions passed in an instruction file.
- Retain `run.json`, SARIF, reports, exact image digest, model alias/revision and network/firewall logs.

Verdict: **belongs as the single primary isolated application-security pilot**, not as a trusted always-on daemon.

## 2. Pentest-Swarm-AI

### What it adds

Its distinct idea is a stigmergic blackboard: independent agents react to weighted findings rather than following only a fixed planner pipeline. It also has useful bug-bounty-oriented scope importers, evidence/report tooling, cleanup registration, rate limiting and assist-mode confirmations.

It supports Ollama, LM Studio, and arbitrary OpenAI-compatible endpoints without requiring cloud auth. It can fall back to JSON-in-prompt when native tool calling is unreliable.

### Security and maturity findings

Strengths:

- Scope validation runs before exploit execution.
- Shell metacharacters are rejected by the parser.
- An executable allowlist can block shell/interpreter bridges.
- Optional safe mode blocks destructive tokens.
- Assist mode can require confirmation for every executed step.
- Shared-intelligence export is opt-in; configured intelligence sharing defaults off.
- Current source passed `go test ./...` locally.

Blockers:

1. Native command execution uses `exec.CommandContext` on the process host. The roadmap explicitly says per-command Docker sandboxing is deferred. Running the whole binary in Docker improves containment, but this is not the same as a purpose-built per-engagement execution boundary.
2. Safe mode and human confirmation are optional, not invariant.
3. The executable allowlist has an intentional disabled state when empty; every construction path must be proven to wire it.
4. `ValidateCommand` is regex-based and exempts several domains. It cannot be the only network boundary and may miss encoded, resolved, indirect or tool-file targets.
5. Default API service config binds `0.0.0.0`, leaves API auth optional, and permits CORS `*`.
6. The advertised “real swarm” mode is explicitly alpha; several dashboard, benchmark, model and adapter claims remain planned.
7. AGPL-3.0 is acceptable for private internal use but creates source-offer obligations if a modified network service is offered to others. It is less convenient for embedding into Hermes/Desktop than Apache-2.0.
8. Current head's GitHub lint and image publishing checks were failing even though builds/tests passed.
9. The only release, v0.1.0, predates a recon parsing fix for a defect that could report zero findings/low risk incorrectly. A pilot must build a reviewed current commit, not use v0.1.0 or a floating `latest` image.
10. Its MCP `scan_target` surface does not expose safe mode, assist, dry-run, an authorization artifact, or a policy-bound target registry. It must not be connected directly to Hermes.
11. Several image dependencies use moving `latest`/`master` inputs, and the Dockerfile labels the image Apache-2.0 despite the repository being AGPL-3.0.

### Stack fit

Pentest-Swarm overlaps heavily with Strix on recon, web tooling, exploitation and reports. Its blackboard/swarm scheduler is interesting, but not enough to justify another operational agent with its own model loops, traffic, state and toolchain before it proves superior on a named workflow.

Verdict: **does not belong in the operational stack now**. Keep it as a pinned lab-only candidate. Reconsider if its Docker execution sandbox ships and a controlled A/B shows the swarm finds a meaningful authorized chain that Strix/Hermes misses with comparable scope discipline.

Acceptable lab pilot: build a reviewed current commit; run the sequential runner first in a disposable non-root VM/container; force `--safe-mode --assist --strict`; use a local endpoint only; enforce authorization and egress outside the framework; do not expose its MCP server; treat every report as an unverified lead.

## 3. Decepticon

### What it adds

Decepticon is broader than the other two: engagement planning/RoE artifacts, persistent interactive tmux sessions, post-exploitation, AD, cloud, smart contracts, reversing, mobile/IoT/ICS, C2 integrations, Neo4j attack graphs, and many specialist agents. It has the strongest current CI/security-check posture of the three and claims strong XBOW validation-benchmark results.

It also ships an explicit external-agent integration in which Hermes controls a separate Decepticon LangGraph service through an MCP bridge and bundled skill. This is the only acceptable integration seam: Hermes remains outside the appliance and receives bounded artifacts; Decepticon's runtime is never imported into the trusted Hermes host.

It explicitly supports a local llama.cpp OpenAI-compatible server through LiteLLM (`LLAMACPP_API_BASE` and `LLAMACPP_MODEL`), so local inference is technically possible. A single local model is used across tiers.

### Security and privacy findings

Positive controls:

- RoE middleware, audit ledger, command scope checks and sandbox nftables allowlisting.
- Separate management and operational Docker networks.
- Prompt-injection wrapping and restricted specialist workload catalog.
- Container caps and `no-new-privileges` are documented.
- Broad CI includes CodeQL, Semgrep, Trivy, secret scanning and tests.

Disqualifying current-host concerns:

1. It is a large multi-service control plane: LiteLLM, PostgreSQL, Neo4j, LangGraph, Kali sandbox, dashboard, dynamic specialist services and optionally C2/reversing infrastructure.
2. The current OSS design uses one shared root sandbox per host. `/tmp`, `/var/log`, `/root`, tmux state and process namespace can cross engagement boundaries. Per-engagement containers are still a design/future-hardening item.
3. The threat model records open/partial controls around dashboard authentication, shared Neo4j identity, plugin startup code, sandbox authentication and kernel escape.
4. Provider API/OAuth material is centralized in LiteLLM; OAuth files may be bind-mounted. This is unnecessary risk for a local-only deployment.
5. Telemetry policy is internally contradictory. `TELEMETRY.md` says default-off/opt-in, but current onboarding code initializes consent to true, calls it opt-out, and maps the default affirmative path to `research`, which sends masked red-team reasoning. This alone blocks private-stack admission without a patch and network denial.
6. The breadth (C2, AD, post-exploitation, cloud and interactive shells) creates much more blast radius than our immediate HackerOne/appsec need.
7. It substantially duplicates Strix and Hermes orchestration while adding PostgreSQL/Neo4j/LiteLLM/Docker lifecycle debt.
8. RoE machine enforcement defaults to `audit`, not `enforce`; in audit mode out-of-scope calls are logged rather than blocked. Any pilot must force `machine_enforcement.mode: enforce` and independently verify the nftables policy.
9. The latest release v1.1.40 predates important local-Ollama tool-call and Autohunt safety changes. A pilot cannot blindly use either the stale release or moving `main`; it needs a reviewed pinned commit and image digests.
10. The compose footprint includes roughly 13 services, and some control paths can access the host Docker socket. This reinforces the dedicated-VM requirement.

### Stack fit

Decepticon may eventually deserve a **dedicated security appliance VM/host** for broad red-team engagements, especially AD/internal-network/C2 work. It should not share the primary workstation, Hermes credential stores, production local-model router, or Docker daemon.

Revisit only when:

- per-engagement sandbox containers are implemented and verified;
- dashboard auth is mandatory and loopback-only by default;
- telemetry is unambiguously default-off and outbound-blocked;
- local llama.cpp-only operation works with no OAuth/API mounts;
- every image is digest-pinned and SBOM/vulnerability-reviewed;
- a dedicated VM/host has separate Docker, network namespace, storage dataset and disposable secrets;
- its unique AD/C2 workflow is actually needed and authorized.
- RoE is forced to enforcement mode and verified with deliberate out-of-scope negative tests.

Verdict: **does not belong on the current stack/host**. Keep the repository on a research watchlist; do not install it now.

If a unique AD/internal/C2 need arises, its status is **isolated pilot**, not permanent rejection: run a dedicated VM or Docker host/VLAN, expose only a loopback-authenticated Decepticon MCP bridge to Hermes, start with the interview-first Soundwave path, disable telemetry and every cloud fallback, omit C2/phishing/AD profiles unless the engagement specifically authorizes them, and pin an audited commit/image digest.

## Recommended stack architecture

1. Hermes remains the control plane and approval/audit boundary.
2. Strix is the only admitted autonomous application-security executor, launched on demand in a hardened disposable environment.
3. Established deterministic scanners (Nuclei, Semgrep, CodeQL, dependency/secret scanners) remain first-line; autonomous exploitation validates selected findings rather than replacing them.
4. Pentest-Swarm remains a lab A/B candidate for swarm-specific research.
5. Decepticon, if ever needed, runs as a separate appliance with no trust path to Hermes credentials or the main Docker daemon.
6. Findings from any agent are untrusted until reproduced, scope-checked, and reviewed. An empty or budget-truncated run is never evidence that a target is clean.

## Primary sources

- https://github.com/PurpleAILAB/Decepticon
- https://github.com/PurpleAILAB/Decepticon/blob/31e1c8e786c83bb20f5c3d9ebc482cf9fb8ffa06/docs/models.md
- https://github.com/PurpleAILAB/Decepticon/blob/31e1c8e786c83bb20f5c3d9ebc482cf9fb8ffa06/docs/security/sandbox-isolation.md
- https://github.com/PurpleAILAB/Decepticon/blob/31e1c8e786c83bb20f5c3d9ebc482cf9fb8ffa06/docs/security/decepticon-threat-model.md
- https://github.com/PurpleAILAB/Decepticon/blob/31e1c8e786c83bb20f5c3d9ebc482cf9fb8ffa06/TELEMETRY.md
- https://github.com/PurpleAILAB/Decepticon/blob/31e1c8e786c83bb20f5c3d9ebc482cf9fb8ffa06/clients/launcher/cmd/onboard.go
- https://github.com/PurpleAILAB/Decepticon/blob/main/docs/integrations/external-agents.md
- https://github.com/PurpleAILAB/Decepticon/blob/main/docs/security/roe-machine-enforcement.md
- https://github.com/Armur-Ai/Pentest-Swarm-AI
- https://github.com/Armur-Ai/Pentest-Swarm-AI/blob/e57c300addf7b8d9d33386268cae3b27705af9c8/internal/agent/exploit/executor.go
- https://github.com/Armur-Ai/Pentest-Swarm-AI/blob/e57c300addf7b8d9d33386268cae3b27705af9c8/internal/scope/validator.go
- https://github.com/Armur-Ai/Pentest-Swarm-AI/blob/e57c300addf7b8d9d33386268cae3b27705af9c8/config.example.yaml
- https://github.com/Armur-Ai/Pentest-Swarm-AI/blob/main/internal/mcp/tools.go
- https://github.com/Armur-Ai/Pentest-Swarm-AI/issues/66
- https://github.com/usestrix/strix
- https://github.com/usestrix/strix/blob/608ef4a37b710dbeb1a4f724b180cf75b1d78fc9/strix/runtime/docker_client.py
- https://github.com/usestrix/strix/blob/608ef4a37b710dbeb1a4f724b180cf75b1d78fc9/strix/config/settings.py
- https://github.com/usestrix/strix/blob/608ef4a37b710dbeb1a4f724b180cf75b1d78fc9/strix/telemetry/posthog.py
- https://github.com/usestrix/strix/issues/1214
- https://docs.strix.ai/llm-providers/local
