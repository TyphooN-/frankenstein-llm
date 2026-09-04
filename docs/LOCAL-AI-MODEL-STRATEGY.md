# Local and frontier AI model strategy

Last updated: 2026-09-03

This document is the decision record for Hermes, local llama.cpp models, coding, creative generation, and authorized security research on `frankenstein`. The coverage-complete deployment roadmap is maintained in `docs/LOCAL-HERMES-CAPABILITY-COVERAGE-PLAN.md`.

The 2026-09-02 candidate sweep — Qwen3-Coder-Next, Gemma-4-12B Heretic, UI-Mate-9B, WeMM-Embedding-2B, FLUX.2-klein-4B and the models rejected alongside them — is recorded with pinned revisions and source links in `docs/CANDIDATE-MODEL-REVIEW-2026-09-02.md`. That review is the research snapshot. Current download, policy, runtime, and functional state is `docs/CANDIDATE-STATUS-2026-09-03.md`. Phase-three and phase-four artifacts are on disk; they are not all functionally qualified.

## Executive decision

Use a portfolio, not one model for every job:

1. Keep `openai-codex / gpt-5.6-sol` as the Hermes default and control-plane model.
2. Use Claude Code with Opus at Max effort as the primary repository executor while included Claude usage is available.
3. Use the local llama.cpp router for private, offline, unrestricted, creative, and fallback work.
4. Use ComfyUI as the execution layer for image and audio generation; text LLMs write prompts, lyrics, plans, and metadata.
5. For authorized security work, combine a strong frontier model, a cybersecurity agent scaffold, deterministic security tools, and a local unrestricted model. Do not equate “uncensored hacker” branding with demonstrated vulnerability-discovery skill.
6. Keep every local large model on demand behind `--models-max 1`; do not reserve VRAM for several models simultaneously.

## Hardware/runtime boundary

- ROCm0: RX 6900 XT, 16 GiB.
- ROCm1: Radeon Pro V620, 32 GiB.
- ROCm2: RX 6900 XT, 16 GiB.
- Aggregate VRAM: approximately 64 GiB, but frameworks do not automatically aggregate it. llama.cpp has a verified `1,2,1` split; ComfyUI/audio models must be tested with their own placement rules.
- RAM: approximately 94 GiB.
- llama.cpp: HIP/ROCm build `50f068f`.
- Router: `http://127.0.0.1:8080/v1`, loopback only, one resident model.
- Remote Hermes default remains `openai-codex / gpt-5.6-sol`.

## Coding model hierarchy

### 1. Main Hermes model: GPT-5.6 Sol

Keep `gpt-5.6-sol` as Hermes’s default.

Why:

- Excellent terminal, tool-use, research, and control-plane performance.
- Already integrated through the ChatGPT/Codex subscription.
- Strong fit for interpreting intent, supervising agents, integration, and final verification.
- Current terminal-agent evaluations place Sol and Claude Opus 5 extremely close; the winner changes with the benchmark and harness.

Caveat:

- A model being strong at terminal work does not make every autonomous action safe. Preserve approvals and explicit boundaries for destructive, production, credential, release, billing, funds, and consensus actions.

### 2. Primary repository executor: Claude Code Opus, Max effort

Use Claude Code Opus at Max for non-trivial repository discovery, implementation, testing, debugging, and self-review while included usage is available.

Why:

- Anthropic describes Opus 5 as its complex agentic-coding model and reports a 22% improvement over Opus 4.7 on its hardest agentic coding tasks.
- Current published comparisons generally favor Opus 5 on repository issue resolution, broad codebase-specific reasoning, hazard spotting, and several real-world software-engineering suites.
- GPT-5.6 Sol remains competitive or better on DeepSWE and terminal/tool coordination. The models are complements, not a reason to replace Hermes’s default.

Operational contract:

- Opus only; Max effort; fail closed rather than silently downgrading to Sonnet/Haiku.
- Claude edits/tests/reviews; Hermes controls ownership, hard approval boundaries, integration, push, and final reporting.

### 3. Best installed local coder: `heretic`

`heretic` is the best current installed local/private coding assistant:

- Qwen3.8-27B, Q6_K.
- 262K training context; router currently exposes 131K.
- Current-generation coding, agentic, office, and general reasoning base.
- Heretic/abliterated behavior reduces refusal friction.
- Native MTP in the selected GGUF.

Use it for:

- private source/code review;
- offline debugging and second opinions;
- unrestricted security analysis inside explicit authorization;
- drafting scripts and PoCs that are then tested in a controlled environment;
- work that should not leave the workstation.

Do not claim it replaces Opus 5 or GPT-5.6 Sol for long-horizon autonomous repository work. Local quantization, smaller capability, and weaker agent scaffolds remain material.

### 4. Local coder candidates not installed

### GLM-5.3 family assessment

#### Regular GLM-5.3 flagship

- Z.ai's current text flagship is `zai-org/GLM-5.3`: approximately 753B total / 39B active, 1M-token context, 128K maximum output, mandatory reasoning (`low`, `high`, or `max`), and text-only input.
- It is compelling for complex coding and authorized white-box security research. Z.ai reports 88.2 on Terminal Bench 2.1, 66.9 on DeepSWE v1.1, 84.5 on CyberGym, and 54.4 on ExploitBench. These are vendor-reported harness results and must not be treated as directly interchangeable with our local or other providers' runs.
- It cannot fit this workstation locally at useful fidelity. OrcaRouter's 4-bit MLX build is 427 GiB and even its aggressive 2-bit build is 300 GiB with a stated minimum near 340 GB RAM.
- Hosted list pricing checked 2026-08-30 is $1.40/M input, $0.26/M cached input, and $4.40/M output.

Decision: hosted comparison specialist for difficult code/security tasks, not a local model and not a replacement for Sol or Claude Code Opus without controlled task-level evidence.

#### Regular GLM-5.3-Flash — local-only assessment

- `zai-org/GLM-5.3-Flash` is the newest efficient GLM: 320B total / 18B active, 45 layers, 288 routed experts (top-8), native text/image/video input, MTP, 1M-token context, and 128K maximum output under MIT.
- Its hybrid linear+sparse attention lowers attention compute and KV-cache cost, but 18B active affects compute rather than the 320B-weight storage footprint. The source checkpoint is roughly 306 GiB in block FP8.
- Z.ai reports 84.3 on Terminal Bench 2.1, 63.4 on DeepSWE v1.1, and 48.8 on AutomationBench. Those vendor-reported results make the base architecture relevant to coding, agents, document work, and authorized-security experiments, but they do not predict performance after extreme quantization. The viable 6block IQ3_XXS artifact is a text-generation GGUF with no published vision projector, so do not count native image/video capability as available in this local route.
- Live capacity measured 2026-08-30: 94.16 GiB system RAM with 69.38 GiB available, plus 61.55 GiB aggregate VRAM with about 59.64 GiB free across RX 6900 XT 16 GiB + Radeon Pro V620 32 GiB + RX 6900 XT 16 GiB. Current usable combined headroom is about 129 GiB before runtime/context buffers; the theoretical physical total is about 155.7 GiB.
- `6block/GLM-5.3-Flash-GGUF` provides the most decision-useful published local measurements. Its imatrix-calibrated IQ3_XXS is 112 GiB at 3.09 BPW and measured 8.2485 perplexity versus 6.6974 for BF16. IQ2_XS falls sharply to 20.2651 perplexity and IQ1_M to 73.9234; IQ4_XS is 155 GiB and leaves no viable runtime headroom. The protected IQ3_XXS is therefore the only serious fit for this workstation.
- Unsloth's equivalent `UD-IQ3_XXS` shard total is 112.10 GiB. Its `UD-Q3_K_XL` is 137.40 GiB and `UD-IQ4_XS` is 146.05 GiB; neither is a robust active-desktop fit. Sub-3-bit builds fit more easily but lose too much quality for a 320B model intended to improve on the installed 27B models.
- The 112 GiB IQ3_XXS nominally fits only by combining host RAM and VRAM. Static placement attempted an oversized allocation on a 16 GiB GPU. The auto-fit attempt consumed roughly 66 GiB RSS and entered sustained reclaim pressure, but that run overlapped a 44-thread Linux kernel compile and is therefore inconclusive rather than an admission failure.
- Retry 32K auto-fit only on an otherwise idle host. Host retuning remains excluded: do not change ARC, swap, kernel, clocks, or persistent policy to make the model fit.
- The installed llama.cpp commit `50f068f` does not recognize `glm5next`. Support remains in experimental PR/fork code. Build and test it in an isolated checkout; do not replace or destabilize the working router binary. This model's fitting path also requires leaving `-ngl` unset initially, which conflicts with the router's global `gpu-layers = all` and requires a measured alias-specific split before integration.

Decision: defer regular GLM-5.3-Flash IQ3_XXS for one clean idle-host 32K retry on the current 96 GiB/three-GPU workstation. Keep 64K, A/B, and router integration blocked; add no alias. Requalify again after planned hardware upgrades. Do not use IQ1/IQ2 merely to make it fit.

#### OrcaRouter GLM-5.3-Flash-Uncensored — local-only assessment

- Exact source artifact: `orcarouter/GLM-5.3-Flash-Uncensored-FP8`, published 2026-08-29/30, gated on Hugging Face, MIT-tagged, 62 safetensor shards, and approximately 306 GiB. It cannot run on this workstation in source FP8 form.
- This is a weight-edited derivative of GLM-5.3-Flash, not the 753B flagship. OrcaRouter reports refusal reductions from 96% to 11% on MaliciousInstruct, 93% to 12% on JailbreakBench, 97% to 15% on AdvBench, and 93% to 18% on HarmBench, with XSTest benign over-refusal falling from 2.4% to 0.4%.
- OrcaRouter explicitly says standard capability retention is not yet established. Refusal-rate tests do not prove correct code, exploit discovery, tool use, prose quality, or parity with the regular model.
- On 2026-08-30 OrcaRouter stated that GGUF weights were expected in 1–2 days. No GGUF derived from the exact OrcaRouter checkpoint was available at the time checked. The correct local target is its future IQ3_XXS-class GGUF around 112 GiB—not FP8, AWQ/vLLM, MLX, or NVFP4/NVIDIA artifacts.
- A separate repository, `AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF`, quantizes `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`, not OrcaRouter's weights. Its current UD-Q3_K_XL shards total 137.40 GiB and are not a safe fit under the current 129 GiB usable pool; its roughly 186 GiB Q4 build cannot fit. Do not substitute it for the requested OrcaRouter comparison.

Decision: do not pursue an OrcaRouter IQ3_XXS-class derivative until the regular model completes the idle-host 32K fit gate. `heretic` remains the practical unrestricted local model. Reconsider only after hardware requalification and only if it fills a distinct workflow.

#### Wangzhang Qwen3.6-27B two-pass abliterated v2

The user-provided `mradermacher/Qwen3.6-27B-abliterated-v2-GGUF` name is potentially misleading: its declared source is `wangzhang/Qwen3.6-27B-abliterated`, not Triangle104. Wangzhang's checkpoint is a technically credible two-pass iterative Abliterix derivative of `Qwen/Qwen3.6-27B`, using rank-3 LoRA search, projected orthogonal ablation, and a second residual-direction peel. The publisher reports 100/100 base refusals reduced to 10/100 on a held-out judged set, 15/15 coherent hard-prompt responses in English and Chinese, cumulative next-token KL around 0.0242, and negligible response-length deviation. However, the exact upstream base revision was not recorded, the judge was external and lightly stochastic, and no broad post-ablation coding/agent benchmark was published.

The best comparison artifact is the weighted/imatrix Q6_K, not the static quant:

- repository: `mradermacher/Qwen3.6-27B-abliterated-v2-i1-GGUF`;
- file: `Qwen3.6-27B-abliterated-v2.i1-Q6_K.gguf`;
- size: 22,082,529,824 bytes (20.57 GiB);
- Hugging Face LFS SHA-256: `ed153e924c33e6792b44492b84be0ca953e780d5fce3ab68fb9dada166b9abb2`;
- license: Apache-2.0;
- source MTP head is reported untouched, but the GGUF must be inspected to prove the tensors and runtime acceptance;
- no mmproj is published in the checked static repository, so treat this GGUF as text-only despite the source VLM wrapper.

This model may fill a narrower role than `heretic`: an evidence-backed, clean Qwen3.6 coding/general checkpoint with substantially reduced refusal behavior and less fusion/specialization than `fable`. It is not presumptively smarter than the newer Qwen3.8-based `heretic`, and it may duplicate the aggressively unrestricted role of `obliterated`.

Decision: queue one Q6_K download and controlled A/B after the active GLM transfer/benchmark completes; do not contend for network, disk, or inference resources now. Promote it only if it materially beats `heretic` or `obliterated` on refusal-prone authorized-security/tool tasks while preserving code correctness, tool-call format, multilingual coherence, and long-context stability. Otherwise keep the existing five-model stack.

#### Qwen3-Coder-Next

- Strong dedicated coding-agent candidate, and the clearest upgrade available for the repository-agent lane: `Qwen/Qwen3-Coder-Next-GGUF`, revision `b82fb738`, Apache-2.0, 80B MoE with 3B active and 262K context, specifically post-trained for agentic coding and tool calls.
- Official Q4_K_M is roughly 48.4 GB; the Q3 variants are roughly 35-38 GB. Q3 fits GPU0+1; Q4 requires host spill once KV/cache overhead is counted, so it is a 48-64 GB-class model with much less runtime headroom than the installed 27B models.
- It is not an automatic upgrade over Qwen3.8-27B for every task, and it must not replace the Qwen2.5 Coder FIM model: low-latency completion and long-horizon agentic coding are different jobs.
- An uncensored derivative exists (Huihui abliterated, plus a Bartowski GGUF of it), but Huihui's own card describes the method as a crude, proof-of-concept refusal-direction removal with no KL or task-retention evidence.

Decision (updated 2026-09-02): the gap this model fills is now named, so it is promoted from "do not download" to the qualification backlog — **adopt the official model, keep Qwen2.5 Coder for FIM, and watch the abliterated variant only**. It still does not enter a download queue until per-file sizes and SHA-256 digests are collected independently and the Q3 fit is checked against KV overhead, and it is promoted over `heretic` only on local A/B evidence. Details and links: `docs/CANDIDATE-MODEL-REVIEW-2026-09-02.md`.

#### Devstral Small 2

- Strong compact agentic software-engineering model.
- Attractive if speed and smaller VRAM footprint matter more than maximum local general capability.
- TrustedSec’s local offensive-security experiment also found Devstral an effective speed/accuracy point on its limited suite.

Decision: benchmark only if a fast coding/security worker is needed; do not add merely to enlarge the picker.

#### Stock Qwen3.8-27B

- Useful as a control against Heretic modifications.
- It would duplicate most weights/capability already present.

Decision: no install unless a reproducible evaluation shows Heretic lost important coding correctness.

## Pliny OBLITERATUS catalog decision

A live 2026-08-31 inventory captured all 11 public `OBLITERATUS` repositories with zero collection errors. The grounded model-by-model assessment is preserved at `verification/OBLITERATUS-CATALOG-ASSESSMENT-2026-08-31.md`.

Portfolio decision:

- Keep and fully validate the already installed `Qwen3.8-27B-OBLITERATED` Q6_K plus its matching projector and MTP path. It is the strongest and broadest catalog representative.
- Do not download `Qwen3.6-27B-OBLITERATED`; the installed Qwen3.8 model directly supersedes it.
- Keep `Ornith-1.5-9B-OBLITERATED` Q6_K plus projector as the only near-term conditional download. Its distinct value would be a faster, lower-residency multimodal/browser/computer-use worker.
- Keep the structured-output Qwen3 4B as a conditional conversion candidate only if a dedicated low-latency JSON/tool router is proven useful.
- Keep Gemma 4 12B as a conditional cross-family reviewer only after Qwen3.8 multimodal validation.
- Reject the other seven for production admission: they are obsolete/duplicative, lack a publisher GGUF, fail their own quality/coherence signals, or do not fill a named workflow.

Publisher refusal and benchmark claims are not admission proof. Every retained candidate still requires local load, coherence, structured-output, workflow-quality, memory-safety, and clean-unload gates.

## Installed local model roles

| Alias | Primary role | Key tradeoff |
|---|---|---|
| `heretic` | Best local daily driver; private code/security/general work | Near-stock unrestricted Qwen3.8, but still below frontier agents |
| `obliterated` | Maximum refusal/deflection removal | Published capability cost versus stock |
| `ridge` | Compact and fast baseline | Lowest quality/fidelity of the installed set |
| `fable` | Intelligence/continuity-first unrestricted fantasy writer | Qwen3.6 rather than latest Qwen3.8; creative specialization |
| `phr00ty` | Voice/prose/roleplay-first unrestricted writer | 64K preset, no MTP; narrower specialist |

The router exposes only one resident model. Switching models incurs a first-prompt load delay and evicts the previous model.

## Image generation

### Correct division of labor

- Hermes text model: idea development, composition, prompt variants, negative prompts, shot lists, visual continuity notes, filename/metadata plans, and workflow parameter selection.
- ComfyUI: actual image generation, inpainting, img2img, ControlNet, upscaling, and batch execution.

Recommendation:

- Use local ComfyUI on Linux with ROCm.
- Choose the first checkpoint/workflow based on the target aesthetic rather than installing a random model zoo. FLUX/current ComfyUI-native image workflows are a sensible high-quality starting family.
- Stop the llama router before heavy ComfyUI jobs to return model VRAM:

```bash
systemctl --user stop llama-router.service
```

Restore it afterward:

```bash
systemctl --user start llama-router.service
```

No ComfyUI deployment has been performed yet. Installation should be a separate accepted task with a hardware probe, isolated Python environment, one known workflow, dependency audit, and an actual generated image.

## Music and audio generation

### First choice: ACE-Step 1.5 through ComfyUI

ACE-Step 1.5 is the best first deployment for this AMD workstation:

- full songs, vocals or instrumentals;
- lyrics, style, duration, and editing controls;
- native ComfyUI workflows;
- first-party AMD/ROCm support is explicitly documented;
- practical VRAM requirements fit each installed GPU class, though placement must be tested rather than assuming multi-GPU aggregation.

Recommended workflow:

1. Use `fable` for concept, lyrics, structure, meter, fantasy narrative, and explicit themes.
2. Use `phr00ty` for voice, scene texture, sensual tone, and character material.
3. Use `heretic` for concise ACE-Step tags, technical prompt structure, and iteration analysis.
4. Generate in ACE-Step/ComfyUI.
5. Compare several seeds/takes and revise the prompt/lyrics from actual audio evidence.

### Secondary options

- HeartMuLa: interesting open-source full-song model, but the primary documented accelerated path is CUDA-first. A small third-party ROCm port exists; ACE-Step has much stronger first-party AMD/ComfyUI evidence.
- Stable Audio/MusicGen/AudioGen: useful for instrumental beds, ambience, and sound effects; not the first full-song stack.

No audio stack has been installed yet.

## Authorized security, pentesting, red teaming, and HackerOne

### Scope boundary

Use these workflows only against systems you own, CTF/lab targets, or targets covered by explicit written authorization and the program’s current scope/rules. Preserve evidence, rate limits, data-handling rules, and disclosure requirements. A model’s willingness to answer is not authorization.

### The important finding: the system matters more than the label

Real security work requires:

- a capable model;
- a strong scaffold with terminal/browser/proxy/code tools;
- deterministic evidence from Semgrep, CodeQL, nuclei, scanners, tests, logs, and packet/application traces;
- target/program scope encoded as an enforceable boundary;
- hypothesis tracking and deduplication;
- reproducible validation and a human-reviewed report.

Cyber-branded fine-tunes often answer offensive questions readily but do not necessarily discover novel bugs, navigate large repositories, or validate exploitability better than modern general/coding frontier models.

### Best frontier choices

#### GPT-5.6 Sol

- Frontier Evals reports 87.2% on Cybench with trusted-cyber access.
- The same evaluation reports 90.6% refusals without trusted-cyber access.
- Therefore normal subscription access may be capability-gated for offensive tasks even when the underlying model is strong.
- Strong default for defensive review, code analysis, planning, and permitted tasks that do not trigger access controls.

#### Claude Opus 5 / Claude Code

- Best general choice for repository-scale vulnerability research, source tracing, patch review, and careful report construction.
- Strong coding and long-horizon reasoning, but policy refusal can still gate offensive execution.
- Use it for source-first detection and remediation even when another model handles unrestricted exploit brainstorming.

#### Alias Robotics CAI / alias models

- CAI is an open-source cybersecurity agent framework supporting many providers.
- Alias Robotics reports `alias3` at 85% Cybench pass@3 under its stated budget and harness.
- It also reports strong live-CTF and token-efficiency results.
- Treat vendor results as promising, not interchangeable with independent pass@1 results; harness, task subset, budget, and access tier materially change scores.
- CAI is more strategically interesting than adding several cyber GGUFs because it improves tool/scaffold discipline around whichever model is used.

Decision: evaluate CAI in a sandbox before any live bounty use. No paid alias-model access or commercial license will be enabled without explicit approval.

### BountyBench lesson

BountyBench measures detection, exploitation, and patching on reconstructed real bug-bounty systems. Its published leaderboard has historically favored Codex CLI and Claude Code agents. The key result is not a permanent model ranking: detection remains much harder than exploitation when the vulnerability is already identified. No model should be trusted to declare a target “clean.”

### Local security models

#### Existing `heretic` — first local choice

Use `heretic` first. It has a newer Qwen3.8 base than most public cyber fine-tunes, Q6 fidelity, long context, current coding/agentic ability, and unrestricted behavior.

#### HIDra 30B-A3B — controlled comparison candidate

Publisher claims:

- abliterated Qwen3-Coder-30B-A3B base;
- cybersecurity fine-tune;
- 30.5B total / approximately 3.3B active parameters;
- GGUF Q3/Q4/Q5/Q8;
- Apache 2.0;
- direct, no-thinking chat behavior.

Best candidate quant for this machine: Q5_K_M, 21,725,583,424 bytes, publisher LFS SHA-256 `1d1ef6d3de75eba0bee571897fde78174b2cbbb8d9e69012ccc11dbc7770a60f`.

Why not install automatically:

- no independent benchmark found proving it beats Qwen3.8 Heretic on vulnerability discovery or real bounty tasks;
- Qwen3-Coder base is older than Qwen3.8;
- training claims are publisher claims;
- adding 21.7 GB is justified only if it wins a controlled A/B.

Decision: keep as the top local cyber-specialist candidate, not yet installed.

#### Foundation-Sec 8B

Good fit for SOC triage, threat intelligence, ATT&CK mapping, and defensive workflows. It is not the best primary model for exploit discovery or repository-scale bounty work.

Decision: no install for the stated offensive/HackerOne goal.

#### WhiteRabbitNeo 33B

Historically important unrestricted cyber model, but its base and GGUF ecosystem are now old relative to Qwen3.8 and modern frontier agents.

Decision: reject as primary in 2026.

#### Trendyol Cybersecurity Qwen3 32B and other cyber-branded derivatives

Interesting but insufficient independent evidence. Synthetic-looking model-card examples and broad capability claims do not substitute for reproducible discovery/exploit/patch benchmarks.

Decision: reject until independently validated.

## Proposed security evaluation before another download

Run the same bounded suite through `heretic`, optionally HIDra, and one frontier model:

1. Source review: seeded vulnerable repositories with hidden ground truth; measure recall, precision, duplicate findings, and evidence quality.
2. Patch review: detect security regressions in diffs without being told the CWE.
3. Web lab: PortSwigger/Juice Shop targets with exact authorization; measure successful validated findings, false positives, requests, and time.
4. CTF mix: web, crypto, reversing, pwn, and forensics; record pass@1 and tool calls.
5. Report quality: reproducible steps, impact, root cause, minimal PoC, remediation, and no unsupported claims.
6. Refusal/access behavior: distinguish policy refusal from capability failure.
7. Resource efficiency: model-load time, tokens/second, context use, VRAM, and wall time.

Promotion rule: install or retain a specialist only if it materially beats `heretic` on at least one named workflow without unacceptable regression elsewhere.

## Recommended HackerOne workflow

1. Snapshot the program scope, exclusions, rate limits, safe-harbor language, and disclosure rules.
2. Create an isolated workspace per program/asset.
3. Perform passive/source-first recon and deterministic enumeration.
4. Use the model to generate ranked hypotheses tied to actual evidence.
5. Validate only in-scope hypotheses with bounded requests and logs.
6. Require a reproducible proof and eliminate duplicates/known issues.
7. Use a second model as critic for impact inflation, missing preconditions, and remediation accuracy.
8. Human-review the report before submission. Never let an agent submit reports or contact a program without explicit approval.

Model routing:

- repo-scale source investigation: Claude Code Opus 5 Max;
- terminal/tool-heavy investigation and orchestration: GPT-5.6 Sol where access permits;
- private unrestricted second opinion/PoC drafting: `heretic`;
- potential later cyber specialist: HIDra only after A/B proof;
- scaffold candidate: CAI in an isolated lab.

## Current verification state

- ZFS recovery scrub: 0B repaired, 0 errors, no known data errors.
- Three AMD GPUs detected, including Radeon Pro V620.
- `ridge`, `obliterated`, `heretic`: publisher hashes verified; direct router and Hermes alias smoke tests passed.
- `fable`: publisher hash verified; direct router and Hermes alias smoke tests passed.
- `phr00ty`: publisher SHA-256 verified; direct router inference passed; Hermes completed one API call through the alias at its 65,536-token preset.
- Packaged Hermes Desktop: launched and visibly listed all five configured local aliases under `LOCAL LLAMA.CPP ROUTER`.

## Sources

Primary/project sources:

- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
- Claude Opus 5 announcement: https://www.anthropic.com/news/claude-opus-5
- Claude Opus 5 system card: https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf
- Cybench: https://cybench.github.io/
- Frontier cyber evaluation: https://evals.frontier.security
- BountyBench repository: https://github.com/bountybench/bountybench
- CAI: https://github.com/aliasrobotics/CAI
- CAI benchmarking claims/method references: https://aliasrobotics.com/benchmarking.php
- HIDra model and publisher artifacts: https://huggingface.co/SAPSAN-SKLEP/HIDra-30B-A3B-GGUF-uncensored-cybersec
- GLM-5.3 official model: https://huggingface.co/zai-org/GLM-5.3
- GLM-5.3 launch and benchmarks: https://z.ai/blog/glm-5.3
- GLM-5.3-Flash official model: https://huggingface.co/zai-org/GLM-5.3-Flash
- GLM-5.3-Flash guide and pricing: https://docs.z.ai/guides/llm/glm-5.3-flash and https://docs.z.ai/guides/overview/pricing
- OrcaRouter uncensored Flash checkpoint: https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8
- Regular Flash GGUF quant evidence: https://unsloth.ai/docs/models/glm-5.3-flash
- TrustedSec local offensive-model experiment: https://trustedsec.com/blog/benchmarking-self-hosted-llms-for-offensive-security
- 2026-09-02 candidate sweep (pinned revisions and per-model source links): `docs/CANDIDATE-MODEL-REVIEW-2026-09-02.md`
- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5
- AMD ACE-Step/ROCm guidance: https://www.amd.com/en/blogs/2026/commercial-grade-ai-music-generation-on-amd-ryzen-ai-and-radeon-ace-step-1-5.html
- ComfyUI ACE-Step documentation: https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1

Interpretation warning: benchmark scores are not directly comparable when models use different agents, task subsets, access tiers, token/tool budgets, epochs, pass@k, or hidden guidance. This strategy uses them directionally and requires local A/B proof before new model promotion.
