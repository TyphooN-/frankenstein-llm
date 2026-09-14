# Frankenstein-LLM — Rust-First Hybrid Migration Brief for Hermes

**Repository:** `https://github.com/TyphooN-/frankenstein-llm`  
**Date:** 2026-09-11  
**Target branch:** `hybrid-rust-control-plane`

---

# Mission

Begin the hybrid migration **immediately**.

The architectural decision is:

> **Use Rust everywhere it makes engineering and performance sense. Keep Python only where the surrounding ML ecosystem makes Python the better boundary.**

This is a **Rust-first hybrid**, not a Python project with a few Rust helpers.

The priorities are:

1. **Maximum practical end-to-end performance**
2. **Reliability and fail-closed behavior**
3. **Strongly typed state and configuration**
4. **Low runtime overhead**
5. **Low process-startup and orchestration overhead**
6. **Preservation of existing evidence and safety guarantees**
7. **Incremental migration without breaking the working system**

`llama.cpp` remains the native inference engine. Do not rewrite inference kernels in Rust.

Python remains permitted where replacing it would mean fighting PyTorch, Transformers, publisher-supplied model code, ComfyUI, or another Python-native ML ecosystem for little or no measurable performance benefit.

The intended end state is **mostly Rust + upstream native C/C++ inference + small isolated Python ML workers**.

---

# 1. Start Now — Do Not Wait

Do not wait for current optimization work to finish.

First inspect the current repository state:

```bash
git status --short --branch
git log -10 --oneline --decorate
```

Never destroy or overwrite in-progress optimization work.

## If the working tree is clean

```bash
git switch -c hybrid-rust-control-plane
```

## If the working tree contains active work

Create a separate worktree immediately:

```bash
git worktree add ../frankenstein-llm-hybrid \
  -b hybrid-rust-control-plane \
  HEAD

cd ../frankenstein-llm-hybrid
```

Do **not** use destructive commands such as:

```text
git reset --hard
git clean -fd
git checkout -- .
```

unless the human operator explicitly requests them.

The hybrid work starts from the latest safe commit available now. Optimization commits can be merged/rebased/cherry-picked later.

---

# 2. Architectural Direction

The desired architecture is approximately:

```text
                       Hermes / user / services
                                |
                                v
                    +-----------------------+
                    |      frankenctl       |
                    |         Rust          |
                    +-----------+-----------+
                                |
               +----------------+----------------+
               |                |                |
               v                v                v
         configuration       runtime          qualification
         model catalog       lifecycle        supervision
         policy              GPU state        evidence
         placement           downloads        caching
         validation          process mgmt     reporting
         benchmarking        locking          admission
               |                |                |
               +----------------+----------------+
                                |
                +---------------+---------------+
                |                               |
                v                               v
       upstream native binaries          isolated Python workers
          llama-server                   only where justified
          llama-bench                    PyTorch / Transformers
          ROCm / HIP                     publisher model code
                                         ComfyUI
                                         Python-only ML stacks
```

The normal control path should not require a Python interpreter.

The latency-sensitive serving path should not depend on Python.

The high-throughput acquisition, hashing, state-management, policy, orchestration, and reporting paths should move to Rust where practical.

---

# 3. Repository Facts That Must Be Preserved

Before changing behavior, verify these facts against the live repository. They describe the current architecture at the start of this migration:

- The project is **AMD ROCm/HIP only**.
- `upstream/llama.cpp` is the pinned inference implementation.
- The router uses `llama-server`.
- The current stack uses explicit `ROCm<N>` device selectors.
- Model acquisition uses pinned queue manifests.
- Downloads currently enforce exclusive writers, resumable partials, exact sizes, SHA-256 where available, durable promotion, and completion stamps.
- Qualification is serialized and resumable.
- Missing evidence is not success.
- Functional gates and benchmark evidence are intentionally distinct.
- Runtime state/model weights remain outside Git.

Treat the repository itself and its tests/docs as the source of truth. If this brief conflicts with newer code, preserve the newer contract and document the discrepancy.

---

# 4. Language Policy

## Rust is the default for new non-ML implementation

For new code, choose Rust unless there is a specific reason Python is materially better.

Rust should be strongly preferred for:

- command-line tooling
- service orchestration
- process management
- process groups
- signal handling
- `/proc` inspection
- GPU telemetry
- GPU placement
- config parsing
- model catalogs
- manifest parsing
- admission policy
- deterministic validation
- download scheduling
- HTTP transfer coordination
- hashing
- resume state
- locks
- atomic filesystem publication
- evidence validation
- evidence aggregation
- qualification supervision
- status commands
- report generation
- benchmark orchestration
- router/sidecar lifecycle management
- system checks
- static repository checks
- deterministic functional scoring that does not require Python ML libraries
- JSON handling
- concurrent I/O
- persistent daemon/control-plane functionality

## Python is an exception boundary

Python is justified when the code fundamentally depends on a Python-native ML runtime or publisher implementation.

Examples:

- PyTorch model execution not already handled by `llama.cpp`
- Hugging Face Transformers model code
- `trust_remote_code` / publisher-supplied Python
- Python-only ASR/TTS implementations
- ComfyUI internals and workflows
- Python-only OCR/model libraries
- rapidly changing experimental ML code whose dependencies are Python-first
- model-specific code that would otherwise require reimplementing a large ML runtime

Do not keep code in Python merely because it is already written in Python.

Also do not port Python merely to achieve a language purity metric.

Every retained Python component should have a concrete justification.

---

# 5. No Python on the Fast Path

The desired steady-state request path is:

```text
client
  -> native server/router
  -> llama.cpp
  -> ROCm/HIP
```

Avoid:

```text
client
  -> Python orchestration
  -> Python HTTP proxy
  -> Python validation
  -> native inference
```

unless measurement proves that Python layer is necessary and insignificant.

A Python interpreter should not be started just to:

- parse config
- select a model
- build command arguments
- inspect GPUs
- check status
- proxy normal inference traffic
- parse JSON
- hash files
- supervise `llama-server`
- produce ordinary reports

Those belong in Rust.

---

# 6. Keep `llama.cpp` Native

Do not rewrite inference in Rust.

`llama.cpp` is already C/C++ and already handles the compute-heavy path.

Initially preserve the current process boundary:

```text
Rust control plane
    -> llama-server
    -> llama.cpp
    -> ROCm/HIP
```

This allows the Rust migration to proceed without coupling it to the internals of the pinned upstream engine.

## Later performance investigation: direct FFI

Because the project explicitly prioritizes ultimate performance, direct Rust-to-`llama.cpp` integration is **not forbidden**.

It should be evaluated later through measurement.

Create a future benchmark/ADR comparing:

```text
A. Rust -> llama-server over loopback HTTP
B. Rust -> llama.cpp C API via FFI
```

Measure:

- time to first token
- steady-state tokens/sec
- request serialization overhead
- CPU utilization
- memory footprint
- concurrency behavior
- model-switch latency
- cancellation latency
- operational complexity
- compatibility cost when updating the pinned upstream revision

Only adopt FFI if the measured end-to-end benefit justifies owning the tighter integration.

Do not assume FFI is faster enough to matter. Prove it.

---

# 7. Replace Shell/Python Orchestration with One Rust CLI

Create:

```text
frankenctl
```

The goal is for operational shell and Python wrappers to shrink over time into either:

- `frankenctl` subcommands, or
- tiny compatibility shims that invoke `frankenctl`

Possible eventual command tree:

```text
frankenctl
├── doctor
├── config
│   ├── check
│   └── show
├── model
│   ├── list
│   └── status
├── serve
├── router
│   ├── status
│   ├── start
│   ├── stop
│   └── reload
├── sidecar
├── gpu
│   ├── inspect
│   └── plan
├── download
├── qualify
├── qualification
│   ├── status
│   └── detail
├── bench
├── proof
└── report
```

Do not implement empty abstractions for every command on day one.

Implement real functionality incrementally.

---

# 8. Initial Rust Layout

Start simple.

```text
rust/
├── Cargo.toml
├── Cargo.lock
└── src/
    ├── main.rs
    ├── cli.rs
    ├── config.rs
    ├── error.rs
    ├── fs.rs
    ├── gpu.rs
    ├── process.rs
    ├── state.rs
    └── model.rs
```

Split into a workspace only after actual module boundaries justify it.

A likely later structure:

```text
rust/
├── Cargo.toml
└── crates/
    ├── frankenctl/
    ├── franken-core/
    ├── franken-state/
    ├── franken-process/
    ├── franken-gpu/
    ├── franken-download/
    └── franken-qualification/
```

Do not spend migration time creating architecture that has no implementation behind it.

---

# 9. Rust Dependency Policy

Use mature dependencies and keep the graph controlled.

Likely candidates:

```text
clap
serde
serde_json
thiserror
anyhow
tracing
tracing-subscriber
nix
libc
sha2
tempfile
```

For asynchronous/network-heavy work, evaluate:

```text
tokio
reqwest
hyper
```

Do not automatically make the entire codebase async.

Use async where it improves real concurrent network/process workloads.

For deterministic local calculations and simple CLI operations, synchronous Rust may be simpler and faster to reason about.

Do not introduce exotic allocators, async runtimes, lock-free structures, or unsafe code merely because they may sound faster.

Benchmark first.

---

# 10. Performance Build Profile

The project should have an explicitly benchmarked production profile.

Start from a conventional release build and test optimizations such as:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
```

Evaluate rather than blindly enable:

```text
panic = "abort"
strip
target-cpu=native
alternative allocators
PGO
BOLT
```

A host-specific `target-cpu=native` build may be appropriate for this single-host project, but record the tradeoff because the resulting binary may not be portable.

Never sacrifice required cleanup/durability semantics for a tiny synthetic speed gain.

---

# 11. Performance Measurement Rules

Performance is a first-class migration requirement, but measure the correct things.

Track separately:

## Inference performance

Dominated by `llama.cpp`/ROCm:

- prompt processing throughput
- generation tokens/sec
- TTFT
- model load time
- model switch time
- VRAM usage

## Control-plane performance

Affected by Rust migration:

- CLI cold-start latency
- config/catalog parse time
- qualification scheduling overhead
- evidence parse/validation time
- status-query latency
- download scheduler overhead
- SHA-256 throughput
- concurrent transfer throughput
- process-launch/supervision overhead
- report generation time
- memory footprint

## End-to-end performance

Measure:

```text
user request -> first useful response
```

not just isolated function speed.

Do not claim a Rust port improves inference throughput when the heavy work remains in `llama.cpp`.

---

# 12. Migration Priority

Use this order unless repository inspection reveals a dependency reason to change it.

## Phase 0 — Baseline and characterization

Before rewriting behavior:

1. inventory current Python and shell entry points
2. classify each as Rust-target / Python-bound / wrapper / obsolete
3. identify inputs and outputs
4. identify environment variables
5. identify filesystem effects
6. identify locks
7. identify child processes
8. identify exit codes
9. identify evidence schemas
10. identify tests
11. collect current performance baselines

Create:

```text
docs/RUST-HYBRID-MIGRATION.md
```

with a table:

```text
component | current language | target language | reason | contract | parity status | perf baseline
```

Do not let documentation block implementation.

---

## Phase 1 — `frankenctl` foundation

Create the Rust project and implement real shared primitives:

- CLI
- typed errors
- logging
- path types
- config loading
- atomic write helper
- lock helper
- child-process helper
- process-group helper
- signal helper
- JSON helpers

Required checks:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Add them to the repository's normal checks.

---

## Phase 2 — `doctor`, status, config, catalog

Port lightweight operational reads first.

Candidates include behavior currently spread through scripts such as:

```text
model_catalog.py
local_model_status.py
qualification_status.py
qualification_detail.py
gpu_vram.py
verify_router_models.py
```

Confirm actual current paths before changing code.

These should not need Python long term unless a specific dependency requires it.

Goals:

- very fast cold start
- typed parsing
- consistent errors
- one shared implementation
- no repeated Python interpreter startup

---

## Phase 3 — serving orchestration

Port the serving orchestration currently associated with:

```text
scripts/serve-model.py
serve-*.sh
systemd execution wrappers
```

Preserve:

- preset resolution
- model selection
- path checks
- environment
- ROCm selectors
- ports
- `llama-server` arguments
- failure semantics
- process replacement/supervision behavior

The result should be:

```bash
frankenctl serve <model>
```

Compatibility shell scripts may temporarily become tiny exec wrappers.

Do not put Python on the serving control path.

---

## Phase 4 — GPU telemetry and placement

Port:

```text
gpu_vram.py
gpu_placement.py
```

and related deterministic placement checks.

Separate:

```text
hardware observation
```

from:

```text
placement algorithm
```

so the planner can be tested from fixtures without physical GPUs.

Use domain types:

```rust
struct GpuId(u32);
struct Bytes(u64);
struct VramBytes(Bytes);
struct ModelBytes(Bytes);
struct LayerCount(u32);
```

Avoid raw untyped integers where units matter.

Add parity fixtures comparing Rust to the current implementation.

After parity, use Rust as the authoritative planner.

---

## Phase 5 — acquisition/download system

Port the downloader early because it combines performance, concurrency, and correctness.

Preserve all current semantics, including:

- one writer per queue/artifact
- collision detection after resolved-path normalization
- resumable `.partial` handling
- quarantine of invalid existing files
- exact byte-size validation
- SHA-256 validation
- concurrent independent files
- bounded total connection budget
- durable file flush
- atomic promotion
- parent-directory durability where required
- completion stamps
- prerequisite-phase validation

The first Rust implementation may continue invoking `aria2c` when that is the fastest proven transfer engine.

Then benchmark a fully native Rust range downloader against the current `aria2c` path.

Do not remove `aria2c` simply for purity if it remains faster.

Ultimate performance means choosing the fastest reliable implementation, not maximizing Rust percentage.

---

## Phase 6 — qualification supervisor

Port the **supervisor** before porting individual ML gates.

Rust should own:

- serialized scheduling
- resume
- locks
- child lifecycle
- process groups
- signal forwarding
- timeout enforcement
- `/proc` identity validation where required
- quiet-host/resource checks
- cache/state
- evidence validation
- terminal state publication

Model-specific Python gates remain subprocesses when justified.

Boundary:

```text
Rust supervisor
    -> launches Python ML gate only if necessary
    -> captures logs
    -> supervises process group
    -> enforces timeout
    -> validates exit status
    -> validates evidence
    -> publishes authoritative state
```

Python may produce candidate evidence.

Rust decides whether the evidence is trusted.

---

## Phase 7 — policy, evidence, reporting, deterministic gates

Port Python modules that are mostly:

```text
JSON
filesystem
hashing
policy
HTTP
string parsing
deterministic scoring
report generation
```

even if they are currently under `verification/`.

Examples worth evaluating include:

- candidate policy
- remote-code digest review
- prompt-corpus admission
- capability ledger/reporting
- proof layout
- static manifest checks
- structural RAG checks
- router response validation
- deterministic embedding/reranker result validation after the model response already exists

Keep a component Python only when the model/library execution itself genuinely requires Python.

This phase should substantially reduce the remaining Python surface.

---

## Phase 8 — Python gate minimization

Inventory every Python runtime that remains.

For each one, record:

```text
why Python is required
which Python package/runtime requires it
whether it is on a latency-sensitive path
whether a stable native alternative exists
expected benefit of porting
expected maintenance cost
```

If there is no strong justification, migrate it to Rust.

Expected legitimate survivors may include:

- Transformers remote-code models
- PyTorch-only models
- ComfyUI
- selected ASR/TTS/OCR stacks
- experimental ML code

Everything else should be challenged.

---

# 13. Strong State Modeling

Use Rust to eliminate invalid combinations.

Prefer:

```rust
enum QualificationState {
    Waiting(WaitingState),
    Running(RunningState),
    Passed(PassedState),
    Failed(FailedState),
    Stopped(StoppedState),
}
```

with:

```rust
struct PassedState {
    completed_at: Timestamp,
    evidence: ValidatedEvidence,
}
```

over:

```rust
struct State {
    status: String,
    evidence: Option<Evidence>,
    error: Option<String>,
}
```

A `Passed` value should structurally require valid evidence.

State transitions should be explicit and tested.

Example legal transitions:

```text
Waiting -> Running
Running -> Passed
Running -> Failed
Running -> Stopped
```

Reject illegal transitions.

---

# 14. Filesystem and Durability Rules

The current repository deliberately treats filesystem operations as part of correctness.

Preserve:

- exclusive ownership
- `flock` semantics where applicable
- safe resolved-path handling
- temporary sibling files
- exact verification before promotion
- `fsync` where durability is currently promised
- atomic rename
- parent-directory sync where required
- durable stamps only after durable artifacts

Use RAII for locks and file handles.

Never weaken durability to make a benchmark look faster.

If durability cost is material, benchmark it separately and optimize the implementation without changing the guarantee.

---

# 15. Python Worker Contract

Remaining Python should be isolated behind explicit process contracts.

Each Python worker/gate should define:

```text
executable/interpreter
arguments
environment
stdin contract
stdout contract
stderr contract
exit-code contract
evidence output path
evidence schema/version
timeout
temporary files
cleanup requirements
GPU visibility
```

Prefer process isolation over embedding CPython into Rust.

Advantages:

- crash isolation
- clean timeout enforcement
- clear ownership
- independent virtual environments
- no GIL in the Rust control plane
- no Python runtime in ordinary operations
- publisher code stays quarantined

Do not use PyO3 merely to say the system is integrated.

Use it only if a benchmarked workload clearly benefits from in-process Python.

---

# 16. Compatibility Strategy

Do not perform a flag-day rewrite.

For each subsystem:

1. characterize current behavior
2. add missing tests
3. implement Rust
4. run both implementations on identical fixtures
5. compare contractual results
6. benchmark both where performance matters
7. switch default to Rust after parity
8. retain Python briefly as a fallback/reference
9. remove fallback after confidence is high

Compare:

```text
exit code
stdout/stderr where contractual
JSON
filesystem state
permissions
locks
process behavior
subprocess arguments
environment
state transitions
evidence
performance
```

The goal is eventually **one implementation**, not permanent dual maintenance.

---

# 17. Fail-Closed Rules

These are non-negotiable.

Treat as failure/not-admitted:

- missing evidence
- malformed evidence
- unknown status
- invalid timestamp
- stale required state
- wrong digest
- wrong size
- unknown model alias
- unknown privilege policy
- interrupted qualification
- failed child process
- inability to measure a required cleanup property
- ambiguous process ownership
- lock acquisition failure when exclusive ownership is required

Never translate "unknown" into "probably okay".

---

# 18. Unsafe Rust Policy

Performance is not permission to spread `unsafe`.

Default to safe Rust.

`unsafe` is acceptable only when:

1. required for FFI or a demonstrated hot path
2. the safe alternative is insufficient
3. the boundary is tiny
4. invariants are documented directly above it
5. tests exercise the boundary
6. benchmarks justify it when performance is the reason

Centralize `unsafe` code into narrow modules.

Do not optimize away Rust's safety advantage without evidence.

---

# 19. Benchmark-Driven Optimization

After parity, profile before tuning.

Useful tools may include:

```text
hyperfine
perf
cargo flamegraph
strace
time
systemd-analyze
rocprof / ROCm profiling tools where relevant
```

Choose tools actually available on the host.

Potential optimization work after profiling:

- reduce repeated config parsing
- persistent control daemon if repeated CLI startup becomes measurable
- cache immutable manifest/catalog data safely
- mmap large read-only metadata where useful
- parallelize independent hashing/verification
- avoid unnecessary JSON reserialization
- reduce subprocess churn
- batch GPU sysfs reads
- remove redundant filesystem scans
- reuse HTTP connections
- reduce unnecessary copies
- native range-download scheduling
- FFI to `llama.cpp` only if loopback/server overhead is measurable

Never optimize based only on intuition.

---

# 20. Optional Future `frankend`

If repeated orchestration startup or repeated host inspection becomes measurable, evaluate a persistent Rust daemon:

```text
frankend
```

Potential responsibilities:

- cached immutable catalog/config
- live GPU telemetry
- model lifecycle
- sidecar lifecycle
- qualification queue
- download queue
- status/event stream
- lock ownership
- persistent process supervision

`frankenctl` would become a thin client.

Do not introduce a daemon before measurement shows it improves latency, observability, or lifecycle correctness.

A daemon is a possible performance optimization, not a prerequisite.

---

# 21. Shell Script Policy

Shell is acceptable for:

- tiny bootstrapping
- package installation glue
- compatibility wrappers
- commands that are literally a few stable shell operations

Shell should not remain the home of complex business logic.

If a shell script has:

- state machines
- parsing
- retries
- complicated branching
- process supervision
- JSON manipulation
- substantial validation

move that logic into Rust.

---

# 22. First-Day Work

Hermes should begin implementation today.

## A. Reconnaissance

Inspect current:

```text
scripts/
verification/
services/
config/
docs/reference/
llama-models.ini
upstream/llama-cpp.lock.json
pytest configuration
```

Produce the migration matrix.

## B. Create the Rust crate

Create the minimal real `frankenctl`.

Add:

```text
clap
serde
serde_json
thiserror
anyhow
tracing
tracing-subscriber
```

Add platform dependencies as required by real code.

## C. Implement `frankenctl doctor`

It must perform real repository/runtime checks derived from existing behavior.

Candidates:

- repository/config readability
- pinned llama.cpp binary presence
- upstream pin sanity
- ROCm sysfs visibility
- expected GPU inventory
- required runtime paths
- optional Python worker environments
- service/runtime prerequisites

Mandatory failures return non-zero.

## D. Port model/config/catalog reads

Move at least one real existing parser into typed Rust.

Add fixtures using current tracked configuration.

## E. Characterize serving

Capture the exact `llama-server` launch specification for representative models.

Implement Rust generation of the same command.

Do not switch production serving until parity is verified.

## F. Commit real progress

Suggested commits:

```text
rust: initialize frankenctl
rust: add typed repository configuration
rust: add doctor command
test: characterize llama-server launch plans
rust: implement serve launch planning
```

---

# 23. First Milestone Acceptance Criteria

The first milestone is complete when:

- `hybrid-rust-control-plane` exists
- current optimization work remains safe
- Rust builds successfully
- `cargo fmt --check` passes
- Clippy with warnings denied passes
- Rust tests pass
- `frankenctl doctor` performs real checks
- at least one existing config/catalog path is implemented in Rust
- serving behavior is characterized
- Rust can construct a parity-checked `llama-server` command
- no Python is introduced on the normal serving path
- existing Python functionality still works
- existing repository tests remain healthy
- baseline performance numbers are recorded

---

# 24. Second Milestone

Port GPU inspection and placement.

Acceptance:

- GPU observation is Rust
- placement math is Rust
- calculation is fixture-testable without GPUs
- parity with current decisions is demonstrated
- performance is measured
- Python implementation becomes fallback/reference only

---

# 25. Third Milestone

Port acquisition/download orchestration.

Acceptance:

- durable semantics preserved
- concurrency preserved or improved
- hash/size validation preserved
- interruption/resume tests pass
- transfer throughput is no worse than current implementation
- Rust implementation becomes default if it meets reliability/performance targets

Benchmark native Rust transfers against `aria2c` before deciding whether to remove the external engine.

---

# 26. Fourth Milestone

Port qualification supervision.

Acceptance:

- Rust owns process lifecycle and trusted state
- Python gates run only when their ML dependency justifies Python
- state transitions are typed
- timeouts/signals/process groups behave correctly
- resume/cache/evidence behavior matches current contracts
- failure tests pass

---

# 27. Fifth Milestone

Aggressively reduce non-ML Python.

Audit every `.py` file.

Classify:

```text
KEEP_PYTHON_ML
PORT_TO_RUST
COMPAT_WRAPPER
DELETE_OBSOLETE
```

For every `KEEP_PYTHON_ML`, add a short rationale to the migration document.

The final Python surface should be intentional, not historical.

---

# 28. Definition of the Target End State

The migration is mature when:

- Rust is the default operational/control language.
- Normal status/config/serve/download/qualification/report operations do not start Python.
- Python runs only for explicitly Python-bound ML workloads.
- `llama.cpp` remains responsible for inference.
- Rust supervises Python rather than depending on Python for supervision.
- Shell contains little or no complex logic.
- Evidence/state/durability behavior remains fail-closed.
- The full system is benchmarked end-to-end.
- Performance-sensitive choices are based on measurement.
- There are no permanent duplicate Python/Rust implementations without a documented reason.
- Direct `llama.cpp` FFI has been benchmarked against the current native-server boundary and decided by evidence.

The desired language hierarchy is:

```text
1. Rust for Frankenstein-owned production logic
2. C/C++ where upstream llama.cpp/ROCm already owns the optimized native path
3. Python only for ML ecosystem boundaries that genuinely justify it
4. Shell only for thin glue
```

---

# 29. Hermes Decision Rule

When deciding whether to port a component, ask in this order:

### Question 1
Does it fundamentally require a Python-only ML/library ecosystem?

- **Yes:** keep a narrow Python worker.
- **No:** continue.

### Question 2
Is it Frankenstein-owned production logic?

- **Yes:** prefer Rust.
- **No:** continue.

### Question 3
Is a mature external native tool already clearly better/faster?

- **Yes:** orchestrate that tool from Rust.
- **No:** implement in Rust.

### Question 4
Would replacing the external/native boundary improve performance?

- Benchmark it.
- Do not guess.

This rule should make Rust the dominant language naturally without rewriting high-performance upstream software for ideological reasons.

---

# 30. Final Instruction to Hermes

Proceed immediately.

Do not create a giant speculative rewrite plan and stop there.

Inspect the live repository, protect current work, establish the Rust branch/worktree, create `frankenctl`, implement a real command, add parity tests, and start moving production orchestration into Rust.

Optimize for:

```text
correctness
then parity
then measurement
then performance
then deletion of the old implementation
```

But treat performance as a core architectural requirement from the beginning.

The final system should feel native:

```text
Rust control plane
+ native llama.cpp inference
+ ROCm/HIP
+ isolated Python only where the ML ecosystem requires it
```

That is the target.
