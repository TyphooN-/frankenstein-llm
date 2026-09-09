# Repository checks

Undated operating note. These checks do not load a model, start a service, touch
a GPU, download anything, or measure throughput. The repository-agent tests do
create private Linux namespaces through Bubblewrap, so run the complete suite
only after active builds and other host pressure have drained; a busy or
RCU-stalled host is not a valid sandbox test environment.

None of them is a functional verdict. Every one of them answers "is this
workspace internally consistent", and consistency is what makes a live gate
worth running -- not a substitute for one. Capability claims come only from the
serialized functional gates and their evidence artifacts.

## Offline test suite

```
python3 -m pytest
```

`pytest.ini` scopes collection to `verification/`. It excludes the untracked
upstream ComfyUI checkout under `tools/`, the local virtualenvs, the gate
evidence directories, and `verification/repository-agent/fixture/` -- whose
three failing tests are the *oracle* for the repository-agent gate and are
defective on purpose. The gate copies that fixture to a temporary directory and
runs it with `unittest` inside Bubblewrap. It is unaffected by this pytest
configuration.

The sandbox is the whole reason that gate is safe to run: `run_tests` executes
candidate-written Python, and import-time code in a "repair" runs before any
assertion does. It gets a read-only `/usr`, a private PID namespace, a private
network namespace so the router on host loopback is unreachable, a size-capped
private `/tmp`, all capabilities dropped, a cleared environment, no ability to
nest a further user namespace, and one writable bind: the disposable fixture.

Two rules make that fail closed rather than best-effort. There is no unisolated
fallback -- if `bwrap` is missing the gate refuses to start. And a sandbox that
cannot start, cannot be executed, or does not finish aborts the qualification
instead of returning a retryable tool error, because a host-side failure
reported as a red suite is a false verdict about the candidate.

The oracle is read-only for the same reason: "make the tests pass" is trivially
satisfiable by deleting the assertions. `run_tests` re-checks the suite's
SHA-256 after every run, a rewritten or removed suite is recorded as tampering,
and a completed `write_file` invalidates an earlier green result -- a passing
run describes one exact generation of the workspace and nothing later.

## Pinned llama.cpp submodule

```
python3 -m pytest verification/upstream-pin/test_llama_cpp_pin.py
```

Which llama.cpp revision this stack runs is written down three times -- the outer
repository's gitlink, `upstream/llama-cpp.lock.json`, and the prose in ADR 0005 --
and only the first of those is what a fresh clone actually checks out. The gitlink
records whatever the submodule worktree happened to be sitting on at `git add`
time, so staging it after an upstream fetch pins that revision for everyone else
while the lock, the documents, and every locally built binary still say v0.4.0.
Nothing reports it: the build script compares the *worktree* `HEAD` against the
lock and is satisfied, because the worktree is right and the index is not. This
suite compares the staged gitlink against the lock directly.

It also holds the build entry point to the lock -- backend, `gfx1030`, Ninja,
Release, the four required targets, and job count discovered through `nproc`
rather than a number that outlives the machine it was measured on -- and checks
that no tracked file still executes the removed `/home/typhoon/src` checkout or
one of the `~/.local/bin` shims that now dangle. Documentation stays free to name
those paths in order to forbid them; prose is exempt and fenced command blocks
are not.

The one part that is not purely static runs `llama-server --help` -- no model, no
GPU, no server -- and checks every key in `llama-models.ini` against the option
list that build prints. llama.cpp loads `--models-preset` with unknown keys
fatal, so a key a release drops does not degrade one alias: `load_from_ini`
throws and the router never finishes starting. Those two cases skip rather than
fail when the submodule has not been built yet.

## Qualification supervisor contracts

```
python3 -m pytest verification/qualification-supervisor/test_qualification_supervisor.py
```

The supervisor decides when a serialized gate may start, and that decision is
made from `/proc`. It never reads `/proc/<pid>/cmdline`: Linux serves that file
through `access_remote_vm`, so a task holding its own mmap write lock can wedge
anything that merely tries to inspect it -- which is exactly how a
command-line sweep of the host gets stuck behind a browser. Only `comm`, the
`cwd` symlink and `cgroup` are read, all three tolerate the task exiting
underneath them, and the proc root is injectable so the classification is tested
against fixtures rather than against whatever the host happens to be running.

Classification is deliberately coarse and errs towards reporting: a build, a
transfer or a second model server is named on the strength of its task name
alone. `comm` is truncated to 15 characters by the kernel, so long names are
matched the way they actually arrive. The one exclusion is the managed
`llama-router.service`, and it is excused by its cgroup rather than its name --
`llama-server` started by hand is still a conflict.

## Static generative-media preflight

```
python3 verification/generative-media/preflight.py
```

Checks artifact presence and exact size, the ComfyUI node contracts, loopback
and display-GPU policy, and -- since the pinned API graphs name weights by bare
filename -- resolves every graph reference through the categories declared in
`extra_model_paths.yaml`. A reference that does not resolve, resolves to two
files, resolves to an uninventoried file, or is not declared by its workflow
claim fails the preflight.

This catches drift that is otherwise invisible until a live run rejects the
graph: the FLUX.2 Klein graph loads the Qwen3-4B text encoder that the Z-Image
lane owns, so the editing lane declares that artifact even though the headline
weight belongs elsewhere.

It proves the graphs *could* load. It does not prove any workflow works;
`workflows_functionally_proven` stays empty until a live run says otherwise.

## Capability ledger

```
python3 verification/local-coverage-foundation/build_capability_ledger.py
```

Rebuilds `evidence/capability-ledger.json` (ignored) from the pinned download
queues and the gate evidence artifacts. Fail-closed in three ways: a capability
with no declared evidence is never qualified, an interrupted run is not a
verdict, and evidence recorded before the weights it judges is reported as
`evidence-stale` rather than as a pass.

That last rule is the reason the generator is tracked. The previous ledger was
written by a script that is not in this repository, so it could not be rebuilt,
and it went on reporting a 2026-09-02 reading after phases three and four landed.

`pass` in the ledger is a statement about the ledger, not about the stack.

## Shared GUI-grounding contract

```
python3 -m pytest verification/computer-use-grounding/test_grounding_contract.py
```

UI-TARS-1.5-7B and UI-Mate-9B disagree before either sees a pixel: patch 14 vs
16, a `min_pixels`/`max_pixels` budget vs a `size.shortest_edge`/`longest_edge`
one, `Thought:`/`Action:` text vs XML tool calls with a leading `<think>` block.
A 1280x800 screenshot resizes to 1288x812 for one and stays 1280x800 for the
other, so a coordinate scored in the wrong frame reads as a model that grounds
slightly badly.

`groundlib.py` reads each geometry from that checkpoint's own
`preprocessor_config.json`, refuses a config that does not state its budget, and
normalises both action spaces to one shape so a single scorer grades both on the
same fixtures. It also judges prompt-injection compliance through the action
space the model was actually given -- a tool-calling model that obeyed an
injection emits `<function=type>` and no `type(`, which the text-only judge
scores as a refusal.

Neither model has a grounding verdict. UI-TARS was interrupted mid-run and
UI-Mate has no baseline to be compared against until that is finished.
