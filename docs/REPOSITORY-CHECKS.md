# Repository checks

Undated operating note. These are the checks that run without loading a model,
starting a service, touching a GPU, downloading anything, or measuring
throughput. They are safe on a busy host, including during a kernel build.

None of them is a functional verdict. Every one of them answers "is this
workspace internally consistent", and consistency is what makes a live gate
worth running -- not a substitute for one. Capability claims come only from the
serialized functional gates and their evidence artifacts.

## Offline test suite

```
python3 -m pytest
```

`pytest.ini` scopes collection to `verification/`. It excludes the untracked
upstream ComfyUI checkout under `tools/`, the local virtualenvs, and
`verification/repository-agent/fixture/` -- whose three failing tests are the
*oracle* for the repository-agent gate and are defective on purpose. The gate
copies that fixture to a temporary directory and runs it with `unittest`, so it
is unaffected by this configuration.

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
