# How a request actually runs: GPU layer split and model loading

Operator reference for two questions that come up whenever the local router
feels slower or emptier than three GPUs suggest it should:

1. Three cards are listed and each preset configures a split — what runs where,
   how was that split chosen, and why does adding a card not add speed?
2. The model picker shows a model. Is it loaded?

Everything below is grounded in the pinned runtime (`upstream/llama.cpp`, v0.4.0)
and this repository's own configuration. Nothing here is a measurement: no
tokens/sec figure appears, and none may be produced without the separate
authorization described in [model runs](../MODEL-RUNS.md).

## Which card is which

Every number below is keyed to a ROCm device index, because that is what
`llama-models.ini` names and what `--tensor-split` is positional over. A ROCm
index is *not* a sysfs `card` number: it is a position in the HIP device list,
built from the KFD topology and then filtered by `ROCR_VISIBLE_DEVICES` and
`HIP_VISIBLE_DEVICES`. They coincide on this host and would stop coinciding the
moment a non-KFD display device appeared or either variable were set.

`scripts/gpu_vram.py` derives the mapping instead of assuming it, joining each
KFD node to sysfs by both the render minor and the PCI address it publishes and
reporting a disagreement rather than picking a side:

```text
$ python3 scripts/gpu_vram.py
ROCm0 (card0, 0000:03:00.0, GPU-6b3c785d187d9cbf): ... role=headless
ROCm1 (card1, 0000:07:00.0, GPU-a21e268c0b0a73d7): ... role=headless
ROCm2 (card2, 0000:0a:00.0, GPU-4d1c68aee3a84b64): ... role=display (card2-DP-4,card2-DP-5)
```

| Device | Card | PCI | VRAM | Role |
|---|---|---|---|---|
| ROCm0 | card0 | 0000:03:00.0 | 15.98 GiB | RX 6900 XT, headless |
| ROCm1 | card1 | 0000:07:00.0 | 29.98 GiB | Radeon Pro V620, headless |
| ROCm2 | card2 | 0000:0a:00.0 | 15.98 GiB | RX 6900 XT, **drives the desktop** |

The UUIDs are the same strings `rocminfo` prints as each agent's `Uuid`, which
is how a reader checks this table against ROCm's own tooling rather than against
this document. The display role is read from connected DRM connectors, not
configured: a moved cable changes which card needs headroom, and a label in a
document would not notice.

## The configured placement

Placement is decided per preset by `scripts/gpu_placement.py` against the policy
in `config/gpu-placement.json`, and written into `llama-models.ini`. The two are
compared by a test, so a preset cannot drift into a placement nobody evaluated.

```bash
python3 scripts/gpu_placement.py            # every preset, current vs proposed
python3 scripts/gpu_placement.py --scan     # also weights on disk with no preset
python3 scripts/gpu_placement.py --verify   # compare against live residency
```

### What the numbers mean

`--tensor-split` is documented upstream as "fraction of the model to offload to
each GPU" (`common/arg.cpp:2858`). `src/llama-model.cpp` turns the values into
layer boundaries: it copies them into `splits`, replaces each entry with the
running total, and divides by the total sum (lines 1462-1473). Each repeating
layer is then assigned to exactly one device by locating it in that cumulative
list (line 1483). `--split-mode` is not set, so the runtime default applies:
`common/common.h:483` initialises `split_mode = LLAMA_SPLIT_MODE_LAYER`.

Only the ratio matters. `1,1,1` and `6,6,6` are the same placement written two
ways — equal *weights*, not gigabytes — and `config/model-runs.json` may hold
either form for the benchmark path, which a test compares by ratio for exactly
that reason.

### How each split was chosen

Two facts per preset, neither of them copied from a document:

- **Weight bytes** are the sizes of the files on disk, every shard of a split
  model included. Sizing `qwen3-coder-next` from its configured path alone
  understates it by 30 GiB.
- **KV-cache bytes** come from the model's own GGUF header — block count, KV
  head count, key and value head width — times the context and cache types the
  preset configures. `scripts/gguf_header.py` reads that header without loading
  the model or mapping its tensor data.

Those are checked against per-device budgets: card total, minus a 1 GiB
per-device reserve, minus a further 2.5 GiB on whichever card drives the
desktop. Each rule then plans against *working headroom* — the budget less a
1 GiB margin — so a placement is never chosen that sits on its last modelled
byte. The rules, in order:

1. **One preferred card, if the model fits on one.** Splitting a model that
   fits on a single card buys nothing and costs something: layer mode is
   pipelined, so a split adds device-to-device handoffs, and it holds budget on
   cards that would otherwise be free.
2. **Across the two RX 6900 XTs, if it fits on those.** The V620 carries none of
   it.
3. **Equal across all three**, once the model is too large for that pair.
4. **The 6900 XTs filled to their working headroom, remainder on the V620** —
   the last resort, and only `qwen3-coder-next` reaches it.

The preference for the RX 6900 XTs over the larger V620 is the operator's, on
thermal grounds, and is recorded in
[ADR 0006](../decisions/0006-placement-policy-prefers-the-rx-6900-xts.md). It is
not a measured performance claim and this document makes none.

### The result, for every preset

Sizes are weights plus the KV-cache upper bound at each preset's configured
context. Regenerate with `python3 scripts/gpu_placement.py`.

| Preset | Size | Split (ROCm0,1,2) | Rule |
|---|---|---|---|
| `qwen3-coder-next` | 46.77 GiB | `5,8,4` | fills the pair, remainder to V620 |
| `heretic` | 30.13 GiB | `1,1,1` | equal |
| `obliterated` | 30.03 GiB | `1,1,1` | equal |
| `phr00ty` | 29.54 GiB | `1,1,1` | equal |
| `fable` | 26.95 GiB | `1,1,1` | equal |
| `obliterated-vision` | 23.18 GiB | `5,0,4` | 6900 XT pair |
| `ridge` | 20.87 GiB | `1,0,1` | 6900 XT pair |
| `gemma4-heretic` | 14.88 GiB (9.88 GiB measured) | `1,0,0` | single card, measured |
| `gemma4-heretic-vision` | 12.00 GiB | `1,0,0` | single card |
| `qwen25-coder-7b-fim` | 8.47 GiB | `1,0,0` | single card |
| `qwen3-embedding-8b` | 5.79 GiB | `1,0,0` | single card |
| `qwen3-reranker-8b` | 5.79 GiB | `1,0,0` | single card |

Both vision presets pin their projector with `mmproj-device`, which
`--tensor-split` does not distribute. Both now pin it to `ROCm0`, a card the
model already occupies; on `ROCm1` the projector was the only reason either
preset touched the V620 at all. Those bytes are withheld from that card's
headroom *before* a split is chosen, not merely subtracted afterwards:
`obliterated-vision` sized on the split alone comes out `3,0,2` and leaves
0.21 GiB free on a 16 GiB card, which fits the arithmetic and not the card.
Reserving the projector first gives `5,0,4` and 1.24 GiB.

Two caveats belong with this table rather than under it:

- **The gemma4 figures are generous over-estimates.** Gemma 4 uses
  sliding-window attention with a 1,024-token window, and the KV column charges
  every block the full context. The real allocation is smaller by several
  gigabytes. Over-estimating is the safe direction for a budget, and the tool
  labels it rather than smoothing it.

  For `gemma4-heretic` that gap decides the placement, so it was measured rather
  than argued: loaded alone on ROCm0 at 32,768 context the preset holds
  **9.88 GiB**, against the 14.88 GiB the column charges and the 13.98 GiB of
  working headroom that card has. It fits, it unloads to its baseline, and rule 1
  applies. The preset is therefore configured `1,0,0` while this tool still
  proposes `1,0,1`; the divergence is the over-estimate, not a second policy, and
  `--verify` is the check that keeps it honest. The gain is large enough to be
  worth the exception -- decode runs 23.60 tok/s split against 40.30 tok/s on one
  card, a 70.8% gain, with prefill unchanged at 935.8 against 928.3 -- because
  layer split is pipelined and every decoded token pays a device hop. See
  [the benchmark artifacts](../benchmarks/README.md) and
  [PLACEMENT-MEASUREMENTS.md](PLACEMENT-MEASUREMENTS.md).
- **`qwen3-coder-next` is the only preset that needs the V620**, and it needs it
  for capacity, not preference. Four more presets -- `heretic`, `obliterated`,
  `fable` and `phr00ty` -- take the equal three-way split because their
  *configured context* pushes them past the pair, not because their weights do.
  What that costs is now measured rather than assumed; see
  [PLACEMENT-MEASUREMENTS.md](PLACEMENT-MEASUREMENTS.md).

### What the estimate is worth

`--verify` compares the arithmetic against live sysfs residency for whatever the
router currently holds. Measured on 2026-09-06 with `heretic` resident under the
previous `3,6,2` placement, at 131,072 context:

| Quantity | Bytes | GiB |
|---|---|---|
| Estimate (weights + KV upper bound) | 32,348,521,952 | 30.13 |
| Measured across all three cards | 32,804,032,512 | 30.55 |
| Less configured idle baseline | 30,746,726,400 | 28.63 |
| **Residual (estimate − attributed)** | **1,601,795,552** | **1.49** |

In that comparison the estimate was about 5% high, which is the direction a
budget wants to err in and the reason the reserves above are sized as they are.
Read it as one observation, not as a bound: the arithmetic spreads bytes in
proportion to layers, an allocator does not, and non-repeating tensors, compute
buffers and fragmentation are free to depart from that proportion by more than
5% on a preset nobody has measured. The residual also carries whatever the
configured idle baselines are stale by; it is not a reading taken with the model
unloaded. "Fits" in the table above means "inside the budget this policy
reserved" — a planning statement. The only proof that a preset loads is loading
it, which is the functional mission's job.

This measurement was taken before the 2026-09-06 reboot and has not been
repeated since. The idle baselines it rests on were re-read after that reboot,
on an idle router: card0 and card1 came back within a megabyte of the figures
`config/gpu-placement.json` records, and card2 read between 1.72 and 1.79 GiB
across readings minutes apart against a recorded 1.51 GiB — the desktop's own
allocation moves, which is the reason its reserve is 2.5 GiB and not the
measurement.

### Superseded

The `3,6,2` default this file previously documented sized every model to card
capacity — 16 / 32 / 16 GiB became 3 / 6 / 2 — which is a sound rule for
*fitting* a model and a poor one for deciding where work should go. It gave the
V620 54.5% of the layers of every preset in the catalog, including the ones
small enough to fit on a single 6900 XT. `gemma4-heretic` went further and ran
`0,1,0`, entirely on the V620.

Measured residency under that placement, taken 2026-09-05 while the router held
`heretic`, is kept because it is the calibration point above and because it
shows why a layer share is not a memory share:

| Card | Role | Used | Card total | Share of card |
|---|---|---|---|---|
| card0 | ROCm0 headless | 7.48 GiB | 15.98 GiB | 46.8% |
| card1 | ROCm1 headless, largest | 13.86 GiB | 29.98 GiB | 46.2% |
| card2 | ROCm2 **display** | 8.97 GiB | 15.98 GiB | 56.1% |

The display card was the most loaded card by proportion even though `3,6,2` gave
it the smallest layer share, because the KV cache, the compute buffers and the
desktop's own ~1.5 GiB all landed there too. `ridge` loading cold onto an idle
set of cards moved card0 from 26 MiB to 5,118 MiB, card1 from 385 MiB to
8,755 MiB and card2 from 1,551 MiB to 7,755 MiB; those idle figures are the
baselines `config/gpu-placement.json` subtracts.

Absolute residency is the check, not the ratio. The proportions requested in a
preset and the bytes that end up on each card are different quantities, and only
the second one can exhaust a card.

## Why a third card adds capacity, not speed

llama.cpp's own option help states the distinction (`common/arg.cpp:2835-2839`):

- `layer` (the default, and what this host uses): "split layers and KV across
  GPUs (**pipelined**)"
- `row`: "split weight across GPUs by rows (**parallelized**)"
- `tensor`: "split weights and KV across GPUs (parallelized, EXPERIMENTAL)"

In layer mode every layer lives on one device, and a transformer evaluates its
layers in order. For a single request the work therefore walks the cards in
sequence: ROCm0 computes its share while ROCm1 and ROCm2 wait, then hands the
activations to ROCm1, and so on. Only one GPU is doing useful work on that
request at any instant. A pipeline can be filled by *other* concurrent requests,
but this host sets `parallel = 1` — one slot, because qualification is
serialized — so there is no second request to overlap with.

The practical consequences:

- Three cards let a 22 GiB Q6_K model and its KV cache fit at 131,072 context.
  On this host, configured as it is, that is the benefit — and it is a real one.
- Three cards do not make one reply arrive sooner than a hypothetical single card
  that could hold the whole model. Inter-device handoffs add to it.
- A card that holds a larger share of layers spends proportionally more of the
  sequence being the one that is busy.

`split-mode = tensor` **must not** be added: Qwen3.8 MTP backend sampling is
incompatible with that path (see [configuration](CONFIGURATION.md#router-presets)
and `docs/local-hermes-models.md`). `row` mode is likewise not in use here.
Neither should be changed on the strength of this document.

The split proportions above are a different matter: they are computed, tested
against the policy, and regenerable. Change the policy in
`config/gpu-placement.json` and rerun `scripts/gpu_placement.py` rather than
editing `llama-models.ini` by hand — a hand edit is what the comparison test
will report. What has *not* happened for the current proportions is a load of
every preset under them; that is the functional mission's job
(`local-ai-functional-mission.service`), it is gated on an uncontended host, and
until it has run these placements are evaluated but not exercised.

### A stall this host can produce

Loading a 22.5 GB model at 131,072 context while a 13 GB LLVM linker held host
memory left `llama-server` with one thread parked in
`amdgpu_amdkfd_gpuvm_map_memory_to_gpu` — uninterruptible `D` state, 48 other
threads idle, VRAM already allocated, generation never starting. Memory pressure
read `some avg10=17.65` and I/O pressure `some avg10=33.85`.

This is the ROCm KFD mapping path blocking under host memory pressure, not a
model or preset defect. It cannot be signalled away: a `D`-state thread does not
take SIGTERM, and killing the process or restarting the driver is the wrong
response. Wait for the competing build to finish. It is also the concrete reason
this repository's gates refuse to run beside a build rather than measuring
through one.

## What still runs on the CPU

Even with `gpu-layers = all`, work remains on the host processor:

- The input layer is never offloaded. `src/llama-model.cpp` assigns it to the CPU
  device unconditionally, with the comment "there is very little benefit to
  offloading the input layer, so always keep it on the CPU" (lines 1488-1490).
- The router process itself, request handling, tokenization, sampling and the
  draft-MTP speculative path are host-side work.
- Any layer that does not fit falls back to the CPU device (lines 1479-1481), which is
  why an over-large model degrades rather than failing outright.

This matters on this host specifically. A wide parallel kernel build or a Cargo
build saturates the same cores and memory bandwidth that the CPU-side portion of
a request needs, so a model can appear slow while every GPU looks healthy. That
is why the repository's gates refuse to run next to a build rather than
measuring through it: `verification/local-coverage-foundation/post_reboot_gate.py`,
`verification/router-functional/gate_router_models.py` and
`verification/generative-media/preflight.py` all classify active build processes
as blockers. They read `/proc/<pid>/comm` and the `cwd` symlink to do it, never
`cmdline` — Linux serves `cmdline` through `access_remote_vm`, so inspecting a
compiler that holds its own mmap write lock could wedge the inspector.

## Picking a model is not loading a model

The router runs with `--models-preset llama-models.ini --models-max 1`
(`services/systemd/llama-router.service`). Model states are the pinned runtime's
own six-value vocabulary (`tools/server/server-models.h:51-59`).

**Listing does not load.** `GET /models` reports metadata and status. On a
healthy idle router every preset reads `unloaded`:

```text
$ scripts/local-model-status.sh
Router: ok (http://127.0.0.1:8080)
Models: 12 (unloaded=12)
Loaded: none
  RVN-Q6_K-multilingual-mtp.gguf (General-purpose Hermes agent and tool use)
    unloaded [text -> text]  [compatibility alias: heretic]
  ...
```

The weight filename leads and the alias trails it, labelled: the alias is what
the API takes, and the filename is what answered.

`scripts/local_model_status.py` reports `loads_models: false` for exactly this
reason: it is an observer.

**A Hermes model picker is a client-side selection.** Choosing an entry changes
which `model` field the *next* request will carry. Until such a request is sent,
the router has been told nothing and the previously resident model — if any —
stays resident.

**The load happens on the first request that names the model.** The router's
chat-completions handler calls `ensure_model_ready(name, ...)` before proxying
(`tools/server/server-models.cpp:1931`), and that call "load[s] the model and
blocking wait[s] until it's ready" (`server-models.h:284-289`). Autoload is on by
default (`common/common.h:687`) and the unit does not disable it. This is why the
first reply after switching models is slow and later replies are not: the wait is
a model load, not generation.

**`--models-max 1` evicts before it loads.** When the limit is reached the router
unloads the least recently used model first and waits for that unload to complete
(`unload_lru()`, `server-models.cpp:938-962`). One large model is resident at a
time, by design and per ADR 0003.

So, to answer "is it loaded?": ask the router, not the picker.

```bash
scripts/local-model-status.sh          # weight filename and state per alias
scripts/local-model-status.sh --json   # same, machine-readable
```

`loading` and `sleeping` both hold a child process and therefore the single
resident slot; `downloaded` only means weights reached local disk. See
[operations → checking the backend](OPERATIONS.md#checking-the-backend).

## Several agents, one router

Nothing in this stack is single-client. Two editors, a shell script and an agent
loop can all point at `127.0.0.1:8080` at once. What they get depends entirely on
whether they name the *same preset*.

All line references are into the pinned submodule, llama.cpp `v0.4.0`
(`upstream/llama-cpp.lock.json`, commit `5266f24da`), which
`verification/upstream-pin/test_llama_cpp_pin.py` holds the staged gitlink to.

**One preset, many callers: one load.** Three separate guards, all under the
router's single mutex, make concurrent requests for one alias produce exactly one
child process:

1. `ensure_model_ready` returns immediately when the entry is already `loaded` or
   `sleeping` (`tools/server/server-models.cpp:1374-1379`).
2. Callers that arrive while it is coming up share one queue entry —
   "requests wanting the same model share one entry, so they all need only one
   slot and all get unblocked by the single load that entry performs"
   (`server_lru_sched::join`, `server-models.cpp:103-114`). Only the entry at the
   head of the queue calls `load()` (`try_claim`, `server-models.cpp:132-142`).
3. `load()` itself returns without spawning anything if the entry is not
   `unloaded`, and re-checks `models_max` under the lock before it spawns
   (`server-models.cpp:986-1006`).

**Different presets: a queue, then an eviction.** `has_capacity` is
`count_running() < models_max` (`server-models.cpp:77-80`), and this router runs
`--models-max 1`. A second agent asking for a second alias does not fail and does
not get a second model: it waits. The header says so directly — "if models_max is
reached, the request waits in a queue until a slot frees up"
(`server-models.h:284-289`). The eviction it waits for will not take a model that
is mid-request: `pick_victim` skips any entry whose `req_count` is non-zero or
which is not `is_ready_or_sleep()` (`server-models.cpp:88-100`), and `req_count`
is held up for the whole proxied request including a streamed reply
(`proxy_request`, `server-models.cpp:1503`). So the second agent waits out the
first agent's generation *and then* pays a full unload plus a full load. Two
agents alternating between two aliases reload the model on every turn.

**Same weights, two aliases, two loads.** The router keys its model map by preset
*section name*, and `add_model` rejects a name that collides with an existing name
or alias (`server-models.cpp:432-478`). Two INI sections are therefore two
entries, two child processes and two independent loads even when their `model =`
lines name the identical file on disk. This checkout already contains two such
pairs:

| Preset pair | Shared weight file |
|---|---|
| `obliterated`, `obliterated-vision` | `Qwen3.8-27B-OBLITERATED-Q6_K.gguf` |
| `gemma4-heretic`, `gemma4-heretic-vision` | `Gemma-4-12B-it-heretic-Q6_K.gguf` |

Under `--models-max 1` those pairs cannot be co-resident. Asking for the second
member evicts the first and re-reads the same bytes from disk. If the goal is for
several agents to share one *loaded* model, they must send the same `model`
string — or a name added through that preset's own `alias =` key, which
registers extra names that resolve to the one entry (`server-models.cpp:445-451`,
`has_model` at `server-models.cpp:889-896`). A second section is a second model
as far as the router is concerned.

**`parallel = 1`: requests are serialized inside the child, not refused.** The
`[*]` block in `llama-models.ini` sets `parallel = 1`, and the router renders
preset keys into the child's argv, so every preset gets one slot
(`for (int i = 0; i < params_base.n_parallel; i++)`,
`tools/server/server-context.cpp:1254`). When a task arrives and that slot is
busy, `get_available_slot` returns `nullptr` and the task is deferred rather than
rejected (`server-context.cpp:2388-2394`), then re-queued when the slot frees
(`server-queue.cpp:90-109`). Two agents on one loaded model both get answers; the
second waits for the first's *entire* generation, not for a time slice. The
sidecar unit passes `--parallel 1` explicitly for the same reason.

**Context cache: one slot means one KV cache.** Reuse within the slot is
longest-common-prefix only —
`slot.prompt.tokens.get_common_prefix(input_tokens)` (`server-context.cpp:3203`).
Two agents with different system prompts or different histories share no prefix
past whatever preamble is byte-identical, so each turn re-prefills from the point
they diverge. There is a second, RAM-backed prompt cache: when a task takes over
a slot holding another conversation, the displaced state is saved and a matching
one is restored (`prompt_save`/`prompt_load` in `get_available_slot`,
`server-context.cpp:1636-1653`), bounded by `--cache-ram`, default 8192 MiB
(`common/common.h:632`). That reduces the cost of alternating agents; it does not
remove it, it is host RAM rather than VRAM, and it is evicted under its own
budget. `--slot-prompt-similarity` (default `0.1`, `common/common.h:695`) picks
*which* slot to reuse and therefore changes nothing when there is only one.

No throughput claim is made here. None of the above was measured; it is read from
the pinned runtime's source. What it costs on this host is not recorded anywhere,
and [the benchmark artifacts](../benchmarks/README.md) are single-client
`llama-bench` runs that do not exercise any of these paths.

**Separate server processes.** Every load spawns a child `llama-server` on its own
free loopback port and the router proxies to it (`load()`,
`server-models.cpp:1010-1046`). Sidecars are separate again: independent
`llama-sidecar@.service` instances on ports 8081-8084, started with
`--load-mode none`, never counted against the router's `models_max` and never
evicted by it. A sidecar and the router are genuinely two resident models on the
same cards, which is why a sidecar left resident by a failed gate is a mission
blocker in its own right — see
[architecture → sidecars](ARCHITECTURE.md#sidecars).

**What this means in practice.** Point every agent that can share a model at one
preset. Expect a second alias to serialize behind the first, not to run beside
it. Treat a preset switch as a cold load, not a context switch. And read the
current occupant from the router rather than from any client's idea of what it
selected:

```bash
scripts/local-model-status.sh
scripts/local-model-status.sh --host-sharing   # adds the mission and host view
```

## Reading this correctly

- Nothing here was measured. Sequential layer execution is read from the
  runtime's design and its own documentation, not from a timing run.
- Placement figures are requests made of the loader. Confirm real placement in
  the load log for the model in question.
- Do not retune `tensor-split`, change `split-mode`, or shift work onto GPU2 on
  the strength of this explanation. Those need an uncontended host and the
  authorization path in [model runs](../MODEL-RUNS.md).
