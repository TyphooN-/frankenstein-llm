# Qualitative characterization

What a locally served model will actually write, and how well it writes it, kept
as two separate questions.

```bash
# Offline preview: prints prompts and settings, no network or writes
python3 verification/qualitative-characterization/characterize.py heretic

# Execute the whole corpus against one router alias
python3 verification/qualitative-characterization/characterize.py heretic --execute

# one axis, or one prompt
python3 verification/qualitative-characterization/characterize.py ridge --axis quality --execute
python3 verification/qualitative-characterization/characterize.py ridge --only dark-fantasy-body-horror --execute
```

Artifacts are written to the ignored evidence directory, one subdirectory per
alias:

```
/home/typhoon/git/frankenstein-llm/verification/qualitative-characterization/evidence/<alias>/
  index.md      # human entry point: table of prompts, outcomes, refusal signals
  report.json   # machine record: prompts, provenance, host samples, checks
  <prompt-id>.txt  # the exact model output, verbatim
```

## The two axes, and why they are separate

- **willingness** — will the model produce coarse, irreverent or bleak fiction on
  request? A refusal is a datum about how the weights were tuned. It is not a
  defect, and compliance is not a merit.
- **quality** — is the output coherent, on-instruction and internally consistent?
  Measured on prompts with nothing transgressive in them, so a model cannot look
  good on this axis merely by being permissive.

**A model that will write anything is not thereby a good model.** No score in
these artifacts combines the two axes, and none should be constructed from them.

## What the automated part can and cannot settle

`evaluate()` runs mechanical checks only: word bounds, required and forbidden
substrings, and numbered-point counts. Those are things a regular expression can
settle. Whether the satire lands or the horror works is not, which is why every
output is saved verbatim for a human to read.

Two results deserve care when reading `index.md`:

- **`inconclusive-truncated`** means the reply hit the harness's token ceiling
  before it could satisfy a check. That is this harness's budget, not a model
  failure, and it is never counted as one.
- **Refusal markers** are substrings found in the opening of a reply. They are a
  signal, not a verdict, and `index.md` quotes the sentence that matched. The
  first live run produced the reason why: `ridge` opened a fully compliant,
  fully profane resignation letter with "I cannot survive another second of your
  soul-crushing bullshit", which trips `i cannot` while refusing nothing.

## Corpus scope

`corpus.json` covers profane satire of invented companies and bureaucracies,
irreverence toward wholly fictional institutions and invented belief systems,
dark fantasy and body horror in invented settings, and fictional characters being
crude or cruel to other fictional characters. Every company, ministry, faith,
tavern and person named in it is invented for the corpus.

Absent by construction, not filtered at runtime: hatred or degradation aimed at
protected groups, abuse or sexual content involving real identifiable people, any
sexual content involving minors, actionable violence, and actionable cyber
wrongdoing. `test_characterize.py` asserts the scope declaration and keeps a
tripwire against a real-world target being substituted for an invented one during
an edit.

## Safety properties of the harness itself

- **Model output is inert data.** It is written only to `.txt`, `.json` and `.md`
  with mode 0644, is never executed, sourced or evaluated, and the test suite
  asserts the harness contains no `subprocess`, `exec`, `eval` or shell call.
- **Loopback only.** The endpoint is asserted to be `127.0.0.1`, not assumed.
- **No speed measurement.** No timing, token-rate or usage field is requested or
  recorded. `report.json` carries `throughput_measured: false` and
  `timing_recorded: false`, and a test asserts those fields never appear.

## Resource evidence

Each run samples `MemAvailable`, `SwapFree` and per-card `mem_info_vram_used`
before and after. That is residency evidence — which card holds what — and
nothing else. Host memory headroom is recorded as
`memory_fit_confounded_by_concurrent_builds: true` whenever builds share the
machine, and no timing comparison is possible from these files by construction.
