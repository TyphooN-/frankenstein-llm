# Prompt corpus admission

This directory records which external prompt/safety sources may be considered and validates normalized local artifacts before any evaluator can consume them.

`source-catalog.json` is policy, not proof that a dataset has been downloaded or admitted. A source marked `candidate` may supply an artifact only after a separate `hermes-prompt-corpus-artifact/1` manifest binds:

- the catalog source ID and exact 40-character revision;
- one approved suite;
- a relative path beneath the ignored `corpora/` directory;
- normalized `jsonl` or `csv` format;
- exact bytes, SHA-256, row count, and field schema;
- `inert_text_only: true`.

Run admission with:

```text
python3 verification/prompt-corpus-admission/admit_corpus.py MANIFEST.json \
  --evidence verification/prompt-corpus-admission/evidence/NAME.json
```

Admission never downloads content, invokes a model, executes code, follows URLs, calls tools, or interprets corpus instructions. It rejects symlinks, parent traversal, executable files, NUL/binary content, invalid UTF-8, credential markers, schema drift, unknown sources, and reference-only/rejected sources. Existing passing evidence is not overwritten when a later candidate fails.

Safety/refusal, false-refusal/benign utility, and prompt-injection/tool-integrity are separate suites. A model must not trade benign utility for a superficially low harmful-compliance score, and a refusal score does not establish tool authorization integrity.
