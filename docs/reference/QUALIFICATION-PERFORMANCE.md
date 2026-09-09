# Passive qualification performance

Functional qualification now retains performance observations from inference it
already performs. There are no extra prompts, warmups, repetitions or speed-based
pass criteria. A cached pass does not run inference just to fill telemetry.

## Where observations live

Each supervisor attempt records `performance_log` in its step state. Find it via
`python3 scripts/qualification_status.py --json`. The target is a unique
`verification/qualification-supervisor/<gate>-<timestamp>-<pid>.performance.jsonl`.
Subprocesses inherit the same target (including the TTS ASR round trip). Direct
Python gate execution defaults to a unique `direct-*.performance.jsonl` sidecar;
set `QUALIFICATION_PERFORMANCE_PATH` to an explicit filename to choose the target.
An explicitly empty value disables collection. Directories must already exist.

The sidecar is separate from the quality artifact. New quality artifacts use
`benchmark_performed: false`; this does not mean timing is unavailable. The old
`throughput_measured` policy marker is read only for historical compatibility by
the capability ledger. Passive observations never promote a failed qualification.

## Fields and coverage

- Router chat/vision and repository-agent turns: allowlisted runtime `timings`
  retain prompt/decode tok/s, durations and token counts when supplied. Usage
  counts and request wall time yield explicitly labeled end-to-end tok/s.
- FIM, embeddings and reranking: shared HTTP instrumentation retains exposed
  token timings; embedding inputs and ranking documents yield items/sec.
- Computer-use and ASR: actual generated tensor dimensions exclude the prompt;
  synchronized generation wall time includes prefill and decode, not decode-only.
  ASR also records audio duration and real-time factor.
- TTS: synchronized synthesis duration and output-audio duration yield real-time
  factor. No codec-token count is invented when the API returns waveforms only.
- WeMM: embedding batch output rows and synchronized execution yield items/sec.
- Generative media: workflow wall time includes submission, queueing and polling;
  completed image/audio/video output entries supply an output-item count. This
  is not a sampler-only rate, frame-rate benchmark, or cold-load separation.
- Policy/preflight/service actions do not infer: token rates are not applicable.
  Cached/refused attempts may have no sidecar. An absent sidecar is unavailable
  telemetry, never a zero speed or proof of non-execution.

Samples retain model identity when the caller exposes it, operation, timestamp,
PID and measurement-source labels. Join to the supervisor's command, boot,
input fingerprint and gate evidence for full model/runtime provenance. Cold/warm,
cache and host-load conditions are not controlled or inferred; do not rank models
from these short, variable workloads. No TTFT or decode-only rate is invented
from non-streaming total latency. Historical runs cannot be retroactively timed.

## Bounds and failure behavior

At most 256 samples of at most 8192 bytes are attempted per process. A multi-process
step can have more samples than one process; the existing gate subprocess bounds
still apply. Every append is one write. Sidecars are ignored runtime evidence and
may be archived with the gate logs; there is no automatic deletion of evidence.
No prompts, completions, headers, tool payloads or arbitrary response fields are
included. Missing, negative, non-finite and non-numeric fields remain unavailable.
Inference exceptions retain elapsed observations and are re-raised. Telemetry
write failure never changes a gate verdict. GPU synchronization is used only
around already-authorized in-process inference, never as a new workload.

Offline regression tests exercise actual loopback HTTP callers, failed requests,
malformed metadata, count semantics, output preservation and sample bounds. Those
tests establish instrumentation correctness, not live model performance.
