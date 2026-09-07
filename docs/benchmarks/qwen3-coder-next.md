# Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf

Native `llama-bench` throughput for the weight file `/home/typhoon/git/frankenstein-llm/models/repository-agent/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf`, served by this checkout as the `qwen3-coder-next` compatibility alias.

## Artifact

| Field | Value |
|---|---|
| Weight file | `Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf` |
| Full path | `/home/typhoon/git/frankenstein-llm/models/repository-agent/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf` |
| Compatibility alias | `qwen3-coder-next` |
| Quantization (as the runtime read it) | `qwen3next 80B.A3B Q4_K - Medium` |
| File size | 45.08 GiB |
| Parameters | 79.67 B |
| Projector loaded when serving | none |

## Host and run identity

| Field | Value |
|---|---|
| Kernel | `7.2.3-273-tkg-eevdf-llvm` |
| Boot id | `9b83466d-2582-4fad-ba3c-00af2cedbdd8` |
| Runtime build | `5266f24da` |
| GPUs reported by the runtime | AMD Radeon RX 6900 XT, AMD Radeon Pro V620, AMD Radeon RX 6900 XT |
| Backends | ROCm |
| Native report directory | `logs/model-runs/benchmark-6p_i9tzo` |

## Benchmark configuration

| Field | Value |
|---|---|
| Devices | `ROCm0/ROCm1/ROCm2` |
| Tensor split (proportions, not gibibytes) | `5/8/4` |
| Tensor split this alias serves | `5,8,4` |
| GPU layers (`-ngl`) | `999` |
| Threads | `44` |
| Prompt tokens | 512 |
| Generated tokens | 128 |
| Repetitions | 3 |

Exact argv, as executed:

```
/home/typhoon/git/frankenstein-llm/upstream/llama.cpp/build/bin/llama-bench -m /home/typhoon/git/frankenstein-llm/models/repository-agent/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf -dev ROCm0/ROCm1/ROCm2 -ts 5/8/4 -t 44 -ngl 999 -p 512 -n 128 -r 3 -o json
```

## What the benchmark did not configure

`llama-bench` takes a weight file and a placement. It does not read `llama-models.ini`, so the rows below are served settings the measurement did not apply.

| Setting | Served as | Benchmarked as |
|---|---|---|
| Speculative decoding (`spec-type`) | none | none; `llama-bench` applies no draft model |
| KV cache key type | `q4_0` | `f16` |
| KV cache value type | `q4_0` | `f16` |
| Flash attention | `on` | `auto (runtime chooses)` |
| Context size | `65536` | sized to the benchmark workload, not to the serving context |

## Measured throughput

Averages and standard deviations are `llama-bench`'s own, read from the native JSON in `logs/model-runs/benchmark-6p_i9tzo/stdout.log`. Nothing here is derived from wall-clock timing of anything else.

| Test | Rate |
|---|---|
| Prompt processing (pp512) | 1052.34 tok/s ± 67.71 |
| Token generation (tg128) | 36.33 tok/s ± 0.20 |

## Memory

- Sampled every 2.0 s across 54 readings while the benchmark ran; the figures below are peaks, not allocation requests.
  - `ROCm0` peak VRAM used: 14.74 GiB
  - `ROCm1` peak VRAM used: 21.77 GiB
  - `ROCm2` peak VRAM used: 12.43 GiB
  - lowest `MemAvailable` observed: 37.73 GiB

## Usability, as the functional gates recorded it

- `router-functional` verdict: **PASS** (recorded 2026-09-07T00:28:08-0400 on kernel `7.2.3-273-tkg-eevdf-llvm`).
  - `coherence`: pass
  - `structured_output`: pass
  - `tool_call`: pass
  - required checks for this preset: `coherence`, `structured_output`, `tool_call`
- These are functional verdicts, not scores. A `FAIL` here is a recorded gate outcome and is never restated as a pass.

## Caveats

- **Serving preset:** this alias serves without speculative decoding. `llama-bench` does not apply the router's chat or speculative-decoding preset. A `spec-type = draft-mtp` preset is served with multi-token prediction and benchmarked without it, so the rate recorded here is not the one that preset delivers through the router, and it may not be compared against a non-MTP preset as a ranking of the weights.
- **Context:** the benchmark does not allocate this preset's serving context (`ctx-size = 65536`), so the residency of a served session is larger than anything measured here.
- **Smoke tests:** The repository's functional gates emit a handful of tokens per check. Their timings are not throughput and are never quoted as a rate: a two-token `PONG` measures whether a preset answers, not how fast it runs.
- **Scope:** prompt processing and token generation only. This is not a measurement of answer quality, agent reliability or tool use.
- **Comparability:** only rows measured with an identical workload, quantization, placement and serving-MTP class may be compared.
