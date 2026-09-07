# qwen2.5-coder-7b-q8_0.gguf

Native `llama-bench` throughput for the weight file `/home/typhoon/git/frankenstein-llm/models/fim/qwen2.5-coder-7b-q8_0.gguf`, served by this checkout as the `qwen25-coder-7b-fim` compatibility alias.

## Artifact

| Field | Value |
|---|---|
| Weight file | `qwen2.5-coder-7b-q8_0.gguf` |
| Full path | `/home/typhoon/git/frankenstein-llm/models/fim/qwen2.5-coder-7b-q8_0.gguf` |
| Compatibility alias | `qwen25-coder-7b-fim` |
| Quantization (as the runtime read it) | `qwen2 7B Q8_0` |
| File size | 7.54 GiB |
| Parameters | 7.62 B |
| Projector loaded when serving | none |

## Host and run identity

| Field | Value |
|---|---|
| Kernel | `7.2.3-273-tkg-eevdf-llvm` |
| Boot id | `9b83466d-2582-4fad-ba3c-00af2cedbdd8` |
| Runtime build | `5266f24da` |
| GPUs reported by the runtime | AMD Radeon RX 6900 XT, AMD Radeon Pro V620, AMD Radeon RX 6900 XT |
| Backends | ROCm |
| Native report directory | `logs/model-runs/benchmark-bbjlrx22` |

## Benchmark configuration

| Field | Value |
|---|---|
| Devices | `ROCm0/ROCm1/ROCm2` |
| Tensor split (proportions, not gibibytes) | `1/0/0` |
| Tensor split this alias serves | `1,0,0` |
| GPU layers (`-ngl`) | `999` |
| Threads | `44` |
| Prompt tokens | 512 |
| Generated tokens | 128 |
| Repetitions | 3 |

Exact argv, as executed:

```
/home/typhoon/git/frankenstein-llm/upstream/llama.cpp/build/bin/llama-bench -m /home/typhoon/git/frankenstein-llm/models/fim/qwen2.5-coder-7b-q8_0.gguf -dev ROCm0/ROCm1/ROCm2 -ts 1/0/0 -t 44 -ngl 999 -p 512 -n 128 -r 3 -o json
```

## What the benchmark did not configure

`llama-bench` takes a weight file and a placement. It does not read `llama-models.ini`, so the rows below are served settings the measurement did not apply.

| Setting | Served as | Benchmarked as |
|---|---|---|
| Speculative decoding (`spec-type`) | none | none; `llama-bench` applies no draft model |
| KV cache key type | `q8_0` | `f16` |
| KV cache value type | `q8_0` | `f16` |
| Flash attention | `on` | `auto (runtime chooses)` |
| Context size | `32768` | sized to the benchmark workload, not to the serving context |

## Measured throughput

Averages and standard deviations are `llama-bench`'s own, read from the native JSON in `logs/model-runs/benchmark-bbjlrx22/stdout.log`. Nothing here is derived from wall-clock timing of anything else.

| Test | Rate |
|---|---|
| Prompt processing (pp512) | 2467.06 tok/s ± 171.61 |
| Token generation (tg128) | 58.17 tok/s ± 0.09 |

## Memory

- Sampled every 2.0 s across 11 readings while the benchmark ran; the figures below are peaks, not allocation requests.
  - `ROCm0` peak VRAM used: 7.67 GiB
  - `ROCm1` peak VRAM used: 0.55 GiB
  - `ROCm2` peak VRAM used: 1.99 GiB
  - lowest `MemAvailable` observed: 41.02 GiB

## Usability, as the functional gates recorded it

- No `router-functional` result for this alias in the current evidence file. Absence of evidence is not a pass.

## Caveats

- **Serving preset:** this alias serves without speculative decoding. `llama-bench` does not apply the router's chat or speculative-decoding preset. A `spec-type = draft-mtp` preset is served with multi-token prediction and benchmarked without it, so the rate recorded here is not the one that preset delivers through the router, and it may not be compared against a non-MTP preset as a ranking of the weights.
- **Context:** the benchmark does not allocate this preset's serving context (`ctx-size = 32768`), so the residency of a served session is larger than anything measured here.
- **Smoke tests:** The repository's functional gates emit a handful of tokens per check. Their timings are not throughput and are never quoted as a rate: a two-token `PONG` measures whether a preset answers, not how fast it runs.
- **Scope:** prompt processing and token generation only. This is not a measurement of answer quality, agent reliability or tool use.
- **Comparability:** only rows measured with an identical workload, quantization, placement and serving-MTP class may be compared.
