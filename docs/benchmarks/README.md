# Native throughput artifacts

One file per weight file that was actually benchmarked with the native `llama-bench` binary. Throughput is deliberately **not** part of `local-ai-qualification.service`, which records `benchmark_performed: false`; these runs are a separate lane that an operator authorizes with `--confirm-kernel --execute` after confirming the intended kernel. See [Scripted model operation](../MODEL-RUNS.md).

Rows are ordered by alias, not by speed. A model absent from this table was not benchmarked; nothing here is estimated. The functional column is the router gate's own verdict, carried here so a fast preset that failed a gate cannot be read as a recommendation.

| Artifact | Alias | Quant | Split | pp | tg | Serving MTP | Functional gate |
|---|---|---|---|---|---|---|---|
| [`Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-Q6_K.gguf`](fable.md) | `fable` | `qwen35 27B Q6_K` | `1/1/1` | 385.84 tok/s | 13.25 tok/s | yes | **PASS** |
| [`Gemma-4-12B-it-heretic-Q6_K.gguf`](gemma4-heretic.md) | `gemma4-heretic` | `gemma4 ?B Q6_K` | `1/0/1` | 923.08 tok/s | 24.13 tok/s | no | **PASS** |
| [`Gemma-4-12B-it-heretic-Q6_K.gguf`](gemma4-heretic-vision.md) | `gemma4-heretic-vision` | `gemma4 ?B Q6_K` | `1/0/0` | 934.93 tok/s | 40.34 tok/s | no | **PASS** |
| [`RVN-Q6_K-multilingual-mtp.gguf`](heretic.md) | `heretic` | `qwen35 27B Q6_K` | `1/1/1` | 367.42 tok/s | 13.78 tok/s | yes | **PASS** |
| [`Qwen3.8-27B-OBLITERATED-Q6_K.gguf`](obliterated.md) | `obliterated` | `qwen35 27B Q6_K` | `1/1/1` | 386.51 tok/s | 13.66 tok/s | yes | **FAIL** |
| [`Qwen3.8-27B-OBLITERATED-Q6_K.gguf`](obliterated-vision.md) | `obliterated-vision` | `qwen35 27B Q6_K` | `5/0/4` | 407.95 tok/s | 14.58 tok/s | yes | **PASS** |
| [`Phr00tyMix-v4-32B-imat-Q6_K.gguf`](phr00ty.md) | `phr00ty` | `qwen2 32B Q6_K` | `1/1/1` | 333.70 tok/s | 12.52 tok/s | no | **FAIL** |
| [`qwen2.5-coder-7b-q8_0.gguf`](qwen25-coder-7b-fim.md) | `qwen25-coder-7b-fim` | `qwen2 7B Q8_0` | `1/0/0` | 2467.06 tok/s | 58.17 tok/s | no | not recorded |
| [`Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf`](qwen3-coder-next.md) | `qwen3-coder-next` | `qwen3next 80B.A3B Q4_K - Medium` | `5/8/4` | 1052.34 tok/s | 36.33 tok/s | no | **PASS** |
| [`Qwen3-Embedding-8B-Q6_K.gguf`](qwen3-embedding-8b.md) | `qwen3-embedding-8b` | `qwen3 8B Q6_K` | `1/0/0` | 1430.10 tok/s | 65.32 tok/s | no | not recorded |
| [`Qwen3-Reranker-8B-Q6_K.gguf`](qwen3-reranker-8b.md) | `qwen3-reranker-8b` | `qwen3 8B Q6_K` | `1/0/0` | 1425.14 tok/s | 65.31 tok/s | no | not recorded |
| [`Qwen3.8-27B-Ridge-3.7bpw.gguf`](ridge.md) | `ridge` | `qwen35 27B IQ2_M - 2.7 bpw` | `1/0/1` | 520.35 tok/s | 18.14 tok/s | yes | **PASS** |

## Ordering within one serving-MTP class

Every rate above was measured **without** multi-token prediction, because `llama-bench` does not apply the router's preset. Ordering an MTP-served preset against a non-MTP-served one by these numbers would read as a ranking of the weights, so the orderings below never cross that line, and even within a class they compare a configuration rather than a model's quality.

Presets served with `draft-mtp`, by measured generation rate:

1. `ridge` (Qwen3.8-27B-Ridge-3.7bpw.gguf, 11.72 GiB, qwen35 27B IQ2_M - 2.7 bpw, split `1/0/1`) — 18.14 tok/s
2. `obliterated-vision` (Qwen3.8-27B-OBLITERATED-Q6_K.gguf, 20.88 GiB, qwen35 27B Q6_K, split `5/0/4`) — 14.58 tok/s
3. `heretic` (RVN-Q6_K-multilingual-mtp.gguf, 20.98 GiB, qwen35 27B Q6_K, split `1/1/1`) — 13.78 tok/s
4. `obliterated` (Qwen3.8-27B-OBLITERATED-Q6_K.gguf, 20.88 GiB, qwen35 27B Q6_K, split `1/1/1`) — 13.66 tok/s
5. `fable` (Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-Q6_K.gguf, 22.37 GiB, qwen35 27B Q6_K, split `1/1/1`) — 13.25 tok/s

Presets served without speculative decoding, by measured generation rate:

1. `qwen3-embedding-8b` (Qwen3-Embedding-8B-Q6_K.gguf, 5.78 GiB, qwen3 8B Q6_K, split `1/0/0`) — 65.32 tok/s
2. `qwen3-reranker-8b` (Qwen3-Reranker-8B-Q6_K.gguf, 5.78 GiB, qwen3 8B Q6_K, split `1/0/0`) — 65.31 tok/s
3. `qwen25-coder-7b-fim` (qwen2.5-coder-7b-q8_0.gguf, 7.54 GiB, qwen2 7B Q8_0, split `1/0/0`) — 58.17 tok/s
4. `gemma4-heretic-vision` (Gemma-4-12B-it-heretic-Q6_K.gguf, 9.10 GiB, gemma4 ?B Q6_K, split `1/0/0`) — 40.34 tok/s
5. `qwen3-coder-next` (Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf, 45.08 GiB, qwen3next 80B.A3B Q4_K - Medium, split `5/8/4`) — 36.33 tok/s
6. `gemma4-heretic` (Gemma-4-12B-it-heretic-Q6_K.gguf, 9.10 GiB, gemma4 ?B Q6_K, split `1/0/1`) — 24.13 tok/s
7. `phr00ty` (Phr00tyMix-v4-32B-imat-Q6_K.gguf, 25.03 GiB, qwen2 32B Q6_K, split `1/1/1`) — 12.52 tok/s

## What these numbers are not

- The repository's functional gates emit a handful of tokens per check. Their timings are not throughput and are never quoted as a rate: a two-token `PONG` measures whether a preset answers, not how fast it runs.
- Functional pass/fail comes from `verification/router-functional/`. A fast preset that failed a gate is reported as failed.
- Native JSON stays in `logs/model-runs/`, which git ignores. Each artifact names the report directory it was written from.
- The `Split` column is the `-ts` each run was measured with, not the split its preset serves today. `gemma4-heretic` was retuned to `1,0,0` after the `1/0/1` row below was recorded, so that row is evidence for the comparison in [placement, measured](../reference/PLACEMENT-MEASUREMENTS.md) rather than a description of the current preset. The live values are in [configuration → aliases](../reference/CONFIGURATION.md#aliases).
