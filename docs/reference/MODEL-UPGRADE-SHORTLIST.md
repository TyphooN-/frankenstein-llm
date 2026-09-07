# Larger-model upgrade shortlist: provisional artifact assessment

Primary cards reviewed: https://huggingface.co/Qwen/Qwen3.5-122B-A10B , https://huggingface.co/Qwen/Qwen3.5-35B-A3B , https://huggingface.co/Qwen/Qwen3.8-Flash-Next . Publisher claims are not locally reproduced results. No candidate is admitted by this document.

## Decisions

- Qwen3.5-35B-A3B Q6_K or Q8_0: evaluate as a compact MoE alternative to existing general/tool and multimodal models. It need not fill VRAM to be useful; validate quality and native tools against installed baselines.
- Qwen3.5-122B-A10B Q4_K_M: prioritize as an isolated RAM-offload candidate. Its weights alone exceed nominal aggregate VRAM; 10B activated parameters do not make it a 10B storage footprint. Q6_K is a later fidelity comparison only after the smaller artifact demonstrates a useful role.
- Qwen3.8-Flash-Next: research-only pending exact smaller-quant selection and runtime proof. The card calls it an experimental architecture preview. The verified high-fidelity artifacts below are too large to recommend casually on this desktop; do not infer storage from headline active parameters.

## Verified publisher file sizes

These are API-declared shard sizes, not downloaded/hash-verified weights, load measurements, or proof of usable fit. Separate MTP artifacts are excluded. Projectors, cache, workspaces and host/desktop reserves are additional. Shard-count completeness is not content integrity.

| Repository | Quant | Files | Bytes | GiB | Inventory |
|---|---|---:|---:|---:|---|
| unsloth/Qwen3.5-122B-A10B-GGUF | Q4_K_M | 3 | 76536964608 | 71.28 | complete by filename count |
| unsloth/Qwen3.5-122B-A10B-GGUF | Q6_K | 4 | 101009782432 | 94.07 | complete by filename count |
| unsloth/Qwen3.5-122B-A10B-GGUF | Q8_0 | 4 | 129871935104 | 120.95 | complete by filename count |
| unsloth/Qwen3.5-122B-A10B-GGUF | UD-Q6_K_XL | 4 | 112401249920 | 104.68 | complete by filename count |
| unsloth/Qwen3.5-35B-A3B-GGUF | Q4_K_M | 1 | 22016023168 | 20.50 | complete by filename count |
| unsloth/Qwen3.5-35B-A3B-GGUF | Q6_K | 1 | 28852861568 | 26.87 | complete by filename count |
| unsloth/Qwen3.5-35B-A3B-GGUF | Q8_0 | 1 | 36903139968 | 34.37 | complete by filename count |
| unsloth/Qwen3.5-35B-A3B-GGUF | UD-Q6_K_S | 1 | 28515105440 | 26.56 | complete by filename count |
| unsloth/Qwen3.5-35B-A3B-GGUF | UD-Q6_K_XL | 1 | 32071842432 | 29.87 | complete by filename count |
| unsloth/Qwen3.8-Flash-Next-GGUF | Q8_0 | 6 | 188225033248 | 175.30 | complete by filename count |
| unsloth/Qwen3.8-Flash-Next-GGUF | UD-Q6_K_XL | 6 | 169165382688 | 157.55 | complete by filename count |

## Corrections and next gate

The interrupted research suggested the 122B Q8_0 set was incomplete. The fresh API lookup lists all four shards: that earlier claim must not be repeated. Metadata can change; pin revisions before transfer.

Require the [evaluation strategy](MODEL-UPGRADE-EVALUATION.md), exact backend/binary support, measured available RAM and per-device VRAM, and a healthy uncontended host before execution. No throughput results exist for this shortlist. No weights were downloaded during this assessment.

## Pinned metadata revisions

- https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/tree/51eab4d59d53f573fb9206cb3ce613f1d0aa392b
- https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/tree/bc014a17be43adabd7066b7a86075ff935c6a4e2
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66
