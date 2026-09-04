# WeMM-Embedding-2B pinned remote-code review — 2026-09-03

- Repository: `tencent/WeMM-Embedding-2B`
- Reviewed revision: `bbd6cd4bf52cfc6716f752a2df80b2706720bd95`
- Snapshot reviewed: `models/embedding/WeMM-Embedding-2B` (phase four, size-verified)
- Reviewer method: read the files; nothing was imported, executed, or loaded.
- Machine-readable form: `wemm_remote_code_review.py` (`REVIEWED_FILES`), enforced
  by `verification/candidate-qualification/wemm_remote_code_review.py`.

## Why this review exists

`config.json` carries

```json
"auto_map": {
  "AutoModel": "modeling_wemm_embedding.WeMMEmbedding",
  "AutoModelForCausalLM": "modeling_wemm_embedding.WeMMEmbedding"
}
```

and `modules.json` points Sentence Transformers at
`modeling_st_wemm.WeMMTransformer`. Loading this model with
`trust_remote_code=True` therefore imports and runs publisher-authored Python
from the model directory. That is code execution triggered by a *download*, so
the bytes are pinned by digest and any change revokes approval.

## Verdicts

| File | SHA-256 | Verdict |
|---|---|---|
| `modeling_wemm_embedding.py` | `ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f` | approved for import |
| `modeling_st_wemm.py` | `521d02c1c60ae727cc9dc6500cdb0b28c53b259e0ce3d37197920a33ba4dd333` | approved for import |
| `patch_sglang_video.py` | `c20c73e803a634e5dc54c39bc1c20cec0e7929c0320121769268a3dfb2e58a2d` | **never execute** |

These three are the only `.py`, `.pyc`, `.so` or `.sh` files in the snapshot. The
gate re-derives that list on every run; an executable file appearing that this
document does not cover fails the gate rather than being ignored.

## `modeling_wemm_embedding.py` — approved

Subclasses `Qwen3_5ForConditionalGeneration` and adds one method, `embedding()`,
which clears `self.model.rope_deltas`, runs a forward pass, gathers the last
unmasked position per row, and L2-normalises. Reviewed for the things that make
remote code dangerous:

- no `open`, `Path`, `os`, `shutil`, or any other filesystem access;
- no `urllib`, `requests`, `socket`, or other network access;
- no `subprocess`, `os.system`, `eval`, `exec`, `compile`, or `__import__`;
- no import-time side effects — the module body only defines the class;
- no pickle, no `torch.load`, no dynamic attribute dispatch on user input.

Only `torch` and `torch.nn.functional` are imported. Nothing here reaches
outside the tensor computation it claims to perform.

## `modeling_st_wemm.py` — approved, with one dependency note

A Sentence Transformers `Transformer` subclass that renders the chat template and
prepares images/videos. Same negative findings as above: no filesystem, network,
subprocess or `eval`/`exec`, and no import-time side effects.

One thing to carry forward rather than a defect: `_apply_chat_template` performs a
function-local `from qwen_vl_utils import process_vision_info`. `qwen_vl_utils` is
a third-party package that the snapshot does not vendor and `config.json` does not
declare, so a Sentence Transformers path would resolve it from whatever is
installed in the venv at call time. If the multimodal index is ever built through
Sentence Transformers rather than through `AutoModel`, that dependency has to be
pinned explicitly like any other supply-chain input. The plain `AutoModel` path
approved for the embedding gate does not reach this file at all.

## `patch_sglang_video.py` — refused

This file is not imported by `auto_map` or `modules.json`; it is a standalone
`__main__` utility. It:

1. resolves the installed `sglang` package directory,
2. rewrites `sglang/srt/multimodal/processors/qwen_vl.py` in place,
3. leaves a `.original` backup beside it, and
4. byte-compiles the result.

That is an in-place mutation of an installed third-party package — a host change
shipped inside a model download. SGLang is not part of this stack, the repository
serves embeddings through llama.cpp sidecars and Transformers, and nothing in
this workspace has a reason to run it. It is recorded as `never-execute`. Its
presence is not itself a failure; running it would be.

## Standing conditions on any later execution

1. **Approval is per-digest.** A publisher re-upload at the same revision, or a
   local edit, changes the digest and the gate refuses the import until this
   document is re-reviewed and updated.
2. **Separate index.** WeMM is a 2048-D multimodal space; the admitted text
   space is 4096-D Qwen3-Embedding-8B. `candidate_policy.MULTIMODAL_INDEX` keeps
   them in different databases under different aliases, and
   `candidate_policy.index_conflicts` refuses a configuration that merges them.
3. **`trust_remote_code` stays scoped to this directory.** The approval covers
   the two files named above at the digests above and nothing else in the
   snapshot or the environment.
