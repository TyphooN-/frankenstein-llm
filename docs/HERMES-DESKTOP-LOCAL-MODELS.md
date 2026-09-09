# Hermes Desktop local uncensored models

## Everyday use

The local router is already running. Cloud remains the Hermes default. To use a local model, pick it in Desktop or type `/model heretic` in a new chat. The first reply after a switch waits for the GGUF to load; later replies do not.

Do not wait for the full qualification qualification. Chat, writing, and local coding through the router are ready. Grounding, ComfyUI, TTS, and WeMM image retrieval are not required for that.

1. Confirm the backend:

```bash
systemctl --user is-active llama-router.service
/home/typhoon/git/frankenstein-llm/scripts/local-model-status.sh
# Machine-readable bounded summary (never dumps router launch arguments):
/home/typhoon/git/frankenstein-llm/scripts/local-model-status.sh --json
```

2. Start Hermes Desktop:

```bash
hermes desktop --skip-build
```

3. Open the model picker or type one of these in a chat:

```text
/model heretic
/model obliterated
/model ridge
/model fable
/model phr00ty
/model qwen3-coder-next
/model gemma4-heretic
```

4. Start a fresh chat after switching model families.

## Which model to pick

- `heretic` loads `RVN-Q6_K-multilingual-mtp.gguf`: recommended uncensored daily driver. Q6, multilingual/code calibrated, embedded MTP, closest-to-stock published behavior evidence of the installed unrestricted models.
- `obliterated` loads `Qwen3.8-27B-OBLITERATED-Q6_K.gguf`: aggressive OBLITERATUS V3. Better when refusal/deflection removal is the priority; published MMLU is 2.1 percentage points below stock.
- `ridge` loads `Qwen3.8-27B-Ridge-3.7bpw.gguf`: compact 3.7-bpw option. Fastest and smallest, but not the quality-first choice on this hardware.
- `fable` loads `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-Q6_K.gguf`: unrestricted Qwen3.6 27B Q6_K fantasy-writing generalist. Prefer it for plot logic, continuity, instruction following, and explicit prose.
- `phr00ty` loads `Phr00tyMix-v4-32B-imat-Q6_K.gguf`: unrestricted Phr00tyMix v4 32B Q6_K prose/RP specialist. Prefer it when voice, scene texture, and spicy roleplay matter most.
- `qwen3-coder-next` loads `Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf`: official 80B-A3B repository-agent candidate. Use for local coding/tool work. Keep Qwen2.5 Coder for FIM.
- `gemma4-heretic` loads `Gemma-4-12B-it-heretic-Q6_K.gguf`: low-privilege multimodal reader. Do not grant it executable tools.

The picker lists aliases; the filename after each one is the artifact that
actually loads. `python3 /home/typhoon/git/frankenstein-llm/scripts/serve-model.py --list`
prints the same mapping for every preset, including the vision pair.

The picker shows the bare alias and not the artifact name, and that is a
boundary rather than a defect. Desktop discovers models from the router
(`discover_models` against `http://127.0.0.1:8080/v1`), and llama.cpp's
OpenAI-compatible `/v1/models` returns an `id` per preset with no name or
description field, so there is nothing richer for a client to show. Everything
this repository owns — `serve-model.py --list`, `local-model-status.sh`, the
tables in this file — leads with the artifact instead. Changing what the picker
itself displays would mean editing the pinned upstream runtime or the Hermes
configuration, neither of which is this repository's to change.

`fable` and `phr00ty` use 65,536-token router presets. Phr00ty's GGUF declares a 131,072-token native training context but has no MTP tensors; its preset therefore omits draft-MTP intentionally.

## What happens during switching

A llama.cpp router listens only on `127.0.0.1:8080`. It keeps at most one large model loaded. Selecting another model unloads the least-recently-used model and loads the requested GGUF. The first answer after a switch therefore has a model-load delay.

## Check or repair the backend

```bash
/home/typhoon/git/frankenstein-llm/scripts/local-model-status.sh
systemctl --user status llama-router.service
systemctl --user restart llama-router.service
journalctl --user -u llama-router.service -f
```

If Desktop cannot see local models, verify `http://127.0.0.1:8080/v1/models` through the status script, then restart Desktop. Desktop, TUI, and CLI all use `~/.hermes/config.yaml`; there is no separate Desktop model registry.

After an unclean power loss, do not trust a model merely because its byte size is complete. Re-run the published SHA-256 check before loading it. The Heretic downloader at `/home/typhoon/git/frankenstein-llm/scripts/download-heretic-clean.sh` performs range-resume, whole-file checksum verification, and atomic promotion.

## GPU coexistence

The loaded 27B model uses all three AMD GPUs. Before heavy ComfyUI image/video work:

```bash
systemctl --user stop llama-router.service
```

Restore it afterward:

```bash
systemctl --user start llama-router.service
```

## Safety boundary

These models have reduced refusal behavior. The server is intentionally loopback-only. Do not expose port 8080 to the LAN or Internet. Model behavior is not an authorization boundary: keep Hermes approvals, tool permissions, secrets, and execution isolation enabled.

Full architecture and evidence ledger:

`docs/local-hermes-models.md`

The full coding, creative-generation, and authorized-security model strategy is at:

`docs/LOCAL-AI-MODEL-STRATEGY.md`
