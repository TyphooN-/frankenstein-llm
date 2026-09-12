#!/usr/bin/env python3
"""Admission policy for the phase-four researched candidates.

Every fact about *what is installed* is read from the pinned download queue
(`download-queue-phase4.json`), which is the artifact the transfer actually
verified. Nothing here re-declares a size or a revision, because a second copy
of a number is a second thing that can drift.

What this module adds on top of the queue is the part a download manifest cannot
express: which candidate is allowed to do what, and what has to be true first.
Three rules are structural rather than advisory, and each is a function that
returns ``False`` when the evidence it needs is missing:

* ``tool_grant_allowed`` -- a low-privilege preset never receives executable
  tools. Gemma-4 Heretic is an ablated model; it is admitted (if at all) as a
  multimodal reader, and a reader that can call ``write_file`` is not a reader.
* ``control_allowed`` -- a GUI candidate may not be given desktop control until
  its own grounding verdict exists and passed. "No verdict yet" is not "safe".
* ``remote_code_execution_allowed`` -- WeMM's pinned ``*.py`` files are executed
  by ``trust_remote_code``; nothing may import them until the tracked security
  review approves those exact bytes.

None of these functions load a model, start a service, or touch a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/typhoon/git/frankenstein-llm")
FOUNDATION = ROOT / "verification" / "local-coverage-foundation"
QUEUE_PATH = FOUNDATION / "download-queue-phase4.json"
PHASE_ONE_QUEUE_PATH = FOUNDATION / "download-queue.json"
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

# Privilege tiers, most restrictive first. The tier is a property of the
# candidate, not of the caller, so a preset cannot be promoted by asking nicely
# from a different gate.
PRIVILEGE_READ_ONLY = "read-only"        # embeddings/perception; no generation authority
PRIVILEGE_LOW = "low"                    # may generate text; never receives tools
PRIVILEGE_TOOL_USING = "tool-using"      # may receive the bounded repository tools
PRIVILEGE_TIERS = (PRIVILEGE_READ_ONLY, PRIVILEGE_LOW, PRIVILEGE_TOOL_USING)

# Runtimes a candidate can be served through. Recorded so a preflight can check
# the right contract: a GGUF preset is a router question, a safetensors snapshot
# is a Transformers question, and a ComfyUI checkpoint is a node-graph question.
RUNTIME_LLAMA_ROUTER = "llama-router"
RUNTIME_TRANSFORMERS = "transformers"
RUNTIME_COMFYUI = "comfyui"

# The text embedding space that already passed its gate. WeMM is 2048-D and
# multimodal; mixing the two in one index would silently return nonsense
# neighbours, so the multimodal index is declared as a separate store here and
# the separation is asserted by ``index_conflicts``.
TEXT_INDEX = {
    "alias": "qwen3-embedding-8b",
    "database": ROOT / "rag" / "hermes-rag.sqlite3",
    "dimension": 4096,
}
MULTIMODAL_INDEX = {
    "alias": "wemm-embedding-2b",
    "database": ROOT / "rag" / "hermes-multimodal.sqlite3",
    # config.json: text_config.hidden_size 2048, matryoshka_dimensions up to 2048.
    "dimension": 2048,
    "matryoshka_dimensions": (64, 128, 256, 512, 1024, 2048),
}

CANDIDATES: dict[str, dict] = {
    "qwen3-coder-next": {
        "artifact_key": "repository-agent-qwen3-coder-next-q4km",
        "capability": "repository-agent",
        "runtime": RUNTIME_LLAMA_ROUTER,
        "privilege": PRIVILEGE_TOOL_USING,
        "presets": ("qwen3-coder-next",),
        "challenges": "heretic",
        "abliterated": False,
        "requires_remote_code": False,
        # The FIM lane is a different capability served by a different endpoint.
        # Coder-Next is not a drop-in for it and must not displace it.
        "must_retain": ("qwen25-coder-7b-fim",),
        "prerequisites": (),
        "note": "Official Q4_K_M split GGUF, arch qwen3next. A/B against the "
                "incumbent repository-agent preset; Qwen2.5 Coder keeps /infill.",
    },
    "gemma4-heretic": {
        "artifact_key": "uncensored-multimodal-gemma4-heretic-q6k",
        "capability": "uncensored-multimodal",
        "runtime": RUNTIME_LLAMA_ROUTER,
        "privilege": PRIVILEGE_LOW,
        "presets": ("gemma4-heretic", "gemma4-heretic-vision"),
        "challenges": None,
        "abliterated": True,
        "requires_remote_code": False,
        "must_retain": (),
        "prerequisites": (),
        "note": "Heretic ARA+LoRA ablation with a vision+audio projector. Reader "
                "only: never appears in a tool-granting gate at any tier.",
    },
    "ui-mate-9b": {
        "artifact_key": "computer-use-ui-mate-9b",
        "capability": "computer-use-grounding",
        "runtime": RUNTIME_TRANSFORMERS,
        "privilege": PRIVILEGE_READ_ONLY,
        "presets": (),
        "challenges": "ui-tars-1.5-7b",
        "abliterated": False,
        "requires_remote_code": False,
        "must_retain": (),
        # A/B needs a baseline. The UI-TARS grounding gate was killed mid-run and
        # has never produced a verdict, so there is nothing to compare against.
        "prerequisites": ("computer-use-grounding",),
        "note": "Qwen3.5-based GUI policy with a tool-call action space. Grounding "
                "must pass before any control authority, and it is never ablated.",
    },
    "wemm-embedding-2b": {
        "artifact_key": "multimodal-embedding-wemm-2b",
        "capability": "multimodal-embeddings",
        "runtime": RUNTIME_TRANSFORMERS,
        "privilege": PRIVILEGE_READ_ONLY,
        "presets": (),
        "challenges": None,
        "abliterated": False,
        "requires_remote_code": True,
        "must_retain": ("qwen3-embedding-8b", "qwen3-reranker-8b"),
        "prerequisites": ("wemm-remote-code-review",),
        "note": "Separate 2048-D multimodal index. The pinned repository ships "
                "executable model code; review gates every import.",
    },
    "flux2-klein-4b": {
        "artifact_key": "image-edit-flux2-klein-4b",
        "capability": "image-generation-editing",
        "runtime": RUNTIME_COMFYUI,
        "privilege": PRIVILEGE_READ_ONLY,
        "presets": (),
        "challenges": None,
        "abliterated": False,
        "requires_remote_code": False,
        "must_retain": ("z-image-unet",),
        "prerequisites": (),
        "note": "Root checkpoint is native ComfyUI format; encoder/tokenizer/VAE "
                "are the publisher's Diffusers subfolders.",
    },
    "qwen3-asr-1.7b": {
        # Installed by phase one at the same pinned revision the 2026-09-02 sweep
        # reviewed. Phase four records it as already satisfied instead of
        # re-fetching 4 GB, so this candidate owns no phase-four files.
        "artifact_key": None,
        "already_satisfied_key": "asr-qwen3-1.7b-hf",
        "capability": "asr",
        "runtime": RUNTIME_TRANSFORMERS,
        "privilege": PRIVILEGE_READ_ONLY,
        "presets": (),
        "challenges": "qwen3-asr-1.7b",
        "abliterated": False,
        "requires_remote_code": False,
        "must_retain": (),
        "prerequisites": ("asr",),
        "note": "No new bytes. Only the functional comparison against the "
                "already-admitted ASR gate remains.",
    },
}

# Capabilities whose models take real actions on the host. Removing refusals from
# an actor removes a safety boundary rather than an inconvenience, so ablation is
# refused here structurally and not left to reviewer discipline.
ACTING_CAPABILITIES = frozenset({"computer-use-grounding", "computer-use-end-to-end",
                                 "repository-agent"})

# Every llama.cpp router alias, including incumbents that are not phase-four
# candidates. Unknown aliases are absent here on purpose: privilege_for_preset
# returns None and tools are refused. Embedding/rerank/FIM are not chat actors.
INCUMBENT_PRESETS: dict[str, str] = {
    "ridge": PRIVILEGE_TOOL_USING,
    "heretic": PRIVILEGE_TOOL_USING,
    "obliterated": PRIVILEGE_TOOL_USING,
    "obliterated-vision": PRIVILEGE_TOOL_USING,
    "fable": PRIVILEGE_TOOL_USING,
    "phr00ty": PRIVILEGE_TOOL_USING,
    "signal": PRIVILEGE_TOOL_USING,
    "qwopus": PRIVILEGE_TOOL_USING,
    "nex": PRIVILEGE_TOOL_USING,
    "qwen3-embedding-8b": PRIVILEGE_READ_ONLY,
    "qwen3-reranker-8b": PRIVILEGE_READ_ONLY,
    "qwen25-coder-7b-fim": PRIVILEGE_READ_ONLY,
}


def load_queue(path: Path = QUEUE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def queue_artifacts(path: Path = QUEUE_PATH) -> dict[str, dict]:
    return {artifact["key"]: artifact for artifact in load_queue(path)["artifacts"]}


def artifact_for(name: str, path: Path = QUEUE_PATH) -> dict | None:
    """The pinned queue entry backing a candidate, or None when it owns no files."""
    key = CANDIDATES[name]["artifact_key"]
    if key is None:
        return None
    return queue_artifacts(path)[key]


def tool_grant_allowed(name: str) -> bool:
    """May this candidate be handed executable tools?

    Only the tool-using tier may. An unknown name is not a special case that
    deserves a permissive answer, so it is refused rather than raised past a
    caller that might be catching exceptions.
    """
    candidate = CANDIDATES.get(name)
    if candidate is None:
        return False
    return candidate["privilege"] == PRIVILEGE_TOOL_USING


def privilege_for_preset(preset: str) -> str | None:
    """Privilege of a router alias. None means unknown, which is not a grant."""
    candidate = candidate_for_preset(preset)
    if candidate is not None:
        return CANDIDATES[candidate]["privilege"]
    return INCUMBENT_PRESETS.get(preset)


def tool_grant_allowed_for_preset(preset: str) -> bool:
    """May this router alias receive a tools payload? Unknown aliases may not."""
    return privilege_for_preset(preset) == PRIVILEGE_TOOL_USING


def abliteration_allowed(name: str) -> bool:
    """Ablated weights are refused for anything that acts on the host."""
    candidate = CANDIDATES.get(name)
    if candidate is None:
        return False
    return candidate["capability"] not in ACTING_CAPABILITIES


def control_allowed(name: str, grounding_passed: bool | None) -> bool:
    """Desktop control authority for a GUI candidate.

    ``grounding_passed`` is None when no verdict artifact exists. That is the
    state the UI-TARS gate has been in since it was killed mid-run, and it must
    read as "not allowed" -- an absent measurement is not a passing one.
    """
    candidate = CANDIDATES.get(name)
    if candidate is None or grounding_passed is not True:
        return False
    return candidate["capability"] == "computer-use-grounding"


def remote_code_execution_allowed(name: str, review_passed: bool | None) -> bool:
    """Import/exec authority for a candidate that ships executable model code."""
    candidate = CANDIDATES.get(name)
    if candidate is None:
        return False
    if not candidate["requires_remote_code"]:
        return True
    return review_passed is True


def index_conflicts(text_index: dict = TEXT_INDEX,
                    multimodal_index: dict = MULTIMODAL_INDEX) -> list[str]:
    """Refuse any configuration that would pour two vector spaces into one store."""
    problems = []
    if Path(text_index["database"]) == Path(multimodal_index["database"]):
        problems.append("multimodal and text embeddings share one database file")
    if text_index["alias"] == multimodal_index["alias"]:
        problems.append("multimodal and text embeddings share one model alias")
    if text_index["dimension"] == multimodal_index["dimension"]:
        problems.append("index separation is being justified by dimension alone")
    return problems


def installed_inventory(path: Path = QUEUE_PATH) -> dict:
    """Exact-size check of every pinned phase-four file that is on disk.

    Size only. The transfer already verified publisher SHA-256 for every file
    that has one, and re-hashing ~99 GB of ignored weights here would be an
    expensive way to learn the same thing.
    """
    artifacts = {}
    problems = []
    for key, artifact in queue_artifacts(path).items():
        files = []
        for entry in artifact["files"]:
            destination = Path(entry["destination"])
            exists = destination.is_file()
            size = destination.stat().st_size if exists else 0
            ok = exists and size == int(entry["size"])
            files.append({
                "path": str(destination),
                "exists": exists,
                "bytes": size,
                "expected_bytes": int(entry["size"]),
                "ok": ok,
            })
            if not ok:
                problems.append(f"{key}: {destination.name} missing or wrong size")
        artifacts[key] = {
            "repository": artifact["repository"],
            "revision": artifact["revision"],
            "capability": artifact["capability"],
            "files": files,
            "pass": all(item["ok"] for item in files),
        }
    return {"artifacts": artifacts, "problems": problems, "pass": not problems}


def already_satisfied_conflicts(queue_path: Path = QUEUE_PATH,
                                earlier_path: Path = PHASE_ONE_QUEUE_PATH) -> list[str]:
    """Prove the ASR entry really is satisfied instead of merely being claimed.

    Phase four skips Qwen3-ASR because phase one already installed the same
    pinned revision. That claim is only safe if the earlier queue actually
    carries that key at that revision; otherwise the candidate is silently
    missing rather than deduplicated.
    """
    problems = []
    earlier = queue_artifacts(earlier_path)
    for entry in load_queue(queue_path).get("already_satisfied", []):
        key = entry["key"]
        source = earlier.get(key)
        if source is None:
            problems.append(f"{key}: not present in {entry.get('source_queue', 'the named queue')}")
            continue
        if source["revision"] != entry["revision"]:
            problems.append(
                f"{key}: phase one pinned {source['revision']}, phase four claims {entry['revision']}")
        if source["repository"] != entry["repository"]:
            problems.append(
                f"{key}: phase one pinned {source['repository']}, phase four claims {entry['repository']}")
    return problems


def admission_blockers(name: str, gate_results: dict[str, bool | None]) -> list[str]:
    """Everything standing between a candidate and promotion, as strings.

    ``gate_results`` maps a gate name to True/False/None. None means the gate has
    no verdict on disk, which blocks exactly as a failure does.
    """
    candidate = CANDIDATES.get(name)
    if candidate is None:
        return [f"{name}: not a reviewed phase-four candidate"]
    blockers = []
    for gate in candidate["prerequisites"]:
        result = gate_results.get(gate)
        if result is None:
            blockers.append(f"{name}: prerequisite gate {gate!r} has no verdict")
        elif result is not True:
            blockers.append(f"{name}: prerequisite gate {gate!r} failed")
    if candidate["abliterated"] and not abliteration_allowed(name):
        blockers.append(f"{name}: ablated weights are refused for {candidate['capability']}")
    if candidate["requires_remote_code"] and not remote_code_execution_allowed(
            name, gate_results.get("wemm-remote-code-review")):
        blockers.append(f"{name}: pinned remote code is not approved for execution")
    return blockers


def candidate_for_preset(preset: str) -> str | None:
    """Map a router preset name back to the candidate that owns it.

    Presets and candidates are not one-to-one: one candidate can own a text
    preset and a projector preset, and both inherit the candidate's privilege.
    """
    for name, candidate in CANDIDATES.items():
        if preset in candidate["presets"]:
            return name
    return None


def router_preset_names(path: Path = ROOT / "llama-models.ini") -> tuple[str, ...]:
    """Named aliases in the live router INI, excluding the shared [*] section."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and stripped != "[*]":
            names.append(stripped[1:-1])
    return tuple(names)


def uncovered_router_presets(path: Path = ROOT / "llama-models.ini") -> list[str]:
    """Aliases that would otherwise inherit tools by falling through."""
    return [name for name in router_preset_names(path) if privilege_for_preset(name) is None]
