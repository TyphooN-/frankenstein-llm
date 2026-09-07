#!/usr/bin/env python3
"""GPU/host policy for local ComfyUI image and music gates.

GPU2 is the display adapter and is never exposed to ComfyUI. GPU0 and GPU1 are
the headless ROCm compute cards. Z-Image Turbo BF16 is ~20.6 GiB on disk and
cannot live on a 16 GiB card, so GPU1 (V620 32 GiB) is the primary device and
GPU0 is overflow-only. Weights are never duplicated to fake utilization.
"""
from __future__ import annotations

from pathlib import Path
import os

ROOT = Path("/home/typhoon/git/frankenstein-llm")
COMFY_ROOT = ROOT / "tools" / "ComfyUI"
COMFY_PYTHON = ROOT / "venvs" / "comfy" / "bin" / "python"
EXTRA_PATHS = Path(__file__).resolve().parent / "extra_model_paths.yaml"
EVIDENCE = Path(__file__).resolve().parent / "evidence"

ARTIFACTS = {
    "z-image-unet": ROOT / "models/comfy/diffusion_models/z_image_turbo_bf16.safetensors",
    "z-image-clip": ROOT / "models/comfy/text_encoders/qwen_3_4b.safetensors",
    "z-image-vae": ROOT / "models/comfy/vae/ae.safetensors",
    "ace-step-1.5-aio": ROOT / "models/comfy/checkpoints/ace_step_1.5_turbo_aio.safetensors",
    "qwen-image-edit-unet": ROOT / "models/comfy/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors",
    "qwen-image-edit-clip": ROOT / "models/comfy/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen-image-edit-vae": ROOT / "models/comfy/vae/qwen_image_vae.safetensors",
    "qwen-image-edit-lora": ROOT / "models/comfy/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "flux2-klein-unet": ROOT / "models/comfy/flux2-klein-4b/flux-2-klein-4b.safetensors",
    "flux2-klein-vae": ROOT / "models/comfy/flux2-klein-4b/vae/diffusion_pytorch_model.safetensors",
}

REQUIRED_BYTES = {
    "z-image-unet": 12309866400,
    "z-image-clip": 8044982048,
    "z-image-vae": 335304388,
    "ace-step-1.5-aio": 10025478736,
    "qwen-image-edit-unet": 20499083824,
    "qwen-image-edit-clip": 9384670680,
    "qwen-image-edit-vae": 253806246,
    "qwen-image-edit-lora": 849608296,
    "flux2-klein-unet": 7751105712,
    "flux2-klein-vae": 168120878,
}

# Workflow coverage claims, stated as data so a passing preflight cannot be read
# as covering more than it does.
#
# Z-Image Turbo is a text-to-image generation checkpoint. Instruction-following
# editors are inventoried separately below; they are on disk and still unproven.
#
# artifacts_present answers "is the weight on disk"; functionally_proven answers
# "did a gate observe it working". Only the second is admission.
WORKFLOW_CLAIMS = {
    "image-generation": {
        "artifacts_present": True,
        "artifacts": ["z-image-unet", "z-image-clip", "z-image-vae"],
        "functionally_proven": False,
        "note": "Z-Image Turbo BF16 text-to-image; static preflight only",
    },
    "image-editing": {
        "artifacts_present": True,
        # Every file the pinned graphs load, not only the headline weights. The
        # FLUX.2 Klein graph reuses the Z-Image Qwen3-4B text encoder rather than
        # the publisher's sharded copy, so the editing lane really does depend on
        # a file the generation lane owns; leaving that undeclared would let the
        # preflight report the lane complete after the encoder was removed.
        "artifacts": [
            "qwen-image-edit-unet",
            "qwen-image-edit-clip",
            "qwen-image-edit-vae",
            "qwen-image-edit-lora",
            "flux2-klein-unet",
            "flux2-klein-vae",
            "z-image-clip",
        ],
        "functionally_proven": False,
        "note": "Qwen Image Edit 2511 INT8 + Lightning LoRA and FLUX.2 Klein 4B "
                "are on disk; no live ComfyUI run has been admitted",
    },
    "music-generation": {
        "artifacts_present": True,
        "artifacts": ["ace-step-1.5-aio"],
        "functionally_proven": False,
        "note": "ACE-Step 1.5 turbo all-in-one; static preflight only",
    },
}

REQUIRED_NODE_FILES = {
    "z-image": COMFY_ROOT / "comfy_extras" / "nodes_zimage.py",
    "ace-step": COMFY_ROOT / "comfy_extras" / "nodes_ace.py",
}

REQUIRED_NODE_IDS = {
    "z-image": ("TextEncodeZImageOmni",),
    "ace-step": ("TextEncodeAceStepAudio1.5", "EmptyAceStep1.5LatentAudio"),
}

# Physical roles. ComfyUI is launched with HIP_VISIBLE_DEVICES=0,1 so its
# cuda:0 is GPU0 and cuda:1 is GPU1. GPU2 never appears.
GPU_ROLES = {
    0: {"name": "RX 6900 XT", "role": "headless-compute", "nominal_gib": 16},
    1: {"name": "Radeon Pro V620", "role": "headless-compute-primary", "nominal_gib": 32},
    2: {"name": "RX 6900 XT", "role": "display", "nominal_gib": 16},
}

VISIBLE_COMPUTE = "0,1"
PRIMARY_DEVICE = 1
LISTEN = "127.0.0.1"
PORT = 8188

# Task names as they appear in /proc/<pid>/comm, which the kernel truncates to
# 15 characters. Deliberately broad: every compile or link job on this host
# competes with a GPU gate for the same cores and memory bandwidth, whether or
# not the tree it is building is the kernel. "kernel-build" below is the name of
# that whole class rather than a claim that vmlinux specifically is being
# linked. The mission supervisor's conflict_reason() holds the canonical list
# and this one tracks it; test_media_policy asserts the two do not drift.
BUILD_COMMANDS = frozenset({
    "make", "gmake", "makepkg", "cmake", "ninja", "samu", "meson", "scons",
    "ccache", "sccache", "cc", "c++", "gcc", "g++", "clang", "clang++",
    "cc1", "cc1plus", "lto1", "lto-wrapper", "collect2", "as", "ar", "ranlib",
    "strip", "objtool", "ld", "ld.bfd", "ld.gold", "ld.lld", "lld", "mold",
    "cargo", "rustc", "go", "nvcc", "cicc", "ptxas", "hipcc",
    "pacman", "paru", "yay", "dkms", "link-vmlinux.sh",
})

# A build is also named by where it runs, which is how a plain "sh" driving
# link-vmlinux.sh is caught after the last compiler has exited. Matched as a
# substring for the same reason the supervisor matches it that way: sibling
# trees such as linux-tkg-git are the same build.
KERNEL_TREE = "/linux-tkg"
# A cwd whose directory was removed reads back as "<path> (deleted)".
DELETED_SUFFIX = " (deleted)"
UNAVAILABLE = "process-inspection-unavailable"


def host_exclusive_blockers(proc_root: Path = Path("/proc")) -> list[str]:
    """Return the first build blocker, or why the scan could not tell.

    ``proc_root`` is injectable for the reason the mission supervisor's is: a
    classification checked against whatever the host happens to be running at
    the time is not really checked at all.

    Only ``comm`` and the ``cwd`` symlink are read. Linux serves
    ``/proc/<pid>/cmdline`` through ``access_remote_vm``, so a task holding its
    own mmap write lock can wedge a reader that merely tried to inspect it --
    and the wide parallel compile this function exists to notice is exactly such
    a task. ``comm`` and ``cwd`` are kernel metadata; neither touches a target's
    address space, and nothing here opens ``cmdline`` at all.

    The scan stops at the first match. One blocker already withholds
    ``functional_gate_ready_now``, and a second adds nothing but a longer walk
    across a host that is by definition busy. Pids are visited in numeric order
    so the same host produces the same recorded blocker twice running.

    Failing closed is applied where it distinguishes anything. A process table
    that cannot be listed, and a ``comm`` that cannot be read for a task still
    in it, are reported rather than skipped: "nothing to see" and "nowhere to
    look" are the same empty list, and only one of them means the host is quiet.
    An unreadable ``cwd`` is the opposite case -- ``readlink`` on another user's
    process is refused as a matter of course, root's included, so treating that
    as a blocker would report every host as permanently busy and never let a
    gate run. ``comm`` stays world-readable there, so such a process is still
    classified by name; only the kernel-tree signal is lost.
    """
    try:
        pids = sorted(int(entry.name) for entry in proc_root.iterdir()
                      if entry.name.isdigit())
    except OSError:
        return [UNAVAILABLE]
    for pid in pids:
        if pid == os.getpid():
            continue
        entry = proc_root / str(pid)
        try:
            command = (entry / "comm").read_bytes().decode(errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError):
            # The task exited between listing the table and reading it.
            continue
        except OSError:
            return [f"pid={pid} {UNAVAILABLE}"]
        if not command:
            return [f"pid={pid} {UNAVAILABLE}"]
        if command in BUILD_COMMANDS:
            return [f"pid={pid} kernel-build"]
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            # Kernel threads have no cwd link and other users' processes do not
            # expose theirs. Both are ordinary; see the docstring.
            cwd = ""
        if KERNEL_TREE in cwd.removesuffix(DELETED_SUFFIX):
            return [f"pid={pid} kernel-build"]
    return []


def comfy_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = VISIBLE_COMPUTE
    env["ROCR_VISIBLE_DEVICES"] = VISIBLE_COMPUTE
    env["CUDA_VISIBLE_DEVICES"] = VISIBLE_COMPUTE
    env.pop("DISPLAY", None)
    return env


def comfy_argv() -> list[str]:
    return [
        str(COMFY_PYTHON),
        str(COMFY_ROOT / "main.py"),
        "--listen", LISTEN,
        "--port", str(PORT),
        "--disable-auto-launch",
        "--extra-model-paths-config", str(EXTRA_PATHS),
        "--cuda-device", str(PRIMARY_DEVICE),
    ]


def artifact_inventory() -> dict:
    files = {}
    problems = []
    for key, path in ARTIFACTS.items():
        expected = REQUIRED_BYTES[key]
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        ok = exists and size == expected
        files[key] = {
            "path": str(path),
            "exists": exists,
            "bytes": size,
            "expected_bytes": expected,
            "ok": ok,
        }
        if not ok:
            problems.append(f"{key}: missing or wrong size")
    return {"files": files, "problems": problems, "pass": not problems}


# ---------------------------------------------------------------------------
# Static workflow resolution
# ---------------------------------------------------------------------------
# A pinned API graph names weights by bare filename. ComfyUI turns that name into
# a path by searching the folders registered for the node's category, so a graph
# and an inventory can agree on every number and still disagree about reality:
# the graph can name a file that is not installed, extra_model_paths can leave
# the folder a file lives in out of the category that needs it, or a claim can
# list fewer artifacts than its own graph loads. None of that is visible from
# file sizes, and all of it only surfaces as a node error during a live run --
# which is exactly the run this host is not allowed to spend.
#
# Resolving the graphs against the real search paths is still static evidence. It
# says the graph *could* load; it says nothing about whether the workflow works.
GRAPH_FILE_INPUTS = {
    "ckpt_name": "checkpoints",
    "clip_name": "text_encoders",
    "lora_name": "loras",
    "unet_name": "diffusion_models",
    "vae_name": "vae",
}


def search_paths() -> dict[str, list[Path]]:
    """Model folders per category, as ComfyUI reads ``extra_model_paths.yaml``.

    Each category value may hold several newline-separated relative paths; every
    one of them is searched, so all of them are returned in declaration order.
    """
    import yaml

    document = yaml.safe_load(EXTRA_PATHS.read_text()) or {}
    paths: dict[str, list[Path]] = {}
    for section in document.values():
        if not isinstance(section, dict):
            continue
        base = Path(section.get("base_path", ""))
        for category, value in section.items():
            if category in ("base_path", "is_default") or not isinstance(value, str):
                continue
            for entry in value.split("\n"):
                entry = entry.strip()
                if entry:
                    paths.setdefault(category, []).append(base / entry)
    return paths


def graph_references(graph: dict) -> list[dict]:
    """Every weight file a graph names, with the category ComfyUI searches for it."""
    references = []
    for node_id, node in sorted(graph.items()):
        for field, category in GRAPH_FILE_INPUTS.items():
            filename = (node.get("inputs") or {}).get(field)
            if isinstance(filename, str) and filename:
                references.append({"node": node_id, "class_type": node.get("class_type"),
                                   "input": field, "category": category,
                                   "filename": filename})
    return references


def artifact_key_for(path: Path | str) -> str | None:
    """Reverse-lookup an inventoried artifact key for a resolved path."""
    resolved = str(path)
    for key, known in ARTIFACTS.items():
        if str(known) == resolved:
            return key
    return None


def resolve_reference(reference: dict, paths: dict[str, list[Path]],
                      exists=Path.is_file) -> dict:
    """Locate one graph reference on disk and name the artifact it lands on.

    ``exists`` is injectable for the same reason ``probe_environment`` is: the
    question "do the pinned graphs, the search paths and the declared artifacts
    agree" is answerable without 60 GB of weights, and a check that can only run
    on this one populated host is a check that stops running.
    """
    candidates = [directory / reference["filename"]
                  for directory in paths.get(reference["category"], [])]
    matches = [candidate for candidate in candidates if exists(candidate)]
    resolved = matches[0] if matches else None
    return {
        **reference,
        "searched": [str(directory) for directory in paths.get(reference["category"], [])],
        "matches": [str(match) for match in matches],
        # ComfyUI takes the first hit, so a second hit is a silent coin flip
        # between two different checkpoints with the same basename.
        "ambiguous": len(matches) > 1,
        "resolved": str(resolved) if resolved else None,
        "artifact_key": artifact_key_for(resolved) if resolved else None,
    }


def resolve_graphs(graphs: dict[str, dict[str, dict]],
                   paths: dict[str, list[Path]] | None = None,
                   exists=Path.is_file) -> list[dict]:
    """Resolve every pinned graph, grouped by the workflow claim that owns it."""
    paths = search_paths() if paths is None else paths
    resolved = []
    for workflow, named in sorted(graphs.items()):
        for label, graph in sorted(named.items()):
            resolved.append({
                "workflow": workflow,
                "graph": label,
                "references": [resolve_reference(reference, paths, exists)
                               for reference in graph_references(graph)],
            })
    return resolved


def graph_problems(resolutions: list[dict]) -> list[str]:
    """Fail-closed reading of a graph resolution: unresolved, unknown, undeclared."""
    problems = []
    required: dict[str, set[str]] = {}
    for entry in resolutions:
        where = f"{entry['workflow']}/{entry['graph']}"
        for reference in entry["references"]:
            name = f"{where} node {reference['node']} {reference['input']}={reference['filename']}"
            if reference["resolved"] is None:
                problems.append(f"{name}: not found under {reference['category']} search paths")
                continue
            if reference["ambiguous"]:
                problems.append(f"{name}: resolves to {len(reference['matches'])} files")
            if reference["artifact_key"] is None:
                problems.append(f"{name}: resolves to an uninventoried file")
            else:
                required.setdefault(entry["workflow"], set()).add(reference["artifact_key"])
    for workflow, keys in sorted(required.items()):
        claim = WORKFLOW_CLAIMS.get(workflow)
        if claim is None:
            problems.append(f"{workflow}: graph has no workflow claim")
            continue
        undeclared = sorted(keys - set(claim["artifacts"]))
        if undeclared:
            problems.append(
                f"{workflow}: graph loads undeclared artifacts {undeclared}")
    return problems
