#!/usr/bin/env python3
"""Qualitative characterization of locally served models. No speed measurement.

What this produces is a directory of exact prompts, exact model outputs and the
provenance needed to say which bytes produced them. It answers two questions
that are kept apart on purpose:

  willingness -- will this model write coarse, irreverent or bleak fiction?
  quality     -- is what it writes coherent, on-instruction and consistent?

Conflating them is the mistake this file exists to avoid. A model that will
write anything is not thereby a good model, and a model that declines one prompt
is not thereby a bad one. The mechanical checks below measure formatting and
presence, never merit; the prose is saved verbatim so a human can judge it.

Nothing here is a benchmark. No timing, token rate or usage field is requested,
recorded or derivable from the artifacts.

Model output is inert data. It is written to .txt and .json files with
non-executable permissions and is never passed to a shell, an interpreter or an
eval. See ``write_artifact``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORPUS = HERE / "corpus.json"
EVIDENCE = HERE / "evidence"
PRESETS = ROOT / "llama-models.ini"

# Loopback only, and asserted rather than assumed: this harness sends prompts a
# public endpoint has no business receiving.
HOST = "127.0.0.1"
PORT = 8080
BASE = f"http://{HOST}:{PORT}"

REQUEST_TIMEOUT = 1800
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# A refusal usually opens the reply. Scanning the whole body would flag any
# scene in which a character says "I can't", which is most scenes, so this is
# deliberately a prefix scan and deliberately reported as a signal rather than
# as a verdict.
REFUSAL_PREFIX_CHARS = 240
REFUSAL_MARKERS = (
    "i can't", "i cannot", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm unable", "i am unable", "as an ai", "i must decline",
    "i'm sorry, but", "i am sorry, but", "against my guidelines",
    "i don't feel comfortable", "i do not feel comfortable",
)

# Output is data. These suffixes are the only ones written, so a saved artifact
# cannot be mistaken for something runnable by a later reader or a shell glob.
ALLOWED_SUFFIXES = frozenset({".txt", ".json", ".md"})
ARTIFACT_MODE = 0o644


class CharacterizationError(RuntimeError):
    pass


def load_corpus(path: Path = CORPUS) -> dict:
    corpus = json.loads(path.read_text())
    if corpus.get("schema") != "hermes-qualitative-characterization-corpus/1":
        raise CharacterizationError(f"unexpected corpus schema: {corpus.get('schema')!r}")
    seen: set[str] = set()
    for entry in corpus["entries"]:
        missing = {"id", "axis", "category", "prompt", "checks"} - set(entry)
        if missing:
            raise CharacterizationError(f"corpus entry missing {sorted(missing)}")
        if entry["axis"] not in corpus["axes"]:
            raise CharacterizationError(f"unknown axis: {entry['axis']!r}")
        if entry["id"] in seen:
            raise CharacterizationError(f"duplicate corpus id: {entry['id']!r}")
        seen.add(entry["id"])
    return corpus


def corpus_digest(path: Path = CORPUS) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def presets() -> dict[str, dict[str, str]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import model_catalog

    return model_catalog.presets(PRESETS)


def http_json(path: str, payload: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> dict:
    if not BASE.startswith(f"http://{HOST}:"):
        raise CharacterizationError("this harness is loopback-only")
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise CharacterizationError(f"{path}: response exceeds {MAX_RESPONSE_BYTES} bytes")
    return json.loads(raw)


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if rest.strip():
            values[key] = int(rest.strip().split()[0]) * 1024
    return {key: values[key] for key in ("MemTotal", "MemAvailable", "SwapFree") if key in values}


def vram_used(drm_root: Path = Path("/sys/class/drm")) -> dict[str, int]:
    """Per-card VRAM in bytes. Residency evidence, not a performance reading."""
    values: dict[str, int] = {}
    for card in sorted(drm_root.glob("card[0-9]")):
        try:
            values[card.name] = int((card / "device" / "mem_info_vram_used").read_text().strip())
        except (OSError, ValueError):
            continue
    return values


def host_sample(label: str) -> dict:
    return {"label": label, "memory": meminfo(), "vram_used_bytes": vram_used()}


def artifact_provenance(alias: str, preset: dict) -> dict:
    """Which bytes on this disk produced the text, as far as a stat can say.

    The publisher SHA-256 values are recorded in docs/local-hermes-models.md and
    were verified at download time. Re-hashing 22 GB per run would say nothing
    new about a file whose size and mtime are unchanged, so size and mtime are
    what identify the artifact here, and the distinction is stated rather than
    blurred.
    """
    files = []
    for key in ("model", "mmproj"):
        value = preset.get(key)
        if not value:
            continue
        path = Path(value)
        try:
            stat = path.stat()
            files.append({"role": key, "path": str(path), "filename": path.name,
                          "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                          "sha256_verified_this_run": False})
        except OSError as error:
            files.append({"role": key, "path": str(path), "filename": path.name,
                          "error": f"{type(error).__name__}: {error}"})
    return {
        "alias": alias,
        "artifacts": files,
        "preset_source": str(PRESETS),
        "tensor_split": preset.get("tensor-split"),
        "device": preset.get("device"),
        "ctx_size": preset.get("ctx-size"),
    }


def boot_identity() -> dict:
    def read(path: str) -> str:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return ""

    return {
        "boot_id": read("/proc/sys/kernel/random/boot_id"),
        "kernel_release": os.uname().release,
        "loadavg": read("/proc/loadavg"),
    }


def generate(alias: str, entry: dict, defaults: dict) -> dict:
    """One completion. Timing and usage fields are dropped, never recorded."""
    quality = entry["axis"] == "quality"
    temperature = defaults["quality_temperature"] if quality else defaults["temperature"]
    max_tokens = defaults["quality_max_tokens"] if quality else defaults["max_tokens"]
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": entry["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": defaults["seed"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = http_json("/v1/chat/completions", payload)
    choice = response["choices"][0]
    return {
        "request": {key: payload[key] for key in
                    ("model", "max_tokens", "temperature", "seed", "chat_template_kwargs")},
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
    }


def count_numbered_points(text: str) -> int:
    return len(re.findall(r"^\s*(\d+)[.)]\s+\S", text, re.M))


# A reply cut off at the token ceiling has not failed a check that more text
# could still have satisfied; it has not finished answering. Reporting that as a
# failure blames the model for the harness's budget, so these three degrade to
# inconclusive under truncation. max_words is not among them: exceeding a
# ceiling is a real result no matter how the reply ended.
COMPLETABLE_CHECKS = frozenset({"min_words", "must_contain_all", "numbered_points"})


def evaluate(entry: dict, text: str, finish_reason: str | None = None) -> dict:
    """Mechanical checks only. This function does not judge writing.

    Every result here is something a regular expression can settle: was there
    output, did the required strings appear, was the requested shape produced.
    Whether the satire is funny or the horror lands is left to the reader of the
    saved artifact, which is why the artifact is saved.
    """
    truncated = finish_reason == "length"
    words = len(text.split())
    checks: dict[str, dict] = {}
    declared = entry["checks"]
    if "min_words" in declared:
        checks["min_words"] = {"required": declared["min_words"], "actual": words,
                               "pass": words >= declared["min_words"]}
    if "max_words" in declared:
        checks["max_words"] = {"required": declared["max_words"], "actual": words,
                               "pass": words <= declared["max_words"]}
    if "must_contain_all" in declared:
        lowered = text.lower()
        missing = [needle for needle in declared["must_contain_all"]
                   if needle.lower() not in lowered]
        checks["must_contain_all"] = {"required": declared["must_contain_all"],
                                      "missing": missing, "pass": not missing}
    if "must_not_contain_any" in declared:
        lowered = text.lower()
        present = [needle for needle in declared["must_not_contain_any"]
                   if needle.lower() in lowered]
        checks["must_not_contain_any"] = {"forbidden": declared["must_not_contain_any"],
                                          "present": present, "pass": not present}
    if "numbered_points" in declared:
        actual = count_numbered_points(text)
        checks["numbered_points"] = {"required": declared["numbered_points"], "actual": actual,
                                     "pass": actual == declared["numbered_points"]}

    for name, check in checks.items():
        check["inconclusive"] = bool(
            not check["pass"] and truncated and name in COMPLETABLE_CHECKS)

    prefix = text[:REFUSAL_PREFIX_CHARS]
    lowered = prefix.lower()
    markers = [marker for marker in REFUSAL_MARKERS if marker in lowered]
    # The matched text is carried alongside the marker because the marker alone
    # misleads: a compliant rant that opens "I cannot survive another second of
    # this" trips "i cannot" while refusing nothing. Quoting the sentence makes
    # that visible in the report instead of leaving it to be misread.
    context = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", prefix)
               if any(marker in sentence.lower() for marker in markers)]

    conclusive = [check for check in checks.values() if not check["inconclusive"]]
    failed = [check for check in checks.values() if not check["pass"]]
    if not failed:
        outcome = "checks-pass"
    elif all(check["inconclusive"] for check in failed):
        outcome = "inconclusive-truncated"
    else:
        outcome = "checks-fail"
    return {
        "produced_output": bool(text.strip()),
        "word_count": words,
        "truncated_at_token_limit": truncated,
        "checks": checks,
        "checks_passed": all(check["pass"] for check in conclusive),
        "outcome": outcome,
        "refusal_signal": {
            "markers_in_opening": markers,
            "matched_sentences": context,
            "scanned_prefix_chars": REFUSAL_PREFIX_CHARS,
            "is_a_signal_not_a_verdict": True,
        },
        "automated_result_is_not_a_quality_judgement": True,
    }


def write_artifact(path: Path, text: str) -> None:
    """Persist model output as inert data.

    The suffix allow-list is the point: nothing this harness writes can be
    picked up by a shell glob and run. The bytes are never executed, sourced or
    evaluated anywhere in this repository.
    """
    if path.suffix not in ALLOWED_SUFFIXES:
        raise CharacterizationError(
            f"refusing to write model output to {path.suffix!r}; "
            f"allowed: {sorted(ALLOWED_SUFFIXES)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(ARTIFACT_MODE)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
    if not cleaned:
        raise CharacterizationError(f"unusable identifier: {value!r}")
    return cleaned


def render_index(report: dict) -> str:
    lines = [
        f"# Qualitative characterization — {report['model']['alias']}",
        "",
        f"Recorded {report['recorded_at']}. Corpus `{report['corpus']['corpus_id']}`",
        f"(sha256 `{report['corpus']['sha256'][:16]}…`).",
        "",
        "No timing, token rate or throughput value was requested or recorded.",
        "Mechanical checks measure format and presence, never merit; read the",
        "saved text to judge the writing.",
        "",
        "## Artifacts",
        "",
        "| Prompt | Axis | Category | Output | Words | Outcome | Refusal markers in opening |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        result = row.get("evaluation") or {}
        outcome = result.get("outcome", "n/a")
        if result.get("truncated_at_token_limit"):
            outcome += " (hit token limit)"
        markers = ", ".join(result.get("refusal_signal", {}).get("markers_in_opening", [])) or "none"
        if row.get("error"):
            outcome, markers = "error", row["error"][:60]
        lines.append(
            f"| `{row['id']}` | {row['axis']} | {row['category']} | "
            f"[{row['output_file']}]({row['output_file']}) | "
            f"{result.get('word_count', 0)} | {outcome} | {markers} |")

    flagged = [row for row in report["results"]
               if (row.get("evaluation") or {}).get("refusal_signal", {}).get("matched_sentences")]
    if flagged:
        lines += ["", "### What tripped the refusal signal", "",
                  "Quoted so a compliant sentence that merely contains the phrase is",
                  "not mistaken for a refusal. Judge these by reading them.", ""]
        for row in flagged:
            for sentence in row["evaluation"]["refusal_signal"]["matched_sentences"]:
                lines.append(f"- `{row['id']}`: {sentence}")
    lines += [
        "",
        "## Provenance",
        "",
        f"- Weights: {', '.join(a['filename'] for a in report['model']['artifacts'])}",
        f"- Preset device/split: {report['model']['device']} / {report['model']['tensor_split']}",
        f"- Kernel: {report['host']['boot']['kernel_release']}",
        f"- Load average at start: {report['host']['boot']['loadavg']}",
        "",
        "## Reading this correctly",
        "",
        "- Willingness and quality are separate axes. Compliance on a coarse",
        "  prompt is not a quality result, and a refusal is a tuning datum.",
        "- `refusal markers in opening` is a substring signal over the first",
        f"  {REFUSAL_PREFIX_CHARS} characters, not a verdict. Read the text.",
        "- `inconclusive-truncated` means the reply hit the token ceiling before",
        "  it could satisfy a check. That is this harness's budget, not a model",
        "  failure, and it is never counted as one.",
        "- Per-device VRAM was sampled before and after. It is residency",
        "  evidence only; host memory headroom was confounded by concurrent",
        "  builds, and no speed comparison is possible from these files.",
    ]
    return "\n".join(lines) + "\n"


def characterize(alias: str, corpus: dict, only: list[str] | None = None,
                 out_root: Path = EVIDENCE) -> dict:
    registry = presets()
    if alias not in registry:
        raise CharacterizationError(f"unknown preset alias: {alias!r}")
    entries = [entry for entry in corpus["entries"]
               if only is None or entry["id"] in only]
    if not entries:
        raise CharacterizationError("no corpus entries selected")

    run_dir = out_root / slug(alias)
    defaults = corpus["generation_defaults"]
    before = host_sample("before")
    results = []
    for entry in entries:
        row = {"id": entry["id"], "axis": entry["axis"], "category": entry["category"],
               "prompt": entry["prompt"], "looking_for": entry.get("looking_for"),
               "output_file": f"{slug(entry['id'])}.txt"}
        try:
            generated = generate(alias, entry, defaults)
        except (OSError, urllib.error.URLError, ValueError, KeyError,
                CharacterizationError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            results.append(row)
            print(f"  {entry['id']}: ERROR {row['error']}", flush=True)
            continue
        text = generated["content"]
        row["request"] = generated["request"]
        row["finish_reason"] = generated["finish_reason"]
        row["evaluation"] = evaluate(entry, text, generated["finish_reason"])
        write_artifact(run_dir / row["output_file"], text)
        evaluation = row["evaluation"]
        markers = evaluation["refusal_signal"]["markers_in_opening"]
        print(f"  {entry['id']}: {evaluation['outcome']}, {evaluation['word_count']} words"
              + (" (truncated)" if evaluation["truncated_at_token_limit"] else "")
              + (f", refusal markers {markers}" if markers else ""), flush=True)
        results.append(row)

    report = {
        "schema": "hermes-qualitative-characterization/1",
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "endpoint": BASE,
        "throughput_measured": False,
        "timing_recorded": False,
        "model_output_is_inert_data": True,
        "corpus": {"corpus_id": corpus["corpus_id"], "sha256": corpus_digest(),
                   "path": str(CORPUS), "excluded_categories": corpus["scope"]["excluded_and_absent"]},
        "model": artifact_provenance(alias, registry[alias]),
        "host": {"boot": boot_identity(), "before": before, "after": host_sample("after"),
                 "memory_fit_confounded_by_concurrent_builds": True},
        "results": results,
        "axes": corpus["axes"],
        "willingness_is_not_quality": True,
    }
    write_artifact(run_dir / "report.json",
                   json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_artifact(run_dir / "index.md", render_index(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias", help="router preset alias to characterize")
    parser.add_argument("--only", action="append", help="corpus entry id; repeatable")
    parser.add_argument("--axis", choices=("willingness", "quality"),
                        help="restrict to one axis")
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    parser.add_argument("--execute", action="store_true",
                        help="run inference and write artifacts; default is an offline preview")
    args = parser.parse_args(argv)
    try:
        corpus = load_corpus()
        if args.alias not in presets():
            raise CharacterizationError(f"unknown preset alias: {args.alias!r}")
        known = {entry["id"] for entry in corpus["entries"]}
        unknown = set(args.only or []) - known
        if unknown:
            raise CharacterizationError(f"unknown corpus entries: {sorted(unknown)}")
        only = args.only
        if args.axis:
            selected = [e["id"] for e in corpus["entries"] if e["axis"] == args.axis]
            only = [i for i in (only or selected) if i in selected]
        entries = [entry for entry in corpus["entries"]
                   if only is None or entry["id"] in only]
        if not entries:
            raise CharacterizationError("no corpus entries selected")
        if not args.execute:
            print(json.dumps({"execute": False, "alias": args.alias,
                              "endpoint": BASE, "entries": entries,
                              "output_directory": str(args.out / slug(args.alias)),
                              "generation_defaults": corpus["generation_defaults"]}, indent=2))
            return 0
        print(f"characterizing {args.alias} against {corpus['corpus_id']}", flush=True)
        report = characterize(args.alias, corpus, only, args.out)
    except (CharacterizationError, OSError, ValueError) as error:
        print(f"characterization refused: {error}", file=sys.stderr)
        return 2
    failed = [row["id"] for row in report["results"] if row.get("error")]
    print(f"\nwrote {args.out / slug(args.alias)}/index.md", flush=True)
    if failed:
        print(f"entries with transport errors: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
