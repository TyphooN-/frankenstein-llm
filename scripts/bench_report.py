#!/usr/bin/env python3
"""Turn native ``llama-bench`` JSON into one markdown artifact per tested model.

Throughput is not part of the functional mission and never will be: the mission
records ``throughput_measured: false`` on purpose, because a token rate says
nothing about whether a preset answers, obeys a schema or calls a tool. This
module is the separate, explicitly authorized lane. It reads what
``scripts/model-runs.py benchmark`` already wrote under ``logs/model-runs/`` and
publishes it as prose; it never runs a model, and it has no way to produce a
number that a benchmark did not measure.

Three refusals carry the honesty of the output:

* A run whose native JSON is missing, unparsable or incomplete is refused. There
  is no fallback estimate, no "approximately", and no rate derived from a
  smoke-test's wall clock. An artifact either cites a measured row or does not
  exist.
* A ranking that mixes serving-MTP classes is refused. ``heretic``,
  ``obliterated``, ``ridge`` and ``fable`` serve with ``spec-type = draft-mtp``;
  ``llama-bench`` applies no such preset. Ordering an MTP-served preset against
  a non-MTP-served one by a number neither measured with MTP reads as a weight
  ranking, and it is not one.
* Two runs claiming the same alias are refused rather than silently collapsed
  into one file, because the surviving file would carry the other run's argv.

A run also has to have measured the placement the alias actually serves. The
argv recorded in ``report.json`` is compared against ``llama-models.ini`` by
ratio, so ``6/6/6`` and ``1,1,1`` agree and ``1,1,1`` against ``5,8,4`` does
not: an artifact headed ``qwen3-coder-next`` that measured an equal split
measured a placement nothing serves.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_vram  # noqa: E402
import model_catalog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/benchmarks"
# Written by the router functional gate. Ignored by git, so it is quoted with
# its own timestamp rather than treated as a durable repository fact.
FUNCTIONAL = ROOT / "verification/router-functional/evidence/router-functional.json"
# Optional, produced by ``bench_report.py sample`` alongside a run.
HOST_SAMPLE = "host-vram.json"

GIB = 1 << 30

# The llama-bench fields an artifact quotes as a measurement. Everything else is
# read with ``.get`` so an upstream column addition cannot fail a report.
REQUIRED_ROW_FIELDS = ("avg_ts", "n_prompt", "n_gen")

MTP_CAVEAT = (
    "`llama-bench` does not apply the router's chat or speculative-decoding "
    "preset. A `spec-type = draft-mtp` preset is served with multi-token "
    "prediction and benchmarked without it, so the rate recorded here is not the one "
    "that preset delivers through the router, and it may not be compared "
    "against a non-MTP preset as a ranking of the weights."
)
SMOKE_CAVEAT = (
    "The repository's functional gates emit a handful of tokens per check. Their "
    "timings are not throughput and are never quoted as a rate: a two-token "
    "`PONG` measures whether a preset answers, not how fast it runs."
)


class ReportRefused(ValueError):
    """The requested artifact would assert something the evidence does not."""


class MissingNativeResults(ReportRefused):
    """A run has no complete native benchmark JSON to quote."""


class RankingRefused(ReportRefused):
    """An ordering was requested across runs that are not comparable."""


def split_ratio(text: str) -> tuple[Fraction, ...]:
    """Normalized proportions, so two spellings of one placement compare equal.

    ``--tensor-split`` takes proportions, not gibibytes. ``6/6/6`` in
    ``config/model-runs.json`` and ``1,1,1`` in ``llama-models.ini`` are the
    same placement; reading either as a byte budget would size every model to
    18 GiB on cards that hold 16.
    """
    parts = [part for part in re.split(r"[/,]", text.strip()) if part != ""]
    try:
        values = [Fraction(part) for part in parts]
    except (ValueError, ZeroDivisionError) as error:
        raise ReportRefused(f"unparsable tensor split {text!r}") from error
    total = sum(values)
    if not values or any(value < 0 for value in values) or total <= 0:
        raise ReportRefused(f"tensor split {text!r} has no positive share")
    return tuple(value / total for value in values)


def argv_value(command: list[str], flag: str) -> str:
    """One flag's argument out of the argv the run actually executed."""
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise MissingNativeResults(
            f"recorded benchmark command has no {flag} argument") from error


def native_rows(run_dir: Path) -> tuple[dict, list[dict]]:
    """The run report and its native rows, or a refusal.

    Every failure mode here ends in a refusal rather than a partial artifact.
    An interrupted or failed benchmark left a report behind on purpose so an
    operator can see what happened; publishing a rate from one would turn a
    recorded failure into a measurement.
    """
    report_path = run_dir / "report.json"
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MissingNativeResults(f"{report_path}: {error}") from error
    if report.get("mode") != "benchmark":
        raise MissingNativeResults(f"{run_dir.name} is not a benchmark run")
    if report.get("status") != "passed" or report.get("exit_code") != 0:
        raise MissingNativeResults(
            f"{run_dir.name} status is {report.get('status')!r} with exit code "
            f"{report.get('exit_code')!r}; an unsuccessful run has no rate to quote")
    if "native_results" not in report:
        raise MissingNativeResults(
            f"{run_dir.name} recorded no native_results; refusing to invent one")
    native = run_dir / report["native_results"]
    try:
        rows = json.loads(native.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MissingNativeResults(f"{native}: {error}") from error
    if not isinstance(rows, list) or not rows:
        raise MissingNativeResults(f"{native} is not a non-empty JSON array")
    for row in rows:
        if not isinstance(row, dict):
            raise MissingNativeResults(f"{native} contains a non-object row")
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            raise MissingNativeResults(
                f"{native} row is missing measured fields: {', '.join(missing)}")
    return report, rows


def throughput(rows: list[dict]) -> dict[str, dict]:
    """The prompt-processing and token-generation rows of one workload.

    ``-p N -n M`` emits exactly one of each. Anything else is a different
    workload than the one this lane standardized on, and averaging it into a
    single figure would hide which configuration produced which number.
    """
    prompt = [row for row in rows if row["n_prompt"] > 0 and row["n_gen"] == 0]
    generation = [row for row in rows if row["n_gen"] > 0 and row["n_prompt"] == 0]
    if len(prompt) != 1 or len(generation) != 1:
        raise MissingNativeResults(
            f"expected one prompt row and one generation row, got "
            f"{len(prompt)} and {len(generation)}")
    return {"prompt": prompt[0], "generation": generation[0]}


def resolve_alias(model_path: Path, registry: dict, explicit: str | None) -> str:
    """Which preset a run measured.

    A vision preset shares its weight file with its text-only sibling, so a path
    does not always identify one alias. That case is refused rather than guessed
    at: the two presets carry different placements and different context sizes,
    and an artifact under the wrong heading would misreport both.
    """
    matches = sorted(
        alias for alias, preset in registry.items()
        if preset.get("model") and Path(preset["model"]).resolve() == model_path)
    if explicit is not None:
        if explicit not in registry:
            raise ReportRefused(f"{explicit!r} is not a configured preset")
        if explicit not in matches:
            raise ReportRefused(
                f"{explicit!r} serves {Path(registry[explicit].get('model', '')).name!r}, "
                f"but this run benchmarked {model_path.name!r}")
        return explicit
    if len(matches) != 1:
        raise ReportRefused(
            f"{model_path.name} matches presets {matches or ['none']}; "
            f"name one with alias=<run directory>")
    return matches[0]


def cite(run_dir: Path) -> str:
    """How an artifact names the report directory it quotes.

    Real runs live under ``logs/model-runs/`` and are cited relative to the
    checkout so the path is meaningful to a reader. A directory outside the
    checkout keeps its absolute path rather than growing ``../..`` segments.
    """
    resolved = run_dir.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def host_sample(run_dir: Path) -> dict | None:
    """An optional measured VRAM/RAM sample taken beside the run."""
    try:
        sample = json.loads((run_dir / HOST_SAMPLE).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return sample if isinstance(sample, dict) and sample.get("samples") else None


def load_functional(path: Path = FUNCTIONAL) -> dict:
    """Router functional evidence, keyed by alias, or empty if it was never run."""
    try:
        evidence = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(evidence, dict):
        return {}
    models = {item["model"]: item for item in evidence.get("models", [])
              if isinstance(item, dict) and "model" in item}
    return {"finished_at": evidence.get("finished_at"),
            "kernel_release": evidence.get("kernel_release"),
            "boot_id": evidence.get("boot_id"), "models": models}


def load_run(run_dir: Path, registry: dict, alias: str | None = None) -> dict:
    """One publishable record: the argv that ran, and the rows it produced."""
    report, rows = native_rows(run_dir)
    command = report.get("command")
    if not isinstance(command, list) or not command:
        raise MissingNativeResults(f"{run_dir.name} recorded no benchmark command")
    model_path = Path(argv_value(command, "-m")).resolve()
    alias = resolve_alias(model_path, registry, alias)
    preset = registry[alias]
    measured_split = argv_value(command, "-ts")
    preset_split = preset.get("tensor-split", "")
    if split_ratio(measured_split) != split_ratio(preset_split):
        raise ReportRefused(
            f"{alias} serves tensor-split {preset_split!r} but this run measured "
            f"{measured_split!r}; the artifact would name a placement it did not measure")
    measured = throughput(rows)
    first = rows[0]
    return {
        "alias": alias,
        "run_dir": cite(run_dir),
        "command": command,
        "model_path": str(model_path),
        "model_filename": model_path.name,
        # llama-bench reads the quantization out of the file it opened, so this
        # is the model's own report and not a guess from the filename.
        "model_type": first.get("model_type", "unknown"),
        "model_size_bytes": first.get("model_size"),
        "model_n_params": first.get("model_n_params"),
        "build_commit": first.get("build_commit"),
        "gpu_info": first.get("gpu_info"),
        "backends": first.get("backends"),
        "kernel_release": report.get("kernel"),
        "kernel_build": report.get("kernel_build"),
        "boot_id": report.get("boot_id"),
        "devices": argv_value(command, "-dev"),
        "tensor_split": measured_split,
        "preset_tensor_split": preset_split,
        "gpu_layers": argv_value(command, "-ngl"),
        "threads": argv_value(command, "-t"),
        "prompt_tokens": int(argv_value(command, "-p")),
        "generation_tokens": int(argv_value(command, "-n")),
        "repetitions": int(argv_value(command, "-r")),
        "prompt_ts": measured["prompt"]["avg_ts"],
        "prompt_ts_stddev": measured["prompt"].get("stddev_ts"),
        "generation_ts": measured["generation"]["avg_ts"],
        "generation_ts_stddev": measured["generation"].get("stddev_ts"),
        # Everything the native run reported about how it was configured, so the
        # artifact can say which serving knobs the measurement did not cover.
        "runtime_settings": {key: first[key] for key in
                             ("type_k", "type_v", "flash_attn", "n_batch",
                              "n_ubatch", "use_mmap", "split_mode", "main_gpu")
                             if key in first},
        # The serving preset, not the benchmark: this is the MTP disclosure.
        "serving_spec": preset.get("spec-type"),
        "serving_ctx_size": preset.get("ctx-size"),
        "serving_cache_type_k": preset.get("cache-type-k"),
        "serving_cache_type_v": preset.get("cache-type-v"),
        "serving_flash_attn": preset.get("flash-attn"),
        "serving_projector": Path(preset["mmproj"]).name if preset.get("mmproj") else None,
        "host_sample": host_sample(run_dir),
    }


def flash_attention(value) -> str:
    """``llama-bench`` reports -1 for "let the runtime decide"."""
    return {1: "on", 0: "off", -1: "auto (runtime chooses)"}.get(value, str(value))


def serving_differences(record: dict) -> list[tuple[str, str, str]]:
    """Where the benchmarked configuration is not the served one.

    ``llama-bench`` takes a weight file and a placement; it does not take
    ``llama-models.ini``. Every row here is a knob the router sets and the
    benchmark did not, so a reader can see exactly which parts of the served
    configuration the measurement did *not* cover. MTP leads the list because it
    is the difference most likely to move the rate.
    """
    bench = record.get("runtime_settings") or {}
    rows = [("Speculative decoding (`spec-type`)",
             f"`{record['serving_spec']}`" if record.get("serving_spec") else "none",
             "none; `llama-bench` applies no draft model")]
    if record.get("serving_projector"):
        rows.append(("Multimodal projector", f"`{record['serving_projector']}`",
                     "not loaded; the text weights alone were benchmarked"))
    comparisons = (
        ("KV cache key type", "serving_cache_type_k", bench.get("type_k")),
        ("KV cache value type", "serving_cache_type_v", bench.get("type_v")),
        ("Flash attention", "serving_flash_attn",
         flash_attention(bench.get("flash_attn")) if "flash_attn" in bench else None),
    )
    for label, key, measured in comparisons:
        served = record.get(key)
        if served is not None and measured is not None and str(served) != str(measured):
            rows.append((label, f"`{served}`", f"`{measured}`"))
    if record.get("serving_ctx_size"):
        rows.append(("Context size", f"`{record['serving_ctx_size']}`",
                     "sized to the benchmark workload, not to the serving context"))
    return rows


def mtp_class(record: dict) -> str:
    """The serving speculative-decoding class a record belongs to."""
    return record.get("serving_spec") or "none"


def workload(record: dict) -> tuple:
    return (record["prompt_tokens"], record["generation_tokens"],
            record["repetitions"], record["devices"])


def rank_by_generation(records: list[dict]) -> list[dict]:
    """Order comparable records by measured generation rate, fastest first.

    Refuses anything that would read as a weight ranking. Mixed serving-MTP
    classes are the case this exists for: every rate here was measured without
    MTP, so ordering an MTP-served preset against a non-MTP-served one compares
    configurations the router never runs. Differing workloads are refused for
    the same reason -- a longer prompt is not a slower model.
    """
    if not records:
        return []
    classes = sorted({mtp_class(record) for record in records})
    if len(classes) > 1:
        raise RankingRefused(
            "refusing to rank across serving-MTP classes " + ", ".join(classes)
            + ": llama-bench measured every row without MTP, so this ordering "
              "would read as a ranking of the weights")
    workloads = {workload(record) for record in records}
    if len(workloads) > 1:
        raise RankingRefused(
            f"refusing to rank {len(workloads)} different workloads together")
    return sorted(records, key=lambda record: (-record["generation_ts"], record["alias"]))


def rate(value, stddev=None) -> str:
    text = f"{value:.2f} tok/s"
    return text if stddev is None else f"{text} ± {stddev:.2f}"


def verdict(alias: str, functional: dict) -> str:
    """The router gate's recorded verdict, or an explicit absence."""
    entry = (functional.get("models") or {}).get(alias)
    if entry is None:
        return "not recorded"
    return "**PASS**" if entry.get("pass") else "**FAIL**"


def functional_lines(alias: str, functional: dict) -> list[str]:
    """Pass/fail as the router gate recorded it. No quality score is invented."""
    entry = (functional.get("models") or {}).get(alias)
    if entry is None:
        return ["- No `router-functional` result for this alias in the current "
                "evidence file. Absence of evidence is not a pass."]
    recorded = "PASS" if entry.get("pass") else "FAIL"
    lines = [f"- `router-functional` verdict: **{recorded}** "
             f"(recorded {functional.get('finished_at') or 'at an unrecorded time'} "
             f"on kernel `{functional.get('kernel_release') or 'unknown'}`)."]
    for name, check in sorted((entry.get("checks") or {}).items()):
        lines.append(f"  - `{name}`: {'pass' if check.get('pass') else 'fail'}")
    required = entry.get("required_checks")
    if required:
        lines.append(f"  - required checks for this preset: "
                     f"{', '.join(f'`{name}`' for name in required)}")
    if entry.get("tools_offered") is False:
        lines.append("  - reader-only under candidate policy: no tools payload was "
                     "sent, so no tool-call capability is claimed.")
    for problem in (entry.get("problems") or [])[:4]:
        lines.append(f"  - problem: {problem}")
    lines.append("- These are functional verdicts, not scores. A `FAIL` here is a "
                 "recorded gate outcome and is never restated as a pass.")
    return lines


def host_lines(record: dict) -> list[str]:
    sample = record.get("host_sample")
    if not sample:
        return ["- VRAM and RAM were not sampled around this run, so no residency "
                "figure is reported. The allocation request above is not a "
                "measurement of what the cards held."]
    lines = [f"- Sampled every {sample.get('interval_seconds', '?')} s across "
             f"{len(sample['samples'])} readings while the benchmark ran; the "
             f"figures below are peaks, not allocation requests."]
    peaks = sample.get("peak_vram_used_bytes") or {}
    for label in sorted(peaks):
        lines.append(f"  - `{label}` peak VRAM used: {peaks[label] / GIB:.2f} GiB")
    floor = sample.get("min_mem_available_bytes")
    if floor is not None:
        lines.append(f"  - lowest `MemAvailable` observed: {floor / GIB:.2f} GiB")
    return lines


def render_model(record: dict, functional: dict) -> str:
    size = record.get("model_size_bytes")
    params = record.get("model_n_params")
    lines = [
        f"# {record['model_filename']}",
        "",
        f"Native `llama-bench` throughput for the weight file "
        f"`{record['model_path']}`, served by this checkout as the "
        f"`{record['alias']}` compatibility alias.",
        "",
        "## Artifact",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Weight file | `{record['model_filename']}` |",
        f"| Full path | `{record['model_path']}` |",
        f"| Compatibility alias | `{record['alias']}` |",
        f"| Quantization (as the runtime read it) | `{record['model_type']}` |",
        f"| File size | {size / GIB:.2f} GiB |" if size else "| File size | not reported |",
        f"| Parameters | {params / 1e9:.2f} B |" if params else "| Parameters | not reported |",
        f"| Projector loaded when serving | "
        f"`{record['serving_projector']}` |" if record.get("serving_projector")
        else "| Projector loaded when serving | none |",
        "",
        "## Host and run identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Kernel | `{record['kernel_release']}` |",
        f"| Boot id | `{record['boot_id']}` |",
        f"| Runtime build | `{record['build_commit']}` |",
        f"| GPUs reported by the runtime | {record['gpu_info']} |",
        f"| Backends | {record['backends']} |",
        f"| Native report directory | `{record['run_dir']}` |",
        "",
        "## Benchmark configuration",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Devices | `{record['devices']}` |",
        f"| Tensor split (proportions, not gibibytes) | `{record['tensor_split']}` |",
        f"| Tensor split this alias serves | `{record['preset_tensor_split']}` |",
        f"| GPU layers (`-ngl`) | `{record['gpu_layers']}` |",
        f"| Threads | `{record['threads']}` |",
        f"| Prompt tokens | {record['prompt_tokens']} |",
        f"| Generated tokens | {record['generation_tokens']} |",
        f"| Repetitions | {record['repetitions']} |",
        "",
        "Exact argv, as executed:",
        "",
        "```",
        " ".join(record["command"]),
        "```",
        "",
        "## What the benchmark did not configure",
        "",
        "`llama-bench` takes a weight file and a placement. It does not read "
        "`llama-models.ini`, so the rows below are served settings the "
        "measurement did not apply.",
        "",
        "| Setting | Served as | Benchmarked as |",
        "|---|---|---|",
        *[f"| {label} | {served} | {measured} |"
          for label, served, measured in serving_differences(record)],
        "",
        "## Measured throughput",
        "",
        "Averages and standard deviations are `llama-bench`'s own, read from the "
        f"native JSON in `{record['run_dir']}/stdout.log`. Nothing here is "
        "derived from wall-clock timing of anything else.",
        "",
        "| Test | Rate |",
        "|---|---|",
        f"| Prompt processing (pp{record['prompt_tokens']}) | "
        f"{rate(record['prompt_ts'], record['prompt_ts_stddev'])} |",
        f"| Token generation (tg{record['generation_tokens']}) | "
        f"{rate(record['generation_ts'], record['generation_ts_stddev'])} |",
        "",
        "## Memory",
        "",
        *host_lines(record),
        "",
        "## Usability, as the functional gates recorded it",
        "",
        *functional_lines(record["alias"], functional),
        "",
        "## Caveats",
        "",
        f"- **Serving preset:** this alias serves with "
        f"`spec-type = {record['serving_spec']}`. {MTP_CAVEAT}"
        if record.get("serving_spec") else
        "- **Serving preset:** this alias serves without speculative decoding. "
        + MTP_CAVEAT,
        f"- **Context:** the benchmark does not allocate this preset's serving "
        f"context (`ctx-size = {record['serving_ctx_size']}`), so the residency "
        f"of a served session is larger than anything measured here."
        if record.get("serving_ctx_size") else
        "- **Context:** the benchmark does not allocate this preset's serving context.",
        f"- **Smoke tests:** {SMOKE_CAVEAT}",
        "- **Scope:** prompt processing and token generation only. This is not a "
        "measurement of answer quality, agent reliability or tool use.",
        "- **Comparability:** only rows measured with an identical workload, "
        "quantization, placement and serving-MTP class may be compared.",
        "",
    ]
    return "\n".join(lines)


def render_index(records: list[dict], functional: dict) -> str:
    lines = [
        "# Native throughput artifacts",
        "",
        "One file per weight file that was actually benchmarked with the native "
        "`llama-bench` binary. Throughput is deliberately **not** part of "
        "`local-ai-functional-mission.service`, which records "
        "`throughput_measured: false`; these runs are a separate lane that an "
        "operator authorizes with `--confirm-kernel --execute` after confirming "
        "the intended kernel. See [Scripted model operation](../MODEL-RUNS.md).",
        "",
        "Rows are ordered by alias, not by speed. A model absent from this table "
        "was not benchmarked; nothing here is estimated. The functional column is "
        "the router gate's own verdict, carried here so a fast preset that failed "
        "a gate cannot be read as a recommendation.",
        "",
        "| Artifact | Alias | Quant | Split | pp | tg | Serving MTP | Functional gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda item: item["alias"]):
        lines.append(
            f"| [`{record['model_filename']}`]({record['alias']}.md) "
            f"| `{record['alias']}` | `{record['model_type']}` "
            f"| `{record['tensor_split']}` "
            f"| {rate(record['prompt_ts'])} | {rate(record['generation_ts'])} "
            f"| {'yes' if record.get('serving_spec') else 'no'} "
            f"| {verdict(record['alias'], functional)} |")
    lines += ["", "## Ordering within one serving-MTP class", ""]
    if records:
        lines.append(
            "Every rate above was measured **without** multi-token prediction, "
            "because `llama-bench` does not apply the router's preset. Ordering "
            "an MTP-served preset against a non-MTP-served one by these numbers "
            "would read as a ranking of the weights, so the orderings below never "
            "cross that line, and even within a class they compare a "
            "configuration rather than a model's quality.")
        lines.append("")
    classes: dict[str, list[dict]] = {}
    for record in records:
        classes.setdefault(mtp_class(record), []).append(record)
    for name in sorted(classes):
        label = ("served with `draft-mtp`" if name == "draft-mtp"
                 else "served without speculative decoding")
        lines.append(f"Presets {label}, by measured generation rate:")
        lines.append("")
        try:
            ordered = rank_by_generation(classes[name])
        except RankingRefused as refusal:
            lines += [f"- Ordering refused: {refusal}", ""]
            continue
        for position, record in enumerate(ordered, start=1):
            # Size, quantization and placement travel with every ordered line.
            # Without them a 12B on one card reads as "better than" a 32B on
            # three, which is a claim none of these runs measured.
            size = record.get("model_size_bytes")
            lines.append(f"{position}. `{record['alias']}` "
                         f"({record['model_filename']}, "
                         f"{f'{size / GIB:.2f} GiB, ' if size else ''}"
                         f"{record['model_type']}, split `{record['tensor_split']}`) — "
                         f"{rate(record['generation_ts'])}")
        lines.append("")
    lines += [
        "## What these numbers are not",
        "",
        f"- {SMOKE_CAVEAT}",
        "- Functional pass/fail comes from `verification/router-functional/`. A "
        "fast preset that failed a gate is reported as failed.",
        "- Native JSON stays in `logs/model-runs/`, which git ignores. Each "
        "artifact names the report directory it was written from.",
        "",
    ]
    return "\n".join(lines)


def write_reports(records: list[dict], out_dir: Path, functional: dict) -> list[Path]:
    """One markdown file per model, plus the index. Duplicate aliases refuse."""
    aliases = [record["alias"] for record in records]
    duplicates = sorted({alias for alias in aliases if aliases.count(alias) > 1})
    if duplicates:
        raise ReportRefused(
            f"two runs claim the same alias: {', '.join(duplicates)}; one file "
            f"would carry the other run's argv")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for record in records:
        path = out_dir / f"{record['alias']}.md"
        path.write_text(render_model(record, functional))
        written.append(path)
    index = out_dir / "README.md"
    index.write_text(render_index(records, functional))
    written.append(index)
    return written


def parse_run(spec: str) -> tuple[str | None, Path]:
    """``[alias=]directory``; the alias is only needed when a path is shared."""
    alias, separator, path = spec.partition("=")
    return (alias, Path(path)) if separator else (None, Path(spec))


def sample_host(out: Path, interval: float) -> int:
    """Poll VRAM and RAM beside a running benchmark until told to stop.

    Peaks are republished atomically on every tick, so terminating this at any
    moment -- including the moment the benchmark ends -- leaves a valid file
    rather than a truncated one.
    """
    state = {"schema": "frankenstein-bench-host-sample/1",
             "interval_seconds": interval, "samples": [],
             "peak_vram_used_bytes": {}, "min_mem_available_bytes": None}
    stop = False

    def halt(signum, frame):
        nonlocal stop
        stop = True

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, halt)
    target = out / HOST_SAMPLE
    temp = target.with_suffix(f".tmp.{os.getpid()}")
    while not stop:
        rows, _ = gpu_vram.devices()
        available = next(int(line.split()[1]) * 1024
                         for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        reading = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "mem_available_bytes": available,
                   "vram_used_bytes": {f"ROCm{row['rocm_index']}": row["used_bytes"]
                                       for row in rows}}
        state["samples"].append(reading)
        for label, used in reading["vram_used_bytes"].items():
            if used is None:
                continue
            state["peak_vram_used_bytes"][label] = max(
                used, state["peak_vram_used_bytes"].get(label, 0))
        floor = state["min_mem_available_bytes"]
        state["min_mem_available_bytes"] = available if floor is None else min(floor, available)
        temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(temp, target)
        if stop:
            break
        time.sleep(interval)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    write = sub.add_parser("write", help="publish markdown from native run directories")
    write.add_argument("runs", nargs="+", metavar="[ALIAS=]DIR",
                       help="benchmark report directories under logs/model-runs/")
    write.add_argument("--out", default=str(OUT))
    write.add_argument("--presets", default=str(model_catalog.PRESETS))
    write.add_argument("--functional", default=str(FUNCTIONAL))

    watch = sub.add_parser("sample", help="poll VRAM/RAM beside a running benchmark")
    watch.add_argument("--into", required=True, help="the run directory to write into")
    watch.add_argument("--interval", type=float, default=2.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "sample":
        return sample_host(Path(args.into), args.interval)
    try:
        registry = model_catalog.presets(Path(args.presets))
        records = [load_run(directory, registry, alias)
                   for alias, directory in map(parse_run, args.runs)]
        written = write_reports(records, Path(args.out), load_functional(Path(args.functional)))
    except ReportRefused as refusal:
        print(f"Report refused: {refusal}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
