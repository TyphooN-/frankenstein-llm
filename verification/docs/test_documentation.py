"""Offline documentation link and inventory checks; no services or host probes."""
import configparser
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import model_catalog

QUEUED_ADDITIONS = [
    "speach1sdef178/MiniMax-H3-Semantic-Bridge",
    "inclusionAI/LLaDA-Image",
    "GestaltLabs/Qwen3.8-27B-EXL3-11.5GB",
    "Lightricks/LTX-2.5",
    "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF",
    "MohamedAhmedAE/llava-medical-3B-clip-vit-stage2",
    "IFM/K2-Horizon-MoVA-36B-A4B-GGUF",
    "inclusionAI/Ling-3.0-flash-Fin",
    "IFM/K2-Horizon-375B-A23B",
    "mradermacher/Omega_Sapphira_Joyous-L3.3-70B-v1.1-i1-GGUF",
    "Reallexi-llc/lexipix-models",
]


def prose(text):
    return re.sub(r"^```.*?^```[^\n]*", "", text, flags=re.M | re.S)


def anchors(path):
    headings = re.findall(r"^#{1,6}\s+(.+)", prose(path.read_text()), re.M)
    return {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}


def test_reference_links_and_anchors():
    documents = [ROOT / "README.md", ROOT / "docs/README.md",
                 ROOT / "docs/USER-GUIDE.md", ROOT / "docs/decisions/README.md",
                 *sorted((ROOT / "docs/reference").glob("*.md"))]
    errors = []
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", prose(document.read_text())):
            if "://" in link or link.startswith("mailto:"):
                continue
            file, _, anchor = link.partition("#")
            target = (document.parent / file).resolve() if file else document
            if not target.exists():
                errors.append((document.name, link, "missing file"))
            elif anchor and target.suffix == ".md" and anchor not in anchors(target):
                errors.append((document.name, link, "missing anchor"))
    assert not errors, errors


def test_model_choice_documents_name_the_weight_file_each_alias_loads():
    """An alias is not a filename, and a doc that prints only the alias hides
    which artifact a reader is about to load. `heretic` is the case that made
    this a check: the release name and the file on disk do not match."""
    registry = model_catalog.presets(ROOT / "llama-models.ini")
    documents = [ROOT / "docs/USER-GUIDE.md",
                 ROOT / "docs/HERMES-DESKTOP-LOCAL-MODELS.md"]
    errors = []
    for document in documents:
        text = document.read_text()
        # Only the "which model do I pick" entries, which introduce an alias at
        # the start of a table row or list item, must carry the filename.
        for line in text.splitlines():
            match = re.match(r"^(?:\| |- )`([a-z0-9][\w.-]*)`", line)
            if match is None or match.group(1) not in registry:
                continue
            for artifact in model_catalog.artifacts(registry[match.group(1)]):
                if artifact not in line:
                    errors.append((document.name, match.group(1), artifact))
    assert errors == [], errors


def _configuration_alias_rows():
    """The `### Aliases` table in CONFIGURATION.md, keyed by preset name."""
    text = (ROOT / "docs/reference/CONFIGURATION.md").read_text()
    section = text.partition("### Aliases")[2].partition("\n### ")[0]
    rows = {}
    for line in section.splitlines():
        match = re.match(r"^\| `([a-z0-9][\w.-]*)` \| (.*?) \| (.*?) \| (.*?) \|$", line)
        if match:
            rows[match.group(1)] = {"model": match.group(2), "overrides": match.group(3),
                                    "privilege": match.group(4)}
    return rows


def test_configuration_alias_table_matches_the_router_presets():
    """The preset table is a copy of `llama-models.ini`, so it can silently rot.

    It did: `gemma4-heretic` was retuned from `1,0,1` to `1,0,0` on a measured
    residency, and `qwen3-coder-next` moved into a per-model subdirectory, while
    this table went on printing the old values. Both are the kind of drift a
    reader acts on -- one names weights that are not there, the other describes a
    placement the router does not use -- and neither is caught by a link check.
    """
    registry = model_catalog.presets(ROOT / "llama-models.ini")
    # presets() merges the shared [*] block into every alias and drops the "*"
    # key, which is right for serving and wrong for this comparison: it would
    # report the inherited 131072 context as an override the table failed to
    # print. Read the raw sections too, so "the alias sets this" and "the alias
    # inherits this" stay distinguishable.
    raw = configparser.ConfigParser(interpolation=None)
    with (ROOT / "llama-models.ini").open() as handle:
        raw.read_file(handle)
    rows = _configuration_alias_rows()
    assert set(rows) == {name for name in registry if name != "*"}, {
        "undocumented": sorted({n for n in registry if n != "*"} - set(rows)),
        "not a preset": sorted(set(rows) - set(registry)),
    }
    errors = []
    for name, preset in registry.items():
        if name == "*":
            continue
        row = rows[name]
        # A row may say "same GGUF" where a pair shares one weight file; that is
        # the case this repository has two of, and it is not a missing filename.
        if "same GGUF" not in row["model"]:
            expected = preset["model"].replace("/home/typhoon/git/frankenstein-llm/", "")
            documented = row["model"].strip("`")
            head = documented.split("\u2026")[0]
            if not expected.startswith(head) or (
                    "\u2026" not in documented and documented != expected):
                errors.append((name, "model", expected, documented))
        for key, pattern in (("tensor-split", r"tensor-split=([0-9,]+)"),
                             ("ctx-size", r"ctx(?:-size)?=(\d+)")):
            match = re.search(pattern, row["overrides"])
            documented = match.group(1) if match else None
            override = dict(raw[name]).get(key)
            if override is not None:
                # The alias sets it: the table must print that value.
                if documented != override:
                    errors.append((name, key, override, documented))
            elif documented is not None and documented != preset.get(key):
                # The alias inherits it: the table may omit it, but must not
                # print a value that contradicts what the router would use.
                errors.append((name, key, preset.get(key), documented))
    assert errors == [], errors


def test_candidate_research_closeout_covers_queued_repositories():
    closeout = (ROOT / "docs/reference/CANDIDATE-RESEARCH-CLOSEOUT.md").read_text()
    missing = [repo for repo in QUEUED_ADDITIONS if repo not in closeout]
    assert not missing, missing
    inventory = json.loads((ROOT / "docs/reference/candidate-research-inventory.json").read_text())
    queued = {row["repository"] for row in inventory["repositories"] if row["scope"] == "queued"}
    assert queued == set(QUEUED_ADDITIONS)


def test_submitted_intake_lists_each_url_exactly_once():
    """A submitted URL is a queue entry, not a verdict.

    The 2026-09-08 second batch resubmitted three identifiers that already had
    a recorded verdict, which is the case this check exists for: an intake that
    silently restates an existing finding as a new one inflates the queue and
    hides that the question was already answered. Each identifier appears once,
    carries an explicit investigated/qualified pair, and nothing in the intake
    may claim qualification -- that is what the gates decide, not this file.
    """
    inventory = json.loads((ROOT / "docs/reference/candidate-research-inventory.json").read_text())
    intake = inventory["submitted_intake"]
    entries = intake["entries"]
    urls = [entry["url"] for entry in entries]
    repositories = [entry["repository"] for entry in entries]
    assert len(urls) == len(set(urls)) == len(repositories) == len(set(repositories))
    assert intake["submitted_url_count"] == len(urls)
    assert intake["unique_repository_count"] == len(set(repositories))
    assert intake["newly_queued_count"] == sum(
        1 for entry in entries if entry["state"] == "queued")
    assert intake["already_investigated_count"] == sum(
        1 for entry in entries if entry["investigated"])
    assert intake["qualified_count"] == 0
    closeout = (ROOT / "docs/reference/CANDIDATE-RESEARCH-CLOSEOUT.md").read_text()
    problems = []
    for entry in entries:
        # The URL must be the repository it claims to be, not a lookalike.
        if entry["url"] != f"https://huggingface.co/{entry['repository']}":
            problems.append((entry["repository"], "url does not match repository"))
        if entry["qualified"]:
            problems.append((entry["repository"], "intake may not claim qualification"))
        if entry["investigated"] != (entry["state"] == "investigated"):
            problems.append((entry["repository"], "state contradicts investigated"))
        # Queued means no research was done, so it must not cite one.
        if entry["investigated"] and not entry["prior_record"]:
            problems.append((entry["repository"], "investigated without a record"))
        if not entry["investigated"] and entry["prior_record"]:
            problems.append((entry["repository"], "queued but cites a record"))
        if entry["prior_record"] and not (ROOT / entry["prior_record"]).exists():
            problems.append((entry["repository"], "prior record file is missing"))
        if closeout.count(entry["repository"]) < 1:
            problems.append((entry["repository"], "absent from the closeout"))
    assert problems == [], problems


def test_research_queue_resolves_each_submission_to_one_record():
    """A resubmitted identifier resolves to its existing record, never a second row.

    The queue's declared ``unique_repository_count`` had already drifted from its
    rows (15 declared against 16) when the 2026-09-12 batch arrived, and that
    batch resubmitted three identifiers that were already recorded. Both are the
    silent inflation the intake check above guards against, one level up: each
    identifier is recorded in exactly one place in the inventory, every intake
    batch URL resolves to exactly one record read at the revision the batch
    pinned, and nothing in the queue claims qualification.
    """
    inventory = json.loads((ROOT / "docs/reference/candidate-research-inventory.json").read_text())
    queue = inventory["additional_research_queue"]
    sections = {
        "additional_research_queue": {row["repository"]: row for row in queue["entries"]},
        "repositories": {row["repository"]: row for row in inventory["repositories"]},
    }
    identifiers = [name.lower() for section in sections.values() for name in section]
    identifiers += [row["repository"].lower() for row in inventory["submitted_intake"]["entries"]]
    assert len(identifiers) == len(set(identifiers)), "an identifier is recorded twice"
    assert queue["unique_repository_count"] == len(queue["entries"]) == len(
        sections["additional_research_queue"])
    closeout = (ROOT / "docs/reference/CANDIDATE-RESEARCH-CLOSEOUT.md").read_text()
    problems = []
    for row in queue["entries"]:
        if row["url"] != f"https://huggingface.co/{row['repository']}":
            problems.append((row["repository"], "url does not match repository"))
        if row["qualified"]:
            problems.append((row["repository"], "the queue may not claim qualification"))
    for batch in queue["intake_batches"]:
        rows = batch["entries"]
        dispositions = [row["disposition"] for row in rows]
        assert set(dispositions) <= {"new", "already-queued", "already-assessed"}, dispositions
        assert len({row["url"] for row in rows}) == len(rows) == batch["submitted_url_count"]
        assert len({row["repository"].lower() for row in rows}) == batch["unique_repository_count"]
        assert batch["newly_queued_count"] == dispositions.count("new")
        assert batch["already_queued_count"] == dispositions.count("already-queued")
        assert batch["already_assessed_count"] == dispositions.count("already-assessed")
        assert batch["qualified_count"] == 0
        for row in rows:
            section = ("repositories" if row["disposition"] == "already-assessed"
                       else "additional_research_queue")
            record = sections[section].get(row["repository"])
            if row["record"] != section or record is None:
                problems.append((row["repository"], "does not resolve to its record"))
            elif record.get("revision") != row["pinned_revision"]:
                problems.append((row["repository"], "record is not at the pinned revision"))
            elif section == "additional_research_queue" and not (
                    record["investigated"] and record["state"] == "investigated"):
                problems.append((row["repository"], "researched in the batch but not investigated"))
            if row["url"] != f"https://huggingface.co/{row['repository']}":
                problems.append((row["repository"], "url does not match repository"))
            if row["repository"] not in closeout:
                problems.append((row["repository"], "absent from the closeout"))
    assert problems == [], problems


def test_unaccepted_license_keeps_a_research_record_out_of_downloads():
    """Unaccepted license terms keep a researched repository off every download path.

    ukisai/Swift-Qwen3.8-27B-GGUF is the case this exists for. Its card says the
    weights are distributed through gated access under the custom Swift Open
    License v1.0, yet the Hub reports the repository ungated, so nothing upstream
    stands between a client and the weights. A record that carries a
    ``license_review`` must state whether the terms were accepted. Until they
    are, it must declare its download blocked, no download queue may name the
    repository, and no router preset may load one of its recorded files.
    """
    inventory = json.loads((ROOT / "docs/reference/candidate-research-inventory.json").read_text())
    reviewed = [row for row in inventory["additional_research_queue"]["entries"]
                if "license_review" in row]
    assert "ukisai/Swift-Qwen3.8-27B-GGUF" in {row["repository"] for row in reviewed}
    queues = sorted((ROOT / "verification/local-coverage-foundation").glob("download-queue*.json"))
    assert queues, "no download queue manifests to check against"
    queued = {artifact["repository"].lower()
              for queue in queues for artifact in json.loads(queue.read_text())["artifacts"]}
    loaded = {name for preset in model_catalog.presets(ROOT / "llama-models.ini").values()
              for name in model_catalog.artifacts(preset)}
    problems = []
    for row in reviewed:
        accepted = row["license_review"].get("accepted")
        if not isinstance(accepted, bool):
            problems.append((row["repository"], "license review does not state acceptance"))
            continue
        if accepted:
            continue
        if row.get("download_admission") != "blocked":
            problems.append((row["repository"], "terms unaccepted but download not blocked"))
        if row["repository"].lower() in queued:
            problems.append((row["repository"], "named by a download queue"))
        for name in sorted(set(row.get("selected_gguf_bytes", {})) & loaded):
            problems.append((row["repository"], f"a router preset loads {name}"))
    assert problems == [], problems


def test_recorded_research_hashes_pair_with_recorded_sizes():
    """A recorded SHA-256 names exactly the files whose sizes are recorded.

    Swift's publisher ships two checksum files that disagree at the pinned
    revision: ``SHA256SUMS`` matches the LFS SHA-256 of every GGUF, while
    ``SHA256SUMS.quants`` still lists three superseded uploads. The record keeps
    the LFS values. A digest with no size beside it, or a size with no digest, is
    how a later transfer ends up checking one file against another's hash.
    """
    inventory = json.loads((ROOT / "docs/reference/candidate-research-inventory.json").read_text())
    hashed = [row for row in inventory["additional_research_queue"]["entries"]
              if "selected_gguf_sha256" in row]
    assert hashed, "no research record carries pinned hashes"
    problems = []
    for row in hashed:
        hashes, sizes = row["selected_gguf_sha256"], row.get("selected_gguf_bytes", {})
        if set(hashes) != set(sizes):
            problems.append((row["repository"], "hashed and sized files differ"))
        for name, digest in hashes.items():
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                problems.append((row["repository"], name, "not a SHA-256 digest"))
        for name, size in sizes.items():
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                problems.append((row["repository"], name, "not a byte count"))
    assert problems == [], problems


def test_coverage_map_contains_all_outer_tracked_files():
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")) - {"", "upstream/llama.cpp"}
    tracked = {path for path in tracked if not (
        path.startswith('proofs/') and path.endswith('/.gitkeep')
        or (ROOT / path).is_symlink() and (ROOT / path).resolve().is_relative_to(ROOT / 'proofs'))}
    coverage = (ROOT / "docs/reference/COVERAGE-MAP.md").read_text()
    mapped = set(re.findall(r"\| \[`([^`]+)`\]", coverage))
    assert tracked == mapped, {"missing": sorted(tracked - mapped), "obsolete": sorted(mapped - tracked)}


def test_amdgpu_profiles_are_one_current_copy_per_card():
    tracked = [
        path for path in subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
        if path.startswith("config/hardware/amdgpu/")
    ]
    expected = {
        "config/hardware/amdgpu/README.md",
        "config/hardware/amdgpu/SHA256SUMS",
        "config/hardware/amdgpu/amdgpu-custom-state.card0",
        "config/hardware/amdgpu/amdgpu-custom-state.card1",
        "config/hardware/amdgpu/amdgpu-custom-state.card2",
    }
    assert set(tracked) == expected
    assert not any("/20" in path for path in tracked)
    checksums = (ROOT / "config/hardware/amdgpu/SHA256SUMS").read_text().splitlines()
    names = {line.split("  ", 1)[1] for line in checksums if line}
    assert names == {
        "amdgpu-custom-state.card0",
        "amdgpu-custom-state.card1",
        "amdgpu-custom-state.card2",
    }


def test_documentation_audit_inventory_and_disposition_counts():
    audit = json.loads((ROOT / "docs/reference/documentation-audit-2026-09-07.json").read_text())
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    expected = {path for path in tracked if path.endswith(".md") or (
        path.startswith("docs/reference/") and path.endswith(".json"))}
    entries = audit["entries"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths)) == audit["inventory_count"]
    assert set(paths) == expected
    counts = {}
    for entry in entries:
        disposition = entry["disposition"]
        assert disposition in audit["dispositions"]
        assert entry["source_evidence"] and entry["finding"] and entry["change"]
        counts[disposition] = counts.get(disposition, 0) + 1
    assert counts == audit["disposition_counts"]


def test_agent_worktrees_are_ignored_and_not_tracked():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    assert not any(path.startswith(".claude/worktrees/") for path in tracked)
    probe = ".claude/worktrees/documentation-ignore-probe/README.md"
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", probe], cwd=ROOT,
        capture_output=True, text=True, check=False)
    assert ignored.returncode == 0, ignored.stderr
    assert ignored.stdout.strip() == probe


def test_download_intake_does_not_claim_transfer_admission():
    inventory = json.loads((ROOT / "docs/reference/candidate-research-inventory.json").read_text())
    queue = inventory["additional_research_queue"]
    batch = next(row for row in queue["intake_batches"] if row["batch"] == "2026-09-14")
    records = {row["repository"]: row for row in queue["entries"]}
    assert batch["requested_actions"] == ["research", "investigate", "download"]
    assert len(batch["entries"]) == 3
    for submitted in batch["entries"]:
        record = records[submitted["repository"]]
        request = record["download_request"]
        assert request["requested"] is True
        assert request["state"] == "blocked" and request["blockers"]
        assert request["downloaded"] is False and request["admitted"] is False
        assert record["qualified"] is False
        for artifact in request["artifact_candidates"]:
            assert artifact["size"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
            assert not artifact["path"].endswith("-v2.gguf")


def test_gate_output_directories_are_not_tracked():
    """Gate output is evidence about one run, not repository state.

    Every ``evidence/`` directory is ignored, and so is the TTS gate's
    ``artifacts/``, which holds waveforms it re-synthesises on every run. Four of
    those were committed by an unscoped ``git add`` in e86383b. Two things broke:
    the tracked-file inventory above went red, and -- less visibly -- the gate
    started dirtying the working tree that
    ``run_qualification.qualification_inputs_fingerprint`` hashes, so running it
    invalidated every step that had already passed.

    The coverage map alone does not catch this; a second unscoped commit that
    also added a row would satisfy it. Tracked-ness is the property that matters.
    """
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")) - {""}
    generated = sorted(
        path for path in tracked
        if (re.search(r"(^|/)(evidence|verification/tts-local/artifacts)/", path)
            or path.startswith('proofs/'))
        and not (path.startswith('proofs/') and path.endswith('/.gitkeep')))
    assert generated == [], generated

    # The ignore rule has to be narrow enough to leave the gate's *inputs* alone:
    # the reference clips the TTS gate clones a voice from are tracked fixtures.
    fixtures = sorted(path for path in tracked if path.endswith(".wav"))
    assert fixtures == [
        "verification/local-coverage-foundation/fixtures/asr/librispeech-mr-quilter.wav",
        "verification/local-coverage-foundation/fixtures/asr/qwen-asr-en.wav",
    ], fixtures
    # check-ignore echoes only the paths it would ignore, so an empty result is
    # the assertion. --quiet is not usable here: it takes a single pathname.
    ignored = subprocess.run(["git", "check-ignore", *fixtures],
                             cwd=ROOT, check=False, capture_output=True, text=True)
    assert ignored.stdout == "", ignored.stdout


def test_signalled_gate_exit_is_not_documented_as_a_pass():
    """A non-zero or signalled exit is a failed run until proven otherwise.

    An artifact persists between runs, so a gate that crashes can leave an older
    ``"pass": true`` on disk -- the same hazard the runner's exit 5 already fails
    closed on. Documentation that says to prefer the artifact over the exit code
    without first pinning the artifact to *this* run teaches operators to launder
    a crash into a verdict, which on a host that is retiring corrupt pages is how
    a hardware fault gets recorded as a passing model.
    """
    troubleshooting = (ROOT / "docs/reference/TROUBLESHOOTING.md").read_text()
    for phrase in ("Trust the artifact, not the exit code",
                   "trust the artifact, not the exit code"):
        assert phrase not in troubleshooting, phrase

    section = troubleshooting.partition("## A failed process outranks a passing artifact")[2]
    section = section.partition("\n## ")[0]
    assert section, "the section the signalled-exit row points at is missing"
    for required in ("recorded_at", "boot_id", "exit_after_verdict",
                     "BUG: Bad page state"):
        assert required in section, required
