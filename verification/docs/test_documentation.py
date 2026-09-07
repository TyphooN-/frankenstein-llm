"""Offline documentation link and inventory checks; no services or host probes."""
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


def test_candidate_research_closeout_covers_queued_repositories():
    closeout = (ROOT / "docs/reference/CANDIDATE-RESEARCH-CLOSEOUT.md").read_text()
    missing = [repo for repo in QUEUED_ADDITIONS if repo not in closeout]
    assert not missing, missing
    inventory = json.loads((ROOT / "docs/reference/candidate-research-inventory.json").read_text())
    queued = {row["repository"] for row in inventory["repositories"] if row["scope"] == "queued"}
    assert queued == set(QUEUED_ADDITIONS)


def test_coverage_map_contains_all_outer_tracked_files():
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")) - {"", "upstream/llama.cpp"}
    coverage = (ROOT / "docs/reference/COVERAGE-MAP.md").read_text()
    mapped = set(re.findall(r"\| \[`([^`]+)`\]", coverage))
    assert tracked == mapped, {"missing": sorted(tracked - mapped), "obsolete": sorted(mapped - tracked)}


def test_gate_output_directories_are_not_tracked():
    """Gate output is evidence about one run, not repository state.

    Every ``evidence/`` directory is ignored, and so is the TTS gate's
    ``artifacts/``, which holds waveforms it re-synthesises on every run. Four of
    those were committed by an unscoped ``git add`` in e86383b. Two things broke:
    the tracked-file inventory above went red, and -- less visibly -- the gate
    started dirtying the working tree that
    ``run_functional_mission.mission_inputs_fingerprint`` hashes, so running it
    invalidated every step that had already passed.

    The coverage map alone does not catch this; a second unscoped commit that
    also added a row would satisfy it. Tracked-ness is the property that matters.
    """
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")) - {""}
    generated = sorted(
        path for path in tracked
        if re.search(r"(^|/)(evidence|verification/tts-local/artifacts)/", path))
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
