"""Offline documentation link and inventory checks; no services or host probes."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import model_catalog


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


def test_coverage_map_contains_all_outer_tracked_files():
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")) - {"", "upstream/llama.cpp"}
    coverage = (ROOT / "docs/reference/COVERAGE-MAP.md").read_text()
    mapped = set(re.findall(r"\| \[`([^`]+)`\]", coverage))
    assert tracked == mapped, {"missing": sorted(tracked - mapped), "obsolete": sorted(mapped - tracked)}
