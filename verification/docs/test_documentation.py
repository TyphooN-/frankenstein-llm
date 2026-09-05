"""Offline documentation link and inventory checks; no services or host probes."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


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


def test_coverage_map_contains_all_outer_tracked_files():
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")) - {"", "upstream/llama.cpp"}
    coverage = (ROOT / "docs/reference/COVERAGE-MAP.md").read_text()
    mapped = set(re.findall(r"\| \[`([^`]+)`\]", coverage))
    assert tracked == mapped, {"missing": sorted(tracked - mapped), "obsolete": sorted(mapped - tracked)}
