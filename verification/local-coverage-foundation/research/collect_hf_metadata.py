#!/usr/bin/env python3
"""Collect exact Hugging Face artifact metadata (revision, files, sizes, LFS SHA-256).

Metadata only: this never downloads model weights. Output is one JSON document per
repository under research/hf/, plus a flat TSV index of every LFS file found.
"""
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

OUT = Path(__file__).resolve().parent / "hf"
INDEX = Path(__file__).resolve().parent / "hf-lfs-index.tsv"
API = "https://huggingface.co/api/models"
UA = {"User-Agent": "hermes-local-coverage-foundation/1.0 (metadata-only)"}


def get(url: str, tries: int = 4):
    last = None
    for attempt in range(1, tries + 1):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                return {"__http_error__": error.code}
            last = error
        except Exception as error:  # noqa: BLE001 - network transport variety
            last = error
        time.sleep(2 * attempt)
    return {"__error__": repr(last)}


def collect(repo: str) -> dict:
    info = get(f"{API}/{repo}")
    if "__http_error__" in info or "__error__" in info:
        return {"repository": repo, "status": info}
    revision = info.get("sha")
    tree = get(f"{API}/{repo}/tree/{revision}?recursive=true&expand=true")
    files = []
    if isinstance(tree, list):
        for entry in tree:
            if entry.get("type") != "file":
                continue
            lfs = entry.get("lfs") or {}
            files.append({
                "path": entry.get("path"),
                "size": lfs.get("size", entry.get("size")),
                "sha256": lfs.get("oid") if lfs else None,
                "lfs": bool(lfs),
            })
    return {
        "repository": repo,
        "revision": revision,
        "last_modified": info.get("lastModified"),
        "gated": info.get("gated"),
        "downloads": info.get("downloads"),
        "license": next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), None),
        "pipeline_tag": info.get("pipeline_tag"),
        "library": info.get("library_name"),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": sorted(files, key=lambda f: f["path"]),
    }


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo in argv:
        document = collect(repo)
        slug = repo.replace("/", "__")
        (OUT / f"{slug}.json").write_text(json.dumps(document, indent=2) + "\n")
        status = document.get("status")
        if status:
            print(f"{repo}\tUNAVAILABLE\t{status}", flush=True)
            continue
        total = sum(f["size"] or 0 for f in document["files"])
        print(f"{repo}\t{document['revision'][:12]}\tfiles={len(document['files'])}\tbytes={total}", flush=True)
        for f in document["files"]:
            if f["lfs"]:
                rows.append(f"{repo}\t{document['revision']}\t{f['path']}\t{f['size']}\t{f['sha256']}")
    if rows:
        existing = INDEX.read_text().splitlines() if INDEX.exists() else []
        merged = sorted(set(existing) | set(rows))
        INDEX.write_text("\n".join(merged) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
