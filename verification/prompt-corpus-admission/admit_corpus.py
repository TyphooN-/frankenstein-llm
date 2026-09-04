#!/usr/bin/env python3
"""Fail-closed admission for inert prompt-corpus artifacts.

This module does not download data, invoke models, or execute corpus content. An
artifact becomes eligible for later evaluation only after its local bytes,
source revision, license disposition, normalized row count, and exact schema
match a reviewed manifest.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Iterator

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "source-catalog.json"
CORPORA = HERE / "corpora"
EVIDENCE = HERE / "evidence"
CATALOG_SCHEMA = "hermes-prompt-source-catalog/1"
ARTIFACT_SCHEMA = "hermes-prompt-corpus-artifact/1"
EVIDENCE_SCHEMA = "hermes-prompt-corpus-admission/1"
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024
MAX_ROWS = 1_000_000
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_KEYS = {
    "id", "url", "revision", "license_spdx", "license_status",
    "disposition", "suites", "execution_policy", "reason",
}
DISPOSITIONS = {"candidate", "reference-only", "rejected"}
LICENSE_STATUSES = {"verified", "unverified", "mixed-or-incomplete"}
SUITES = {
    "false-refusal", "harmful-refusal", "benign-utility", "tool-integrity",
    "multimodal-safety", "response-judge", "mcp-tool-integrity",
    "computer-use-safety", "reference-catalog", "prompt-research",
}
CREDENTIAL_PATTERNS = (
    re.compile(rb"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP) PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[opusr]_[A-Za-z0-9]{32,255}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,255}\b"),
)


class AdmissionError(ValueError):
    """The catalog, manifest, or local artifact failed admission."""


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdmissionError(f"{field} must be a non-empty string")
    return value


def load_catalog(path: Path = CATALOG) -> dict[str, dict]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(document) != {"schema", "sources"}:
        raise AdmissionError("catalog has unknown or missing top-level fields")
    if document["schema"] != CATALOG_SCHEMA:
        raise AdmissionError(f"unsupported catalog schema: {document['schema']!r}")
    if not isinstance(document["sources"], list):
        raise AdmissionError("catalog sources must be a list")

    sources: dict[str, dict] = {}
    for index, source in enumerate(document["sources"]):
        if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
            raise AdmissionError(f"catalog source {index} has unknown or missing fields")
        source_id = _require_string(source["id"], f"sources[{index}].id")
        if source_id in sources:
            raise AdmissionError(f"duplicate source id: {source_id}")
        url = _require_string(source["url"], f"{source_id}.url")
        if not url.startswith("https://"):
            raise AdmissionError(f"{source_id} URL must use https")
        if not REVISION_RE.fullmatch(str(source["revision"])):
            raise AdmissionError(f"{source_id} revision must be a 40-character lowercase commit")
        if source["disposition"] not in DISPOSITIONS:
            raise AdmissionError(f"{source_id} has unknown disposition")
        if source["license_status"] not in LICENSE_STATUSES:
            raise AdmissionError(f"{source_id} has unknown license status")
        _require_string(source["license_spdx"], f"{source_id}.license_spdx")
        _require_string(source["reason"], f"{source_id}.reason")
        suites = source["suites"]
        if (not isinstance(suites, list) or not suites
                or any(not isinstance(item, str) or item not in SUITES for item in suites)
                or len(suites) != len(set(suites))):
            raise AdmissionError(f"{source_id} has invalid or duplicate suites")
        expected_policy = {
            "candidate": "inert-text-only",
            "reference-only": "never-execute",
            "rejected": "never-ingest",
        }[source["disposition"]]
        if source["execution_policy"] != expected_policy:
            raise AdmissionError(f"{source_id} execution policy contradicts disposition")
        if source["disposition"] == "candidate" and source["license_status"] != "verified":
            raise AdmissionError(f"candidate source {source_id} lacks verified licensing")
        sources[source_id] = source
    return sources


def load_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema", "source_id", "revision", "suite", "artifact_path",
        "format", "bytes", "sha256", "rows", "fields", "inert_text_only",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise AdmissionError("artifact manifest has unknown or missing fields")
    if manifest["schema"] != ARTIFACT_SCHEMA:
        raise AdmissionError(f"unsupported artifact schema: {manifest['schema']!r}")
    if manifest["format"] not in {"jsonl", "csv"}:
        raise AdmissionError("normalized artifact format must be jsonl or csv")
    if not isinstance(manifest["bytes"], int) or not 0 < manifest["bytes"] <= MAX_ARTIFACT_BYTES:
        raise AdmissionError("artifact byte count is invalid or exceeds the admission bound")
    if not SHA256_RE.fullmatch(str(manifest["sha256"])):
        raise AdmissionError("artifact sha256 must be 64 lowercase hexadecimal characters")
    if not isinstance(manifest["rows"], int) or not 0 < manifest["rows"] <= MAX_ROWS:
        raise AdmissionError("artifact row count is invalid or exceeds the admission bound")
    fields = manifest["fields"]
    if (not isinstance(fields, list) or not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or len(fields) != len(set(fields))):
        raise AdmissionError("artifact fields must be unique non-empty strings")
    if manifest["inert_text_only"] is not True:
        raise AdmissionError("artifact must explicitly declare inert_text_only=true")
    return manifest


def _open_artifact(root: Path, relative: object) -> tuple[Path, int, os.stat_result]:
    text = _require_string(relative, "artifact_path")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise AdmissionError("artifact_path must be relative and may not traverse parents")
    root = root.resolve(strict=True)
    candidate = root / path
    try:
        candidate.parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise AdmissionError("artifact parent resolves outside the corpus root") from exc
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise AdmissionError(f"artifact is not an accessible regular file: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        opened = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
    except OSError:
        os.close(descriptor)
        raise AdmissionError("opened artifact cannot be inspected safely") from None
    try:
        opened.relative_to(root)
    except ValueError:
        os.close(descriptor)
        raise AdmissionError("opened artifact resolves outside the corpus root") from None
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise AdmissionError("artifact must be a regular file, not a special file")
    if metadata.st_mode & 0o111:
        os.close(descriptor)
        raise AdmissionError("artifact must not have executable permission bits")
    return opened, descriptor, metadata


def _hash_and_scan(handle, expected_bytes: int) -> str:
    digest = hashlib.sha256()
    total = 0
    overlap = b""
    while chunk := handle.read(1024 * 1024):
        total += len(chunk)
        if total > expected_bytes or total > MAX_ARTIFACT_BYTES:
            raise AdmissionError("artifact grew beyond its declared or maximum size")
        if b"\x00" in chunk:
            raise AdmissionError("artifact contains NUL bytes and is not admitted text")
        scan_window = overlap + chunk
        if any(pattern.search(scan_window) for pattern in CREDENTIAL_PATTERNS):
            raise AdmissionError("artifact contains high-confidence credential material")
        overlap = scan_window[-512:]
        digest.update(chunk)
    if total != expected_bytes:
        raise AdmissionError(f"artifact size mismatch: expected {expected_bytes}, got {total}")
    return digest.hexdigest()


def _validate_row(row: object, fields: list[str], number: int) -> None:
    if not isinstance(row, dict) or set(row) != set(fields):
        raise AdmissionError(f"row {number} does not match the exact declared field schema")
    for field, value in row.items():
        if isinstance(value, (dict, list)):
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        elif value is None or isinstance(value, (str, int, float, bool)):
            encoded = str(value).encode("utf-8")
        else:
            raise AdmissionError(f"row {number} field {field!r} has unsupported value type")
        if len(encoded) > MAX_RECORD_BYTES:
            raise AdmissionError(f"row {number} field {field!r} exceeds the record bound")


def _rows(handle, format_name: str, fields: list[str]) -> Iterator[dict]:
    text = io.TextIOWrapper(handle, encoding="utf-8", errors="strict", newline="")
    if format_name == "jsonl":
        try:
            for number, line in enumerate(text, 1):
                if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                    raise AdmissionError(f"row {number} exceeds the record bound")
                if not line.strip():
                    raise AdmissionError(f"row {number} is blank")
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AdmissionError(f"row {number} is not valid JSON") from exc
        finally:
            text.detach()
        return

    previous_limit = csv.field_size_limit()
    csv.field_size_limit(MAX_RECORD_BYTES)
    try:
        reader = csv.DictReader(text)
        if reader.fieldnames != fields:
            raise AdmissionError("CSV header does not match the exact declared field order")
        yield from reader
    finally:
        csv.field_size_limit(previous_limit)
        text.detach()


def validate_artifact(manifest_path: Path, *, catalog_path: Path = CATALOG,
                      corpus_root: Path = CORPORA) -> dict:
    sources = load_catalog(catalog_path)
    manifest = load_manifest(manifest_path)
    source_id = _require_string(manifest["source_id"], "source_id")
    source = sources.get(source_id)
    if source is None:
        raise AdmissionError(f"unknown source id: {source_id}")
    if source["disposition"] != "candidate":
        raise AdmissionError(f"source {source_id} is not approved for corpus admission")
    if manifest["revision"] != source["revision"]:
        raise AdmissionError("artifact revision does not match the pinned source revision")
    if manifest["suite"] not in source["suites"]:
        raise AdmissionError("artifact suite is not approved for this source")

    _, descriptor, initial_metadata = _open_artifact(corpus_root, manifest["artifact_path"])
    count = 0
    with os.fdopen(descriptor, "rb") as handle:
        if initial_metadata.st_size != manifest["bytes"]:
            raise AdmissionError("artifact size does not match the manifest")
        digest = _hash_and_scan(handle, manifest["bytes"])
        if digest != manifest["sha256"]:
            raise AdmissionError("artifact SHA-256 does not match the manifest")
        handle.seek(0)
        try:
            for count, row in enumerate(
                    _rows(handle, manifest["format"], manifest["fields"]), 1):
                if count > MAX_ROWS:
                    raise AdmissionError("artifact exceeds the maximum row count")
                _validate_row(row, manifest["fields"], count)
        except UnicodeDecodeError as exc:
            raise AdmissionError("artifact is not strict UTF-8 text") from exc
        handle.seek(0)
        final_digest = _hash_and_scan(handle, manifest["bytes"])
        final_metadata = os.fstat(handle.fileno())
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(initial_metadata) != identity(final_metadata) or final_digest != digest:
        raise AdmissionError("artifact changed during admission")
    if count != manifest["rows"]:
        raise AdmissionError(f"artifact row mismatch: expected {manifest['rows']}, got {count}")

    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "admitted",
        "source_id": source_id,
        "source_url": source["url"],
        "revision": source["revision"],
        "license_spdx": source["license_spdx"],
        "suite": manifest["suite"],
        "artifact_path": manifest["artifact_path"],
        "format": manifest["format"],
        "bytes": manifest["bytes"],
        "sha256": digest,
        "rows": count,
        "fields": manifest["fields"],
        "execution_policy": "inert-text-only",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--corpus-root", type=Path, default=CORPORA)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_artifact(args.manifest, catalog_path=args.catalog,
                                   corpus_root=args.corpus_root)
        if args.evidence:
            atomic_json(args.evidence, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (AdmissionError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"schema": EVIDENCE_SCHEMA, "status": "rejected",
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
