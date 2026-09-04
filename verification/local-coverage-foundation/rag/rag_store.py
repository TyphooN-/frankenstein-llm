#!/usr/bin/env python3
"""Bounded, local, provenance-carrying document store for Hermes RAG.

Deliberate constraints:
  * SQLite plus NumPy brute-force cosine. No vector-database daemon, because a
    personal corpus does not need one and every extra service is another thing
    that can hold VRAM or drift out of sync with the metadata.
  * Every chunk keeps its source path, content hash, ordinal and character span,
    so any retrieved sentence can be pointed back at exact bytes on disk.
  * Update and delete are first-class. Re-ingesting a changed file atomically
    replaces its chunks; a removed file's chunks stop being retrievable in the
    same transaction that marks the document deleted.
  * Retrieved text is data, never instructions. build_context() wraps chunks in a
    labelled envelope and the caller is expected to keep that framing.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.request

import numpy as np

DEFAULT_DB = Path("/home/typhoon/git/frankenstein-llm/rag/hermes-rag.sqlite3")
EMBED_URL = os.environ.get("HERMES_EMBED_URL", "http://127.0.0.1:8080/v1/embeddings")
EMBED_MODEL = os.environ.get("HERMES_EMBED_MODEL", "qwen3-embedding-8b")

# Bounds. These exist so a runaway ingest cannot exhaust RAM or silently build a
# prompt larger than the answering model's usable context.
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_CHARS = 12000
MAX_CONTEXT_CHUNKS = 12
EMBED_BATCH = 8
DEFAULT_INGEST_PATTERNS = ("*.md", "*.txt", "*.html", "*.htm")
TEXT_SUFFIXES = {".md", ".txt", ".rst"}
HTML_SUFFIXES = {".html", ".htm"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id         INTEGER PRIMARY KEY,
    source_path    TEXT NOT NULL UNIQUE,
    content_sha256 TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    mtime_ns       INTEGER NOT NULL,
    ingested_at    TEXT NOT NULL,
    deleted_at     TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    embedding  BLOB NOT NULL,
    UNIQUE (doc_id, ordinal)
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS documents_live ON documents(deleted_at);
"""


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(SCHEMA)
    return connection


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _VisibleHTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def extract_text(path: Path, data: bytes) -> str | None:
    """Return visible text, or None when the file must not enter the index.

    Unknown binary types are skipped rather than UTF-8-replaced into garbage
    chunks that would later be cited as documents.
    """
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return data.decode("utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        parser = _VisibleHTMLText()
        parser.feed(data.decode("utf-8", errors="replace"))
        text = " ".join(part.strip() for part in parser.parts if part.strip())
        return text or None
    return None


def chunk_text(text: str) -> list[tuple[int, int, str]]:
    """Split on paragraph boundaries, then hard-wrap anything still oversized.

    Character spans are returned alongside each chunk so citations can quote an
    exact offset range rather than "somewhere in this file".
    """
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    length = len(text)
    while cursor < length:
        end = min(cursor + MAX_CHUNK_CHARS, length)
        if end < length:
            window = text.rfind("\n\n", cursor + MAX_CHUNK_CHARS // 2, end)
            if window == -1:
                window = text.rfind("\n", cursor + MAX_CHUNK_CHARS // 2, end)
            if window == -1:
                window = text.rfind(" ", cursor + MAX_CHUNK_CHARS // 2, end)
            if window != -1:
                end = window
        piece = text[cursor:end].strip()
        if piece:
            start_offset = cursor + (len(text[cursor:end]) - len(text[cursor:end].lstrip()))
            spans.append((start_offset, start_offset + len(piece), piece))
        if end >= length:
            break
        cursor = max(end - CHUNK_OVERLAP_CHARS, end) if CHUNK_OVERLAP_CHARS == 0 else max(cursor + 1, end - CHUNK_OVERLAP_CHARS)
    return spans


def embed(texts: list[str], url: str = EMBED_URL, model: str = EMBED_MODEL) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        request = urllib.request.Request(
            url,
            data=json.dumps({"input": batch, "model": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
        for item in sorted(payload["data"], key=lambda d: d["index"]):
            vectors.append(item["embedding"])
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def ingest_file(connection: sqlite3.Connection, path: Path) -> dict:
    """Insert or update one document. Returns what actually changed."""
    resolved = str(path.resolve())
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        return {"path": resolved, "action": "skipped", "reason": f"exceeds {MAX_FILE_BYTES} bytes"}
    digest = sha256_bytes(data)
    stat = path.stat()

    row = connection.execute(
        "SELECT doc_id, content_sha256, deleted_at FROM documents WHERE source_path = ?", (resolved,)
    ).fetchone()
    if row and row["content_sha256"] == digest and row["deleted_at"] is None:
        return {"path": resolved, "action": "unchanged", "doc_id": row["doc_id"]}

    text = extract_text(path, data)
    if text is None:
        return {"path": resolved, "action": "skipped", "reason": "unsupported or non-text file"}
    spans = chunk_text(text)
    if not spans:
        return {"path": resolved, "action": "skipped", "reason": "no extractable text"}
    matrix = embed([piece for _, _, piece in spans])

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with connection:
        if row:
            doc_id = row["doc_id"]
            # Replace, never append: a stale chunk from a previous revision would
            # otherwise stay retrievable and be cited as current.
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            connection.execute(
                "UPDATE documents SET content_sha256=?, size_bytes=?, mtime_ns=?, ingested_at=?, deleted_at=NULL "
                "WHERE doc_id=?",
                (digest, len(data), stat.st_mtime_ns, now, doc_id),
            )
            action = "updated"
        else:
            cursor = connection.execute(
                "INSERT INTO documents (source_path, content_sha256, size_bytes, mtime_ns, ingested_at) "
                "VALUES (?,?,?,?,?)",
                (resolved, digest, len(data), stat.st_mtime_ns, now),
            )
            doc_id = cursor.lastrowid
            action = "inserted"
        connection.executemany(
            "INSERT INTO chunks (doc_id, ordinal, char_start, char_end, text, dim, embedding) VALUES (?,?,?,?,?,?,?)",
            [
                (doc_id, ordinal, start, end, piece, matrix.shape[1], matrix[ordinal].tobytes())
                for ordinal, (start, end, piece) in enumerate(spans)
            ],
        )
    return {"path": resolved, "action": action, "doc_id": doc_id, "chunks": len(spans), "sha256": digest}


def delete_document(connection: sqlite3.Connection, path: Path) -> dict:
    """Remove a document from retrieval in one transaction."""
    resolved = str(path.resolve())
    with connection:
        row = connection.execute(
            "SELECT doc_id FROM documents WHERE source_path = ?", (resolved,)
        ).fetchone()
        if not row:
            return {"path": resolved, "action": "absent"}
        connection.execute("DELETE FROM chunks WHERE doc_id = ?", (row["doc_id"],))
        connection.execute(
            "UPDATE documents SET deleted_at = ? WHERE doc_id = ?",
            (time.strftime("%Y-%m-%dT%H:%M:%S%z"), row["doc_id"]),
        )
    return {"path": resolved, "action": "deleted", "doc_id": row["doc_id"]}


def sync_directory(connection: sqlite3.Connection, root: Path,
                   patterns: tuple[str, ...] = DEFAULT_INGEST_PATTERNS) -> dict:
    """Ingest new/changed files under root and delete records whose file is gone."""
    seen: set[str] = set()
    results = []
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                seen.add(str(path.resolve()))
                results.append(ingest_file(connection, path))
    stale = [
        row["source_path"]
        for row in connection.execute(
            "SELECT source_path FROM documents WHERE deleted_at IS NULL"
        ).fetchall()
        if row["source_path"].startswith(str(root.resolve())) and row["source_path"] not in seen
    ]
    for source_path in stale:
        results.append(delete_document(connection, Path(source_path)))
    return {"root": str(root.resolve()), "results": results}


def search(connection: sqlite3.Connection, query: str, top_k: int = 24) -> list[dict]:
    """Dense retrieval over live chunks only."""
    rows = connection.execute(
        "SELECT c.chunk_id, c.doc_id, c.ordinal, c.char_start, c.char_end, c.text, c.dim, c.embedding, "
        "       d.source_path, d.content_sha256 "
        "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE d.deleted_at IS NULL"
    ).fetchall()
    if not rows:
        return []
    matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    vector = embed([query])[0]
    scores = matrix @ vector
    order = np.argsort(-scores)[:top_k]
    return [
        {
            "chunk_id": rows[i]["chunk_id"],
            "doc_id": rows[i]["doc_id"],
            "ordinal": rows[i]["ordinal"],
            "source_path": rows[i]["source_path"],
            "content_sha256": rows[i]["content_sha256"],
            "char_start": rows[i]["char_start"],
            "char_end": rows[i]["char_end"],
            "text": rows[i]["text"],
            "dense_score": float(scores[i]),
        }
        for i in order
    ]


def render_passage(index: int, passage: dict) -> str:
    """One numbered, provenance-stamped, explicitly-untrusted passage block."""
    return (
        f"[{index}] source={passage['source_path']} "
        f"chars={passage['char_start']}-{passage['char_end']} "
        f"sha256={passage['content_sha256'][:12]}\n"
        f"<<<BEGIN UNTRUSTED DOCUMENT DATA {index}>>>\n"
        f"{passage['text']}\n"
        f"<<<END UNTRUSTED DOCUMENT DATA {index}>>>"
    )


def build_context(passages: list[dict]) -> tuple[str, list[dict]]:
    """Assemble a bounded, numbered, explicitly-untrusted context block.

    The budget is measured against the *rendered* text, envelope and provenance
    header included. Counting only the raw passage body would let the per-passage
    framing push the real prompt past the cap.
    """
    selected: list[dict] = []
    blocks: list[str] = []
    total = 0
    separator = len("\n\n")
    for passage in passages:
        if len(selected) >= MAX_CONTEXT_CHUNKS:
            break
        block = render_passage(len(selected) + 1, passage)
        cost = len(block) + (separator if blocks else 0)
        if total + cost > MAX_CONTEXT_CHARS:
            continue
        total += cost
        blocks.append(block)
        selected.append(passage)
    return "\n\n".join(blocks), selected


def stats(connection: sqlite3.Connection) -> dict:
    live_documents = connection.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE deleted_at IS NULL"
    ).fetchone()["n"]
    deleted_documents = connection.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE deleted_at IS NOT NULL"
    ).fetchone()["n"]
    chunk_count = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    return {"live_documents": live_documents, "deleted_documents": deleted_documents, "chunks": chunk_count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes local RAG store")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="ingest/update/delete a directory tree")
    sync.add_argument("root", type=Path)
    ingest = sub.add_parser("ingest", help="ingest one file")
    ingest.add_argument("path", type=Path)
    remove = sub.add_parser("delete", help="remove one file from retrieval")
    remove.add_argument("path", type=Path)
    query = sub.add_parser("search", help="dense search")
    query.add_argument("text")
    query.add_argument("--top-k", type=int, default=8)
    sub.add_parser("stats", help="show corpus counts")

    args = parser.parse_args()
    connection = connect(args.db)
    if args.command == "sync":
        print(json.dumps(sync_directory(connection, args.root), indent=2))
    elif args.command == "ingest":
        print(json.dumps(ingest_file(connection, args.path), indent=2))
    elif args.command == "delete":
        print(json.dumps(delete_document(connection, args.path), indent=2))
    elif args.command == "search":
        print(json.dumps(search(connection, args.text, args.top_k), indent=2))
    else:
        print(json.dumps(stats(connection), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
