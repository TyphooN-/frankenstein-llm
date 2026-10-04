#!/usr/bin/env python3
"""Generate portable BitTorrent v1 metainfo, without any network activity.

Weights are opened read-only. Only adjacent tool-owned sidecars and a report
are written. Metadata generation is not provenance, licensing or seed approval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.parse import quote

SCHEMA = "model-torrent-sidecars/1"
MODEL_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".ckpt"})
MAX_PIECES = 65536
MAX_META_BYTES = 2 * 1024 * 1024
COMMENT = ("Local metainfo only; provenance and redistribution permissions are not verified. "
           "Preserve the model's license, notices, configuration and companion files before sharing.")


def bencode(value):
    """Canonical BEP 3 encoding; SHA-1 is used only as required by BitTorrent v1."""
    if isinstance(value, bytes):
        return str(len(value)).encode("ascii") + b":" + value
    if type(value) is int:
        return b"i" + str(value).encode("ascii") + b"e"
    if isinstance(value, (list, tuple)):
        return b"l" + b"".join(bencode(item) for item in value) + b"e"
    if isinstance(value, dict) and all(isinstance(key, bytes) for key in value):
        return b"d" + b"".join(bencode(key) + bencode(value[key]) for key in sorted(value)) + b"e"
    raise TypeError("bencoding accepts bytes, integers, sequences and byte-keyed dictionaries")


def choose_piece_length(size):
    piece = 256 * 1024
    while (size + piece - 1) // piece > 8192 and piece < 16 * 1024 * 1024:
        piece *= 2
    if (size + piece - 1) // piece > MAX_PIECES:
        raise ValueError("file exceeds the bounded metainfo budget")
    return piece


def source_signature(path):
    s = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(s.st_mode):
        raise ValueError(f"source symlink refused: {path}")
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f"not a regular source file: {path}")
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def scan_models(root):
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("model root must be a real directory, not a symlink")
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(base) / d).is_symlink())
        for name in sorted(names):
            p = Path(base) / name
            if (p.suffix.lower() in MODEL_SUFFIXES and not p.is_symlink()
                    and not name.startswith(".")
                    and not any(tag in name.lower() for tag in (".partial", ".part.", ".tmp.", ".incomplete", ".download"))):
                yield p


@contextmanager
def root_lock(root):
    path = Path(root) / ".torrent-sidecars.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def write_atomic(path, data):
    """Publish a complete local output, never follow an existing output symlink."""
    if path.is_symlink():
        raise ValueError(f"sidecar symlink refused: {path}")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise ValueError(f"sidecar symlink refused: {path}")
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def hash_source(path, piece_length):
    """One serial read pass; memory is bounded by one piece plus MAX_PIECES hashes."""
    pieces = bytearray()
    whole = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        while block := stream.read(piece_length):
            pieces.extend(hashlib.sha1(block).digest())
            whole.update(block)
            if len(pieces) > MAX_PIECES * 20:
                raise ValueError("source grew beyond metainfo budget")
    return bytes(pieces), whole.hexdigest()


def generate_one(path, *, min_age_seconds=300, piece_length=None):
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"source parent symlink refused: {path}")
    signature = source_signature(path)
    size = signature[2]
    if size == 0 or time.time_ns() - signature[3] < min_age_seconds * 1_000_000_000:
        return {"source": str(path), "size": size, "status": "skipped", "reason": "empty or still settling"}
    piece = choose_piece_length(size) if piece_length is None else piece_length
    if (not 16384 <= piece <= 16777216 or piece & (piece - 1)
            or (size + piece - 1) // piece > MAX_PIECES):
        raise ValueError("invalid or excessive piece length")
    torrent = path.with_name(path.name + ".torrent")
    magnet_file = path.with_name(path.name + ".magnet")
    receipt = path.with_name(path.name + ".torrent.json")
    for output in (torrent, magnet_file, receipt):
        if output.is_symlink():
            raise ValueError(f"sidecar symlink refused: {output}")
        if output.exists() and not output.is_file():
            raise ValueError(f"nonregular sidecar refused: {output}")
    previous = None
    if receipt.exists():
        if receipt.stat().st_size > MAX_META_BYTES:
            raise ValueError("oversized ownership receipt refused")
        previous = json.loads(receipt.read_text())
        if not isinstance(previous, dict) or previous.get("schema") != SCHEMA or previous.get("name") != path.name:
            raise FileExistsError(f"unowned sidecar receipt: {receipt}")
    elif torrent.exists() or magnet_file.exists():
        raise FileExistsError(f"unowned sidecars preserved: {path}")
    if (previous and previous.get("complete") is True
            and previous.get("source_signature") == signature
            and previous.get("piece_length") == piece
            and isinstance(previous.get("infohash"), str)
            and re.fullmatch(r"[0-9a-f]{40}", previous.get("infohash", ""))
            and torrent.is_file() and magnet_file.is_file()
            and torrent.stat().st_size <= MAX_META_BYTES
            and magnet_file.stat().st_size <= 16384):
        expected_magnet = "magnet:?xt=urn:btih:" + previous["infohash"] + "&dn=" + quote(path.name, safe="") + "&xl=" + str(size)
        if (hashlib.sha256(torrent.read_bytes()).hexdigest() == previous.get("metainfo_sha256")
                and magnet_file.read_text() == expected_magnet + "\n"):
            return {**previous, "status": "unchanged", "source": str(path), "torrent": str(torrent), "magnet_path": str(magnet_file)}
    pieces, sha256 = hash_source(path, piece)
    if source_signature(path) != signature:
        raise RuntimeError(f"source changed during hashing: {path}")
    expected_pieces = (size + piece - 1) // piece
    if len(pieces) != expected_pieces * 20:
        raise RuntimeError(f"source length changed during hashing: {path}")
    info = {b"name": path.name.encode("utf-8"), b"length": size,
            b"piece length": piece, b"pieces": pieces}
    infohash = hashlib.sha1(bencode(info)).hexdigest()
    metainfo = bencode({b"comment": COMMENT.encode(), b"created by": SCHEMA.encode(), b"info": info})
    magnet = "magnet:?xt=urn:btih:" + infohash + "&dn=" + quote(path.name, safe="") + "&xl=" + str(size)
    row = {"schema": SCHEMA, "name": path.name, "source": str(path), "source_signature": signature,
           "size": size, "piece_length": piece, "pieces": expected_pieces, "sha256": sha256,
           "infohash": infohash, "magnet": magnet, "torrent": str(torrent), "magnet_path": str(magnet_file),
           "metainfo_sha256": hashlib.sha256(metainfo).hexdigest(), "redistribution_review": "not-performed",
           "complete": False, "status": "created"}
    # Establish recoverable ownership before output publication. An interrupted
    # file can be rebuilt; existing unrelated torrents are never adopted.
    write_atomic(receipt, (json.dumps(row, indent=2) + "\n").encode())
    write_atomic(torrent, metainfo)
    write_atomic(magnet_file, (magnet + "\n").encode())
    row["complete"] = True
    write_atomic(receipt, (json.dumps(row, indent=2) + "\n").encode())
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Model tree (regular complete weight files only)")
    parser.add_argument("--report", type=Path, help="Local index (default: model root/.torrent-sidecars-index.json)")
    parser.add_argument("--min-age-seconds", type=int, default=300, help="Ignore files modified within this interval")
    parser.add_argument("--dry-run", action="store_true", help="List selected files without hashing or writing anything")
    args = parser.parse_args(argv)
    if args.min_age_seconds < 0:
        parser.error("minimum age must be nonnegative")
    root = args.root.absolute()
    paths = list(scan_models(root))
    if args.dry_run:
        print(json.dumps({"status": "planned", "files": [str(p) for p in paths], "count": len(paths),
                          "bytes": sum(p.stat().st_size for p in paths)}, indent=2))
        return 0
    report_path = args.report.absolute() if args.report else root / ".torrent-sidecars-index.json"
    if report_path.suffix != ".json" or report_path.is_symlink() or any(p.is_symlink() for p in report_path.parents):
        parser.error("report must be a regular .json output, not a symlink or a model weight")
    if report_path.is_relative_to(root) and report_path.name != ".torrent-sidecars-index.json":
        parser.error("inside the model tree, only .torrent-sidecars-index.json is a report destination")
    if report_path.exists():
        if report_path.stat().st_size > 16 * 1024 * 1024:
            parser.error("oversized existing report refused")
        previous_report = json.loads(report_path.read_text())
        if not isinstance(previous_report, dict) or previous_report.get("schema") != SCHEMA:
            parser.error("unowned existing report preserved")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "root": str(root), "started_at": dt.datetime.now().astimezone().isoformat(),
              "selected": len(paths), "network_activity": False, "seeding": False,
              "redistribution_review": "not-performed", "files": []}
    try:
        with root_lock(root):
            for index, path in enumerate(paths, 1):
                print(json.dumps({"event": "hash_or_check", "index": index, "total": len(paths), "source": str(path)}), flush=True)
                try:
                    row = generate_one(path, min_age_seconds=args.min_age_seconds)
                except (OSError, ValueError, RuntimeError) as error:
                    row = {"source": str(path), "status": "error", "error": str(error)}
                report["files"].append(row)
                print(json.dumps({"event": "file_done", "index": index, "total": len(paths), "status": row["status"],
                                  "source": str(path), "infohash": row.get("infohash"), "error": row.get("error")}), flush=True)
            report["counts"] = {key: sum(row["status"] == key for row in report["files"])
                                for key in ("created", "unchanged", "skipped", "error")}
            report["finished_at"] = dt.datetime.now().astimezone().isoformat()
            write_atomic(report_path, (json.dumps(report, indent=2) + "\n").encode())
    except BlockingIOError:
        print(json.dumps({"status": "busy", "reason": "Another generator owns the root; next timer retries."}), flush=True)
        return 0
    print(json.dumps({"event": "finished", "selected": len(paths), "counts": report["counts"], "report": str(report_path)}), flush=True)
    if report["counts"]["error"]:
        return 1
    return 2 if report["counts"]["skipped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
