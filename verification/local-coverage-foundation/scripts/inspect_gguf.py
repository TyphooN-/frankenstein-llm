#!/usr/bin/env python3
"""Read GGUF header metadata without loading the model onto a GPU.

Used to prove structural facts a model card cannot: that a reranker conversion
actually carries pooling_type=RANK, that an embedding model reports the expected
dimension, that a FIM model declares its prefix/suffix/middle token ids.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

# GGUF value type enum, from the format specification.
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
FIXED = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}


class Reader:
    def __init__(self, handle) -> None:
        self.handle = handle

    def raw(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise ValueError("unexpected end of file")
        return data

    def scalar(self, kind: int):
        fmt, size = FIXED[kind]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        length = struct.unpack("<Q", self.raw(8))[0]
        return self.raw(length).decode("utf-8", errors="replace")

    def value(self, kind: int, array_limit: int = 8):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item_kind = struct.unpack("<I", self.raw(4))[0]
            count = struct.unpack("<Q", self.raw(8))[0]
            items = []
            for index in range(count):
                item = self.value(item_kind)
                if index < array_limit:
                    items.append(item)
            return {"array_type": item_kind, "count": count, "head": items}
        return self.scalar(kind)


def read_metadata(path: Path, array_limit: int = 8) -> dict:
    with path.open("rb") as handle:
        reader = Reader(handle)
        magic = reader.raw(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {magic!r}")
        version = struct.unpack("<I", reader.raw(4))[0]
        tensor_count = struct.unpack("<Q", reader.raw(8))[0]
        kv_count = struct.unpack("<Q", reader.raw(8))[0]
        metadata = {}
        for _ in range(kv_count):
            key = reader.string()
            kind = struct.unpack("<I", reader.raw(4))[0]
            metadata[key] = reader.value(kind, array_limit)
    return {
        "path": str(path),
        "gguf_version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "metadata": metadata,
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: inspect_gguf.py <file.gguf> [key ...]", file=sys.stderr)
        return 2
    document = read_metadata(Path(argv[0]))
    keys = argv[1:]
    if keys:
        selected = {k: document["metadata"].get(k) for k in keys}
        print(json.dumps(selected, indent=2))
    else:
        summary = {
            k: v for k, v in document["metadata"].items()
            if not isinstance(v, dict) or v.get("count", 0) <= 8
        }
        print(json.dumps({
            "path": document["path"],
            "gguf_version": document["gguf_version"],
            "tensor_count": document["tensor_count"],
            "kv_count": document["kv_count"],
            "scalar_metadata": summary,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
