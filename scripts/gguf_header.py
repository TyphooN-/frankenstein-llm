#!/usr/bin/env python3
"""Read a GGUF file's metadata header without loading the model.

Placement arithmetic needs facts that only the weight file carries: how many
blocks a model has, how wide its key/value heads are, what context it was
trained for. Every other source for those numbers is a copy that can drift --
a README, a catalog entry, a remembered figure from a different quantization of
the same release.

Only the header is read. The KV block sits at the front of the file, the tensor
data behind it, so this stops at the first byte of tensor data and never faults
in a 25 GiB mapping. Nothing here executes anything from the file; it is a
bounded parse of a documented binary layout, which is what makes it safe to
point at weights whose provenance is a download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = frozenset({2, 3})

# Value type ids from the GGUF specification. ARRAY nests one of the others.
STRING = 8
ARRAY = 9
SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

# A metadata block larger than this is not a header this tool should keep
# reading: the parse is meant to be bounded, and a corrupt length field would
# otherwise turn a header read into an unbounded allocation.
MAX_STRING_BYTES = 1 << 24
MAX_ARRAY_ITEMS = 1 << 26
# GGUF arrays may nest, so a hostile or corrupt file can describe an array of
# arrays of arrays without limit. Left alone that is unbounded recursion, which
# surfaces as a RecursionError rather than as the GGUFError every caller here
# catches -- an unreadable file would take the caller down instead of being
# reported as unreadable. Real headers nest one level: an array of scalars, or
# an array of strings.
MAX_ARRAY_DEPTH = 8


class GGUFError(ValueError):
    """The file is not a GGUF header this reader will parse."""


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def raw(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise GGUFError(f"header truncated: wanted {count} bytes, got {len(data)}")
        return data

    def scalar(self, kind: int):
        fmt, size = SCALARS[kind]
        return struct.unpack(fmt, self.raw(size))[0]

    def u32(self) -> int:
        return self.scalar(4)

    def u64(self) -> int:
        return self.scalar(10)

    def string(self) -> str:
        length = self.u64()
        if length > MAX_STRING_BYTES:
            raise GGUFError(f"implausible string length {length}")
        return self.raw(length).decode("utf-8", errors="replace")

    def value(self, kind: int, depth: int = 0):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            if depth >= MAX_ARRAY_DEPTH:
                raise GGUFError(f"array nested deeper than {MAX_ARRAY_DEPTH} levels")
            element = self.u32()
            count = self.u64()
            if count > MAX_ARRAY_ITEMS:
                raise GGUFError(f"implausible array length {count}")
            if element == STRING:
                return [self.string() for _ in range(count)]
            if element == ARRAY:
                return [self.value(ARRAY, depth + 1) for _ in range(count)]
            if element not in SCALARS:
                raise GGUFError(f"unknown array element type {element}")
            fmt, size = SCALARS[element]
            return list(struct.unpack(f"<{count}{fmt[1]}", self.raw(count * size)))
        if kind not in SCALARS:
            raise GGUFError(f"unknown value type {kind}")
        return self.scalar(kind)


def read_header(path) -> dict:
    """``{"version", "tensor_count", "metadata"}`` for one GGUF file."""
    with Path(path).open("rb") as handle:
        reader = _Reader(handle)
        magic = reader.raw(4)
        if magic != MAGIC:
            raise GGUFError(f"not a GGUF file: magic {magic!r}")
        version = reader.u32()
        if version not in SUPPORTED_VERSIONS:
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = reader.u64()
        pairs = reader.u64()
        if pairs > MAX_ARRAY_ITEMS:
            raise GGUFError(f"implausible metadata count {pairs}")
        metadata: dict[str, object] = {}
        for _ in range(pairs):
            key = reader.string()
            metadata[key] = reader.value(reader.u32())
        return {"version": version, "tensor_count": tensor_count,
                "metadata": metadata}


def architecture(metadata: dict) -> str:
    value = metadata.get("general.architecture")
    return value if isinstance(value, str) else "unknown"


def arch_value(metadata: dict, suffix: str, default=None):
    """One ``<architecture>.<suffix>`` entry, which is how GGUF namespaces these."""
    return metadata.get(f"{architecture(metadata)}.{suffix}", default)


def _int(value, default=None):
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def shape(header: dict) -> dict:
    """The fields KV-cache arithmetic needs, or ``None`` where the file is silent.

    ``head_count_kv`` is per-layer on some architectures and a single number on
    others; GGUF stores the per-layer form as an array. Both are returned as
    written, because collapsing an array to its maximum would silently overstate
    a model whose layers differ -- which is exactly the case (sliding-window
    attention) where an estimate needs the caller's attention, not a smoothed
    number.
    """
    metadata = header["metadata"]
    heads = arch_value(metadata, "attention.head_count")
    heads_kv = arch_value(metadata, "attention.head_count_kv", heads)
    embedding = _int(arch_value(metadata, "embedding_length"))
    head_count = _int(heads if not isinstance(heads, list) else max(heads))
    derived = (embedding // head_count
               if embedding and head_count else None)
    return {
        "architecture": architecture(metadata),
        "name": metadata.get("general.name") if isinstance(
            metadata.get("general.name"), str) else None,
        "size_label": metadata.get("general.size_label") if isinstance(
            metadata.get("general.size_label"), str) else None,
        "file_type": _int(metadata.get("general.file_type")),
        "block_count": _int(arch_value(metadata, "block_count")),
        "context_length": _int(arch_value(metadata, "context_length")),
        "embedding_length": embedding,
        "head_count": heads,
        "head_count_kv": heads_kv,
        "key_length": _int(arch_value(metadata, "attention.key_length"), derived),
        "value_length": _int(arch_value(metadata, "attention.value_length"), derived),
        "sliding_window": _int(arch_value(metadata, "attention.sliding_window")),
        "expert_count": _int(arch_value(metadata, "expert_count")),
        "expert_used_count": _int(arch_value(metadata, "expert_used_count")),
        "tensor_count": header["tensor_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="+", help="GGUF file; first shard for split models")
    parser.add_argument("--all", action="store_true",
                        help="print every metadata key, not the placement fields")
    args = parser.parse_args(argv)
    failed = False
    report = {}
    for path in args.path:
        try:
            header = read_header(path)
        except (OSError, GGUFError) as error:
            report[path] = {"error": f"{type(error).__name__}: {error}"}
            failed = True
            continue
        if args.all:
            report[path] = {key: value for key, value in header["metadata"].items()
                            if not key.startswith("tokenizer.")}
        else:
            report[path] = shape(header)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
