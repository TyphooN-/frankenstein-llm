"""Offline metainfo, source preservation, caching and refusal contracts."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("model_sidecars", ROOT / "scripts/model-torrent-sidecars.py")
assert spec is not None and spec.loader is not None
sidecars = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sidecars
spec.loader.exec_module(sidecars)


def decode(data: bytes) -> dict[bytes, Any]:
    """Independent fixture decoder; actual interoperability uses torf."""
    def take(pos: int) -> tuple[Any, int]:
        token = data[pos:pos + 1]
        if token == b"i":
            end = data.index(b"e", pos)
            return int(data[pos + 1:end]), end + 1
        if token in (b"d", b"l"):
            result: Any = {} if token == b"d" else []
            pos += 1
            while data[pos:pos + 1] != b"e":
                value, pos = take(pos)
                if token == b"d":
                    other, pos = take(pos)
                    result[value] = other
                else:
                    result.append(value)
            return result, pos + 1
        sep = data.index(b":", pos)
        size = int(data[pos:sep])
        return data[sep + 1:sep + 1 + size], sep + 1 + size
    result, end = take(0)
    assert end == len(data)
    return result


def model(tmp_path, name="model.gguf", data=b"GGUF" + b"x" * 40000):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def make(p, **kwargs):
    return sidecars.generate_one(p, min_age_seconds=0, **kwargs)


def test_canonical_bencoding():
    assert sidecars.bencode({b"spam": [b"a", b"b"], b"cow": b"moo"}) == b"d3:cow3:moo4:spaml1:a1:bee"
    assert sidecars.bencode(0) == b"i0e"
    with pytest.raises(TypeError):
        sidecars.bencode({"nonbytes": b"no"})


def test_real_single_file_hashes_and_portable_magnet(tmp_path):
    p = model(tmp_path, "weights ü space.gguf")
    original = p.read_bytes()
    before = p.stat()
    row = make(p, piece_length=16384)
    torrent = p.with_name(p.name + ".torrent")
    meta = decode(torrent.read_bytes())
    info = meta[b"info"]
    assert info[b"name"].decode() == p.name
    assert info[b"length"] == len(original)
    assert info[b"piece length"] == 16384
    assert info[b"pieces"] == b"".join(hashlib.sha1(original[i:i + 16384]).digest() for i in range(0, len(original), 16384))
    assert set(info) == {b"name", b"length", b"piece length", b"pieces"}
    assert not ({b"announce", b"announce-list", b"url-list", b"nodes"} & meta.keys())
    assert str(tmp_path).encode() not in torrent.read_bytes()
    ih = hashlib.sha1(sidecars.bencode(info)).hexdigest()
    magnet = p.with_name(p.name + ".magnet").read_text().strip()
    query = parse_qs(urlsplit(magnet).query)
    assert query["xt"] == ["urn:btih:" + ih]
    assert query["dn"] == [p.name]
    assert query["xl"] == [str(len(original))]
    assert "tr" not in query
    assert row["infohash"] == ih and row["sha256"] == hashlib.sha256(original).hexdigest()
    assert row["status"] == "created"
    assert p.read_bytes() == original and p.stat().st_mtime_ns == before.st_mtime_ns
    assert row["redistribution_review"] == "not-performed"


def test_unchanged_files_are_not_rehashed(tmp_path, monkeypatch):
    p = model(tmp_path)
    first = make(p)
    before = p.with_name(p.name + ".torrent").stat().st_mtime_ns
    def forbidden(*args, **kwargs):
        pytest.fail("unchanged content was rehashed")
    monkeypatch.setattr(sidecars, "hash_source", forbidden)
    second = make(p)
    assert second["status"] == "unchanged"
    assert second["infohash"] == first["infohash"]
    assert p.with_name(p.name + ".torrent").stat().st_mtime_ns == before


def test_changed_source_gets_a_new_infohash(tmp_path):
    p = model(tmp_path)
    first = make(p)
    p.write_bytes(b"GGUF" + b"new content")
    second = make(p)
    assert second["status"] == "created"
    assert second["infohash"] != first["infohash"]


def test_damaged_owned_sidecars_are_repaired(tmp_path):
    p = model(tmp_path)
    first = make(p)
    p.with_name(p.name + ".torrent").write_bytes(b"broken")
    p.with_name(p.name + ".magnet").write_text("broken")
    second = make(p)
    assert second["status"] == "created"
    assert second["infohash"] == first["infohash"]
    assert decode(p.with_name(p.name + ".torrent").read_bytes())[b"info"][b"name"] == p.name.encode()


def test_unowned_torrent_is_not_overwritten(tmp_path):
    p = model(tmp_path)
    t = p.with_name(p.name + ".torrent")
    t.write_bytes(b"operator-owned metainfo")
    with pytest.raises(FileExistsError):
        make(p)
    assert t.read_bytes() == b"operator-owned metainfo"


@pytest.mark.parametrize("extension", [".torrent", ".magnet", ".torrent.json"])
def test_sidecar_symlinks_cannot_write_outside_model_tree(tmp_path, extension):
    p = model(tmp_path)
    outside = tmp_path / "unrelated"
    outside.write_text("keep")
    p.with_name(p.name + extension).symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        make(p)
    assert outside.read_text() == "keep"


def test_source_symlinks_are_refused(tmp_path):
    p = model(tmp_path)
    link = tmp_path / "other.gguf"
    link.symlink_to(p)
    with pytest.raises(ValueError, match="symlink"):
        make(link)
    assert not link.with_name(link.name + ".torrent").exists()


def test_mid_hash_mutation_never_publishes(tmp_path, monkeypatch):
    p = model(tmp_path)
    original = sidecars.hash_source
    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        p.write_bytes(b"changed during hashing")
        return result
    monkeypatch.setattr(sidecars, "hash_source", change)
    with pytest.raises(RuntimeError, match="changed"):
        make(p)
    assert not p.with_name(p.name + ".torrent").exists()


def test_still_settling_or_empty_sources_are_skipped(tmp_path):
    p = model(tmp_path)
    assert sidecars.generate_one(p, min_age_seconds=3600)["status"] == "skipped"
    p.write_bytes(b"")
    assert make(p)["status"] == "skipped"
    assert not p.with_name(p.name + ".torrent").exists()


def test_scan_includes_weights_not_partial_files_or_links(tmp_path):
    expected = [model(tmp_path, "weights" + ext) for ext in sidecars.MODEL_SUFFIXES]
    model(tmp_path, "download.gguf.partial")
    model(tmp_path, "unfinished.partial.gguf")
    model(tmp_path, "README.md")
    (tmp_path / "alias.gguf").symlink_to(expected[0])
    hidden = tmp_path / ".staging"
    hidden.mkdir()
    model(hidden, "secret.gguf")
    (tmp_path / "escape").symlink_to(hidden, target_is_directory=True)
    assert set(sidecars.scan_models(tmp_path)) == set(expected)


def test_piece_sizes_are_power_of_two_and_bounded():
    for size in (1, 40000, 30 * 1024**3, 800 * 1024**3):
        piece = sidecars.choose_piece_length(size)
        assert 16384 <= piece <= 16777216 and piece & (piece - 1) == 0
        assert (size + piece - 1) // piece <= 65536
    with pytest.raises(ValueError):
        sidecars.choose_piece_length(2 * 1024**4)


def test_root_lock_prevents_duplicate_writers(tmp_path):
    import fcntl
    with sidecars.root_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with sidecars.root_lock(tmp_path):
                pytest.fail("duplicate lock acquired")


def test_hash_read_failure_does_not_publish(tmp_path, monkeypatch):
    p = model(tmp_path)
    def unreadable(*args, **kwargs):
        raise OSError("fixture EIO")
    monkeypatch.setattr(sidecars, "hash_source", unreadable)
    with pytest.raises(OSError, match="EIO"):
        make(p)
    assert not p.with_name(p.name + ".torrent").exists()


def test_cli_report_cannot_overwrite_weights(tmp_path):
    p = model(tmp_path)
    before = p.read_bytes()
    with pytest.raises(SystemExit) as exit_info:
        sidecars.main(["--root", str(tmp_path), "--report", str(p), "--min-age-seconds", "0"])
    assert exit_info.value.code == 2 and p.read_bytes() == before


def test_cli_preserves_unowned_report(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    p = model(models)
    report = tmp_path / "user.json"
    report.write_text('{"user": "keep"}')
    with pytest.raises(SystemExit):
        sidecars.main(["--root", str(models), "--report", str(report)])
    assert report.read_text() == '{"user": "keep"}'
    assert not p.with_name(p.name + ".torrent").exists()


def test_cli_dry_run_does_not_write(tmp_path):
    p = model(tmp_path)
    assert sidecars.main(["--root", str(tmp_path), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == [p]


def test_cli_is_network_free_and_writes_real_index(tmp_path, monkeypatch):
    import json
    import socket
    def network_forbidden(*args, **kwargs):
        pytest.fail("network operation attempted")
    monkeypatch.setattr(socket, "socket", network_forbidden)
    p = model(tmp_path)
    assert sidecars.main(["--root", str(tmp_path), "--min-age-seconds", "0"]) == 0
    report = json.loads((tmp_path / ".torrent-sidecars-index.json").read_text())
    assert report["selected"] == 1
    assert report["counts"] == {"created": 1, "unchanged": 0, "skipped": 0, "error": 0}
    assert report["network_activity"] is False and report["seeding"] is False
    assert report["files"][0]["source"] == str(p)


def test_nonregular_sidecar_is_refused_without_blocking(tmp_path):
    import os
    p = model(tmp_path)
    receipt = p.with_name(p.name + ".torrent.json")
    os.mkfifo(receipt)
    with pytest.raises(ValueError, match="nonregular"):
        make(p)
    assert not p.with_name(p.name + ".torrent").exists()
