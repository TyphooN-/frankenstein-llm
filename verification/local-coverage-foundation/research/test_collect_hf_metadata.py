#!/usr/bin/env python3
"""Offline tests for the Hugging Face metadata collector. No network, no weights.

Two failures this module exists to prevent, both of which produce a document
that *looks* complete:

1. The tree API caps a response at 50 entries and hands the rest back through a
   ``Link: rel="next"`` header. Reading one page understates the shard list of
   every large repository, and a shard total is the number the download and
   fit decisions are made from.
2. A crash between create and write leaves a zero-byte file. That is how an
   earlier collection run was lost, so publishing goes through a temporary file,
   an fsync, a rename, and a directory fsync.

Every network call is replaced with a recorded page list, so a mid-walk failure
can be exercised as easily as the happy path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_hf_metadata as collector  # noqa: E402


def file_entry(path: str, size: int) -> dict:
    return {"type": "file", "path": path, "size": size,
            "lfs": {"size": size, "oid": "0" * 64}}


def pages(*responses):
    """Turn recorded (payload, next_url) pairs into a fetch() stand-in."""
    calls = []

    def fake_fetch(url: str, tries: int = 4):
        calls.append(url)
        return responses[len(calls) - 1]

    return fake_fetch, calls


class NextLinkTests(unittest.TestCase):
    def test_a_real_next_header_yields_the_cursor_url(self):
        header = ('<https://huggingface.co/api/models/a/b/tree/rev'
                  '?expand=true&recursive=true&limit=50&cursor=abc%3D%3D>; rel="next"')
        match = collector.NEXT_LINK.search(header)
        self.assertIsNotNone(match)
        self.assertTrue(match.group(1).endswith("cursor=abc%3D%3D"))

    def test_a_header_without_a_next_relation_is_not_followed(self):
        self.assertIsNone(collector.NEXT_LINK.search('<https://example/x>; rel="prev"'))

    def test_an_absent_header_is_not_followed(self):
        self.assertIsNone(collector.NEXT_LINK.search(""))


class TreePaginationTests(unittest.TestCase):
    def test_every_page_is_walked_and_the_pages_are_concatenated(self):
        first = [file_entry(f"shard-{i}.safetensors", 1000) for i in range(50)]
        second = [file_entry(f"shard-{i}.safetensors", 1000) for i in range(50, 61)]
        fake, calls = pages((first, "https://next/page2"), (second, None))
        with unittest.mock.patch.object(collector, "fetch", fake):
            entries, error = collector.get_tree("org/repo", "rev")
        self.assertIsNone(error)
        self.assertEqual(61, len(entries), "the second page was dropped")
        self.assertEqual("https://next/page2", calls[1])

    def test_a_failure_midway_is_reported_rather_than_returned_as_complete(self):
        first = [file_entry("shard-0.safetensors", 1000)]
        fake, _ = pages((first, "https://next/page2"), ({"__error__": "boom"}, None))
        with unittest.mock.patch.object(collector, "fetch", fake):
            entries, error = collector.get_tree("org/repo", "rev")
        self.assertEqual({"__error__": "boom"}, error)
        self.assertEqual(1, len(entries))

    def test_collect_refuses_to_publish_a_truncated_listing_as_a_document(self):
        info = {"sha": "rev", "tags": ["license:mit"]}
        fake, _ = pages(([file_entry("a.safetensors", 7)], "https://next/page2"),
                        ({"__error__": "boom"}, None))
        with unittest.mock.patch.object(collector, "get", lambda url, tries=4: info), \
             unittest.mock.patch.object(collector, "fetch", fake):
            document = collector.collect("org/repo")
        self.assertIn("status", document, "a partial walk was reported as a full one")
        self.assertNotIn("files", document)
        self.assertEqual(1, len(document["partial_files"]))

    def test_a_complete_walk_records_files_sizes_and_the_license(self):
        info = {"sha": "rev", "tags": ["license:apache-2.0"], "pipeline_tag": "text-generation"}
        fake, _ = pages(([file_entry("b.gguf", 5), file_entry("a.gguf", 3),
                          {"type": "directory", "path": "assets"}], None))
        with unittest.mock.patch.object(collector, "get", lambda url, tries=4: info), \
             unittest.mock.patch.object(collector, "fetch", fake):
            document = collector.collect("org/repo")
        self.assertEqual(["a.gguf", "b.gguf"], [f["path"] for f in document["files"]],
                         "directories must be dropped and files sorted")
        self.assertEqual(8, sum(f["size"] for f in document["files"]))
        self.assertEqual("apache-2.0", document["license"])

    def test_an_unavailable_repository_keeps_its_http_status(self):
        with unittest.mock.patch.object(collector, "get",
                                        lambda url, tries=4: {"__http_error__": 404}):
            document = collector.collect("org/missing")
        self.assertEqual({"__http_error__": 404}, document["status"])


class PublishTests(unittest.TestCase):
    def test_the_file_and_its_directory_entry_are_both_fsynced(self):
        synced = []
        real_fsync = os.fsync

        def recording(descriptor):
            synced.append(os.fstat(descriptor).st_mode & 0o170000)
            real_fsync(descriptor)

        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "out.json"
            with unittest.mock.patch.object(collector.os, "fsync", recording):
                collector.publish(target, '{"a": 1}\n')
            self.assertEqual('{"a": 1}\n', target.read_text())
        self.assertIn(0o100000, synced, "the document was never fsynced")
        self.assertIn(0o040000, synced, "the directory entry was never fsynced")

    def test_a_filesystem_that_refuses_directory_fsync_still_publishes(self):
        def only_directory_fails(descriptor):
            if os.fstat(descriptor).st_mode & 0o170000 == 0o040000:
                raise OSError("directory fsync unsupported")

        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "out.json"
            with unittest.mock.patch.object(collector.os, "fsync", only_directory_fails):
                collector.publish(target, "kept\n")
            self.assertEqual("kept\n", target.read_text())

    def test_no_temporary_file_is_left_beside_the_published_document(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "out.json"
            collector.publish(target, "x\n")
            self.assertEqual(["out.json"], sorted(p.name for p in Path(root).iterdir()))

    def test_publishing_over_an_existing_document_replaces_it_whole(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "out.json"
            collector.publish(target, json.dumps({"old": True}) + "\n")
            collector.publish(target, json.dumps({"new": True}) + "\n")
            self.assertEqual({"new": True}, json.loads(target.read_text()))


if __name__ == "__main__":
    unittest.main()
