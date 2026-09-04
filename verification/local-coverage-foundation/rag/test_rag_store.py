#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from rag_store import DEFAULT_INGEST_PATTERNS, extract_text


class ExtractTextTests(unittest.TestCase):
    def test_markdown_is_decoded_as_utf8(self):
        text = extract_text(Path("note.md"), b"# ZFS scrub repaired zero bytes\n")
        self.assertIn("ZFS scrub repaired zero bytes", text)

    def test_html_keeps_visible_text_and_drops_script(self):
        raw = (
            b"<html><head><script>stealSecrets()</script></head>"
            b"<body><p>ZFS scrub repaired zero bytes</p></body></html>"
        )
        text = extract_text(Path("page.html"), raw)
        self.assertIn("ZFS scrub repaired zero bytes", text)
        self.assertNotIn("stealSecrets", text)

    def test_png_is_not_ingested_as_replacement_text(self):
        self.assertIsNone(extract_text(Path("shot.png"), b"\x89PNG\r\n\x1a\nnot-text"))

    def test_pdf_without_an_extractor_is_skipped_not_decoded(self):
        self.assertIsNone(extract_text(Path("doc.pdf"), b"%PDF-1.4 binary-junk"))


class SyncPatternTests(unittest.TestCase):
    def test_default_patterns_include_html(self):
        from rag_store import DEFAULT_INGEST_PATTERNS
        self.assertIn("*.html", DEFAULT_INGEST_PATTERNS)
        self.assertIn("*.md", DEFAULT_INGEST_PATTERNS)


if __name__ == "__main__":
    unittest.main()
