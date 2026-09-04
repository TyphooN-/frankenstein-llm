#!/usr/bin/env python3
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import rag_store
from rag_store import (
    DEFAULT_INGEST_PATTERNS,
    SKIP_BINARY,
    SKIP_CORRUPT_ARCHIVE,
    SKIP_EMPTY,
    SKIP_OCR_REQUIRED,
    SKIP_OPTIONAL_UNAVAILABLE,
    SKIP_OVERSIZED_ARCHIVE,
    connect,
    extract_text,
    extract_text_with_reason,
    ingest_file,
)


def make_ooxml(parts: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)
    return output.getvalue()


class ExtractTextTests(unittest.TestCase):
    def test_markdown_is_decoded_as_utf8(self):
        text = extract_text(Path("note.md"), b"# ZFS scrub repaired zero bytes\n")
        self.assertIn("ZFS scrub repaired zero bytes", text)

    def test_binary_disguised_as_text_is_rejected_with_reason(self):
        text, reason = extract_text_with_reason(Path("note.txt"), b"\xff\xfe\xfd\x00")
        self.assertIsNone(text)
        self.assertEqual(reason, SKIP_BINARY)

    def test_html_keeps_visible_text_and_drops_script(self):
        raw = (
            b"<html><head><script>stealSecrets()</script></head>"
            b"<body><p>ZFS scrub repaired zero bytes</p></body></html>"
        )
        text = extract_text(Path("page.html"), raw)
        self.assertIn("ZFS scrub repaired zero bytes", text)
        self.assertNotIn("stealSecrets", text)

    def test_docx_extracts_document_runs(self):
        raw = make_ooxml({
            "word/document.xml": (
                b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                b"<w:body><w:p><w:r><w:t>First paragraph</w:t></w:r>"
                b"<w:r><w:t>continued</w:t></w:r></w:p></w:body></w:document>"
            )
        })
        self.assertEqual(extract_text(Path("brief.docx"), raw), "First paragraph continued")

    def test_pptx_extracts_all_slide_runs(self):
        namespace = b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
        raw = make_ooxml({
            "ppt/slides/slide1.xml": b"<a:sld " + namespace + b"><a:t>Alpha</a:t></a:sld>",
            "ppt/slides/slide2.xml": b"<a:sld " + namespace + b"><a:t>Beta</a:t></a:sld>",
        })
        self.assertEqual(extract_text(Path("deck.pptx"), raw), "Alpha Beta")

    def test_xlsx_extracts_shared_strings(self):
        raw = make_ooxml({
            "xl/sharedStrings.xml": (
                b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                b"<si><t>Symbol</t></si><si><t>Position</t></si></sst>"
            )
        })
        self.assertEqual(extract_text(Path("book.xlsx"), raw), "Symbol Position")

    def test_archive_expansion_is_capped_before_member_read(self):
        raw = make_ooxml({"word/document.xml": b"<document>large payload</document>"})
        with patch.object(rag_store, "MAX_EXPANDED_BYTES", 8):
            text, reason = extract_text_with_reason(Path("bomb.docx"), raw)
        self.assertIsNone(text)
        self.assertEqual(reason, SKIP_OVERSIZED_ARCHIVE)

    def test_corrupt_and_entity_archives_fail_closed(self):
        text, reason = extract_text_with_reason(Path("bad.docx"), b"not a zip")
        self.assertIsNone(text)
        self.assertEqual(reason, SKIP_CORRUPT_ARCHIVE)

        raw = make_ooxml({
            "word/document.xml": (
                b'<!DOCTYPE x [<!ENTITY boom "unsafe">]>'
                b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                b"<w:t>&boom;</w:t></w:document>"
            )
        })
        self.assertEqual(
            extract_text_with_reason(Path("entity.docx"), raw),
            (None, SKIP_CORRUPT_ARCHIVE),
        )

    def test_png_requires_ocr_instead_of_becoming_replacement_text(self):
        text, reason = extract_text_with_reason(Path("shot.png"), b"\x89PNG\r\n\x1a\nnot-text")
        self.assertIsNone(text)
        self.assertEqual(reason, SKIP_OCR_REQUIRED)

    def test_pdf_without_an_extractor_reports_missing_dependency(self):
        with patch.object(rag_store.importlib, "import_module", side_effect=ImportError):
            text, reason = extract_text_with_reason(Path("doc.pdf"), b"%PDF-1.4 binary-junk")
        self.assertIsNone(text)
        self.assertEqual(reason, f"{SKIP_OPTIONAL_UNAVAILABLE}: pypdf")

    def test_empty_text_has_a_stable_reason(self):
        self.assertEqual(extract_text_with_reason(Path("empty.md"), b" \n"), (None, SKIP_EMPTY))


class IngestSafetyTests(unittest.TestCase):
    def test_changed_document_that_becomes_binary_is_retired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "note.md"
            path.write_text("valid document", encoding="utf-8")
            connection = connect(root / "rag.sqlite3")
            self.addCleanup(connection.close)
            with patch.object(rag_store, "embed", return_value=np.asarray([[1.0, 0.0]], dtype=np.float32)):
                inserted = ingest_file(connection, path)
            self.assertEqual(inserted["action"], "inserted")

            path.write_bytes(b"\xff\xfe\xfd\x00")
            skipped = ingest_file(connection, path)
            self.assertEqual(skipped["action"], "skipped")
            self.assertEqual(skipped["reason"], SKIP_BINARY)
            self.assertEqual(skipped["retired_doc_id"], inserted["doc_id"])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)
            self.assertIsNotNone(
                connection.execute("SELECT deleted_at FROM documents").fetchone()[0]
            )


class SyncPatternTests(unittest.TestCase):
    def test_default_patterns_cover_every_in_process_document_format(self):
        for pattern in ("*.md", "*.html", "*.docx", "*.pptx", "*.xlsx", "*.pdf"):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, DEFAULT_INGEST_PATTERNS)
        self.assertNotIn("*.png", DEFAULT_INGEST_PATTERNS)


if __name__ == "__main__":
    unittest.main()
