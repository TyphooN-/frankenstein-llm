from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("admit_corpus", HERE / "admit_corpus.py")
assert SPEC is not None and SPEC.loader is not None
admit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admit)


class PromptCorpusAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "corpora"
        self.root.mkdir()
        self.artifact = self.root / "sample.jsonl"
        self.rows = [
            {"id": "safe-1", "prompt": "Explain why a lock needs a key."},
            {"id": "safe-2", "prompt": "Describe a synthetic tool result."},
        ]
        self._write_rows(self.rows)
        self.manifest = Path(self.temporary.name) / "manifest.json"
        self._write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_rows(self, rows: list[dict]) -> None:
        self.artifact.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _manifest_document(self) -> dict:
        data = self.artifact.read_bytes()
        return {
            "schema": admit.ARTIFACT_SCHEMA,
            "source_id": "phtest",
            "revision": "edb80210812ecc7dab2219aa2e70921710c01888",
            "suite": "false-refusal",
            "artifact_path": self.artifact.name,
            "format": "jsonl",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(self.rows),
            "fields": ["id", "prompt"],
            "inert_text_only": True,
        }

    def _write_manifest(self, update: dict | None = None) -> None:
        document = self._manifest_document()
        document.update(update or {})
        self.manifest.write_text(json.dumps(document), encoding="utf-8")

    def test_tracked_catalog_is_valid_and_has_expected_dispositions(self) -> None:
        sources = admit.load_catalog()
        self.assertEqual(sources["phtest"]["disposition"], "candidate")
        self.assertEqual(sources["agentdojo"]["suites"], ["tool-integrity"])
        self.assertEqual(sources["dobliuw-prompts"]["disposition"], "rejected")
        self.assertEqual(sources["the-big-prompt-library"]["execution_policy"],
                         "never-ingest")

    def test_valid_normalized_jsonl_is_admitted(self) -> None:
        result = admit.validate_artifact(self.manifest, corpus_root=self.root)
        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["execution_policy"], "inert-text-only")
        self.assertEqual(result["source_id"], "phtest")

    def test_unknown_manifest_fields_fail_closed(self) -> None:
        document = self._manifest_document()
        document["surprise"] = "ignored by permissive readers"
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(admit.AdmissionError, "unknown or missing"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_unpinned_revision_fails_closed(self) -> None:
        self._write_manifest({"revision": "0" * 40})
        with self.assertRaisesRegex(admit.AdmissionError, "pinned source revision"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_rejected_source_cannot_be_admitted(self) -> None:
        self._write_manifest({
            "source_id": "dobliuw-prompts",
            "revision": "616d8ab45bb903b04229a90dbca108bce8daacf4",
            "suite": "harmful-refusal",
        })
        with self.assertRaisesRegex(admit.AdmissionError, "not approved"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_reference_source_cannot_be_admitted(self) -> None:
        self._write_manifest({
            "source_id": "offensive-ai-compilation",
            "revision": "48954374ba6cc5a123b5769b98783987e85047cb",
            "suite": "reference-catalog",
        })
        with self.assertRaisesRegex(admit.AdmissionError, "not approved"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_wrong_hash_and_size_fail_closed(self) -> None:
        self._write_manifest({"sha256": "0" * 64})
        with self.assertRaisesRegex(admit.AdmissionError, "SHA-256"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)
        self._write_manifest({"bytes": self.artifact.stat().st_size + 1})
        with self.assertRaisesRegex(admit.AdmissionError, "size"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_wrong_row_count_and_schema_fail_closed(self) -> None:
        self._write_manifest({"rows": 3})
        with self.assertRaisesRegex(admit.AdmissionError, "row mismatch"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)
        self._write_manifest({"fields": ["id", "prompt", "extra"]})
        with self.assertRaisesRegex(admit.AdmissionError, "field schema"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_symlink_parent_escape_and_executable_file_fail_closed(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        external = outside / "sample.jsonl"
        external.write_bytes(self.artifact.read_bytes())
        link = self.root / "escape"
        link.symlink_to(outside, target_is_directory=True)
        self._write_manifest({"artifact_path": "escape/sample.jsonl"})
        with self.assertRaisesRegex(admit.AdmissionError, "outside the corpus root"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

        self._write_manifest()
        self.artifact.chmod(0o755)
        with self.assertRaisesRegex(admit.AdmissionError, "executable"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_parent_traversal_is_rejected_before_resolution(self) -> None:
        self._write_manifest({"artifact_path": "../sample.jsonl"})
        with self.assertRaisesRegex(admit.AdmissionError, "traverse"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_credential_material_and_nul_bytes_fail_closed(self) -> None:
        self.artifact.write_text(
            json.dumps({"id": "unsafe", "prompt": "-----BEGIN OPENSSH PRIVATE KEY-----"}) + "\n",
            encoding="utf-8",
        )
        self.rows = [{"id": "unsafe", "prompt": "marker"}]
        self._write_manifest({"rows": 1})
        with self.assertRaisesRegex(admit.AdmissionError, "credential"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

        self.artifact.write_bytes(b'{"id":"bad","prompt":"x"}\x00\n')
        self.rows = [{"id": "bad", "prompt": "x"}]
        self._write_manifest({"rows": 1})
        with self.assertRaisesRegex(admit.AdmissionError, "NUL"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_non_utf8_and_blank_rows_fail_closed(self) -> None:
        self.artifact.write_bytes(b'{"id":"bad","prompt":"\xff"}\n')
        self.rows = [{"id": "bad", "prompt": "x"}]
        self._write_manifest({"rows": 1})
        with self.assertRaisesRegex(admit.AdmissionError, "UTF-8"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

        self._write_rows(self.rows)
        with self.artifact.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        self._write_manifest({"rows": 1})
        with self.assertRaisesRegex(admit.AdmissionError, "blank"):
            admit.validate_artifact(self.manifest, corpus_root=self.root)

    def test_cli_publishes_atomic_evidence_only_after_admission(self) -> None:
        evidence = Path(self.temporary.name) / "evidence" / "admission.json"
        exit_code = admit.main([
            str(self.manifest), "--corpus-root", str(self.root),
            "--evidence", str(evidence),
        ])
        self.assertEqual(exit_code, 0)
        document = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(document["status"], "admitted")
        self.assertFalse(list(evidence.parent.glob("*.tmp.*")))

        previous = evidence.read_bytes()
        self._write_manifest({"sha256": "0" * 64})
        exit_code = admit.main([
            str(self.manifest), "--corpus-root", str(self.root),
            "--evidence", str(evidence),
        ])
        self.assertEqual(exit_code, 1)
        self.assertEqual(evidence.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
