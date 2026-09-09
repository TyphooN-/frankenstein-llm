from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import asr_roundtrip as rt
import gate_tts


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.wav = Path(self.temp.name) / "generated.wav"
        self.wav.write_bytes(b"RIFF" + b"\0" * 128)

    def tearDown(self):
        self.temp.cleanup()

    def raw_pairs(self, text="The quick brown fox jumps over the lazy dog."):
        return json.dumps([{"name": "short", "path": str(self.wav), "text": text}])

    def test_parse_pairs_rejects_malformed_and_missing_files(self):
        for raw in ("{}", "[]", '[{"name":"x"}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                rt.parse_pairs(raw)
        missing = json.dumps([{"name": "x", "path": str(self.wav) + ".missing", "text": "x"}])
        with self.assertRaises(ValueError):
            rt.parse_pairs(missing)

    def test_evaluate_pairs_passes_intelligible_transcript(self):
        pairs = rt.parse_pairs(self.raw_pairs())
        result = rt.evaluate_pairs(
            pairs, lambda _path: ("The quick brown fox jumps over the lazy dog", "English")
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["cases"][0]["pass"])
        self.assertGreaterEqual(result["cases"][0]["similarity"], rt.SIMILARITY_THRESHOLD)
        self.assertFalse(result["benchmark_performed"])

    def test_evaluate_pairs_fails_low_similarity(self):
        pairs = rt.parse_pairs(self.raw_pairs())
        result = rt.evaluate_pairs(pairs, lambda _path: ("completely unrelated words", "English"))
        self.assertFalse(result["pass"])
        self.assertFalse(result["cases"][0]["pass"])

    def test_aggregate_requires_every_case(self):
        second = Path(self.temp.name) / "second.wav"
        second.write_bytes(b"RIFF" + b"\0" * 128)
        raw = json.dumps([
            {"name": "one", "path": str(self.wav), "text": "alpha beta gamma"},
            {"name": "two", "path": str(second), "text": "delta epsilon zeta"},
        ])
        answers = iter([("alpha beta gamma", "English"), ("wrong", "English")])
        result = rt.evaluate_pairs(rt.parse_pairs(raw), lambda _path: next(answers))
        self.assertEqual([True, False], [case["pass"] for case in result["cases"]])
        self.assertFalse(result["pass"])

    def test_main_fails_closed_on_model_error_and_runs_cleanup(self):
        cleaned = []

        def broken(_path):
            raise RuntimeError("mock ASR failure")

        with mock.patch.object(rt, "build_local_transcriber", return_value=(broken, lambda: cleaned.append(True))):
            with mock.patch("builtins.print") as printed:
                rc = rt.main(["--pairs", self.raw_pairs()])
        self.assertEqual(1, rc)
        self.assertEqual([True], cleaned)
        payload = json.loads(printed.call_args.args[0])
        self.assertFalse(payload["pass"])
        self.assertIn("mock ASR failure", payload["error"])


class AdmissionRuleTests(unittest.TestCase):
    """A section that never ran must not be cheaper to satisfy than one that failed."""

    def passing_summary(self, **overrides):
        summary = {key: {"pass": True} for key in gate_tts.SECTION_KEYS}
        summary.update(overrides)
        return summary

    def test_all_sections_passing_admits(self):
        result = gate_tts.finalize(self.passing_summary())
        self.assertTrue(result["pass"])
        self.assertEqual([], result["problems"])
        self.assertTrue(all(result["section_verdicts"].values()))

    def test_missing_section_is_not_a_pass(self):
        summary = self.passing_summary()
        del summary["intelligibility"]
        result = gate_tts.finalize(summary)
        self.assertFalse(result["pass"])
        self.assertFalse(result["section_verdicts"]["intelligibility"])
        self.assertIn("section intelligibility was never recorded", result["problems"])

    def test_skipped_intelligibility_is_not_a_pass(self):
        summary = self.passing_summary(
            intelligibility={"pass": False, "skipped": True, "reason": "--skip-asr"})
        result = gate_tts.finalize(summary)
        self.assertFalse(result["pass"])
        self.assertIn("section intelligibility did not pass", result["problems"])

    def test_failing_section_is_reported_once(self):
        summary = self.passing_summary(unload={"pass": False})
        summary["problems"] = ["section unload did not pass"]
        result = gate_tts.finalize(summary)
        self.assertEqual(["section unload did not pass"], result["problems"])

    def test_note_deduplicates(self):
        problems = []
        gate_tts.note(problems, "same")
        gate_tts.note(problems, "same")
        gate_tts.note(problems, "other")
        self.assertEqual(["same", "other"], problems)

    def test_every_section_key_is_checked(self):
        for key in gate_tts.SECTION_KEYS:
            summary = self.passing_summary(**{key: {"pass": False}})
            with self.subTest(section=key):
                self.assertFalse(gate_tts.finalize(summary)["pass"])


class TextNormalisationTests(unittest.TestCase):
    def test_similarity_ignores_case_and_punctuation(self):
        self.assertEqual(1.0, gate_tts.similarity("Hello, world!", "hello world"))

    def test_similarity_separates_unrelated_text(self):
        self.assertLess(gate_tts.similarity("the quick brown fox", "zzz qqq"), 0.5)


if __name__ == "__main__":
    unittest.main()
