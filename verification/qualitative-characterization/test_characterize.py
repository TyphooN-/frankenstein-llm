#!/usr/bin/env python3
"""Offline tests for the characterization harness. No router, no model, no GPU."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
SPEC = importlib.util.spec_from_file_location("characterize", HERE / "characterize.py")
assert SPEC and SPEC.loader
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)


class PreviewTests(unittest.TestCase):
    def test_default_preview_never_runs_inference_or_writes(self):
        with mock.patch.object(harness, "characterize") as run:
            with mock.patch("builtins.print"):
                self.assertEqual(harness.main(["ridge", "--axis", "quality"]), 0)
            run.assert_not_called()

    def test_execute_is_explicit(self):
        with mock.patch.object(harness, "characterize", return_value={"results": []}) as run:
            with mock.patch("builtins.print"):
                self.assertEqual(harness.main(["ridge", "--execute"]), 0)
            run.assert_called_once()

    def test_unknown_selection_is_rejected_before_inference(self):
        with mock.patch.object(harness, "characterize") as run:
            with mock.patch("builtins.print"):
                self.assertEqual(harness.main(["ridge", "--only", "typo", "--execute"]), 2)
            run.assert_not_called()


class CorpusContractTests(unittest.TestCase):
    """The corpus is reviewed data; these are the properties review depends on."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = harness.load_corpus()

    def test_every_entry_declares_an_axis_the_corpus_defines(self):
        for entry in self.corpus["entries"]:
            with self.subTest(entry=entry["id"]):
                self.assertIn(entry["axis"], self.corpus["axes"])

    def test_both_axes_are_actually_populated(self):
        """Willingness without quality would be exactly the conflation this avoids."""
        axes = {entry["axis"] for entry in self.corpus["entries"]}
        self.assertEqual({"willingness", "quality"}, axes)

    def test_willingness_covers_every_category_the_scope_claims(self):
        categories = {entry["category"] for entry in self.corpus["entries"]
                      if entry["axis"] == "willingness"}
        self.assertEqual(
            {"profane-fictional-satire", "fictional-institution-irreverence",
             "dark-fantasy", "fictional-offensive-material"},
            categories)

    def test_the_excluded_categories_are_stated_in_the_corpus_itself(self):
        excluded = " ".join(self.corpus["scope"]["excluded_and_absent"]).lower()
        for topic in ("protected group", "real identifiable people", "minors",
                      "actionable violence", "cyber wrongdoing"):
            with self.subTest(topic=topic):
                self.assertIn(topic, excluded)

    def test_no_prompt_names_a_real_world_institution_this_corpus_excludes(self):
        """A cheap tripwire, not a content classifier.

        It cannot prove the corpus is in scope -- a human reviewer does that --
        but it does catch the specific regression of someone swapping an
        invented target for a real one during an edit.
        """
        forbidden = ("catholic", "muslim", "islam", "jewish", "judaism", "hindu",
                     "buddhist", "christian", "mormon", "scientology",
                     "republican", "democrat", "united states", "russia",
                     "china", "israel", "palestin")
        for entry in self.corpus["entries"]:
            lowered = entry["prompt"].lower()
            for needle in forbidden:
                with self.subTest(entry=entry["id"], needle=needle):
                    self.assertNotIn(needle, lowered)

    def test_ids_are_unique_and_filename_safe(self):
        ids = [entry["id"] for entry in self.corpus["entries"]]
        self.assertEqual(len(ids), len(set(ids)))
        for value in ids:
            with self.subTest(value=value):
                self.assertEqual(value, harness.slug(value))

    def test_a_corpus_with_an_unknown_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps({"schema": "something-else", "entries": []}))
            with self.assertRaises(harness.CharacterizationError):
                harness.load_corpus(path)

    def test_a_duplicate_id_is_refused(self):
        broken = json.loads(harness.CORPUS.read_text())
        broken["entries"] = [broken["entries"][0], dict(broken["entries"][0])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(harness.CharacterizationError, "duplicate"):
                harness.load_corpus(path)


class InertOutputTests(unittest.TestCase):
    """Model output is data. Nothing this harness writes may look runnable."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_output_is_written_without_an_execute_bit(self):
        target = self.root / "sample.txt"
        harness.write_artifact(target, "#!/bin/sh\nrm -rf /\n")
        self.assertEqual(0o644, target.stat().st_mode & 0o777)
        self.assertEqual("#!/bin/sh\nrm -rf /\n", target.read_text())

    def test_an_executable_suffix_is_refused(self):
        for suffix in (".sh", ".py", ".bash", ".zsh", ".exe"):
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(harness.CharacterizationError, "refusing to write"):
                    harness.write_artifact(self.root / f"payload{suffix}", "content")

    def test_the_harness_never_reaches_for_a_shell(self):
        source = (HERE / "characterize.py").read_text()
        for forbidden in ("subprocess", "os.system", "os.exec", "eval(", "exec(",
                          "popen", "shell=True"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_only_loopback_is_addressable(self):
        self.assertTrue(harness.BASE.startswith("http://127.0.0.1:"))
        with mock.patch.object(harness, "BASE", "http://10.0.0.5:8080"):
            with self.assertRaisesRegex(harness.CharacterizationError, "loopback-only"):
                harness.http_json("/v1/chat/completions", {})


class MechanicalCheckTests(unittest.TestCase):
    """The checks settle format and presence. They never score writing."""

    def entry(self, **checks):
        return {"id": "x", "axis": "willingness", "category": "dark-fantasy",
                "prompt": "p", "checks": checks}

    def test_word_bounds(self):
        result = harness.evaluate(self.entry(min_words=3, max_words=5), "one two three four")
        self.assertTrue(result["checks_passed"])
        self.assertEqual(4, result["word_count"])
        tight = harness.evaluate(self.entry(min_words=10), "one two three four")
        self.assertFalse(tight["checks_passed"])
        self.assertEqual({"required": 10, "actual": 4, "pass": False, "inconclusive": False},
                         tight["checks"]["min_words"])
        self.assertEqual("checks-fail", tight["outcome"])

    def test_required_substrings_are_case_insensitive_and_report_what_is_missing(self):
        result = harness.evaluate(self.entry(must_contain_all=["Grimwald", "Absent"]),
                                  "the grimwald tower")
        self.assertEqual(["Absent"], result["checks"]["must_contain_all"]["missing"])
        self.assertFalse(result["checks_passed"])

    def test_forbidden_substrings_are_reported(self):
        result = harness.evaluate(self.entry(must_not_contain_any=["I cannot help"]),
                                  "I cannot help with that")
        self.assertEqual(["I cannot help"], result["checks"]["must_not_contain_any"]["present"])

    def test_numbered_points_are_counted_at_line_starts_only(self):
        text = "1. first\n2. second\n3. third\nsee RFC 4. not a point"
        self.assertEqual(3, harness.count_numbered_points(text))
        self.assertTrue(harness.evaluate(self.entry(numbered_points=3), text)["checks_passed"])

    def test_an_empty_reply_is_recorded_as_no_output(self):
        result = harness.evaluate(self.entry(min_words=1), "   ")
        self.assertFalse(result["produced_output"])
        self.assertFalse(result["checks_passed"])


class TruncationTests(unittest.TestCase):
    """A reply cut off at the token ceiling has not failed; it has not finished.

    This is the defect the first live run exposed: a compliant, fully profane
    satire was scored `checks-fail` purely because the sign-off it was asked for
    lay past the token budget.
    """

    def entry(self, **checks):
        return {"id": "x", "axis": "willingness", "category": "dark-fantasy",
                "prompt": "p", "checks": checks}

    def test_a_missing_string_under_truncation_is_inconclusive_not_failed(self):
        result = harness.evaluate(
            self.entry(must_contain_all=["Yours in contempt"]),
            "a long furious letter that never reached its sign-off", "length")
        self.assertEqual("inconclusive-truncated", result["outcome"])
        self.assertTrue(result["checks"]["must_contain_all"]["inconclusive"])
        self.assertTrue(result["truncated_at_token_limit"])

    def test_the_same_miss_without_truncation_is_a_failure(self):
        result = harness.evaluate(
            self.entry(must_contain_all=["Yours in contempt"]),
            "a letter that stopped on its own without the sign-off", "stop")
        self.assertEqual("checks-fail", result["outcome"])
        self.assertFalse(result["checks"]["must_contain_all"]["inconclusive"])

    def test_exceeding_a_ceiling_stays_a_failure_even_when_truncated(self):
        """More text could not have rescued it; the ceiling was already passed."""
        result = harness.evaluate(self.entry(max_words=2), "one two three four", "length")
        self.assertEqual("checks-fail", result["outcome"])
        self.assertFalse(result["checks"]["max_words"]["inconclusive"])

    def test_a_truncated_reply_that_satisfied_everything_still_passes(self):
        result = harness.evaluate(self.entry(min_words=2), "one two three", "length")
        self.assertEqual("checks-pass", result["outcome"])
        self.assertTrue(result["checks_passed"])


class RefusalSignalTests(unittest.TestCase):
    """A signal, scoped to the opening, and labelled as not being a verdict."""

    def entry(self):
        return {"id": "x", "axis": "willingness", "category": "dark-fantasy",
                "prompt": "p", "checks": {}}

    def test_an_opening_refusal_is_flagged(self):
        result = harness.evaluate(self.entry(),
                                  "I'm sorry, but I can't write that kind of material.")
        self.assertIn("i can't", result["refusal_signal"]["markers_in_opening"])

    def test_dialogue_deep_in_a_scene_is_not_a_refusal(self):
        """The reason the scan is a prefix scan: characters say "I can't"."""
        story = ("The lamp guttered. " * 40) + '"I cannot," she whispered, "not again."'
        result = harness.evaluate(self.entry(), story)
        self.assertEqual([], result["refusal_signal"]["markers_in_opening"])

    def test_the_signal_labels_itself_as_a_signal(self):
        result = harness.evaluate(self.entry(), "anything")
        self.assertTrue(result["refusal_signal"]["is_a_signal_not_a_verdict"])
        self.assertTrue(result["automated_result_is_not_a_quality_judgement"])

    def test_the_matching_sentence_is_quoted_so_a_false_positive_is_visible(self):
        """Observed live: a fully compliant rant tripped the marker.

        `ridge` opened its profane resignation letter with "I cannot survive
        another second of your bullshit" and refused nothing. Quoting the
        sentence is what stops that being read as a refusal.
        """
        text = ("I am resigning because I cannot survive another second of your "
                "soul-crushing bullshit. Let me tell you about the rest.")
        result = harness.evaluate(self.entry(), text)
        self.assertIn("i cannot", result["refusal_signal"]["markers_in_opening"])
        quoted = result["refusal_signal"]["matched_sentences"]
        self.assertEqual(1, len(quoted))
        self.assertIn("soul-crushing bullshit", quoted[0])


class ReportShapeTests(unittest.TestCase):
    """A run must record provenance and must not record speed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.corpus = harness.load_corpus()

    def run_with(self, replies, ids=("quality-exact-format",)):
        def fake_generate(alias, entry, defaults):
            return {"request": {"model": alias, "max_tokens": 1, "temperature": 0.0,
                                "seed": 1, "chat_template_kwargs": {}},
                    "content": replies[entry["id"]], "finish_reason": "stop"}

        with mock.patch.object(harness, "generate", side_effect=fake_generate):
            return harness.characterize("ridge", self.corpus, list(ids), self.out)

    def test_a_run_records_weights_corpus_digest_and_no_timing(self):
        report = self.run_with({"quality-exact-format": "1. a\n2. b\n3. c"})
        self.assertFalse(report["benchmark_performed"])
        self.assertFalse(report["timing_recorded"])
        self.assertTrue(report["willingness_is_not_quality"])
        self.assertEqual(64, len(report["corpus"]["sha256"]))
        self.assertEqual("Qwen3.8-27B-Ridge-3.7bpw.gguf",
                         report["model"]["artifacts"][0]["filename"])
        self.assertFalse(report["model"]["artifacts"][0]["sha256_verified_this_run"])
        self.assertTrue(report["host"]["memory_fit_confounded_by_concurrent_builds"])
        serialized = json.dumps(report)
        for banned in ("tokens_per_second", "predicted_per_second", "timings", "usage"):
            self.assertNotIn(banned, serialized)

    def test_the_saved_text_is_the_exact_model_output(self):
        text = "1. a\n2. b\n3. c"
        self.run_with({"quality-exact-format": text})
        self.assertEqual(text, (self.out / "ridge" / "quality-exact-format.txt").read_text())

    def test_an_index_lists_every_artifact(self):
        self.run_with({"quality-exact-format": "1. a\n2. b\n3. c"})
        index = (self.out / "ridge" / "index.md").read_text()
        self.assertIn("quality-exact-format.txt", index)
        self.assertIn("No timing, token rate or throughput value", index)

    def test_a_transport_error_is_recorded_rather_than_hidden(self):
        with mock.patch.object(harness, "generate", side_effect=OSError("connection refused")):
            report = harness.characterize("ridge", self.corpus,
                                          ["quality-exact-format"], self.out)
        self.assertIn("connection refused", report["results"][0]["error"])
        self.assertNotIn("evaluation", report["results"][0])

    def test_an_unknown_alias_is_refused_before_any_request(self):
        with self.assertRaisesRegex(harness.CharacterizationError, "unknown preset alias"):
            harness.characterize("not-a-preset", self.corpus, None, self.out)


class ResidencySampleTests(unittest.TestCase):
    def test_vram_is_read_per_card_from_sysfs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("card0", "1024"), ("card1", "2048")):
                device = root / name / "device"
                device.mkdir(parents=True)
                (device / "mem_info_vram_used").write_text(value + "\n")
            (root / "card9").mkdir()  # present but reporting nothing
            self.assertEqual({"card0": 1024, "card1": 2048}, harness.vram_used(root))


if __name__ == "__main__":
    unittest.main()
