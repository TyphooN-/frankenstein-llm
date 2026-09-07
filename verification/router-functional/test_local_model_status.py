#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MODULE = SCRIPTS / "local_model_status.py"
# Running the script puts scripts/ on sys.path; loading it out of band has to
# arrange the same thing, or its shared display module is not importable.
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("local_model_status", MODULE)
assert SPEC is not None and SPEC.loader is not None
status = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = status
SPEC.loader.exec_module(status)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        return json.dumps(self.payload).encode()


class LocalModelStatusTests(unittest.TestCase):
    def test_collects_bounded_summary_without_model_arguments(self):
        replies = iter([
            FakeResponse({"status": "ok"}),
            FakeResponse({
                "data": [
                    {
                        "id": "zeta",
                        "status": {"value": "unloaded", "args": ["--secret-like-noise"]},
                        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    },
                    {
                        "id": "alpha",
                        "status": {"value": "loaded"},
                        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                    },
                ]
            }),
        ])
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)) as request:
            result = status.collect_status(2.0)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["endpoint"], "http://127.0.0.1:8080")
        self.assertEqual(result["model_count"], 2)
        self.assertEqual(result["loaded_models"], ["alpha"])
        self.assertEqual([model["id"] for model in result["models"]], ["alpha", "zeta"])
        self.assertNotIn("args", result["models"][1])
        self.assertFalse(result["loads_models"])

    def test_unhealthy_router_fails_before_models_request(self):
        with patch.object(status.urllib.request, "urlopen", return_value=FakeResponse({"status": "loading"})) as request:
            with self.assertRaisesRegex(status.StatusError, "unhealthy"):
                status.collect_status()
        self.assertEqual(request.call_count, 1)

    def test_malformed_model_payload_fails_closed(self):
        replies = iter([FakeResponse({"status": "ok"}), FakeResponse({"data": [{}]})])
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            with self.assertRaisesRegex(status.StatusError, "malformed model entry"):
                status.collect_status()

    def test_timeout_is_strictly_bounded(self):
        for value in (0, -1, status.MAX_TIMEOUT + 0.1):
            with self.subTest(value=value), self.assertRaises(status.StatusError):
                status.fetch_json("/health", value)

    def test_response_body_is_bounded_before_json_decoding(self):
        class OversizedResponse(FakeResponse):
            def read(self, size=-1):
                self.requested_size = size
                return b"x" * size

        response = OversizedResponse(None)
        with patch.object(status.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(status.StatusError, "response exceeds"):
                status.fetch_json("/models")
        self.assertEqual(response.requested_size, status.MAX_RESPONSE_BYTES + 1)

    def _models_reply(self, *entries):
        return iter([
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [
                {
                    "id": name,
                    "status": {"value": value},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                }
                for name, value in entries
            ]}),
        ])

    def test_only_states_that_hold_a_child_process_count_as_resident(self):
        # llama.cpp v0.4.0 answers /models with six states. "sleeping" is a live
        # child that idled and "loading" is one starting up: both occupy the
        # router's single resident slot. "downloaded" only means weights reached
        # local disk, and the pre-v0.4.0 rule -- resident is anything except
        # "unloaded" -- reported that as a loaded model.
        replies = self._models_reply(
            ("cached", "downloaded"),
            ("fetching", "downloading"),
            ("idle", "sleeping"),
            ("starting", "loading"),
            ("live", "loaded"),
            ("cold", "unloaded"),
        )
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            result = status.collect_status()
        self.assertEqual(["idle", "live", "starting"], result["loaded_models"])
        self.assertEqual(6, result["model_count"])

    def test_status_vocabulary_is_partitioned_and_matches_the_pinned_runtime(self):
        self.assertEqual(frozenset(), status.RESIDENT_STATUSES & status.VACANT_STATUSES)
        self.assertEqual(
            {"downloading", "downloaded", "unloaded", "loading", "loaded", "sleeping"},
            set(status.KNOWN_STATUSES),
        )

    def test_status_outside_the_pinned_vocabulary_fails_closed(self):
        # A state this pin does not define could sit on either side of the
        # resident boundary, so guessing would silently misreport the slot.
        replies = self._models_reply(("mystery", "hibernating"))
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            with self.assertRaisesRegex(status.StatusError, "unknown status"):
                status.collect_status()

    def test_text_render_does_not_dump_router_launch_arguments(self):
        summary = {
            "endpoint": status.BASE_URL,
            "health": "ok",
            "model_count": 1,
            "status_counts": {"unloaded": 1},
            "loaded_models": [],
            "models": [{
                "id": "heretic",
                "status": "unloaded",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            }],
        }
        rendered = status.render_text(summary)
        self.assertIn("unloaded [text -> text]", rendered)
        self.assertIn("[compatibility alias: heretic]", rendered)
        self.assertNotIn("--model", rendered)

    def test_a_router_id_is_resolved_to_the_weight_file_it_loads(self):
        # "heretic" is the alias; RVN-Q6_K-multilingual-mtp.gguf is the file.
        replies = self._models_reply(("heretic", "loaded"), ("ridge", "unloaded"))
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            result = status.collect_status()
        by_id = {model["id"]: model for model in result["models"]}
        self.assertEqual(
            "RVN-Q6_K-multilingual-mtp.gguf (General-purpose Hermes agent and tool use)",
            by_id["heretic"]["description"])
        self.assertEqual(
            ["RVN-Q6_K-multilingual-mtp.gguf (General-purpose Hermes agent and tool use)"
             "  [compatibility alias: heretic]"],
            result["loaded_descriptions"])
        rendered = status.render_text(result)
        # Identity leads every line; the router's own key trails it, labelled.
        self.assertIn("  RVN-Q6_K-multilingual-mtp.gguf (General-purpose", rendered)
        self.assertIn("Loaded: RVN-Q6_K-multilingual-mtp.gguf", rendered)
        for line in rendered.splitlines():
            self.assertFalse(line.strip().startswith("heretic:"), line)

    def test_an_id_this_checkout_does_not_configure_is_said_so_not_hidden(self):
        replies = self._models_reply(("not-a-preset", "unloaded"))
        with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
            result = status.collect_status()
        self.assertEqual("not-a-preset (not a configured preset)",
                         result["models"][0]["description"])

    def test_unreadable_local_configuration_still_reports_the_router(self):
        """A status read describes a live service; it is not a config gate."""
        with patch.object(status, "local_registry", return_value=({}, {})):
            replies = self._models_reply(("heretic", "loaded"))
            with patch.object(status.urllib.request, "urlopen", side_effect=lambda *_a, **_k: next(replies)):
                result = status.collect_status()
        self.assertEqual(["heretic"], result["loaded_models"])
        self.assertEqual("heretic (not a configured preset)",
                         result["models"][0]["description"])


if __name__ == "__main__":
    unittest.main()
