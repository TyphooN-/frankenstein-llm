#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[2] / "scripts" / "local_model_status.py"
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
        self.assertIn("heretic: unloaded [text -> text]", rendered)
        self.assertNotIn("--model", rendered)


if __name__ == "__main__":
    unittest.main()
