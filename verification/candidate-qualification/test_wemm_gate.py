#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate_wemm", HERE / "gate_wemm.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WeMMGateAdmissionTests(unittest.TestCase):
    def test_refuses_to_import_when_remote_code_is_not_approved(self):
        report = {"pass": False, "problems": ["changed bytes"], "approved_for_import": []}
        with mock.patch.object(module.review, "scan", return_value=report), \
             mock.patch.object(module.review, "execution_permitted", return_value=False), \
             mock.patch.object(module.policy, "index_conflicts", return_value=[]):
            result = module.run()
        self.assertFalse(result["pass"])
        self.assertIn("remote-code review refused execution", result["problems"])
        self.assertNotIn("text_dimension", result)

    def test_refuses_a_merged_vector_index_before_loading(self):
        with mock.patch.object(module.policy, "index_conflicts",
                               return_value=["multimodal and text embeddings share one database file"]):
            result = module.run()
        self.assertFalse(result["pass"])
        self.assertTrue(any("share one database" in item for item in result["problems"]))
