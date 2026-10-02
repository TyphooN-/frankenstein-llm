#!/usr/bin/env python3
"""Offline physical inventory contracts; no CUDA runtime or NVIDIA hardware proof."""
from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_gpu_vram import three_gpu_host

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import gpu_vram

UUID = "GPU-12345678-1234-5678-9abc-123456789abc"
FIELDS = ["7", UUID, "00000000:0A:00.0", "RTX 3060", "12288", "256",
          "Disabled", "Disabled", "[N/A]"]


def csv_text(*rows):
    stream = io.StringIO()
    csv.writer(stream).writerows(rows)
    return stream.getvalue()


def response(text=None):
    return subprocess.CompletedProcess([], 0, csv_text(FIELDS) if text is None else text, "")


class PhysicalInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.query_run = Mock(return_value=response())

    def inventory(self, environ=None):
        return gpu_vram.physical_inventory(self.root / "kfd", self.root / "drm",
                                           environ={} if environ is None else environ,
                                           run=self.query_run)

    def test_physical_nvidia_identity_and_vram_do_not_claim_a_cuda_index(self):
        inventory = self.inventory()
        self.assertEqual(inventory["schema"], "frankenstein-gpu-physical-inventory/1")
        self.assertEqual(inventory["problems"], [])
        self.assertEqual(len(inventory["devices"]), 1)
        row = inventory["devices"][0]
        self.assertEqual((row["vendor"], row["backend"]), ("nvidia", "CUDA"))
        self.assertEqual(row["raw"]["pci.bus_id"], "00000000:0A:00.0")
        self.assertEqual(row["raw"]["uuid"], UUID)
        self.assertEqual(row["pci"], "0000:0a:00.0")
        self.assertEqual(row["uuid"], UUID)
        self.assertEqual(row["nvidia_smi_index"], 7)
        self.assertEqual(row["total_bytes"], 12288 * gpu_vram.MIB)
        self.assertEqual(row["used_bytes"], 256 * gpu_vram.MIB)
        self.assertIs(row["display_active"], False)
        self.assertIs(row["display_mode"], False)
        self.assertIsNone(row["runtime_index"])
        self.assertIn("unresolved", row["runtime_index_error"])
        self.assertIs(row["placement_usable"], False)
        self.assertEqual(row["problems"], [])
        args, kwargs = self.query_run.call_args
        self.assertEqual(args[0], ["nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,display_active,display_mode,mig.mode.current",
            "--format=csv,noheader,nounits"])
        self.assertEqual(kwargs, {"capture_output": True, "text": True,
                                  "timeout": 10, "check": False})

    def test_unavailable_or_failed_query_reports_inventory_failure_not_absence(self):
        failures = [
            (FileNotFoundError("missing"), "FileNotFoundError"),
            (PermissionError("unreadable tool"), "PermissionError"),
            (subprocess.TimeoutExpired("nvidia-smi", 10, output=csv_text(FIELDS)), "timed out"),
            (UnicodeError("bad output encoding"), "UnicodeError"),
            (subprocess.CompletedProcess([], 9, csv_text(FIELDS), "driver not loaded"), "exited 9"),
        ]
        for failure, reason in failures:
            with self.subTest(reason=reason):
                self.query_run.side_effect = failure if isinstance(failure, Exception) else None
                if not isinstance(failure, Exception):
                    self.query_run.return_value = failure
                inventory = self.inventory()
                self.assertEqual(inventory["devices"], [])
                self.assertTrue(any(reason in p for p in inventory["problems"]))

    def test_unreadable_or_invalid_memory_stays_unknown_without_losing_the_row(self):
        for column, key in ((4, "total_bytes"), (5, "used_bytes")):
            for value in ("[N/A]", "[Not Supported]", "", "-1", "NaN", "inf", "1.5", "bad"):
                with self.subTest(column=column, value=value):
                    fields = FIELDS.copy()
                    fields[column] = value
                    self.query_run.return_value = response(csv_text(fields))
                    row = self.inventory()["devices"][0]
                    self.assertIsNone(row[key])
                    self.assertTrue(any("VRAM" in p for p in row["problems"]))
                    self.assertEqual(row["raw"][gpu_vram.NVIDIA_FIELDS[column]], value)
                    self.assertFalse(row["placement_usable"])
        for total, used in (("0", "0"), ("100", "101")):
            with self.subTest(total=total, used=used):
                fields = FIELDS.copy()
                fields[4:6] = [total, used]
                self.query_run.return_value = response(csv_text(fields))
                row = self.inventory()["devices"][0]
                self.assertIsNone(row["total_bytes"])
                self.assertTrue(any("VRAM" in p for p in row["problems"]))

    def test_invalid_identity_is_preserved_raw_but_not_accepted_as_a_join_key(self):
        cases = [(0, "-1", "nvidia_smi_index"), (0, "seven", "nvidia_smi_index"),
                 (1, "GPU-truncated", "uuid"), (1, "MIG-12345678-1234-5678-9abc-123456789abc", "uuid"),
                 (1, "", "uuid"), (2, "00010000:0a:00.0", "pci"),
                 (2, "0000:0a:20.0", "pci"), (2, "0000:0a:00.8", "pci"),
                 (2, "0a:00.0", "pci"), (2, "nonsense", "pci")]
        for column, value, key in cases:
            with self.subTest(value=value):
                fields = FIELDS.copy()
                fields[column] = value
                self.query_run.return_value = response(csv_text(fields))
                row = self.inventory()["devices"][0]
                self.assertIsNone(row[key])
                self.assertEqual(row["raw"][gpu_vram.NVIDIA_FIELDS[column]], value)
                self.assertTrue(any("identity" in p for p in row["problems"]))

    def test_valid_csv_quoting_and_four_digit_pci_domain_are_supported(self):
        fields = FIELDS.copy()
        fields[2:4] = ["0001:AB:1f.7", "Quadro, quoted name"]
        self.query_run.return_value = response(csv_text(fields))
        row = self.inventory()["devices"][0]
        self.assertEqual(row["pci"], "0001:ab:1f.7")
        self.assertEqual(row["name"], "Quadro, quoted name")
        self.assertEqual(row["problems"], [])

    def test_malformed_csv_keeps_bad_rows_without_shifting_good_rows(self):
        for bad in (FIELDS[:-1], FIELDS + ["unexpected"], ["not CSV"]):
            with self.subTest(bad=bad):
                self.query_run.return_value = response(csv_text(bad, FIELDS))
                rows = self.inventory()["devices"]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["raw_fields"], bad)
                self.assertIsNone(rows[0]["pci"])
                self.assertIsNone(rows[0]["total_bytes"])
                self.assertTrue(any("malformed CSV" in p for p in rows[0]["problems"]))
                self.assertEqual(rows[1]["pci"], "0000:0a:00.0")
                self.assertEqual(rows[1]["nvidia_smi_index"], 7)

    def test_broken_csv_and_empty_success_are_not_silent_empty_inventories(self):
        for text in ('"unterminated', "", "\n\n"):
            with self.subTest(text=text):
                self.query_run.return_value = response(text)
                inventory = self.inventory()
                self.assertEqual(inventory["devices"], [])
                self.assertTrue(inventory["problems"])
        self.query_run.return_value = response(csv_text(FIELDS) + '"unterminated')
        inventory = self.inventory()
        self.assertEqual(len(inventory["devices"]), 1)
        self.assertTrue(any("CSV" in p for p in inventory["problems"]))

    def test_duplicate_physical_identities_invalidate_every_contested_row(self):
        second = FIELDS.copy()
        second[:3] = ["8", "GPU-abcdefab-1234-5678-9abc-123456789abc", "0000:0b:00.0"]
        third = FIELDS.copy()
        third[:3] = ["9", "GPU-99999999-1234-5678-9abc-123456789abc", "0000:0c:00.0"]
        for column, duplicate in ((0, "07"), (1, UUID.upper()), (2, "0000:0a:00.0")):
            with self.subTest(column=column):
                bad = second.copy()
                bad[column] = duplicate
                self.query_run.return_value = response(csv_text(FIELDS, bad, third))
                rows = self.inventory()["devices"]
                self.assertEqual(len(rows), 3)
                for row in rows[:2]:
                    self.assertTrue(any("duplicate physical identity" in p for p in row["problems"]))
                    self.assertIsNone(row["pci"])
                    self.assertIsNone(row["uuid"])
                    self.assertIsNone(row["nvidia_smi_index"])
                self.assertEqual(rows[1]["raw_fields"], bad)
                self.assertEqual(rows[2]["nvidia_smi_index"], 9)
                self.assertEqual(rows[2]["problems"], [])

    def test_display_metadata_is_observed_not_guessed_from_presence(self):
        cases = [("Enabled", True), ("Disabled", False), ("[N/A]", None),
                 ("[Not Supported]", None), ("", None), ("unexpected", None)]
        for column, key in ((6, "display_active"), (7, "display_mode")):
            for value, expected in cases:
                with self.subTest(column=column, value=value):
                    fields = FIELDS.copy()
                    fields[column] = value
                    self.query_run.return_value = response(csv_text(fields))
                    row = self.inventory()["devices"][0]
                    self.assertIs(row[key], expected)
                    self.assertEqual(row["raw"][gpu_vram.NVIDIA_FIELDS[column]], value)
                    if value == "unexpected":
                        self.assertTrue(row["problems"])

    def test_mig_parent_is_never_interpreted_as_a_runtime_device(self):
        for value, mode in (("Enabled", "enabled"), ("Disabled", "disabled"),
                            ("[N/A]", None), ("unexpected", None)):
            with self.subTest(value=value):
                fields = FIELDS.copy()
                fields[8] = value
                self.query_run.return_value = response(csv_text(fields))
                row = self.inventory({"CUDA_VISIBLE_DEVICES": "MIG-arbitrary-instance"})["devices"][0]
                self.assertEqual(row["mig_mode"], mode)
                self.assertIsNone(row["runtime_index"])
                self.assertFalse(row["placement_usable"])
                if mode == "enabled":
                    self.assertIn("MIG", row["runtime_index_error"])
                if value == "unexpected":
                    self.assertTrue(row["problems"])

    def test_mixed_inventory_ignores_runtime_visibility_without_changing_legacy_amd(self):
        host = three_gpu_host(self.root)
        environ = {"ROCR_VISIBLE_DEVICES": "2,0", "HIP_VISIBLE_DEVICES": "1",
                   "CUDA_VISIBLE_DEVICES": "7", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "NVIDIA_VISIBLE_DEVICES": UUID}
        legacy, applied = gpu_vram.devices(host.kfd, host.drm, environ)
        self.assertEqual([(r["rocm_index"], r["card"]) for r in legacy], [(0, "card0")])
        inventory = gpu_vram.physical_inventory(host.kfd, host.drm, environ, run=self.query_run)
        self.assertEqual(inventory["runtime_environment"], environ)
        self.assertFalse(inventory["runtime_visibility_applied"])
        rows = inventory["devices"]
        self.assertEqual([r["vendor"] for r in rows], ["amd", "amd", "amd", "nvidia"])
        self.assertEqual([r["card"] for r in rows[:3]], ["card0", "card1", "card2"])
        self.assertEqual([r["rocm_physical_index"] for r in rows[:3]], [0, 1, 2])
        self.assertEqual(rows[1]["total_bytes"], 32 << 30)
        self.assertIs(rows[2]["display_active"], True)
        for row in rows:
            self.assertNotIn("rocm_index", row)
            self.assertIsNone(row["runtime_index"])
            self.assertTrue(row["runtime_index_error"])
            self.assertFalse(row["placement_usable"])
        self.assertEqual(gpu_vram.devices(host.kfd, host.drm, environ), (legacy, applied))

    def test_empty_and_ambiguous_cuda_visibility_never_filters_or_maps_physical_rows(self):
        for value in ("", "0", "7", "7,7", "GPU-123", UUID, "MIG-unknown", "-1"):
            with self.subTest(value=value):
                environ = {"CUDA_VISIBLE_DEVICES": value, "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
                inventory = self.inventory(environ)
                self.assertEqual(inventory["runtime_environment"], environ)
                self.assertEqual(len(inventory["devices"]), 1)
                row = inventory["devices"][0]
                self.assertIsNone(row["runtime_index"])
                self.assertFalse(row["placement_usable"])
                self.assertEqual(row["nvidia_smi_index"], 7)

    def test_explicit_physical_cli_reports_unresolved_mappings_but_inventory_exit_status(self):
        inventory = self.inventory()
        output = io.StringIO()
        with patch.object(gpu_vram, "physical_inventory", return_value=inventory), redirect_stdout(output):
            code = gpu_vram.main(["--physical-inventory", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), inventory)
        output = io.StringIO()
        with patch.object(gpu_vram, "physical_inventory", return_value=inventory), redirect_stdout(output):
            code = gpu_vram.main(["--physical-inventory"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("nvidia-smi index=7", text)
        self.assertIn("runtime=unresolved", text)
        self.assertIn("placement=unusable", text)
        self.assertNotIn("CUDA7", text)
        for problems in (inventory["problems"], inventory["devices"][0]["problems"]):
            problems.append("fixture inventory failure")
            with patch.object(gpu_vram, "physical_inventory", return_value=inventory), redirect_stdout(io.StringIO()):
                self.assertEqual(gpu_vram.main(["--physical-inventory", "--json"]), 1)
            problems.clear()

    def test_legacy_cli_schema_and_visibility_do_not_invoke_nvidia_inventory(self):
        host = three_gpu_host(self.root)
        rows, applied = gpu_vram.devices(host.kfd, host.drm, {"HIP_VISIBLE_DEVICES": "2"})
        output = io.StringIO()
        with patch.object(gpu_vram, "devices", return_value=(rows, applied)), \
                patch.object(gpu_vram, "physical_inventory") as physical, redirect_stdout(output):
            self.assertEqual(gpu_vram.main(["--json"]), 0)
            physical.assert_not_called()
        data = json.loads(output.getvalue())
        self.assertEqual(data, {"schema": "frankenstein-gpu-vram/1", "devices": rows,
                               "visibility_filters": [{"variable": "HIP_VISIBLE_DEVICES", "value": "2"}]})
        output = io.StringIO()
        with patch.object(gpu_vram, "devices", return_value=(rows, applied)), \
                patch.object(gpu_vram, "physical_inventory") as physical, redirect_stdout(output):
            self.assertEqual(gpu_vram.main([]), 0)
            physical.assert_not_called()
        self.assertIn("ROCm0 (card2", output.getvalue())
        self.assertNotIn("nvidia", output.getvalue())

    def test_nvidia_failure_and_unreadable_amd_memory_keep_physical_amd_rows(self):
        host = three_gpu_host(self.root)
        (host.devices / "0000:07:00.0/mem_info_vram_used").unlink()
        self.query_run.side_effect = FileNotFoundError("no NVIDIA tool")
        inventory = gpu_vram.physical_inventory(host.kfd, host.drm,
                                                {"HIP_VISIBLE_DEVICES": ""}, run=self.query_run)
        rows = inventory["devices"]
        self.assertEqual([r["rocm_physical_index"] for r in rows], [0, 1, 2])
        self.assertIsNone(rows[1]["used_bytes"])
        self.assertTrue(any("unreadable" in p for p in rows[1]["problems"]))
        self.assertEqual(rows[2]["total_bytes"], 16 << 30)
        self.assertTrue(any("FileNotFoundError" in p for p in inventory["problems"]))

    def test_three_way_and_transitive_identity_collisions_are_all_refused(self):
        second = FIELDS.copy()
        second[0] = "8"
        third = FIELDS.copy()
        third[:3] = ["9", "GPU-99999999-1234-5678-9abc-123456789abc", "0000:0c:00.0"]
        for records in ((FIELDS, second, FIELDS), (FIELDS, second, ["8", *third[1:]])):
            with self.subTest(records=records):
                self.query_run.return_value = response(csv_text(*records))
                rows = self.inventory()["devices"]
                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertIsNone(row["pci"])
                    self.assertIsNone(row["uuid"])
                    self.assertTrue(any("duplicate physical identity" in p for p in row["problems"]))

    def test_real_management_subprocess_preserves_csv_padding_and_environment_metadata(self):
        fields = [" 7", " " + UUID.upper(), " 00000000:0A:00.0", *FIELDS[3:]]
        tool = self.root / "nvidia-smi"
        tool.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write({csv_text(fields)!r})\n")
        tool.chmod(0o755)
        environ = {"PATH": str(self.root), "CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        with patch.dict(os.environ, environ, clear=True):
            inventory = gpu_vram.physical_inventory(self.root / "kfd", self.root / "drm")
        row = inventory["devices"][0]
        self.assertEqual(row["raw_fields"], fields)
        self.assertEqual(row["uuid"], UUID)
        self.assertEqual(row["pci"], "0000:0a:00.0")
        self.assertEqual(inventory["runtime_environment"], {k: v for k, v in environ.items() if k != "PATH"})
        self.assertIsNone(row["runtime_index"])
        self.assertFalse(row["placement_usable"])
        self.assertEqual(inventory["problems"], [])


if __name__ == "__main__":
    unittest.main()
