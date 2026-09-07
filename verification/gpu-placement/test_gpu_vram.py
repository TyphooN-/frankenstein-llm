#!/usr/bin/env python3
"""Identity mapping for the VRAM report.

Every test here builds a synthetic sysfs tree rather than reading the host, so
the cases that matter -- a DRM card that is not a KFD GPU, a visibility variable
in the environment, a device that stopped answering -- can be exercised on a
machine where none of them are true.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("gpu_vram", SCRIPTS / "gpu_vram.py")
assert SPEC is not None and SPEC.loader is not None
gpu_vram = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gpu_vram
SPEC.loader.exec_module(gpu_vram)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class Host:
    """A synthetic ``/sys`` holding just what the mapping reads."""

    def __init__(self, root: Path):
        self.root = root
        self.kfd = root / "class/kfd/kfd/topology/nodes"
        self.drm = root / "class/drm"
        self.devices = root / "devices/pci0000:00"
        self.kfd.mkdir(parents=True, exist_ok=True)
        self.drm.mkdir(parents=True, exist_ok=True)
        write(self.kfd / "0/properties", "cpu_cores_count 44\nsimd_count 0\n")

    def add_kfd_node(self, node: int, *, bus: int, render_minor: int,
                     unique_id: int, simd_count: int = 160,
                     domain: int = 0, name: str = "sienna_cichlid") -> None:
        location_id = bus << 8
        write(self.kfd / f"{node}/properties",
              f"simd_count {simd_count}\n"
              f"location_id {location_id}\n"
              f"domain {domain}\n"
              f"drm_render_minor {render_minor}\n"
              f"unique_id {unique_id}\n"
              "gfx_target_version 100300\n")
        write(self.kfd / f"{node}/name", f"{name}\n")

    def add_card(self, card: str, *, bus: int, render_minor: int | None = None,
                 used: int | None = 0, total: int | None = 0,
                 connected: tuple[str, ...] = (), domain: int = 0) -> None:
        address = f"{domain:04x}:{bus:02x}:00.0"
        device = self.devices / address
        device.mkdir(parents=True, exist_ok=True)
        if used is not None and total is not None:
            write(device / "mem_info_vram_used", f"{used}\n")
            write(device / "mem_info_vram_total", f"{total}\n")
        for node in (card, *(() if render_minor is None else (f"renderD{render_minor}",))):
            (self.drm / node).mkdir(parents=True, exist_ok=True)
            link = self.drm / node / "device"
            if not link.exists():
                link.symlink_to(device)
        for index, connector in enumerate(connected):
            write(self.drm / f"{card}-{connector}/status", "connected\n")
            write(self.drm / f"{card}-spare{index}/status", "disconnected\n")

    def devices_report(self, environ=None):
        return gpu_vram.devices(self.kfd, self.drm, environ or {})


def three_gpu_host(root: Path) -> Host:
    """The shape of this workspace: two 16 GiB cards and a 32 GiB card."""
    host = Host(root)
    host.add_kfd_node(1, bus=0x03, render_minor=128, unique_id=0x6b3c785d187d9cbf)
    host.add_kfd_node(2, bus=0x07, render_minor=129, unique_id=0xa21e268c0b0a73d7,
                      simd_count=144)
    host.add_kfd_node(3, bus=0x0A, render_minor=130, unique_id=0x4d1c68aee3a84b64)
    host.add_card("card0", bus=0x03, render_minor=128, used=1 << 30, total=16 << 30)
    host.add_card("card1", bus=0x07, render_minor=129, used=2 << 30, total=32 << 30)
    host.add_card("card2", bus=0x0A, render_minor=130, used=3 << 30, total=16 << 30,
                  connected=("DP-4", "DP-5"))
    return host


class AddressTests(unittest.TestCase):
    def test_location_id_unpacks_to_the_sysfs_address_form(self):
        # KFD packs (bus << 8) | devfn, which is how these three hosts' cards
        # appear in /sys/bus/pci/devices.
        self.assertEqual(gpu_vram.pci_address(0, 768), "0000:03:00.0")
        self.assertEqual(gpu_vram.pci_address(0, 1792), "0000:07:00.0")
        self.assertEqual(gpu_vram.pci_address(0, 2560), "0000:0a:00.0")

    def test_device_and_function_bits_are_not_dropped(self):
        self.assertEqual(gpu_vram.pci_address(1, (0x81 << 8) | (0x03 << 3) | 2),
                         "0001:81:03.2")


class IdentityTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_rows_are_rocm_indices_joined_to_the_matching_card(self):
        rows, applied = three_gpu_host(self.root).devices_report()
        self.assertEqual(applied, [])
        self.assertEqual([row["rocm_index"] for row in rows], [0, 1, 2])
        self.assertEqual([row["card"] for row in rows], ["card0", "card1", "card2"])
        self.assertEqual([row["uuid"] for row in rows],
                         ["GPU-6b3c785d187d9cbf", "GPU-a21e268c0b0a73d7",
                          "GPU-4d1c68aee3a84b64"])
        self.assertEqual([row["problems"] for row in rows], [[], [], []])

    def test_cpu_node_never_consumes_a_rocm_index(self):
        # Node 0 has simd_count 0. Counting nodes instead of GPUs would label
        # the first GPU ROCm1 and shift every device behind it.
        rows, _ = three_gpu_host(self.root).devices_report()
        self.assertEqual(rows[0]["kfd_node"], 1)
        self.assertEqual(rows[0]["rocm_index"], 0)

    def test_a_drm_card_without_a_kfd_node_does_not_shift_indices(self):
        """The bug this module replaced: counting sysfs cards, not ROCm devices.

        A display adapter with no KFD node -- a second-vendor card, or an
        amdgpu device that failed to bind KFD -- occupies a low card number and
        no ROCm index. Enumerating cards would have labelled it ROCm0 and named
        every real device one index too high.
        """
        host = Host(self.root)
        host.add_card("card0", bus=0x01, used=1 << 20, total=256 << 20)
        host.add_kfd_node(1, bus=0x03, render_minor=128, unique_id=0x1111)
        host.add_kfd_node(2, bus=0x0A, render_minor=129, unique_id=0x2222)
        host.add_card("card1", bus=0x03, render_minor=128, used=1 << 30, total=16 << 30)
        host.add_card("card2", bus=0x0A, render_minor=129, used=3 << 30, total=16 << 30)
        rows, _ = host.devices_report()
        self.assertEqual([(row["rocm_index"], row["card"]) for row in rows],
                         [(0, "card1"), (1, "card2")])
        self.assertEqual(rows[1]["total_bytes"], 16 << 30)

    def test_display_role_comes_from_connected_outputs(self):
        rows, _ = three_gpu_host(self.root).devices_report()
        self.assertEqual(rows[0]["display_connectors"], [])
        self.assertEqual(rows[1]["display_connectors"], [])
        self.assertEqual(rows[2]["display_connectors"], ["card2-DP-4", "card2-DP-5"])


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.host = three_gpu_host(self.root)

    def test_rocr_visible_devices_reorders_the_indices(self):
        rows, applied = self.host.devices_report({"ROCR_VISIBLE_DEVICES": "2,0"})
        self.assertEqual([(row["rocm_index"], row["card"]) for row in rows],
                         [(0, "card2"), (1, "card0")])
        self.assertEqual(applied, [("ROCR_VISIBLE_DEVICES", "2,0")])

    def test_hip_filter_applies_on_top_of_the_rocr_filter(self):
        # ROCr hides card0 first; HIP index 1 then means the second survivor.
        rows, applied = self.host.devices_report(
            {"ROCR_VISIBLE_DEVICES": "1,2", "HIP_VISIBLE_DEVICES": "1"})
        self.assertEqual([(row["rocm_index"], row["card"]) for row in rows],
                         [(0, "card2")])
        self.assertEqual([name for name, _ in applied],
                         ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"])

    def test_cuda_spelling_is_only_a_fallback(self):
        rows, applied = self.host.devices_report(
            {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "2"})
        self.assertEqual([row["card"] for row in rows], ["card0"])
        self.assertEqual([name for name, _ in applied], ["HIP_VISIBLE_DEVICES"])
        rows, applied = self.host.devices_report({"CUDA_VISIBLE_DEVICES": "2"})
        self.assertEqual([row["card"] for row in rows], ["card2"])
        self.assertEqual([name for name, _ in applied], ["CUDA_VISIBLE_DEVICES"])

    def test_uuid_entries_select_the_same_device_as_indices(self):
        rows, _ = self.host.devices_report(
            {"HIP_VISIBLE_DEVICES": "GPU-4d1c68aee3a84b64"})
        self.assertEqual([row["card"] for row in rows], ["card2"])

    def test_list_truncates_at_the_first_unusable_entry(self):
        # HIP stops parsing there, so reporting the tail would name devices the
        # runtime will not hand out.
        for spec in ("0,9,1", "0,nonsense,1", "0,0,1"):
            with self.subTest(spec=spec):
                rows, _ = self.host.devices_report({"HIP_VISIBLE_DEVICES": spec})
                self.assertEqual([row["card"] for row in rows], ["card0"])

    def test_empty_list_hides_every_device(self):
        rows, _ = self.host.devices_report({"HIP_VISIBLE_DEVICES": ""})
        self.assertEqual(rows, [])


class MissingDeviceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_unreadable_counters_keep_the_row_and_report_the_reason(self):
        host = three_gpu_host(self.root)
        (host.devices / "0000:07:00.0/mem_info_vram_used").unlink()
        rows, _ = host.devices_report()
        self.assertEqual([row["rocm_index"] for row in rows], [0, 1, 2])
        self.assertIsNone(rows[1]["used_bytes"])
        self.assertEqual(rows[1]["card"], "card1")
        self.assertTrue(any("unreadable" in problem for problem in rows[1]["problems"]))
        # The device behind the failure keeps its own index and its own bytes.
        self.assertEqual(rows[2]["card"], "card2")
        self.assertEqual(rows[2]["total_bytes"], 16 << 30)

    def test_kfd_node_with_no_drm_card_is_reported_not_dropped(self):
        host = Host(self.root)
        host.add_kfd_node(1, bus=0x03, render_minor=128, unique_id=0x1111)
        host.add_kfd_node(2, bus=0x07, render_minor=129, unique_id=0x2222)
        host.add_card("card0", bus=0x03, render_minor=128, used=0, total=16 << 30)
        rows, _ = host.devices_report()
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[1]["card"])
        self.assertEqual(rows[1]["rocm_index"], 1)
        self.assertIn("no DRM card", rows[1]["problems"][0])

    def test_disagreeing_joins_refuse_to_name_a_card(self):
        """PCI address and render minor must point at the same device."""
        host = three_gpu_host(self.root)
        (host.drm / "renderD129/device").unlink()
        (host.drm / "renderD129/device").symlink_to(host.devices / "0000:0a:00.0")
        rows, _ = host.devices_report()
        self.assertIsNone(rows[1]["card"])
        self.assertIn("disagree", rows[1]["problems"][0])

    def test_render_minor_alone_still_identifies_a_card(self):
        # location_id can be absent on a node the kernel exports differently;
        # the render node is a second, independent path to the same device.
        host = three_gpu_host(self.root)
        properties = host.kfd / "2/properties"
        properties.write_text(properties.read_text().replace("location_id 1792",
                                                            "location_id 0"))
        rows, _ = host.devices_report()
        self.assertEqual(rows[1]["card"], "card1")
        self.assertEqual(rows[1]["total_bytes"], 32 << 30)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_format_names_the_index_the_card_and_the_role(self):
        rows, _ = three_gpu_host(self.root).devices_report()
        rendered = gpu_vram.format_row(rows[2])
        self.assertIn("ROCm2 (card2, 0000:0a:00.0, GPU-4d1c68aee3a84b64)", rendered)
        self.assertIn("role=display", rendered)
        self.assertIn("free=   13312 MiB", rendered)

    def test_format_marks_absent_counters_rather_than_printing_zero(self):
        host = three_gpu_host(self.root)
        (host.devices / "0000:03:00.0/mem_info_vram_used").unlink()
        rows, _ = host.devices_report()
        rendered = gpu_vram.format_row(rows[0])
        self.assertIn("used=       ?", rendered)
        self.assertNotIn("used=       0", rendered)
        self.assertIn("! VRAM counters unreadable", rendered)


if __name__ == "__main__":
    unittest.main()
