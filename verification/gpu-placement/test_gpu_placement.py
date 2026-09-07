#!/usr/bin/env python3
"""Placement arithmetic and the policy order it implements.

The device rows are constructed rather than read, so the rules can be exercised
at sizes this host does not currently hold. The GGUF reader is exercised against
files this module writes byte by byte, for the same reason.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gguf_header = _load("gguf_header")
placement = _load("gpu_placement")

GIB = 1 << 30
POLICY = {
    "schema": 1,
    "per_device_reserve_bytes": 1 * GIB,
    "display_reserve_bytes": 2 * GIB,
    "prefer": [0, 2],
    "fallback": [1],
    "equal_split_when_it_fits": True,
    "split_scale": 96,
}


def rows(*budgets: int) -> list[dict]:
    return [{"rocm_index": index, "card": f"card{index}", "budget_bytes": budget,
             "total_bytes": budget, "display": index == 2, "problems": [],
             "reserved_bytes": 0, "uuid": None}
            for index, budget in enumerate(budgets)]


# The shape of this workspace after reserves: two RX 6900 XTs, one of them the
# display, and a 32 GiB V620 as the fallback.
HOST = rows(15 * GIB, 29 * GIB, 12 * GIB)


def write_gguf(path: Path, metadata: dict, tensor_count: int = 3) -> Path:
    """A GGUF header with no tensor data behind it."""
    def string(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for key, (kind, value) in metadata.items():
        body += string(key) + struct.pack("<I", kind)
        if kind == gguf_header.STRING:
            body += string(value)
        elif kind == gguf_header.ARRAY:
            element, items = value
            body += struct.pack("<IQ", element, len(items))
            body += b"".join(struct.pack("<I", item) for item in items)
        else:
            body += struct.pack(gguf_header.SCALARS[kind][0], value)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, tensor_count, len(metadata)) + body)
    return path


U32 = 4
STR = gguf_header.STRING
ARR = gguf_header.ARRAY


class GGUFHeaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_the_fields_kv_sizing_needs(self):
        path = write_gguf(self.dir / "m.gguf", {
            "general.architecture": (STR, "qwen35"),
            "qwen35.block_count": (U32, 65),
            "qwen35.attention.head_count": (U32, 24),
            "qwen35.attention.head_count_kv": (U32, 4),
            "qwen35.attention.key_length": (U32, 256),
            "qwen35.attention.value_length": (U32, 256),
            "qwen35.context_length": (U32, 262144),
        })
        shape = gguf_header.shape(gguf_header.read_header(path))
        self.assertEqual(shape["architecture"], "qwen35")
        self.assertEqual(shape["block_count"], 65)
        self.assertEqual(shape["head_count_kv"], 4)
        self.assertEqual(shape["key_length"], 256)

    def test_head_width_falls_back_to_embedding_over_head_count(self):
        path = write_gguf(self.dir / "m.gguf", {
            "general.architecture": (STR, "llama"),
            "llama.block_count": (U32, 32),
            "llama.embedding_length": (U32, 4096),
            "llama.attention.head_count": (U32, 32),
        })
        shape = gguf_header.shape(gguf_header.read_header(path))
        self.assertEqual(shape["key_length"], 128)
        self.assertEqual(shape["value_length"], 128)
        # head_count_kv is absent, so it is the head count: no grouped-query
        # assumption is invented for a model that did not declare one.
        self.assertEqual(shape["head_count_kv"], 32)

    def test_per_block_kv_head_arrays_survive_as_arrays(self):
        path = write_gguf(self.dir / "m.gguf", {
            "general.architecture": (STR, "gemma4"),
            "gemma4.block_count": (U32, 6),
            "gemma4.attention.head_count": (U32, 16),
            "gemma4.attention.head_count_kv": (ARR, (U32, [8, 8, 8, 8, 8, 1])),
            "gemma4.attention.key_length": (U32, 512),
            "gemma4.attention.value_length": (U32, 512),
            "gemma4.attention.sliding_window": (U32, 1024),
        })
        shape = gguf_header.shape(gguf_header.read_header(path))
        self.assertEqual(shape["head_count_kv"], [8, 8, 8, 8, 8, 1])
        self.assertEqual(shape["sliding_window"], 1024)

    def test_refuses_a_file_that_is_not_gguf(self):
        path = self.dir / "not.gguf"
        path.write_bytes(b"NOPE" + b"\0" * 64)
        with self.assertRaises(gguf_header.GGUFError):
            gguf_header.read_header(path)

    def test_refuses_an_implausible_length_instead_of_allocating(self):
        path = self.dir / "huge.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
                         + struct.pack("<Q", 1 << 40))
        with self.assertRaises(gguf_header.GGUFError):
            gguf_header.read_header(path)

    def test_endlessly_nested_arrays_are_refused_as_a_gguf_error(self):
        """Bounded means bounded on this axis too.

        A hostile or corrupt header can describe an array of arrays without
        limit. Recursing on it raises RecursionError, which is not what any
        caller here catches, so a weight file would take the placement tool down
        instead of being reported as unreadable.
        """
        path = self.dir / "nested.gguf"
        body = struct.pack("<Q", 1) + b"k" + struct.pack("<I", gguf_header.ARRAY)
        body += (struct.pack("<IQ", gguf_header.ARRAY, 1)
                 * (gguf_header.MAX_ARRAY_DEPTH + 2))
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + body)
        with self.assertRaisesRegex(gguf_header.GGUFError, "nested deeper"):
            gguf_header.read_header(path)

    def test_truncated_header_is_an_error_not_a_partial_shape(self):
        path = self.dir / "cut.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 4) + b"\x05")
        with self.assertRaises(gguf_header.GGUFError):
            gguf_header.read_header(path)


class KVSizingTests(unittest.TestCase):
    def test_matches_the_hand_computation_for_a_grouped_query_model(self):
        shape = {"block_count": 65, "head_count_kv": 4, "key_length": 256,
                 "value_length": 256}
        # 65 blocks x 4 KV heads x (256 + 256) elements x 18/32 bytes x 131072
        expected = int(65 * 4 * (256 + 256) * (18 / 32) * 131072)
        self.assertEqual(kv(shape, 131072, "q4_0", "q4_0"), expected)

    def test_cache_type_changes_the_size_by_its_block_layout(self):
        shape = {"block_count": 2, "head_count_kv": 1, "key_length": 64,
                 "value_length": 64}
        self.assertEqual(kv(shape, 1024, "f16", "f16"), 2 * 1 * 128 * 2 * 1024)
        self.assertEqual(kv(shape, 1024, "q8_0", "q8_0"),
                         int(2 * 1 * 128 * (34 / 32) * 1024))

    def test_per_block_head_counts_are_summed_not_averaged(self):
        shape = {"block_count": 6, "head_count_kv": [8, 8, 8, 8, 8, 1],
                 "key_length": 512, "value_length": 512}
        expected = int(sum([8, 8, 8, 8, 8, 1]) * 1024 * 2.0 * 4096)
        self.assertEqual(kv(shape, 4096, "f16", "f16"), expected)

    def test_mismatched_per_block_list_uses_the_widest_entry(self):
        # An array that does not match the block count cannot be trusted
        # element-wise; over-estimating is the safe direction for a budget.
        shape = {"block_count": 4, "head_count_kv": [8, 1], "key_length": 64,
                 "value_length": 64}
        self.assertEqual(placement.kv_heads_per_block(shape), [8, 8, 8, 8])

    def test_missing_attention_fields_yield_no_estimate(self):
        self.assertIsNone(kv({"block_count": 4}, 1024, "f16", "f16"))

    def test_unknown_cache_type_is_refused_not_guessed(self):
        shape = {"block_count": 1, "head_count_kv": 1, "key_length": 8,
                 "value_length": 8}
        with self.assertRaises(placement.PolicyError):
            kv(shape, 16, "q3_k_s", "f16")


def kv(shape, context, type_k, type_v):
    return placement.kv_bytes(shape, context, type_k, type_v)


class PolicyOrderTests(unittest.TestCase):
    def propose(self, gib: float, device_rows=None, policy=None):
        return placement.propose(int(gib * GIB), device_rows or HOST, policy or POLICY)

    def test_a_model_that_fits_on_one_preferred_device_is_not_split(self):
        result = self.propose(8)
        self.assertEqual(result["split"], [1, 0, 0])
        self.assertTrue(result["fits"])
        self.assertIn("no split needed", result["reasons"][0])

    def test_a_single_device_placement_never_lands_on_the_fallback(self):
        """The rule this policy exists for.

        A 20 GiB model fits on the 29 GiB fallback card alone and on neither
        preferred card alone. Choosing the fallback would be the smaller split
        and the wrong placement: the operator wants the preferred pair carrying
        work and the fallback idle.
        """
        result = self.propose(20)
        self.assertEqual(result["split"][1], 0)
        self.assertGreater(result["split"][0], 0)
        self.assertGreater(result["split"][2], 0)
        self.assertIn("fallback device carries none of it", " ".join(result["reasons"]))

    def test_preferred_pair_share_follows_working_headroom(self):
        # 24 GiB cannot be halved onto 14 + 11 GiB of headroom, so the larger
        # card has to take the larger share rather than the split equalising.
        result = self.propose(24)
        projected = result["projected_bytes"]
        self.assertEqual(result["split"][1], 0)
        self.assertGreater(projected[0], projected[2])
        self.assertLessEqual(projected[0], 15 * GIB)
        self.assertLessEqual(projected[2], 12 * GIB)

    def test_equal_split_only_once_the_preferred_pair_is_too_small(self):
        # The pair holds 14 + 11 GiB of working headroom, not 15 + 12 GiB of
        # budget: 24 GiB fits on it, 30 GiB does not, and an equal 10 GiB share
        # is inside every device's headroom.
        self.assertEqual(self.propose(24)["split"][1], 0)
        equal = self.propose(30)
        self.assertEqual(equal["split"], [1, 1, 1])
        self.assertIn("equal split", " ".join(equal["reasons"]))

    def test_oversized_model_fills_preferred_then_spills_to_the_fallback(self):
        result = self.propose(46)
        self.assertTrue(result["fits"], result["reasons"])
        projected = result["projected_bytes"]
        self.assertLessEqual(projected[0], 15 * GIB)
        self.assertLessEqual(projected[2], 12 * GIB)
        self.assertGreater(projected[1], 15 * GIB)

    def test_a_model_too_large_for_the_host_is_reported_as_not_fitting(self):
        result = self.propose(80)
        self.assertFalse(result["fits"])
        self.assertIn("does not fit", " ".join(result["reasons"]))

    def test_rounding_never_pushes_a_device_past_its_budget(self):
        """A split is coarse; the plan behind it is not.

        Largest-remainder rounding alone put ROCm0 at 15.59 GiB of a 14.98 GiB
        budget for the repository-agent model -- a plan that fit in bytes,
        emitted as a split that did not.
        """
        for size in range(20, 56):
            with self.subTest(gib=size):
                result = self.propose(size)
                if not result["fits"]:
                    continue
                for index, projected in result["projected_bytes"].items():
                    self.assertLessEqual(round(projected), HOST[index]["budget_bytes"],
                                         f"{size} GiB overfilled ROCm{index}")

    def test_split_is_reduced_to_its_smallest_equivalent_form(self):
        result = self.propose(30)
        self.assertEqual(result["split"], [1, 1, 1])

    def test_marginal_placements_are_called_out(self):
        # 53.5 GiB against 53 GiB of total working headroom: it still fits
        # inside the 56 GiB of budget, but only by eating into the margin.
        result = self.propose(53.5)
        self.assertTrue(result["fits"])
        self.assertIn("marginal", " ".join(result["reasons"]))

    def test_unknown_size_proposes_nothing_rather_than_a_default(self):
        result = placement.propose(None, HOST, POLICY)
        self.assertNotIn("split", result)
        self.assertIn("unknown", " ".join(result["reasons"]))

    def test_a_device_with_no_budget_is_skipped_without_shifting_indices(self):
        degraded = rows(15 * GIB, 29 * GIB, 12 * GIB)
        degraded[2]["budget_bytes"] = None
        degraded[2]["problems"] = ["VRAM counters unreadable"]
        result = placement.propose(10 * GIB, degraded, POLICY)
        self.assertEqual(len(result["split"]), 3)
        self.assertEqual(result["split"][2], 0)

    def test_no_usable_device_proposes_nothing(self):
        blank = rows(0, 0, 0)
        result = placement.propose(10 * GIB, blank, POLICY)
        self.assertNotIn("split", result)
        self.assertIn("no device", " ".join(result["reasons"]))


class BudgetTests(unittest.TestCase):
    def test_display_device_is_charged_the_extra_reserve(self):
        devices = [
            {"rocm_index": 0, "card": "card0", "total_bytes": 16 * GIB,
             "display_connectors": [], "problems": []},
            {"rocm_index": 2, "card": "card2", "total_bytes": 16 * GIB,
             "display_connectors": ["card2-DP-4"], "problems": []},
        ]
        computed = placement.budgets(devices, POLICY)
        self.assertEqual(computed[0]["budget_bytes"], 15 * GIB)
        self.assertEqual(computed[1]["budget_bytes"], 13 * GIB)
        self.assertTrue(computed[1]["display"])

    def test_a_device_with_no_total_reports_no_budget_rather_than_zero(self):
        computed = placement.budgets(
            [{"rocm_index": 0, "card": None, "total_bytes": None,
              "display_connectors": [], "problems": ["no DRM card"]}], POLICY)
        self.assertIsNone(computed[0]["budget_bytes"])
        self.assertEqual(computed[0]["problems"], ["no DRM card"])


class PolicyFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, policy: dict) -> Path:
        path = self.dir / "policy.json"
        path.write_text(json.dumps(policy))
        return path

    def test_the_repository_policy_loads(self):
        loaded = placement.load_policy(placement.POLICY)
        self.assertEqual(loaded["prefer"], [0, 2])
        self.assertEqual(loaded["fallback"], [1])

    def test_a_device_cannot_be_preferred_and_fallback_at_once(self):
        broken = dict(POLICY, prefer=[0, 1], fallback=[1])
        with self.assertRaises(placement.PolicyError):
            placement.load_policy(self.write(broken))

    def test_a_policy_with_no_preferred_device_is_refused(self):
        with self.assertRaises(placement.PolicyError):
            placement.load_policy(self.write(dict(POLICY, prefer=[])))

    def test_unknown_schema_is_refused(self):
        with self.assertRaises(placement.PolicyError):
            placement.load_policy(self.write(dict(POLICY, schema=99)))

    def test_missing_reserve_is_refused_rather_than_defaulted(self):
        partial = {key: value for key, value in POLICY.items()
                   if key != "display_reserve_bytes"}
        with self.assertRaises(placement.PolicyError):
            placement.load_policy(self.write(partial))


class ShardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_split_model_is_sized_from_every_shard(self):
        """The configured path is one file; the model is four.

        Sizing from the ``model =`` value alone understates the repository-agent
        preset by roughly 30 GiB, which is the difference between a plan that
        fits and one that does not.
        """
        for index in range(1, 5):
            (self.dir / f"M-{index:05d}-of-00004.gguf").write_bytes(b"x" * 1000)
        found = placement.shards(self.dir / "M-00001-of-00004.gguf")
        self.assertEqual(len(found), 4)

    def test_an_unsharded_model_is_itself(self):
        path = self.dir / "solo.gguf"
        path.write_bytes(b"x")
        self.assertEqual(placement.shards(path), [path])

    def test_a_first_shard_with_no_siblings_still_reports_itself(self):
        path = self.dir / "M-00001-of-00004.gguf"
        self.assertEqual(placement.shards(path), [path])


class PresetMeasurementTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.model = write_gguf(self.dir / "m.gguf", {
            "general.architecture": (STR, "qwen35"),
            "qwen35.block_count": (U32, 4),
            "qwen35.attention.head_count": (U32, 8),
            "qwen35.attention.head_count_kv": (U32, 2),
            "qwen35.attention.key_length": (U32, 128),
            "qwen35.attention.value_length": (U32, 128),
            "qwen35.context_length": (U32, 8192),
        })

    def test_context_comes_from_the_preset_not_the_model_maximum(self):
        record = placement.measure("t", {"model": str(self.model), "ctx-size": "2048",
                                         "cache-type-k": "q4_0", "cache-type-v": "q4_0"})
        self.assertEqual(record["context"], 2048)
        self.assertEqual(record["kv_bytes"],
                         int(4 * 2 * 256 * (18 / 32) * 2048))

    def test_absent_context_falls_back_to_the_model_maximum(self):
        record = placement.measure("t", {"model": str(self.model)})
        self.assertEqual(record["context"], 8192)

    def test_embedding_presets_are_not_charged_a_kv_cache(self):
        record = placement.measure("e", {"model": str(self.model), "ctx-size": "4096",
                                         "embedding": "1"})
        self.assertEqual(record["kv_bytes"], 0)
        self.assertIn("no persistent KV cache", " ".join(record["notes"]))

    def test_a_missing_weight_file_is_reported_not_silently_zero(self):
        record = placement.measure("t", {"model": str(self.dir / "absent.gguf")})
        self.assertIn("missing_files", record)
        self.assertIn("weight files missing", " ".join(record["notes"]))
        self.assertNotIn("required_bytes", record)

    def test_a_projector_is_reported_separately_from_the_split(self):
        projector = self.dir / "mmproj.gguf"
        projector.write_bytes(b"x" * 4096)
        record = placement.measure("v", {"model": str(self.model),
                                         "mmproj": str(projector),
                                         "mmproj-device": "ROCm1"})
        self.assertEqual(record["projector_bytes"], 4096)
        self.assertEqual(record["projector_device"], "ROCm1")
        self.assertIn("pinned by mmproj-device", " ".join(record["notes"]))

    def test_a_projector_with_no_device_is_not_described_as_pinned(self):
        """The note has to match the preset, not the usual case.

        Claiming "pinned by mmproj-device" for a preset that sets no such key
        asserts a placement nobody configured, in the one report an operator
        would consult before trusting the split.
        """
        projector = self.dir / "mmproj.gguf"
        projector.write_bytes(b"x" * 4096)
        record = placement.measure("v", {"model": str(self.model),
                                         "mmproj": str(projector)})
        self.assertIsNone(record["projector_device"])
        joined = " ".join(record["notes"])
        self.assertIn("sets no mmproj-device", joined)
        self.assertNotIn("bytes are pinned by mmproj-device", joined)

    def test_sliding_window_models_are_labelled_as_over_estimates(self):
        model = write_gguf(self.dir / "swa.gguf", {
            "general.architecture": (STR, "gemma4"),
            "gemma4.block_count": (U32, 6),
            "gemma4.attention.head_count": (U32, 16),
            "gemma4.attention.head_count_kv": (ARR, (U32, [8, 8, 8, 8, 8, 1])),
            "gemma4.attention.key_length": (U32, 512),
            "gemma4.attention.value_length": (U32, 512),
            "gemma4.attention.sliding_window": (U32, 1024),
        })
        record = placement.measure("g", {"model": str(model), "ctx-size": "32768"})
        joined = " ".join(record["notes"])
        self.assertIn("sliding-window", joined)
        self.assertIn("overstates", joined)


class ConfiguredSplitTests(unittest.TestCase):
    def test_reads_comma_and_slash_forms(self):
        self.assertEqual(placement.configured_split({"tensor-split": "3,6,2"}, 3),
                         [3, 6, 2])
        self.assertEqual(placement.configured_split({"tensor-split": "3/6/2"}, 3),
                         [3, 6, 2])

    def test_a_split_of_the_wrong_length_is_not_reported_as_configured(self):
        self.assertIsNone(placement.configured_split({"tensor-split": "1,1"}, 3))

    def test_absent_or_malformed_splits_are_none(self):
        self.assertIsNone(placement.configured_split({}, 3))
        self.assertIsNone(placement.configured_split({"tensor-split": "a,b,c"}, 3))


class ProjectorChargeTests(unittest.TestCase):
    def proposal(self, gib: float):
        return placement.propose(int(gib * GIB), HOST, POLICY)

    def test_projector_is_charged_to_the_device_it_is_pinned_to(self):
        proposal = self.proposal(8)
        record = {"projector_device": "ROCm1", "projector_bytes": 1 << 30}
        placement.charge_projector(record, proposal)
        self.assertEqual(proposal["projector_charged_to"], 1)
        self.assertEqual(proposal["slack_bytes"][1], 28 * GIB)
        self.assertTrue(proposal["fits"])

    def test_a_projector_that_overflows_its_device_flips_the_verdict(self):
        """The blind spot this closes.

        tensor-split does not distribute the projector, so a preset can look
        like it fits while the card pinned to hold the projector cannot.
        """
        proposal = self.proposal(24)
        self.assertTrue(proposal["fits"])
        record = {"projector_device": "ROCm0", "projector_bytes": 2 << 30}
        placement.charge_projector(record, proposal)
        self.assertFalse(proposal["fits"])
        self.assertIn("over budget", " ".join(proposal["reasons"]))

    def test_case_is_not_part_of_the_device_name(self):
        proposal = self.proposal(8)
        placement.charge_projector(
            {"projector_device": "rocm2", "projector_bytes": 1 << 30}, proposal)
        self.assertEqual(proposal["projector_charged_to"], 2)

    def test_a_device_name_that_is_not_an_index_is_reported_not_ignored(self):
        proposal = self.proposal(8)
        placement.charge_projector(
            {"projector_device": "CPU", "projector_bytes": 1 << 30}, proposal)
        self.assertNotIn("projector_charged_to", proposal)
        self.assertIn("not a ROCm index", " ".join(proposal["reasons"]))

    def test_a_pin_to_an_absent_device_is_reported(self):
        proposal = self.proposal(8)
        placement.charge_projector(
            {"projector_device": "ROCm7", "projector_bytes": 1 << 30}, proposal)
        self.assertIn("not a visible device", " ".join(proposal["reasons"]))

    def test_a_preset_with_no_projector_is_left_alone(self):
        proposal = self.proposal(8)
        before = dict(proposal["slack_bytes"])
        placement.charge_projector({}, proposal)
        self.assertEqual(proposal["slack_bytes"], before)

    def test_a_pinned_projector_is_reserved_before_the_split_is_chosen(self):
        """The knife edge this closes.

        Charging the projector only after a split was chosen let the split hand
        its device a share that fit until the projector landed on top of it:
        ``obliterated-vision`` was reported as fitting with 0.21 GiB left on a
        16 GiB card. Reserving the bytes while the rules are still running moves
        the share off that edge instead of reporting it from it.
        """
        record = {"projector_device": "ROCm0", "projector_bytes": 1 << 30}
        required = 23 * GIB
        blind = placement.propose(required, HOST, POLICY)
        placement.charge_projector(record, blind, POLICY["per_device_reserve_bytes"])
        planned = placement.propose(required, HOST, POLICY,
                                    placement.pinned_bytes(record))
        placement.charge_projector(record, planned, POLICY["per_device_reserve_bytes"])
        self.assertTrue(blind["fits"])
        self.assertTrue(planned["fits"])
        self.assertGreater(planned["slack_bytes"][0], blind["slack_bytes"][0])
        self.assertIn("held back from its share", " ".join(planned["reasons"]))

    def test_a_charge_that_leaves_less_than_the_margin_is_called_marginal(self):
        proposal = self.proposal(24)
        placement.charge_projector({"projector_device": "ROCm0",
                                    "projector_bytes": 1 << 30}, proposal, 1 << 30)
        self.assertTrue(proposal["fits"])
        self.assertIn("marginal", " ".join(proposal["reasons"]))

    def test_no_margin_is_asserted_when_the_caller_supplies_none(self):
        proposal = self.proposal(24)
        placement.charge_projector(
            {"projector_device": "ROCm0", "projector_bytes": 1 << 30}, proposal)
        self.assertNotIn("marginal", " ".join(proposal["reasons"]))

    def test_an_unpinned_projector_reserves_nothing_because_no_device_is_named(self):
        # The runtime chooses the card, so there is no device to hold bytes back
        # from. Guessing one would plan around a placement nobody configured.
        self.assertEqual({}, placement.pinned_bytes({"projector_bytes": 1 << 30}))
        self.assertEqual({}, placement.pinned_bytes(
            {"projector_device": "CPU", "projector_bytes": 1 << 30}))
        self.assertEqual({0: 1 << 30}, placement.pinned_bytes(
            {"projector_device": "rocm0", "projector_bytes": 1 << 30}))

    def test_an_unpinned_projector_is_reported_rather_than_dropped(self):
        """Real bytes with no configured device must not read as a clean fit.

        Without mmproj-device the runtime picks the card, so no slack figure can
        honestly carry these bytes -- but silently returning left a preset
        reporting "fits" with most of a gigabyte unaccounted anywhere.
        """
        proposal = self.proposal(8)
        before = dict(proposal["slack_bytes"])
        placement.charge_projector({"projector_bytes": 1 << 30}, proposal)
        self.assertEqual(proposal["slack_bytes"], before)
        self.assertNotIn("projector_charged_to", proposal)
        self.assertIn("not pinned by mmproj-device", " ".join(proposal["reasons"]))


class VerifyBaselineTests(unittest.TestCase):
    """``--verify`` is the only thing that gives the estimate any standing.

    Its residual is quoted in the reference documentation and the ADR as the
    calibration of every budget in the policy, so an arithmetic slip here is a
    slip in a published number rather than in a debug line.
    """

    POLICY = dict(POLICY, idle_baseline_bytes={"0": 1 * GIB, "1": 1 * GIB, "2": 2 * GIB})

    def setUp(self):
        # verify() reaches for the router through this module; a stub keeps the
        # arithmetic under test and the network out of it.
        stub = type(sys)("local_model_status")
        stub.collect_status = lambda: {"endpoint": "stub", "loaded_models": ["m"]}
        sys.modules["local_model_status"] = stub
        self.addCleanup(sys.modules.pop, "local_model_status", None)
        self.report = [{"alias": "m", "required_bytes": 30 * GIB,
                        "configured_split": [1, 1, 1]}]

    def devices(self, *used):
        return [{"rocm_index": index, "used_bytes": value}
                for index, value in enumerate(used)]

    def test_residual_is_the_estimate_less_measured_bytes_net_of_baseline(self):
        result = placement.verify(HOST, self.devices(11 * GIB, 11 * GIB, 12 * GIB),
                                  self.report, self.POLICY)
        comparison = result["comparison"]
        self.assertEqual(comparison["measured_bytes"], 34 * GIB)
        self.assertEqual(comparison["attributed_bytes"], 30 * GIB)
        self.assertEqual(comparison["residual_bytes"], 0)
        self.assertEqual(comparison["unread_devices"], [])

    def test_an_unread_device_drops_out_of_both_sides_not_one(self):
        """The flaw this closes.

        A card whose residency cannot be read contributes no bytes to the
        measurement. Subtracting its idle baseline anyway charged the desktop's
        allocation against a card that reported nothing, inflating the residual
        by that baseline and making the estimate look more conservative than it
        is.
        """
        result = placement.verify(HOST, self.devices(11 * GIB, 11 * GIB, None),
                                  self.report, self.POLICY)
        comparison = result["comparison"]
        self.assertEqual(comparison["measured_bytes"], 22 * GIB)
        # Only ROCm0 and ROCm1's baselines, never the unread ROCm2's 2 GiB.
        self.assertEqual(comparison["attributed_bytes"], 20 * GIB)
        self.assertEqual(comparison["unread_devices"], [2])
        self.assertEqual(comparison["compared_devices"], [0, 1])
        self.assertIn("covers only part of the model", comparison["note"])


class ScanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def touch(self, relative: str) -> Path:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return path

    def test_finds_weights_no_preset_loads(self):
        served = self.touch("served.gguf")
        spare = self.touch("spare.gguf")
        found = placement.unconfigured_models({"a": {"model": str(served)}}, self.dir)
        self.assertEqual(found, [spare])

    def test_a_projector_a_preset_loads_is_not_reported_as_unconfigured(self):
        model = self.touch("m.gguf")
        projector = self.touch("mmproj.gguf")
        found = placement.unconfigured_models(
            {"v": {"model": str(model), "mmproj": str(projector)}}, self.dir)
        self.assertEqual(found, [])

    def test_a_split_model_is_reported_once_by_its_first_shard(self):
        for index in range(1, 4):
            self.touch(f"S-{index:05d}-of-00003.gguf")
        found = placement.unconfigured_models({}, self.dir)
        self.assertEqual([path.name for path in found], ["S-00001-of-00003.gguf"])

    def test_later_shards_of_a_configured_model_are_not_reported(self):
        for index in range(1, 4):
            self.touch(f"S-{index:05d}-of-00003.gguf")
        found = placement.unconfigured_models(
            {"a": {"model": str(self.dir / "S-00001-of-00003.gguf")}}, self.dir)
        self.assertEqual(found, [])

    def test_staging_directories_are_skipped(self):
        self.touch(".staging/partial.gguf")
        kept = self.touch("real.gguf")
        self.assertEqual(placement.unconfigured_models({}, self.dir), [kept])


class VerdictTests(unittest.TestCase):
    def test_an_unsized_model_reads_as_not_sized_not_as_a_fit_failure(self):
        record = {"alias": "mmproj", "weight_bytes": 1 << 20, "notes": [],
                  "proposal": placement.propose(None, HOST, POLICY)}
        rendered = placement.format_models([record])
        self.assertIn("[not sized]", rendered)
        self.assertNotIn("DOES NOT FIT", rendered)

    def test_an_oversized_model_still_reads_as_a_fit_failure(self):
        record = {"alias": "huge", "weight_bytes": 200 << 30, "notes": [],
                  "required_bytes": 200 << 30,
                  "proposal": placement.propose(200 << 30, HOST, POLICY)}
        self.assertIn("DOES NOT FIT", placement.format_models([record]))


class LivePresetTests(unittest.TestCase):
    """The repository's own presets, sized against constructed device rows.

    This reads ``llama-models.ini`` and the weight files that are actually on
    this disk. It asserts the properties a placement must have rather than the
    numbers of the day, so it stays meaningful as models are added or requantized.
    """

    def setUp(self):
        self.presets = placement.model_catalog.presets()
        self.policy = placement.load_policy(placement.POLICY)

    def test_every_preset_names_a_weight_file_that_exists(self):
        for alias, preset in self.presets.items():
            with self.subTest(alias=alias):
                record = placement.measure(alias, preset)
                self.assertNotIn("missing_files", record,
                                 f"{alias} references weights that are not on disk")

    def test_every_preset_can_be_sized_and_placed(self):
        for alias, preset in self.presets.items():
            with self.subTest(alias=alias):
                record = placement.measure(alias, preset)
                self.assertIsNotNone(record.get("required_bytes"),
                                     f"{alias} could not be sized")
                proposal = placement.propose(record["required_bytes"], HOST, self.policy)
                self.assertIn("split", proposal)
                self.assertEqual(len(proposal["split"]), 3)

    def test_configured_placement_matches_what_the_policy_proposes(self):
        """The regression this whole module exists to make possible.

        ``llama-models.ini`` is edited by hand and the policy is not consulted
        when it is. Comparing the two here means a preset whose context, cache
        type or weights change -- or a policy whose reserves change -- fails
        loudly instead of drifting into a placement nobody evaluated.

        It reads the live device set, because the budgets are a property of the
        cards actually present. On a host without them there is nothing to
        compare and the check says so rather than inventing a topology.
        """
        devices, _ = placement.gpu_vram.devices()
        if len(devices) != 3 or any(d.get("total_bytes") is None for d in devices):
            self.skipTest(f"needs the three-GPU host; found {len(devices)} usable devices")
        device_rows = placement.budgets(devices, self.policy)
        margin = self.policy.get("min_slack_bytes",
                                 self.policy["per_device_reserve_bytes"])
        for alias, preset in self.presets.items():
            with self.subTest(alias=alias):
                record = placement.evaluate_preset(alias, preset, device_rows,
                                                   self.policy)
                proposal = record["proposal"]
                configured = record["configured_split"]
                self.assertEqual(
                    configured, proposal["split"],
                    f"{alias} is configured {configured} but the policy proposes "
                    f"{proposal['split']}: {'; '.join(proposal['reasons'])}")
                self.assertTrue(proposal["fits"],
                                f"{alias} does not fit: {'; '.join(proposal['reasons'])}")
                # No preset may ship on the edge of a card. A placement that
                # only fits until a compute buffer grows is one this evaluation
                # is supposed to have moved off, not one it is supposed to
                # report from.
                for index, slack in proposal["slack_bytes"].items():
                    if proposal["split"][index] or index in placement.pinned_bytes(record):
                        self.assertGreaterEqual(
                            slack, margin,
                            f"{alias} leaves ROCm{index} only {slack} bytes: "
                            f"{'; '.join(proposal['reasons'])}")

    def test_the_fallback_device_is_used_only_when_the_pair_cannot_hold_it(self):
        """The operator preference, asserted rather than described.

        A proposal may use the V620 -- four of these models are too big not to
        -- but never while the two RX 6900 XTs could have carried the model
        inside their own working headroom.
        """
        margin = self.policy.get("min_slack_bytes",
                                 self.policy["per_device_reserve_bytes"])
        pair = sum(max(HOST[index]["budget_bytes"] - margin, 0)
                   for index in self.policy["prefer"])
        for alias, preset in self.presets.items():
            with self.subTest(alias=alias):
                record = placement.measure(alias, preset)
                required = record.get("required_bytes")
                proposal = placement.propose(required, HOST, self.policy)
                if "split" not in proposal:
                    # A preset that could not be sized has no placement to judge.
                    # Saying why beats a KeyError: the sizing read is file I/O on
                    # a multi-gigabyte header and can fail transiently on a host
                    # under heavy build load, which is a different problem from
                    # the fallback device being misused.
                    self.fail(f"{alias} was not sized, so its placement cannot be "
                              f"judged: {'; '.join(proposal['reasons'])}; "
                              f"{'; '.join(record.get('notes', []))}")
                if proposal["split"][1] > 0:
                    self.assertGreater(required, pair,
                                       f"{alias} used the fallback device while the "
                                       "preferred pair had headroom for all of it")


if __name__ == "__main__":
    unittest.main()
