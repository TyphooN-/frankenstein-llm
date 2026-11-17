"""Source-only coverage for the declared convolution overlay, not native proof.

An upstream refresh can add raw convolution callers or new tensor names while
an older patch still applies cleanly. These checks do not compile or run C++.
"""
from pathlib import Path
import re
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "upstream/llama.cpp"


class ConvolutionOverlayCoverageTests(unittest.TestCase):
    def test_all_model_convolutions_use_the_shared_conversion_helper(self):
        models = sorted((SOURCE / "src/models").rglob("*.cpp"))
        self.assertTrue(models, "model source inventory is empty")
        raw_callers = [
            str(path.relative_to(SOURCE))
            for path in models
            if re.search(r"\bggml_ssm_conv\s*\(", path.read_text())
        ]
        self.assertEqual([], raw_callers,
                         "model convolution callers bypass the shared F32 conversion")

    def test_inkling_convolution_names_are_in_the_typed_regression(self):
        names = {
            "blk.%d.attn_k_conv.weight",
            "blk.%d.attn_v_conv.weight",
            "blk.%d.attn_post_conv.weight",
            "blk.%d.ffn_post_conv.weight",
        }
        source = (SOURCE / "tests/test-llama-archs.cpp").read_text()
        table = re.search(r"static const char \* conv_kernel_names\[\] = \{(.*?)\};",
                          source, re.DOTALL)
        if table is None:
            self.fail("typed convolution regression table is missing")
        actual = set(re.findall(r'"([^"]+)"', table.group(1)))
        self.assertEqual(set(), names - actual,
                         "typed regression would skip Inkling2 convolution tensors")


if __name__ == "__main__":
    unittest.main()
