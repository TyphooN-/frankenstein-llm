#!/usr/bin/env python3
"""The chat templates this repository substitutes for broken shipped ones.

No server and no GPU. These render the actual files in config/chat-templates/
against the message shapes the router really sends, so the properties that made
the override necessary cannot regress silently:

* the tools branch exists, so function signatures reach the model at all;
* later turns survive, so multi-turn serving is not truncated;
* a `tool` result is rendered, so a tool call can be answered;
* string content passes through verbatim, which is what llama.cpp's multimodal
  path relies on -- it substitutes a media marker into the text before the
  template ever runs.

The templates are extracted verbatim from in-repository weights, so these also
pin the provenance: a file whose sha256 has drifted is no longer the template
those weights ship, whatever it may say in a comment.
"""
from __future__ import annotations

import configparser
import hashlib
from pathlib import Path
import unittest

jinja2 = None
try:
    import jinja2 as _jinja2
    jinja2 = _jinja2
except ImportError:                                             # pragma: no cover
    pass

ROOT = Path("/home/typhoon/git/frankenstein-llm")
TEMPLATES = ROOT / "config/chat-templates"

# sha256 of the template as extracted. qwen3.8-27b-tools.jinja came out of
# Qwen3.8-27B-Ridge-3.7bpw.gguf and is byte-identical to the one inside
# RVN-Q6_K-multilingual-mtp.gguf; qwen2.5-tools.jinja came out of
# fim/qwen2.5-coder-7b-q8_0.gguf.
EXPECTED = {
    "qwen3.8-27b-tools.jinja":
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "qwen2.5-tools.jinja":
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
}

TOOLS = [{"type": "function", "function": {
    "name": "lookup_ticket",
    "parameters": {"type": "object",
                   "properties": {"ticket_id": {"type": "integer"}}}}}]

# The conversation the 88-character template phr00ty shipped destroyed: it
# rendered messages[0] and dropped the other three.
CONVERSATION = [
    {"role": "user", "content": "Look up ticket 4172."},
    {"role": "assistant", "content": "",
     "tool_calls": [{"function": {"name": "lookup_ticket",
                                  "arguments": {"ticket_id": 4172}}}]},
    {"role": "tool", "content": '{"status":"open"}'},
    {"role": "user", "content": "Summarise it."},
]


def render(name: str, **context) -> str:
    """Render one template the way llama.cpp would, with its raise_exception global."""
    environment = jinja2.Environment(loader=jinja2.BaseLoader(),
                                     undefined=jinja2.ChainableUndefined)

    def raise_exception(message):                               # pragma: no cover
        raise RuntimeError(message)

    environment.globals["raise_exception"] = raise_exception
    return environment.from_string((TEMPLATES / name).read_text()).render(**context)


class TemplateProvenanceTests(unittest.TestCase):
    """These files are copies of published templates, not local authorship."""

    def test_every_template_matches_the_weights_it_was_taken_from(self):
        for name, digest in EXPECTED.items():
            with self.subTest(template=name):
                path = TEMPLATES / name
                self.assertTrue(path.exists(), f"{name} is missing")
                self.assertEqual(
                    digest, hashlib.sha256(path.read_bytes()).hexdigest(),
                    f"{name} no longer matches the template extracted from its "
                    f"source weights; re-extract rather than hand-editing")

    def test_every_referenced_template_exists_and_is_declared(self):
        # A preset pointing at a template this suite does not pin would serve an
        # unchecked prompt format under a checked alias.
        parser = configparser.ConfigParser()
        parser.read(ROOT / "llama-models.ini")
        referenced = set()
        for section in parser.sections():
            value = parser[section].get("chat-template-file")
            if value:
                referenced.add(Path(value).name)
                self.assertTrue(Path(value).exists(), f"{section}: {value} missing")
        self.assertTrue(referenced <= set(EXPECTED),
                        f"unpinned templates in use: {sorted(referenced - set(EXPECTED))}")


@unittest.skipIf(jinja2 is None, "jinja2 is not installed")
class TemplateBehaviourTests(unittest.TestCase):
    """What the overrides have to do that the shipped templates did not."""

    def test_both_templates_render_the_tool_definitions(self):
        # This is the whole reason the override exists. Without a tools branch
        # the signatures never reach the model and no sampler setting can produce
        # a native call.
        for name in EXPECTED:
            with self.subTest(template=name):
                out = render(name, messages=[{"role": "user", "content": "hi"}],
                             tools=TOOLS, add_generation_prompt=True)
                self.assertIn("lookup_ticket", out)

    def test_both_templates_keep_every_turn(self):
        for name in EXPECTED:
            with self.subTest(template=name):
                out = render(name, messages=CONVERSATION, tools=TOOLS,
                             add_generation_prompt=True)
                self.assertIn("Look up ticket 4172.", out)
                self.assertIn("Summarise it.", out)

    def test_both_templates_render_a_tool_result(self):
        # A model that can call a tool but never sees the result cannot finish
        # the round trip; gate_native_tool_use.py checks exactly that path.
        for name in EXPECTED:
            with self.subTest(template=name):
                out = render(name, messages=CONVERSATION, tools=TOOLS,
                             add_generation_prompt=True)
                self.assertIn('{"status":"open"}', out)

    def test_both_templates_close_their_blocks(self):
        for name in EXPECTED:
            with self.subTest(template=name):
                out = render(name, messages=CONVERSATION, tools=TOOLS,
                             add_generation_prompt=True)
                self.assertEqual(out.count("<|im_start|>"), out.count("<|im_end|>") + 1,
                                 "every block but the trailing generation prompt "
                                 "must be closed")

    def test_string_content_passes_through_for_the_multimodal_path(self):
        # llama.cpp extracts images before templating and leaves a marker in the
        # text, so obliterated-vision depends on string content being emitted
        # verbatim rather than reformatted.
        out = render("qwen3.8-27b-tools.jinja",
                     messages=[{"role": "user", "content": "<__media__>Label?"}],
                     add_generation_prompt=True)
        self.assertIn("<__media__>Label?", out)

    def test_typed_image_content_emits_the_vision_tokens(self):
        out = render("qwen3.8-27b-tools.jinja", add_generation_prompt=True,
                     messages=[{"role": "user", "content": [
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                         {"type": "text", "text": "Label?"}]}])
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", out)


if __name__ == "__main__":
    unittest.main()
