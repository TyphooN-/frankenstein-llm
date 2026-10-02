import configparser
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'config/chat-templates/cyber-frost-3.8.jinja'

class CyberFrostTemplateTests(unittest.TestCase):
    def test_preserves_request_thinking_control_and_tool_format(self):
        self.assertTrue(TEMPLATE.is_file(), 'Repository template asset is missing')
        text = TEMPLATE.read_text()
        self.assertNotRegex(text, r'set\s+enable_thinking\s*=\s*true')
        self.assertIn('enable_thinking is defined and enable_thinking is false', text)
        self.assertIn('<think>\\n\\n</think>\\n\\n', text)
        self.assertIn('message.tool_calls', text)
        self.assertIn('<tool_call>', text)
        self.assertIn('<tool_response>', text)

    def test_preset_uses_repository_template_without_changing_full_profile(self):
        config = configparser.ConfigParser(interpolation=None)
        config.read(ROOT / 'llama-models.ini')
        preset = config['cyber-frost-38']
        self.assertEqual(preset['chat-template-file'], 'config/chat-templates/cyber-frost-3.8.jinja')
        self.assertEqual(preset['ctx-size'], '32768')
        self.assertEqual(preset['tensor-split'], '13,27,10')
        self.assertEqual(preset['override-tensor'], 'per_layer_token_embd=CPU')
        self.assertEqual(preset['cache-type-k'], 'f16')
        self.assertEqual(preset['cache-type-v'], 'f16')

if __name__ == '__main__':
    unittest.main()
