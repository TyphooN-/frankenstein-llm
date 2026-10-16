"""Offline contracts. Fake only the host context and the model transport."""
import importlib.util
import json
from pathlib import Path
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('local_worker_under_test', ROOT / '__init__.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self):
        self.tools = {}
        self.prompts = {}

    def register_tool(self, **kwargs):
        self.tools[kwargs['name']] = kwargs
        return object()

    def register_system_prompt_section(self, name, content, **kwargs):
        self.prompts[name] = content
        return object()


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.context = Context()
        register = getattr(self.module, 'register', None)
        if register is not None:
            register(self.context)

    def test_supervisor_receives_worker_tool_and_automatic_guidance(self):
        self.assertIn('local_worker', self.context.tools)
        prompt = '\n'.join(self.context.prompts.values()).lower()
        self.assertIn('automatically', prompt)
        self.assertIn('untrusted', prompt)
        self.assertNotIn('override', self.context.tools['local_worker'])

    def test_registered_handler_uses_configured_tool_free_transport(self):
        calls = []
        transport = ModuleType('agent.qualified_local_helper')
        def prepare(task, *, messages):
            calls.append((task, messages))
            return SimpleNamespace(model='replaceable-model', provider='llama.cpp')
        setattr(transport, 'prepare_local_call', prepare)
        setattr(transport, 'call_local', lambda call: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"items": 3}'))]))
        with patch.dict('sys.modules', {'agent': ModuleType('agent'), 'agent.qualified_local_helper': transport}):
            result = json.loads(self.context.tools['local_worker']['handler'](
                {'operation': 'extract', 'text': 'There are three items.', 'focus': 'Return JSON.'}))
        self.assertTrue(result['ok'])
        self.assertEqual(result['model'], 'replaceable-model')
        self.assertEqual(result['output'], '{"items": 3}')
        self.assertEqual(result['authority'], 'untrusted_worker_text')
        self.assertEqual(calls[0][0], 'local_helper')
        self.assertEqual([m['role'] for m in calls[0][1]], ['system', 'user'])
        self.assertEqual(calls[0][1][1]['content'], 'There are three items.')

    def test_invalid_wire_options_and_unbounded_text_never_dispatch(self):
        calls = []
        transport = ModuleType('agent.qualified_local_helper')
        def prepare(task, **kwargs):
            calls.append(task)
            return SimpleNamespace(model='replaceable-model')
        setattr(transport, 'prepare_local_call', prepare)
        setattr(transport, 'call_local', lambda call: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='unexpected'))]))
        base = {'operation': 'extract', 'text': 'valid source'}
        invalid = [dict(base, tools=[]), dict(base, api_key='fixture-only'),
            dict(base, model='cloud'), dict(base, base_url='https://example.invalid'),
            dict(base, text=''), dict(base, text='   '), dict(base, text='x' * 12001),
            dict(base, text=['not text']), dict(base, focus=['not text']),
            dict(base, focus='x' * 1001)]
        with patch.dict('sys.modules', {'agent': ModuleType('agent'), 'agent.qualified_local_helper': transport}):
            for arguments in invalid:
                with self.subTest(fields=sorted(arguments), text_type=type(arguments['text']).__name__):
                    result = json.loads(self.context.tools['local_worker']['handler'](arguments))
                    self.assertFalse(result['ok'])
                    self.assertEqual(result['error'], 'invalid_request')
        self.assertEqual(calls, [])

    def test_missing_policy_and_transport_failure_return_sanitized_refusals(self):
        for failure in ('policy', 'prepare', 'transport'):
            with self.subTest(failure=failure):
                transport = ModuleType('agent.qualified_local_helper')
                def prepare(*args, **kwargs):
                    if failure == 'prepare':
                        raise RuntimeError('private-fixture-marker')
                    return None if failure == 'policy' else SimpleNamespace(model='local')
                def send(call):
                    raise RuntimeError('private-fixture-marker')
                setattr(transport, 'prepare_local_call', prepare)
                setattr(transport, 'call_local', send)
                with patch.dict('sys.modules', {'agent': ModuleType('agent'), 'agent.qualified_local_helper': transport}):
                    try:
                        raw = self.module.handle({'operation': 'summarize', 'text': 'supplied source'})
                    except Exception as exc:
                        self.fail('worker boundary must return a sanitized refusal, not raise ' + type(exc).__name__)
                result = json.loads(raw)
                self.assertFalse(result['ok'])
                self.assertEqual(result['error'], 'local_worker_unavailable_or_refused')
                self.assertNotIn('private-fixture-marker', raw)

    def test_invalid_argument_shapes_refuse_without_importing_transport(self):
        for args in (None, [], 'text', {}, {'operation': 'execute', 'text': 'source'},
                     {'operation': ['extract'], 'text': 'source'}):
            with self.subTest(shape=type(args).__name__):
                self.assertEqual(json.loads(self.module.handle(args))['error'], 'invalid_request')

    def test_availability_checks_policy_without_inference(self):
        transport = ModuleType('agent.qualified_local_helper')
        setattr(transport, 'prepare_local_call', lambda *a, **k: SimpleNamespace(model='local'))
        setattr(transport, 'call_local', lambda *a: self.fail('availability must not run inference'))
        with patch.dict('sys.modules', {'agent': ModuleType('agent'), 'agent.qualified_local_helper': transport}):
            self.assertTrue(self.module.available())
            setattr(transport, 'prepare_local_call', lambda *a, **k: None)
            self.assertFalse(self.module.available())


if __name__ == '__main__':
    unittest.main()
