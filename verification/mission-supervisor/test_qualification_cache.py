import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('qualification_cache', Path(__file__).with_name('qualification_cache.py'))
assert spec and spec.loader
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class QualificationReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = cache.Store(self.temp.name)

    def run_target(self, gate='router', model='a', key: str | None='key', force=False, passed=True):
        run = Mock(return_value={'pass': passed, 'model': model})
        result = cache.run_cached(gate, model, key, run, force, self.store)
        return result, run

    def test_pass_reused_without_execution(self):
        self.run_target()
        result, run = self.run_target()
        run.assert_not_called()
        self.assertTrue(result['qualification_reused'])

    def test_new_model_does_not_repeat_successful_model(self):
        self.run_target()
        self.run_target(model='b', passed=False)
        self.run_target()[1].assert_not_called()
        self.run_target(model='b')[1].assert_called_once()

    def test_force_is_targeted_and_failure_revokes_old_pass(self):
        self.run_target()
        self.run_target(model='b')
        self.run_target(force=True, passed=False)[1].assert_called_once()
        self.run_target(model='b')[1].assert_not_called()
        self.assertTrue(self.store.read('router', 'a')['last_pass']['pass'])
        self.run_target()[1].assert_called_once()

    def test_changed_key_requalifies(self):
        self.run_target()
        self.run_target(key='changed')[1].assert_called_once()

    def test_interrupted_attempt_is_not_passed(self):
        self.run_target()
        with self.assertRaises(RuntimeError):
            cache.run_cached('router', 'a', 'key', Mock(side_effect=RuntimeError()), True, self.store)
        self.run_target()[1].assert_called_once()

    def test_corrupt_or_oversized_receipt_is_not_reused(self):
        for raw in ('{', '[]', 'x' * (cache.MAX_RECORD + 1)):
            self.store.path('router', 'a').write_text(raw)
            self.assertIsNone(self.store.reuse('router', 'a', 'key'))

    def test_unknown_identity_never_reuses(self):
        self.run_target(key=None)
        self.run_target(key=None)[1].assert_called_once()

    def test_other_gate_independent(self):
        self.run_target()
        self.run_target(gate='repo')[1].assert_called_once()

    def test_preset_identity_ignores_unrelated_candidate(self):
        root = Path(self.temp.name)
        weights = root / 'weights.gguf'
        weights.write_bytes(b'weights')
        ini = root / 'llama-models.ini'
        ini.write_text(f'[*]\nctx-size=100\n[a]\nmodel={weights}\n')
        with patch.object(cache, 'ROOT', root):
            before = cache.preset_identity('a')
            with ini.open('a') as f:
                f.write('[new]\nmodel=/missing/new.gguf\n')
            self.assertEqual(before, cache.preset_identity('a'))
            weights.write_bytes(b'changed')
            self.assertNotEqual(before, cache.preset_identity('a'))

    def test_gate_runner_has_targeted_cli(self):
        gate = Path(__file__).parents[1] / 'router-functional/gate_router_models.py'
        self.assertIn('--model', gate.read_text())
        self.assertIn('--requalify', gate.read_text())

    def test_actual_router_dispatch_reuses_only_passed_models(self):
        import functools
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        source = Path(__file__).parents[1] / 'router-functional/gate_router_models.py'
        loader = importlib.util.spec_from_file_location('router_cache_fixture', source)
        assert loader and loader.loader
        gate = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(gate)
        calls = []
        def execute(model):
            calls.append(model)
            return {'model': model, 'pass': model == 'ridge', 'problems': []}
        with patch.object(gate, 'CHAT_MODELS', ['ridge', 'heretic']), \
                patch.object(gate, 'VISION_MODELS', []), \
                patch.object(gate, 'model_key', side_effect=lambda _g, m, *a, **kw: m), \
                patch.object(gate, 'run_cached', functools.partial(cache.run_cached, store=self.store)), \
                patch.object(gate, 'checks_for', return_value=()), \
                patch.object(gate, 'check_model', side_effect=execute), \
                patch.object(gate, 'write_atomic'), patch.dict('os.environ', {}, clear=True):
            self.assertEqual(1, gate.main([]))
            self.assertEqual(['ridge', 'heretic'], calls)
            calls.clear()
            self.assertEqual(1, gate.main([]))
            self.assertEqual(['heretic'], calls)
            calls.clear()
            self.assertEqual(0, gate.main(['--model', 'ridge', '--requalify']))
            self.assertEqual(['ridge'], calls)

    def test_real_supervisor_reuses_step_and_force_executes(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        import run_functional_mission as mission
        state = {'steps': {}, 'input_fingerprint': 'unrelated-new-fingerprint'}
        def run(name, command, state):
            state['steps'][name] = {'status': 'passed', 'exit_code': 0}
            return 0
        with patch.object(mission, 'HERE', Path(self.temp.name)), \
                patch.object(mission, 'qualification_key', return_value='stable-input'), \
                patch.object(mission, 'atomic_json'), patch.object(mission, 'log'), \
                patch.object(mission, 'run_step', side_effect=run) as execute:
            self.assertEqual(0, mission.execute_qualification('embeddings', [], state))
            self.assertTrue(mission.reuse_step('embeddings', [], state))
            self.assertFalse(mission.reuse_step('embeddings', [], state, True))
            execute.assert_called_once()

    def test_stability_mode_is_bounded_and_forces_each_selected_run(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        import run_functional_mission as mission
        original = mission.main
        with patch.object(mission, 'main', return_value=0) as child:
            self.assertEqual(0, original(['--gate', 'asr', '--stability-runs', '3']))
            self.assertEqual(3, child.call_count)
            self.assertEqual(['--gate', 'asr', '--requalify'], child.call_args.args[0])
        with patch.object(mission, 'main', return_value=1) as child:
            self.assertEqual(1, original(['--gate', 'asr', '--stability-runs=3']))
            child.assert_called_once()

    def test_legacy_migration_requires_complete_matching_provenance(self):
        source = Path(self.temp.name) / 'gate.py'
        source.write_text('gate fixture')
        command = ['python3', str(source)]
        step = {'status': 'passed', 'exit_code': 0, 'command': command,
                'input_fingerprint': 'old', 'started_at': '2099-01-01T00:00:00+00:00',
                'finished_at': '2099-01-01T00:01:00+00:00'}
        state = {'steps': {'gate': step}, 'input_fingerprint': 'old',
                 'kernel_build_signature': Path('/proc/version').read_text().strip()}
        inputs = {'source': [cache.file_identity(source)]}
        with patch.object(cache, 'step_key', return_value=inputs):
            step['exit_code'] = 1
            self.assertFalse(cache.adopt_legacy_step('gate', command, state, self.store))
            step['exit_code'] = 0
            self.assertTrue(cache.adopt_legacy_step('gate', command, state, self.store))
            self.assertTrue(self.store.reuse('gate', 'gate', cache.digest(inputs))['legacy_migration'])
            self.assertFalse(cache.adopt_legacy_step('gate', command, state, self.store))

    def test_legacy_changed_inputs_do_not_migrate(self):
        source = Path(self.temp.name) / 'gate.py'
        source.write_text('new source')
        command = ['python3', str(source)]
        state = {'steps': {'gate': {'status': 'passed', 'exit_code': 0, 'command': command,
                'input_fingerprint': 'old', 'started_at': '2000-01-01T00:00:00+00:00',
                'finished_at': '2000-01-01T00:01:00+00:00'}}, 'input_fingerprint': 'old',
                 'kernel_build_signature': Path('/proc/version').read_text().strip()}
        with patch.object(cache, 'step_key', return_value={'source': [cache.file_identity(source)]}):
            self.assertFalse(cache.adopt_legacy_step('gate', command, state, self.store))
