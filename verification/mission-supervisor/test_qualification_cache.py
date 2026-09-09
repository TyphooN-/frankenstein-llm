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
                patch.object(gate, 'model_key', side_effect=lambda _g, m, *a, **kw:
                             {'model': m} if kw.get('details') else cache.digest({'model': m})), \
                patch.object(gate, 'blocked_workloads', return_value=[]), \
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
        def run(name, command, state, before_start=None):
            if before_start:
                before_start()
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


class AdmissionVersusVerdictTests(QualificationReuseTests):
    """Refusing to start is not a verdict, and must not spend a receipt.

    ``run_cached`` publishes a ``running`` receipt to revoke reuse before a test
    begins, which is right for a test that begins. The router gate refused a
    confounded qualification from *inside* that wrapped call, so a busy host --
    a download queue re-verifying weights -- deleted a preset's pass and recorded
    a FAIL that was never a statement about the model. These pin the boundary:
    nothing began, nothing is spent.
    """

    def test_refusal_before_work_leaves_the_previous_receipt_untouched(self):
        # The inputs moved, so this receipt is not reusable and the gate would
        # dispatch. A busy host stops it before it does -- and the record of what
        # was last proven stays exactly as it was rather than becoming a FAIL
        # about a model nobody tested.
        self.run_target()
        receipt = self.store.read('router', 'a')
        run = Mock(return_value={'pass': True, 'model': 'a'})
        result = cache.run_cached('router', 'a', 'moved', run, False, self.store,
                                  admit=lambda: [{'pid': 7, 'reason': 'download-queue'}])
        run.assert_not_called()
        self.assertFalse(result['pass'])
        self.assertTrue(result['admission_refused'])
        self.assertEqual('blocked', result['outcome'])
        self.assertEqual([{'pid': 7, 'reason': 'download-queue'}], result['blocked_by'])
        self.assertEqual(receipt, self.store.read('router', 'a'))

    def test_refusal_under_force_does_not_revoke_the_pass_it_never_retested(self):
        # --requalify asks for a real retest. A retest that is refused admission
        # never ran, so the previous pass is still the last thing that was proven.
        self.run_target()
        run = Mock(return_value={'pass': True, 'model': 'a'})
        cache.run_cached('router', 'a', 'key', run, True, self.store,
                         admit=lambda: [{'pid': 7, 'reason': 'kernel-build'}])
        run.assert_not_called()
        self.assertIsNotNone(self.store.reuse('router', 'a', 'key'))

    def test_a_genuine_forced_retest_revokes_before_it_runs(self):
        self.run_target()
        seen = []

        def run():
            seen.append(self.store.reuse('router', 'a', 'key'))
            return {'pass': False, 'model': 'a'}

        cache.run_cached('router', 'a', 'key', run, True, self.store, admit=lambda: [])
        self.assertEqual([None], seen, 'a retest in flight must not expose the old pass')
        self.assertIsNone(self.store.reuse('router', 'a', 'key'))
        self.assertEqual('failed', self.store.read('router', 'a')['status'])

    def test_a_refusal_raised_inside_the_workload_rolls_the_receipt_back(self):
        # The race the outer check cannot close: admission passes, then a queue
        # starts before the first request. The gate still refuses, and because
        # no model was dispatched the receipt returns to exactly what it was.
        self.run_target()
        receipt = self.store.read('router', 'a')
        blocked = {'pass': False, 'model': 'a', 'outcome': 'blocked',
                   'admission_refused': True,
                   'problems': ['refusing confounded qualification']}
        result = cache.run_cached('router', 'a', 'key', lambda: blocked, True, self.store)
        self.assertFalse(result['pass'])
        self.assertEqual(receipt, self.store.read('router', 'a'))
        self.run_target()[1].assert_not_called()

    def test_a_refusal_with_no_previous_receipt_leaves_no_receipt_behind(self):
        blocked = {'pass': False, 'outcome': 'blocked', 'admission_refused': True}
        cache.run_cached('router', 'fresh', 'key', lambda: blocked, False, self.store)
        self.assertFalse(self.store.path('router', 'fresh').exists())

    def test_admission_is_only_consulted_when_a_test_would_actually_run(self):
        self.run_target()
        admit = Mock(return_value=[])
        cache.run_cached('router', 'a', 'key', Mock(), False, self.store, admit=admit)
        admit.assert_not_called()


class UnreadableIdentityTests(QualificationReuseTests):
    """A weight file that is not there yet cannot be a verdict about a model.

    The download queue quarantines a file whose SHA-256 does not match and
    re-downloads it into a ``.partial`` sibling, so a preset's weights are
    routinely absent for hours. ``preset_identity`` stats them, and the
    ``FileNotFoundError`` escaped both model gates -- exit 1, which every runner
    reads as "this model failed".
    """

    def preset(self, present=True):
        root = Path(self.temp.name)
        weights = root / 'weights.gguf'
        if present:
            weights.write_bytes(b'weights')
        (root / 'llama-models.ini').write_text(
            f'[*]\nctx-size=100\n[a]\nmodel={weights}\nmmproj={weights}\n')
        return patch.object(cache, 'ROOT', root)

    def test_preset_values_answers_without_reading_the_weights(self):
        with self.preset(present=False):
            self.assertEqual('100', cache.preset_values('a')['ctx-size'])
            with self.assertRaises(FileNotFoundError):
                cache.preset_identity('a')

    def test_preset_values_still_rejects_an_unknown_preset(self):
        with self.preset():
            with self.assertRaises(ValueError):
                cache.preset_values('not-configured')

    def test_preset_values_and_preset_identity_agree_when_the_weights_exist(self):
        with self.preset():
            self.assertEqual(cache.preset_values('a'), cache.preset_identity('a')['preset'])

    def test_an_unreadable_identity_is_blocked_rather_than_failed(self):
        result = cache.unreadable_identity('router-models', 'a',
                                           FileNotFoundError(2, 'No such file'))
        self.assertEqual('blocked', cache.outcome_of(result))
        self.assertIs(False, result['pass'])
        self.assertFalse(result['qualification_reused'])
        self.assertEqual({}, result['checks'])
        self.assertEqual('qualification-inputs-unreadable', result['blocked_by'][0]['reason'])
        self.assertIn('FileNotFoundError', result['blocked_by'][0]['error'])

    def test_the_refusal_record_stays_bounded(self):
        result = cache.unreadable_identity('g', 'm', OSError('x' * 5000))
        self.assertLessEqual(len(result['blocked_by'][0]['error']), 300)


class ReceiptLockContentionTests(QualificationReuseTests):
    """Losing the race for a receipt is an admission outcome, not a verdict.

    ``run_cached`` takes the receipt lock non-blocking so two qualifications
    cannot interleave writes to one record. The loser used to receive the raw
    ``BlockingIOError``, which left the gate to die on a traceback and exit 1 --
    the code every runner reads as "this model failed". Nothing ran, so nothing
    may be spent and nothing may be claimed.
    """

    def held(self, gate='router', model='a'):
        """Hold the same lock ``run_cached`` will try to take."""
        import fcntl
        self.store.root.mkdir(parents=True, exist_ok=True)
        handle = self.store.path(gate, model).with_suffix('.lock').open('a')
        self.addCleanup(handle.close)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle

    def test_a_locked_receipt_blocks_instead_of_raising(self):
        self.run_target()
        receipt = self.store.read('router', 'a')
        self.held()
        run = Mock(return_value={'pass': True, 'model': 'a'})
        result = cache.run_cached('router', 'a', 'moved', run, False, self.store)
        run.assert_not_called()
        self.assertEqual('blocked', cache.outcome_of(result))
        self.assertTrue(result['admission_refused'])
        self.assertIs(False, result['pass'])
        self.assertEqual('qualification-receipt-locked', result['blocked_by'][0]['reason'])
        self.assertEqual(receipt, self.store.read('router', 'a'),
                         'a lost race must not touch the receipt it could not open')

    def test_a_locked_receipt_under_force_keeps_the_pass_it_never_retested(self):
        self.run_target()
        self.held()
        cache.run_cached('router', 'a', 'key', Mock(), True, self.store)
        self.assertIsNotNone(self.store.reuse('router', 'a', 'key'))


class InconclusiveOutcomeTests(QualificationReuseTests):
    """A run whose inputs moved under it is not a model that failed."""

    def inconclusive(self, reason='qualification inputs changed during execution'):
        return cache.mark_inconclusive({'pass': True, 'model': 'a', 'problems': []}, reason)

    def test_marking_a_result_inconclusive_clears_its_pass(self):
        result = self.inconclusive()
        self.assertFalse(result['pass'])
        self.assertEqual('inconclusive', result['outcome'])
        self.assertIn('qualification inputs changed during execution', result['problems'])
        self.assertEqual('inconclusive', cache.outcome_of(result))

    def test_inconclusive_is_recorded_as_neither_passed_nor_failed(self):
        self.run_target()
        cache.run_cached('router', 'a', 'key', self.inconclusive, True, self.store)
        record = self.store.read('router', 'a')
        self.assertEqual('inconclusive', record['status'])
        self.assertTrue(record['last_pass']['pass'], 'the historical pass is still recorded')

    def test_an_inconclusive_retest_never_reuses_the_success_it_invalidated(self):
        self.run_target()
        cache.run_cached('router', 'a', 'key', self.inconclusive, True, self.store)
        self.assertIsNone(self.store.reuse('router', 'a', 'key'))
        self.run_target()[1].assert_called_once()

    def test_outcome_of_classifies_plain_results(self):
        self.assertEqual('passed', cache.outcome_of({'pass': True}))
        self.assertEqual('failed', cache.outcome_of({'pass': False}))
        self.assertEqual('blocked', cache.outcome_of({'pass': False, 'outcome': 'blocked'}))
        self.assertEqual('failed', cache.outcome_of(None))


def test_supervisor_preserves_pass_on_refusal_and_wait_failure(tmp_path, monkeypatch):
    import run_functional_mission as mission
    store = cache.Store(tmp_path / 'qualification-cache')
    store.publish('asr', 'asr', 'same', {'pass': True, 'step': {'status': 'passed'}})
    original = store.read('asr', 'asr')
    monkeypatch.setattr(mission, 'HERE', tmp_path)
    monkeypatch.setattr(mission, 'qualification_key', lambda *a: 'same')
    monkeypatch.setattr(mission, 'qualification_components', lambda *a: {'runtime': 'x'})
    monkeypatch.setattr(mission, 'atomic_json', lambda *a: None)
    state = {'steps': {}}
    def refused(name, command, state, before_start=None):
        assert before_start is not None
        before_start()
        state['steps'][name] = {'status': 'failed'}
        return cache.EXIT_ADMISSION_REFUSED
    monkeypatch.setattr(mission, 'run_step', refused)
    assert mission.execute_qualification('asr', [], state) == 75
    assert state['steps']['asr']['status'] == 'blocked'
    assert store.read('asr', 'asr') == original
    def wait_failed(*args, **kwargs):
        raise RuntimeError('host unavailable before admission')
    monkeypatch.setattr(mission, 'run_step', wait_failed)
    import pytest
    with pytest.raises(RuntimeError):
        mission.execute_qualification('asr', [], state)
    assert store.read('asr', 'asr') == original


def test_supervisor_interruption_is_inconclusive_not_a_model_failure(tmp_path, monkeypatch):
    """An operator stop is forwarded to the gate, which then exits non-zero.

    The mission state already called that ``interrupted``. The receipt called it
    ``failed``, so the two records of one event disagreed and the durable one
    blamed the model for being stopped.
    """
    import run_functional_mission as mission
    monkeypatch.setattr(mission, 'HERE', tmp_path)
    monkeypatch.setattr(mission, 'qualification_key', lambda *a: 'same')
    monkeypatch.setattr(mission, 'qualification_components', lambda *a: {'runtime': 'x'})
    monkeypatch.setattr(mission, 'atomic_json', lambda *a: None)
    monkeypatch.setattr(mission, 'stop_signal', 15)

    def stopped(name, command, state, before_start=None):
        before_start()
        state['steps'][name] = {'status': 'failed', 'exit_code': 143}
        return 143

    monkeypatch.setattr(mission, 'run_step', stopped)
    state = {'steps': {}}
    assert mission.execute_qualification('asr', [], state) == 143
    assert state['steps']['asr']['status'] == 'interrupted'
    store = cache.Store(tmp_path / 'qualification-cache')
    assert store.read('asr', 'asr')['status'] == 'inconclusive'
    assert store.reuse('asr', 'asr', 'same') is None


def test_supervisor_interruption_does_not_demote_a_gate_that_finished(tmp_path, monkeypatch):
    import run_functional_mission as mission
    monkeypatch.setattr(mission, 'HERE', tmp_path)
    monkeypatch.setattr(mission, 'qualification_key', lambda *a: 'same')
    monkeypatch.setattr(mission, 'qualification_components', lambda *a: {'runtime': 'x'})
    monkeypatch.setattr(mission, 'atomic_json', lambda *a: None)
    monkeypatch.setattr(mission, 'stop_signal', 15)

    def finished(name, command, state, before_start=None):
        before_start()
        state['steps'][name] = {'status': 'passed', 'exit_code': 0}
        return 0

    monkeypatch.setattr(mission, 'run_step', finished)
    assert mission.execute_qualification('asr', [], {'steps': {}}) == 0
    store = cache.Store(tmp_path / 'qualification-cache')
    assert store.read('asr', 'asr')['status'] == 'passed'
    assert store.reuse('asr', 'asr', 'same') is not None


def test_supervisor_changed_inputs_are_inconclusive(tmp_path, monkeypatch):
    import run_functional_mission as mission
    monkeypatch.setattr(mission, 'HERE', tmp_path)
    keys = iter(['before', 'after'])
    monkeypatch.setattr(mission, 'qualification_key', lambda *a: next(keys))
    monkeypatch.setattr(mission, 'qualification_components', lambda *a: {'runtime': 'old'})
    monkeypatch.setattr(mission, 'atomic_json', lambda *a: None)
    monkeypatch.setattr(mission, 'stop_signal', None)
    def passed(name, command, state, before_start=None):
        assert before_start is not None
        before_start()
        state['steps'][name] = {'status': 'passed', 'exit_code': 0}
        return 0
    monkeypatch.setattr(mission, 'run_step', passed)
    state = {'steps': {}}
    assert mission.execute_qualification('asr', [], state) == 76
    assert state['steps']['asr']['status'] == 'inconclusive'
    store = cache.Store(tmp_path / 'qualification-cache')
    assert store.read('asr', 'asr')['status'] == 'inconclusive'
    assert store.reuse('asr', 'asr', 'before') is None


def test_corrupt_pass_receipt_is_not_reusable(tmp_path):
    store = cache.Store(tmp_path)
    store.publish('g', 'm', 'key', {'pass': True})
    record = store.read('g', 'm')
    del record['updated_at']
    store._write('g', 'm', record)
    assert store.reuse('g', 'm', 'key') is None
    assert store.explain('g', 'm', 'key')['reason'] == 'invalid-receipt'


class ReceiptProvenanceTests(QualificationReuseTests):
    """A receipt that cannot say which dependency moved cannot be diagnosed."""

    COMPONENTS = {'source': ['a'], 'artifacts': ['b'], 'runtime': ['c']}

    def publish_with_components(self, components):
        digested = cache.key_components(components)
        cache.run_cached('gate', 'gate', cache.digest(components),
                         lambda: {'pass': True}, False, self.store,
                         components=digested)
        return digested

    def test_the_receipt_stores_a_bounded_digest_per_component(self):
        digested = self.publish_with_components(self.COMPONENTS)
        stored = self.store.read('gate', 'gate')['components']
        self.assertEqual(digested, stored)
        self.assertEqual({'source', 'artifacts', 'runtime'}, set(stored))
        for value in stored.values():
            self.assertRegex(value, r'^[0-9a-f]{64}$')

    def test_explain_names_only_the_dependency_that_changed(self):
        self.publish_with_components(self.COMPONENTS)
        moved = dict(self.COMPONENTS, artifacts=['b2'])
        reason = self.store.explain('gate', 'gate', cache.digest(moved),
                                    cache.key_components(moved))
        self.assertEqual('inputs-changed', reason['reason'])
        self.assertEqual(['artifacts'], reason['changed_components'])

    def test_explain_reports_a_reusable_receipt(self):
        digested = self.publish_with_components(self.COMPONENTS)
        reason = self.store.explain('gate', 'gate', cache.digest(self.COMPONENTS), digested)
        self.assertEqual('reusable', reason['reason'])
        self.assertEqual([], reason['changed_components'])

    def test_a_receipt_predating_component_provenance_is_explained_not_trusted(self):
        # Old receipts carry only the composite key. A matching key is still a
        # match, but a mismatch cannot be attributed, and it must not be excused.
        self.run_target()
        record = self.store.read('router', 'a')
        self.assertIsNone(record.get('components'))
        reason = self.store.explain('router', 'a', 'moved', {'source': 'x'})
        self.assertEqual('inputs-changed', reason['reason'])
        self.assertIsNone(reason['changed_components'])
        self.assertIn('predates', reason['component_provenance'])
        self.assertIsNone(self.store.reuse('router', 'a', 'moved'))

    def test_explain_distinguishes_absent_unidentifiable_and_unpassed(self):
        self.assertEqual('no-receipt', self.store.explain('gate', 'missing', 'k', {})['reason'])
        self.assertEqual('no-identity', self.store.explain('gate', 'missing', None, {})['reason'])
        self.run_target(passed=False)
        self.assertEqual('failed', self.store.explain('router', 'a', 'key', {})['reason'])
        self.assertEqual('forced-retest',
                         self.store.explain('router', 'a', 'key', {}, force=True)['reason'])

    def test_key_components_hashes_each_top_level_component_once(self):
        components = cache.key_components({'source': [1, 2], 'runtime': {'k': 'v'}})
        self.assertEqual({'source', 'runtime'}, set(components))
        self.assertEqual(cache.digest([1, 2]), components['source'])
        self.assertNotEqual(components['source'], components['runtime'])

    def test_model_key_can_report_its_own_components(self):
        root = Path(self.temp.name)
        weights = root / 'weights.gguf'
        weights.write_bytes(b'weights')
        (root / 'llama-models.ini').write_text(f'[a]\nmodel={weights}\n')
        with patch.object(cache, 'ROOT', root), \
                patch.object(cache, 'runtime_identity', return_value={'kernel': 'fixture'}):
            details = cache.model_key('router-models', 'a', [weights], details=True)
            self.assertEqual({'schema', 'gate', 'model', 'preset', 'sources',
                              'runtime', 'extra', 'cache_contract'}, set(details))
            self.assertEqual(cache.digest(details),
                             cache.model_key('router-models', 'a', [weights]))
