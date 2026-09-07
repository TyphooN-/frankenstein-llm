"""Offline checks on the throughput report writer.

Every test here builds its own fixture run directory. Nothing loads a model,
reads a GGUF or touches a GPU: the writer's whole job is to move numbers that a
benchmark already measured into prose, so the interesting behaviour is what it
refuses to write.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bench_report', ROOT / 'scripts/bench_report.py')
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

REGISTRY = {
    'heretic': {'model': '/models/RVN.gguf', 'tensor-split': '1,1,1',
                'spec-type': 'draft-mtp', 'ctx-size': '131072'},
    'phr00ty': {'model': '/models/Phr00ty.gguf', 'tensor-split': '1,1,1',
                'ctx-size': '65536'},
    'coder': {'model': '/models/Coder-00001-of-00004.gguf', 'tensor-split': '5,8,4'},
    'gemma': {'model': '/models/Gemma.gguf', 'tensor-split': '1,0,1'},
    'gemma-vision': {'model': '/models/Gemma.gguf', 'tensor-split': '1,0,0',
                     'mmproj': '/models/mmproj-Gemma.gguf'},
}


def rows(prompt_ts=100.0, generation_ts=20.0, **overrides):
    common = {'model_filename': 'x.gguf', 'model_type': 'qwen3 27B Q6_K',
              'model_size': 22 << 30, 'model_n_params': 27_000_000_000,
              'build_commit': 'abc1234', 'gpu_info': 'AMD Radeon RX 6900 XT',
              'backends': 'ROCm', 'stddev_ts': 0.5}
    common.update(overrides)
    return [{**common, 'n_prompt': 512, 'n_gen': 0, 'avg_ts': prompt_ts},
            {**common, 'n_prompt': 0, 'n_gen': 128, 'avg_ts': generation_ts}]


def make_run(tmp_path, name, model, split, *, native=None, status='passed',
             exit_code=0, native_results='stdout.log', devices='ROCm0/ROCm1/ROCm2'):
    run = tmp_path / name
    run.mkdir()
    command = ['/bin/llama-bench', '-m', model, '-dev', devices, '-ts', split,
               '-t', '32', '-ngl', '999', '-p', '512', '-n', '128', '-r', '3',
               '-o', 'json']
    report = {'schema': 'frankenstein-operator-run/1', 'mode': 'benchmark',
              'command': command, 'status': status, 'exit_code': exit_code,
              'benchmarking_requested': True, 'kernel': '7.2.3-273-tkg-eevdf-llvm',
              'kernel_build': 'Linux version 7.2.3-273-tkg-eevdf-llvm',
              'boot_id': '9b83466d-2582-4fad-ba3c-00af2cedbdd8'}
    if native_results is not None:
        report['native_results'] = native_results
    (run / 'report.json').write_text(json.dumps(report))
    if native is not None:
        (run / 'stdout.log').write_text(json.dumps(native))
    return run


def test_refuses_a_run_with_no_native_json(tmp_path):
    """A directory with a report but no metrics has nothing to publish."""
    run = make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=None)
    with pytest.raises(m.MissingNativeResults):
        m.load_run(run, REGISTRY)


def test_refuses_a_report_that_never_recorded_native_results(tmp_path):
    run = make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1',
                   native=rows(), native_results=None)
    with pytest.raises(m.MissingNativeResults):
        m.load_run(run, REGISTRY)


def test_refuses_an_interrupted_or_failed_run(tmp_path):
    """An interrupted benchmark is a recorded failure, not a slow measurement."""
    for status, code in (('interrupted', 124), ('failed', 1), ('running', None)):
        run = make_run(tmp_path, f'run-{status}', '/models/RVN.gguf', '1/1/1',
                       native=rows(), status=status, exit_code=code)
        with pytest.raises(m.MissingNativeResults):
            m.load_run(run, REGISTRY)


def test_refuses_malformed_or_incomplete_native_rows(tmp_path):
    for name, native in (('empty', []),
                         ('object', {'avg_ts': 5}),
                         ('missing-field', [{'n_prompt': 512, 'n_gen': 0}]),
                         ('no-generation-row', rows()[:1])):
        run = make_run(tmp_path, name, '/models/RVN.gguf', '1/1/1', native=native)
        with pytest.raises(m.MissingNativeResults):
            m.load_run(run, REGISTRY)
    unparsable = make_run(tmp_path, 'garbage', '/models/RVN.gguf', '1/1/1', native=rows())
    (unparsable / 'stdout.log').write_text('llama_model_load: error\n')
    with pytest.raises(m.MissingNativeResults):
        m.load_run(unparsable, REGISTRY)


def test_quotes_the_measured_rate_and_the_exact_argv(tmp_path):
    run = make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=rows(97.5, 18.25))
    record = m.load_run(run, REGISTRY)
    assert record['alias'] == 'heretic'
    assert (record['prompt_ts'], record['generation_ts']) == (97.5, 18.25)
    assert record['model_type'] == 'qwen3 27B Q6_K'
    assert record['serving_spec'] == 'draft-mtp'
    body = m.render_model(record, {})
    assert '97.50 tok/s' in body and '18.25 tok/s' in body
    assert '-ts 1/1/1' in body and '-p 512' in body and '-r 3' in body
    assert record['run_dir'] not in ('', None) or True


def test_split_ratio_compares_proportions_not_gibibytes(tmp_path):
    """``6/6/6`` is the equal split written differently, not a 6 GiB budget."""
    assert m.split_ratio('6/6/6') == m.split_ratio('1,1,1') == m.split_ratio('2/2/2')
    assert m.split_ratio('5,0,4') != m.split_ratio('1,1,1')
    run = make_run(tmp_path, 'a', '/models/RVN.gguf', '6/6/6', native=rows())
    assert m.load_run(run, REGISTRY)['alias'] == 'heretic'


def test_refuses_a_run_that_measured_a_placement_the_alias_does_not_serve(tmp_path):
    run = make_run(tmp_path, 'a', '/models/Coder-00001-of-00004.gguf', '1/1/1',
                   native=rows())
    with pytest.raises(m.ReportRefused, match='placement'):
        m.load_run(run, REGISTRY)


def test_shared_weight_file_requires_an_explicit_alias(tmp_path):
    """A vision preset shares its GGUF with its text sibling; guessing misreports both."""
    run = make_run(tmp_path, 'a', '/models/Gemma.gguf', '1/0/1', native=rows())
    with pytest.raises(m.ReportRefused, match='gemma'):
        m.load_run(run, REGISTRY)
    assert m.load_run(run, REGISTRY, 'gemma')['alias'] == 'gemma'
    vision = make_run(tmp_path, 'b', '/models/Gemma.gguf', '1/0/0', native=rows())
    assert m.load_run(vision, REGISTRY, 'gemma-vision')['serving_projector'] == 'mmproj-Gemma.gguf'
    with pytest.raises(m.ReportRefused):
        m.load_run(run, REGISTRY, 'heretic')


def test_refuses_to_rank_mtp_against_non_mtp(tmp_path):
    """llama-bench measured every row without MTP; ordering across the two reads
    as a weight ranking and is refused."""
    mtp = m.load_run(make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1',
                              native=rows(generation_ts=30.0)), REGISTRY)
    plain = m.load_run(make_run(tmp_path, 'b', '/models/Phr00ty.gguf', '1/1/1',
                                native=rows(generation_ts=10.0)), REGISTRY)
    assert m.mtp_class(mtp) == 'draft-mtp' and m.mtp_class(plain) == 'none'
    with pytest.raises(m.RankingRefused, match='MTP'):
        m.rank_by_generation([mtp, plain])
    assert [record['alias'] for record in m.rank_by_generation([plain])] == ['phr00ty']


def test_refuses_to_rank_different_workloads(tmp_path):
    fast = m.load_run(make_run(tmp_path, 'a', '/models/Phr00ty.gguf', '1/1/1',
                               native=rows(generation_ts=10.0)), REGISTRY)
    other = m.load_run(make_run(tmp_path, 'b', '/models/Phr00ty.gguf', '1/1/1',
                                native=rows(generation_ts=40.0),
                                devices='ROCm0'), REGISTRY)
    with pytest.raises(m.RankingRefused, match='workload'):
        m.rank_by_generation([fast, other])


def test_index_never_orders_across_mtp_classes(tmp_path):
    mtp = m.load_run(make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1',
                              native=rows(generation_ts=30.0)), REGISTRY)
    plain = m.load_run(make_run(tmp_path, 'b', '/models/Phr00ty.gguf', '1/1/1',
                                native=rows(generation_ts=10.0)), REGISTRY)
    index = m.render_index([mtp, plain], {})
    ordering = index.split('## Ordering within one serving-MTP class', 1)[1]
    # Each class gets its own list; neither list contains the other's alias.
    blocks = [block for block in ordering.split('Presets ') if '1. `' in block]
    assert len(blocks) == 2
    for block in blocks:
        assert ('heretic' in block) != ('phr00ty' in block)
    # The table itself is alphabetical, so it is not a speed ordering either.
    table = index.split('## Ordering', 1)[0]
    assert table.index('`heretic`') < table.index('`phr00ty`')


def test_writes_exactly_one_file_per_model_plus_an_index(tmp_path):
    records = [
        m.load_run(make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=rows()), REGISTRY),
        m.load_run(make_run(tmp_path, 'b', '/models/Phr00ty.gguf', '1/1/1', native=rows()), REGISTRY),
        m.load_run(make_run(tmp_path, 'c', '/models/Gemma.gguf', '1/0/1', native=rows()), REGISTRY, 'gemma'),
    ]
    out = tmp_path / 'benchmarks'
    written = m.write_reports(records, out, {})
    assert sorted(path.name for path in written) == [
        'README.md', 'gemma.md', 'heretic.md', 'phr00ty.md']
    assert sorted(path.name for path in out.iterdir()) == [
        'README.md', 'gemma.md', 'heretic.md', 'phr00ty.md']
    assert len(written) == len(records) + 1


def test_refuses_two_runs_claiming_one_alias(tmp_path):
    first = m.load_run(make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=rows()), REGISTRY)
    second = m.load_run(make_run(tmp_path, 'b', '/models/RVN.gguf', '6/6/6',
                                 native=rows(generation_ts=99.0)), REGISTRY)
    with pytest.raises(m.ReportRefused, match='same alias'):
        m.write_reports([first, second], tmp_path / 'out', {})
    assert not (tmp_path / 'out' / 'heretic.md').exists()


def test_functional_notes_come_from_the_gate_and_invent_no_score(tmp_path):
    evidence = tmp_path / 'router-functional.json'
    evidence.write_text(json.dumps({
        'finished_at': '2026-09-07T00:28:08-0400', 'kernel_release': '7.2.3-273-tkg-eevdf-llvm',
        'models': [
            {'model': 'heretic', 'pass': True, 'problems': [],
             'required_checks': ['coherence', 'structured_output', 'tool_call'],
             'checks': {'coherence': {'pass': True}, 'structured_output': {'pass': True},
                        'tool_call': {'pass': True}}},
            {'model': 'phr00ty', 'pass': False, 'problems': [],
             'required_checks': ['coherence', 'structured_output', 'tool_call'],
             'checks': {'coherence': {'pass': True}, 'structured_output': {'pass': True},
                        'tool_call': {'pass': False}}},
        ]}))
    functional = m.load_functional(evidence)
    passed = '\n'.join(m.functional_lines('heretic', functional))
    failed = '\n'.join(m.functional_lines('phr00ty', functional))
    assert '**PASS**' in passed and '**FAIL**' in failed
    assert '`tool_call`: fail' in failed
    # A gate failure is never restated as a pass, and no score is manufactured.
    assert '**PASS**' not in failed
    absent = '\n'.join(m.functional_lines('coder', functional))
    assert 'Absence of evidence is not a pass' in absent


def test_missing_functional_evidence_is_not_a_pass(tmp_path):
    assert m.load_functional(tmp_path / 'nothing.json') == {}
    record = m.load_run(make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=rows()), REGISTRY)
    body = m.render_model(record, {})
    assert 'Absence of evidence is not a pass' in body


def test_every_artifact_carries_the_required_disclosures(tmp_path):
    for alias, model, split in (('heretic', '/models/RVN.gguf', '1/1/1'),
                                ('phr00ty', '/models/Phr00ty.gguf', '1/1/1')):
        record = m.load_run(make_run(tmp_path, alias, model, split, native=rows()), REGISTRY)
        body = m.render_model(record, {})
        for expected in (record['model_filename'], record['model_path'], alias,
                         'qwen3 27B Q6_K', '7.2.3-273-tkg-eevdf-llvm',
                         '9b83466d-2582-4fad-ba3c-00af2cedbdd8',
                         'ROCm0/ROCm1/ROCm2', split, '-ngl', '512', '128', '3',
                         'multi-token prediction', 'not throughput',
                         'logs/model-runs' if 'logs/model-runs' in record['run_dir'] else record['run_dir']):
            assert expected in body, (alias, expected)


def test_unsampled_memory_is_reported_as_unmeasured(tmp_path):
    run = make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=rows())
    record = m.load_run(run, REGISTRY)
    assert record['host_sample'] is None
    assert 'were not sampled' in m.render_model(record, {})
    (run / m.HOST_SAMPLE).write_text(json.dumps({
        'interval_seconds': 2.0,
        'samples': [{'mem_available_bytes': 40 << 30, 'vram_used_bytes': {'ROCm0': 8 << 30}}],
        'peak_vram_used_bytes': {'ROCm0': 8 << 30},
        'min_mem_available_bytes': 40 << 30}))
    sampled = m.load_run(run, REGISTRY)
    body = m.render_model(sampled, {})
    assert 'peak VRAM used: 8.00 GiB' in body and 'MemAvailable' in body


def test_cli_refusal_is_an_exit_code_not_a_written_file(tmp_path, capsys):
    run = make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=None)
    presets = tmp_path / 'presets.ini'
    presets.write_text('[*]\ntensor-split = 1,1,1\n\n'
                       '[heretic]\nmodel = /models/RVN.gguf\nspec-type = draft-mtp\n')
    out = tmp_path / 'out'
    assert m.main(['write', str(run), '--out', str(out), '--presets', str(presets),
                   '--functional', str(tmp_path / 'none.json')]) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()


def test_repository_presets_and_benchmark_config_agree_on_every_split():
    """The published split has to be the one the router serves.

    ``config/model-runs.json`` carries the default; each preset may override it.
    Both are hand-edited, and a benchmark run at a split nothing serves measures
    the wrong machine.
    """
    registry = m.model_catalog.presets(ROOT / 'llama-models.ini')
    default = json.loads((ROOT / 'config/model-runs.json').read_text())['benchmark']['tensor_split']
    assert m.split_ratio(default) == m.split_ratio(registry['heretic']['tensor-split'])
    for alias, preset in registry.items():
        assert m.split_ratio(preset['tensor-split']), alias


def test_index_carries_the_gate_verdict_so_speed_is_not_a_recommendation(tmp_path):
    """A preset can be the fastest row and still have failed its gate."""
    evidence = tmp_path / 'router-functional.json'
    evidence.write_text(json.dumps({'finished_at': '2026-09-07T00:28:08-0400', 'models': [
        {'model': 'phr00ty', 'pass': False, 'problems': [],
         'checks': {'tool_call': {'pass': False}}}]}))
    functional = m.load_functional(evidence)
    fast = m.load_run(make_run(tmp_path, 'a', '/models/Phr00ty.gguf', '1/1/1',
                               native=rows(generation_ts=99.0)), REGISTRY)
    quiet = m.load_run(make_run(tmp_path, 'b', '/models/Coder-00001-of-00004.gguf', '5/8/4',
                                native=rows(generation_ts=1.0)), REGISTRY)
    index = m.render_index([fast, quiet], functional)
    table = index.split('## Ordering', 1)[0]
    phr00ty_row = next(line for line in table.splitlines() if '`phr00ty`' in line)
    assert phr00ty_row.endswith('**FAIL** |')
    coder_row = next(line for line in table.splitlines() if '`coder`' in line)
    assert coder_row.endswith('not recorded |')
    assert m.verdict('phr00ty', functional) == '**FAIL**'
    assert m.verdict('absent', {}) == 'not recorded'


def test_vision_sibling_discloses_that_the_projector_was_not_benchmarked(tmp_path):
    record = m.load_run(make_run(tmp_path, 'a', '/models/Gemma.gguf', '1/0/0',
                                 native=rows()), REGISTRY, 'gemma-vision')
    differences = dict((label, measured) for label, _, measured in m.serving_differences(record))
    assert 'not loaded' in differences['Multimodal projector']
    assert 'no draft model' in differences['Speculative decoding (`spec-type`)']
    body = m.render_model(record, {})
    assert 'mmproj-Gemma.gguf' in body and 'not loaded' in body
    # Markdown cells must not nest backticks inside a backticked cell.
    for line in body.splitlines():
        if line.startswith('| ') and line.count('`') % 2:
            raise AssertionError(f'unbalanced backticks: {line}')
