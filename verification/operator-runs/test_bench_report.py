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


@pytest.mark.parametrize('inside_checkout', [False, True],
                         ids=['absolute-citation', 'checkout-relative-citation'])
def test_quotes_the_measured_rate_and_the_exact_argv(tmp_path, monkeypatch, inside_checkout):
    run = make_run(tmp_path, 'a', '/models/RVN.gguf', '1/1/1', native=rows(97.5, 18.25))
    if inside_checkout:
        monkeypatch.setattr(m, 'ROOT', tmp_path)
    expected_run_dir = 'a' if inside_checkout else str(run.resolve())
    record = m.load_run(run, REGISTRY)
    assert record['alias'] == 'heretic'
    assert (record['prompt_ts'], record['generation_ts']) == (97.5, 18.25)
    assert record['model_type'] == 'qwen3 27B Q6_K'
    assert record['serving_spec'] == 'draft-mtp'
    body = m.render_model(record, {})
    assert '97.50 tok/s' in body and '18.25 tok/s' in body
    assert '-ts 1/1/1' in body and '-p 512' in body and '-r 3' in body
    assert record['run_dir'] == expected_run_dir
    assert f'| Native report directory | `{expected_run_dir}` |' in body
    assert f'native JSON in `{expected_run_dir}/stdout.log`' in body


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


def report_write_argv(tmp_path, run, out):
    presets = tmp_path / 'presets.ini'
    presets.write_text('[*]\ntensor-split = 1,1,1\n\n'
                       '[heretic]\nmodel = /models/RVN.gguf\nspec-type = draft-mtp\n')
    return ['write', str(run), '--out', str(out), '--presets', str(presets),
            '--functional', str(tmp_path / 'none.json')]


@pytest.mark.parametrize('row_index', [0, 1], ids=['prompt', 'generation'])
@pytest.mark.parametrize(('field', 'value'), [
    pytest.param(field, value, id=f'{field}-{label}')
    for field in ('avg_ts', 'stddev_ts')
    for label, value in (
        ('bool', True), ('string', '18.25'), ('array', []), ('object', {}),
        ('nan', float('nan')), ('infinity', float('inf')),
        ('negative-infinity', -float('inf')), ('negative', -0.25),
        ('unrepresentable-integer', 10 ** 400),
    )
] + [pytest.param('avg_ts', None, id='avg_ts-null')] + [
    pytest.param(field, value, id=f'{field}-{label}')
    for field in ('n_prompt', 'n_gen')
    for label, value in (
        ('null', None), ('bool', True), ('string', '512'), ('array', []),
        ('object', {}), ('negative', -1), ('fraction', 0.5),
        ('float', 512.0), ('nan', float('nan')), ('infinity', float('inf')),
    )
])
def test_cli_refuses_invalid_measured_fields_before_publication(
        tmp_path, capsys, row_index, field, value):
    native = rows(97.5, 18.25)
    native[row_index][field] = value
    run = make_run(tmp_path, 'invalid-measurement', '/models/RVN.gguf', '1/1/1',
                   native=native)
    before = {name: (run / name).read_bytes() for name in ('report.json', 'stdout.log')}
    out = tmp_path / 'out'
    assert m.main(report_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()
    assert {name: (run / name).read_bytes() for name in before} == before


@pytest.mark.parametrize(('measured_rate', 'deviation'), [
    pytest.param(0, 0, id='zero-integers'),
    pytest.param(0.0, 0.0, id='zero-floats'),
    pytest.param(20, None, id='integer-rate-null-deviation'),
    pytest.param(18.25, 0.5, id='native-floats'),
    pytest.param(sys.float_info.max, 0.5, id='largest-finite-double'),
    pytest.param(18.25, 'omitted', id='legacy-omitted-deviation'),
])
def test_valid_native_numbers_and_nested_samples_are_preserved(
        tmp_path, capsys, measured_rate, deviation):
    native = rows(measured_rate, measured_rate)
    for row in native:
        if deviation == 'omitted':
            del row['stddev_ts']
        else:
            row['stddev_ts'] = deviation
        row.update(samples_ns=[1000000000, 2000000000], samples_ts=[1.0, 2.0],
                   future_optional_column={'array': ['[log-like content]', 7]})
    run = make_run(tmp_path, 'valid-measurement', '/models/RVN.gguf', '1/1/1',
                   native=native)
    before = {name: (run / name).read_bytes() for name in ('report.json', 'stdout.log')}
    report, loaded = m.native_rows(run)
    assert loaded == native
    record = m.load_run(run, REGISTRY)
    assert record['command'] == report['command']
    assert record['prompt_ts'] == record['generation_ts'] == measured_rate
    assert type(record['generation_ts']) is type(measured_rate)
    assert record['generation_ts_stddev'] == (None if deviation == 'omitted' else deviation)
    out = tmp_path / 'out'
    assert m.main(report_write_argv(tmp_path, run, out)) == 0
    assert capsys.readouterr().err == ''
    assert sorted(path.name for path in out.iterdir()) == ['README.md', 'heretic.md']
    assert m.rate(measured_rate) in (out / 'README.md').read_text()
    assert {name: (run / name).read_bytes() for name in before} == before


def test_legacy_optional_metadata_is_not_a_new_required_schema(tmp_path):
    native = rows()
    for row in native:
        for field in ('stddev_ts', 'gpu_info', 'backends', 'build_commit'):
            row.pop(field)
    run = make_run(tmp_path, 'legacy', '/models/RVN.gguf', '1/1/1', native=native)
    report = json.loads((run / 'report.json').read_text())
    del report['schema']
    (run / 'report.json').write_text(json.dumps(report))
    record = m.load_run(run, REGISTRY)
    assert record['generation_ts'] == 20.0
    assert record['generation_ts_stddev'] is None
    assert record['gpu_info'] is record['backends'] is record['build_commit'] is None


@pytest.mark.parametrize('wire_shape', ['leading-log', 'truncated', 'trailing-log'])
def test_native_json_is_not_salvaged_from_logs_or_truncation(tmp_path, capsys, wire_shape):
    run = make_run(tmp_path, 'bad-wire', '/models/RVN.gguf', '1/1/1', native=rows())
    raw = (run / 'stdout.log').read_text()
    raw = {'leading-log': '[llama] leading log\n' + raw,
           'truncated': raw[:-1], 'trailing-log': raw + '\n[llama] trailing log'}[wire_shape]
    (run / 'stdout.log').write_text(raw)
    out = tmp_path / 'out'
    assert m.main(report_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()


def make_cpu_run(tmp_path, name='cpu', *, native=None):
    """The maintained CPU producer's canonical argv, not a missing-GPU guess."""
    if native is None:
        native = rows(80.0, 7.5, backends='CPU', gpu_info='')
        native[0]['n_prompt'] = 16
        native[1]['n_gen'] = 8
    run = make_run(tmp_path, name, '/models/RVN.gguf', '1/1/1', native=native)
    report = json.loads((run / 'report.json').read_text())
    report['command'] = [
        '/bin/llama-bench', '-m', '/models/RVN.gguf', '-dev', 'none',
        '-t', '2', '-ngl', '0', '-p', '16', '-n', '8', '-r', '1', '-o', 'json',
        '-lm', 'mmap', '-lzm', 'on', '--repack', '0', '--no-warmup', '--offline']
    (run / 'report.json').write_text(json.dumps(report))
    return run


def cpu_write_argv(tmp_path, run, out):
    argv = report_write_argv(tmp_path, run, out)
    return ['cpu-diagnostic', *argv[1:argv.index('--functional')]]


def test_cpu_diagnostic_reports_canonical_cpu_without_serving_claim(tmp_path, capsys):
    run = make_cpu_run(tmp_path)
    before = {name: (run / name).read_bytes() for name in ('report.json', 'stdout.log')}
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'heretic.md').write_text('existing GPU artifact\n')
    (out / 'README.md').write_text('existing GPU index\n')
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 0
    assert capsys.readouterr().err == ''
    body = (out / 'heretic-cpu-diagnostic.md').read_text()
    index = (out / 'README-cpu-diagnostics.md').read_text()
    assert 'CPU-only diagnostic' in body and 'CPU-only diagnostic' in index
    assert 'not a serving-preset measurement' in body
    assert '80.00 tok/s' in body and '7.50 tok/s' in body
    assert 'pp16' in body and 'tg8' in body
    assert json.dumps(json.loads(before['report.json'])['command']) in body
    assert 'Ordering within one serving-MTP class' not in index
    assert '**PASS**' not in body and '**PASS**' not in index
    assert (out / 'heretic.md').read_text() == 'existing GPU artifact\n'
    assert (out / 'README.md').read_text() == 'existing GPU index\n'
    assert {name: (run / name).read_bytes() for name in before} == before


@pytest.mark.parametrize('flag', ['-t', '-p', '-n', '-r'])
@pytest.mark.parametrize('value', [
    pytest.param('0', id='zero'), pytest.param('-1', id='negative'),
    pytest.param('1.5', id='fraction'), pytest.param('unknown', id='nonnumeric'),
    pytest.param('9' * 5000, id='oversized-decimal'),
])
def test_cpu_diagnostic_refuses_invalid_workload_before_publication(
        tmp_path, capsys, flag, value):
    run = make_cpu_run(tmp_path)
    report = json.loads((run / 'report.json').read_text())
    report['command'][report['command'].index(flag) + 1] = value
    (run / 'report.json').write_text(json.dumps(report))
    before = {name: (run / name).read_bytes() for name in ('report.json', 'stdout.log')}
    out = tmp_path / 'out'
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()
    assert {name: (run / name).read_bytes() for name in before} == before


@pytest.mark.parametrize('mutation', [
    'non-cpu-device', 'offload', 'missing-device', 'missing-offload',
    'duplicate-device', 'duplicate-offload', 'tensor-split', 'device-alias',
    'offload-alias', 'non-json', 'missing-value', 'unknown-flag',
    'no-command', 'non-string-argument', 'duplicate-switch',
])
def test_cpu_diagnostic_refuses_ambiguous_or_gpu_argv(tmp_path, capsys, mutation):
    run = make_cpu_run(tmp_path)
    report = json.loads((run / 'report.json').read_text())
    command = report['command']
    replacements = {'non-cpu-device': ('-dev', 'ROCm0'),
                    'offload': ('-ngl', '1'), 'non-json': ('-o', 'csv')}
    removals = {'missing-device': '-dev', 'missing-offload': '-ngl'}
    additions = {'duplicate-device': ['-dev', 'none'],
                 'duplicate-offload': ['-ngl', '999'], 'tensor-split': ['-ts', '1/1/1'],
                 'device-alias': ['--device', 'ROCm0'],
                 'offload-alias': ['--n-gpu-layers', '999'],
                 'unknown-flag': ['--unknown-placement'],
                 'duplicate-switch': ['--no-warmup']}
    if mutation in replacements:
        flag, value = replacements[mutation]
        command[command.index(flag) + 1] = value
    elif mutation in removals:
        index = command.index(removals[mutation])
        del command[index:index + 2]
    elif mutation in additions:
        command.extend(additions[mutation])
    elif mutation == 'missing-value':
        del command[command.index('-o') + 1]
    elif mutation == 'no-command':
        report['command'] = None
    else:
        command[0] = False
    (run / 'report.json').write_text(json.dumps(report))
    out = tmp_path / 'out'
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('row_index', [0, 1])
@pytest.mark.parametrize('layers', [1, -1, True, '0', None, 0.0])
def test_cpu_diagnostic_refuses_contradictory_native_gpu_layers(
        tmp_path, capsys, row_index, layers):
    run = make_cpu_run(tmp_path)
    native = json.loads((run / 'stdout.log').read_text())
    native[row_index]['n_gpu_layers'] = layers
    (run / 'stdout.log').write_text(json.dumps(native))
    out = tmp_path / 'out'
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(('status', 'exit_code'), [
    ('failed', 1), ('interrupted', 124), ('running', None),
    ('passed', 1), ('passed', None), ('passed', False), ('passed', 0.0),
])
def test_cpu_diagnostic_requires_observed_success_before_publication(
        tmp_path, capsys, status, exit_code):
    run = make_cpu_run(tmp_path)
    report = json.loads((run / 'report.json').read_text())
    report.update(status=status, exit_code=exit_code)
    (run / 'report.json').write_text(json.dumps(report))
    out = tmp_path / 'out'
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('mutation', [
    'missing-json', 'leading-log', 'trailing-log', 'truncated', 'empty',
    'non-object-row', 'missing-rate', 'nan-rate', 'bool-token', 'token-mismatch',
    'missing-generation',
])
def test_cpu_diagnostic_keeps_native_admission_and_workload_checks(
        tmp_path, capsys, mutation):
    run = make_cpu_run(tmp_path)
    path = run / 'stdout.log'
    raw = path.read_text()
    native = json.loads(raw)
    if mutation == 'missing-json':
        path.unlink()
    elif mutation in {'leading-log', 'trailing-log', 'truncated'}:
        path.write_text({'leading-log': '[llama] log\n' + raw,
                         'trailing-log': raw + '\n[llama] log', 'truncated': raw[:-1]}[mutation])
    else:
        if mutation == 'empty':
            native = []
        elif mutation == 'non-object-row':
            native[0] = 'not a row'
        elif mutation == 'missing-rate':
            native[0].pop('avg_ts')
        elif mutation == 'nan-rate':
            native[1]['avg_ts'] = float('nan')
        elif mutation == 'bool-token':
            native[0]['n_prompt'] = True
        elif mutation == 'token-mismatch':
            native[0]['n_prompt'] = 32
        else:
            native.pop()
        path.write_text(json.dumps(native))
    out = tmp_path / 'out'
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('metadata', ['legacy', 'compiled-gpu-backend', 'native-zero-layers',
                                     'minimal-command'])
def test_cpu_diagnostic_preserves_optional_native_metadata(tmp_path, metadata):
    run = make_cpu_run(tmp_path)
    native = json.loads((run / 'stdout.log').read_text())
    report = json.loads((run / 'report.json').read_text())
    for row in native:
        row['future_column'] = {'nested': ['unchanged', 7]}
        if metadata == 'legacy':
            for key in ('gpu_info', 'backends', 'build_commit', 'stddev_ts'):
                row.pop(key)
        elif metadata == 'compiled-gpu-backend':
            row['backends'] = 'CPU,ROCm'
            row['gpu_info'] = 'available GPU, not proof of use'
        elif metadata == 'native-zero-layers':
            row['n_gpu_layers'] = 0
    if metadata == 'legacy':
        report.pop('schema')
    if metadata == 'minimal-command':
        report['command'] = report['command'][:report['command'].index('-lm')]
    (run / 'stdout.log').write_text(json.dumps(native))
    (run / 'report.json').write_text(json.dumps(report))
    before = {name: (run / name).read_bytes() for name in ('report.json', 'stdout.log')}
    out = tmp_path / 'out'
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 0
    body = (out / 'heretic-cpu-diagnostic.md').read_text()
    assert json.dumps(native, indent=2) in body
    assert 'not treated as evidence that a GPU was used' in body
    assert {name: (run / name).read_bytes() for name in before} == before


def test_cpu_diagnostic_needs_explicit_mode_and_never_queries_host(
        tmp_path, capsys, monkeypatch):
    run = make_cpu_run(tmp_path)
    out = tmp_path / 'out'
    assert m.main(report_write_argv(tmp_path, run, out)) == 2
    assert not out.exists()
    capsys.readouterr()

    def forbidden(*args, **kwargs):
        pytest.fail('CPU reporting must not sample the host or borrow a functional/preset verdict')

    monkeypatch.setattr(m, 'sample_host', forbidden)
    monkeypatch.setattr(m, 'load_functional', forbidden)
    monkeypatch.setattr(m, 'load_run', forbidden)
    monkeypatch.setattr(m.gpu_vram, 'devices', forbidden)
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 0
    args = m.parser().parse_args(['cpu-diagnostic', str(run)])
    assert Path(args.out) == m.OUT / 'cpu-diagnostics'


def test_cpu_diagnostic_mixed_runs_and_duplicate_aliases_refuse_before_writes(tmp_path, capsys):
    cpu = make_cpu_run(tmp_path, 'cpu')
    gpu = make_run(tmp_path, 'gpu', '/models/RVN.gguf', '1/1/1', native=rows())
    out = tmp_path / 'mixed'
    argv = cpu_write_argv(tmp_path, cpu, out)
    argv.insert(2, str(gpu))
    assert m.main(argv) == 2
    assert not out.exists()
    capsys.readouterr()
    argv = cpu_write_argv(tmp_path, cpu, tmp_path / 'duplicate')
    argv.insert(2, str(cpu))
    assert m.main(argv) == 2
    assert 'same alias' in capsys.readouterr().err
    assert not (tmp_path / 'duplicate').exists()


def test_cpu_diagnostic_rejects_unadmitted_records_and_unsafe_alias_before_writes(tmp_path):
    gpu = m.load_run(make_run(tmp_path, 'gpu', '/models/RVN.gguf', '1/1/1',
                              native=rows()), REGISTRY)
    with pytest.raises(m.ReportRefused, match='admitted CPU-only'):
        m.write_cpu_diagnostics([gpu], tmp_path / 'gpu-out')
    record = m.load_cpu_diagnostic(make_cpu_run(tmp_path), REGISTRY)
    record['alias'] = '../escape'
    with pytest.raises(m.ReportRefused, match='safe artifact name'):
        m.write_cpu_diagnostics([record], tmp_path / 'bad-alias')
    assert not (tmp_path / 'gpu-out').exists()
    assert not (tmp_path / 'bad-alias').exists()


def test_cpu_diagnostic_refuses_an_extra_combined_workload_row(tmp_path, capsys):
    run = make_cpu_run(tmp_path)
    native = json.loads((run / 'stdout.log').read_text())
    native.append({**native[0], 'n_prompt': 16, 'n_gen': 8})
    (run / 'stdout.log').write_text(json.dumps(native))
    out = tmp_path / 'out'
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('filename', ['heretic-cpu-diagnostic.md', 'README-cpu-diagnostics.md'])
def test_cpu_diagnostic_never_clobbers_existing_artifact(tmp_path, capsys, filename):
    run = make_cpu_run(tmp_path)
    out = tmp_path / 'shared'
    out.mkdir()
    existing = out / filename
    existing.write_text('Preexisting serving or diagnostic artifact; preserve exact bytes\n')
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_cpu_diagnostic_exclusive_creation_refuses_late_target(tmp_path, capsys, monkeypatch):
    run = make_cpu_run(tmp_path)
    out = tmp_path / 'late-target'
    target = out / 'heretic-cpu-diagnostic.md'
    original_open = Path.open
    reached = []

    def inject_target(path, mode='r', *args, **kwargs):
        if path == target and mode == 'x':
            reached.append(path)
            with original_open(path, 'w') as stream:
                stream.write('Concurrent-owner fixture; do not replace\n')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', inject_target)
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert reached == [target]
    assert target.read_text() == 'Concurrent-owner fixture; do not replace\n'
    assert not (out / 'README-cpu-diagnostics.md').exists()


def test_cpu_diagnostic_refuses_dangling_target_symlink(tmp_path, capsys):
    run = make_cpu_run(tmp_path)
    out = tmp_path / 'linked-target'
    out.mkdir()
    target = tmp_path / 'uncreated-private-target'
    link = out / 'heretic-cpu-diagnostic.md'
    link.symlink_to(target)
    assert m.main(cpu_write_argv(tmp_path, run, out)) == 2
    assert 'Report refused' in capsys.readouterr().err
    assert link.is_symlink() and link.readlink() == target
    assert not target.exists()
    assert not (out / 'README-cpu-diagnostics.md').exists()


@pytest.mark.parametrize(('exit_code', 'expected'), [
    pytest.param(False, 2, id='boolean-false'),
    pytest.param(0.0, 2, id='floating-zero'),
    pytest.param(0, 0, id='observed-integer-zero'),
])
def test_cli_exit_status_is_integer_zero_not_just_zero_equivalent(
        tmp_path, capsys, exit_code, expected):
    run = make_run(tmp_path, 'exit-status', '/models/RVN.gguf', '1/1/1',
                   native=rows(97.5, 18.25), exit_code=exit_code)
    before = {name: (run / name).read_bytes() for name in ('report.json', 'stdout.log')}
    out = tmp_path / 'out'
    assert m.main(report_write_argv(tmp_path, run, out)) == expected
    stderr = capsys.readouterr().err
    if expected == 2:
        assert 'unsuccessful run has no rate to quote' in stderr
        assert not out.exists()
    else:
        assert stderr == ''
        assert '18.25 tok/s' in (out / 'heretic.md').read_text()
    assert {name: (run / name).read_bytes() for name in before} == before


def assert_input_refusal(tmp_path, capsys, run, mode):
    before = {name: (run / name).read_bytes() for name in ('report.json', 'stdout.log')}
    out = tmp_path / 'out'
    argv = cpu_write_argv(tmp_path, run, out) if mode == 'cpu-diagnostic' else (
        report_write_argv(tmp_path, run, out))
    assert m.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err.startswith('Report refused: ')
    assert run.name in captured.err
    assert not out.exists()
    assert {name: (run / name).read_bytes() for name in before} == before


@pytest.mark.parametrize('mode', ['write', 'cpu-diagnostic'])
@pytest.mark.parametrize('value', [
    pytest.param(None, id='null'), pytest.param(True, id='boolean'),
    pytest.param(0, id='integer'), pytest.param(0.0, id='float'),
    pytest.param('report', id='string'), pytest.param([], id='array'),
])
def test_cli_refuses_non_object_benchmark_report(tmp_path, capsys, mode, value):
    run = make_cpu_run(tmp_path) if mode == 'cpu-diagnostic' else make_run(
        tmp_path, 'invalid-input', '/models/RVN.gguf', '1/1/1', native=rows())
    (run / 'report.json').write_text(json.dumps(value))
    assert_input_refusal(tmp_path, capsys, run, mode)


@pytest.mark.parametrize('mode', ['write', 'cpu-diagnostic'])
@pytest.mark.parametrize('value', [
    pytest.param(None, id='null'), pytest.param(True, id='boolean'),
    pytest.param(0, id='integer'), pytest.param(0.0, id='float'),
    pytest.param([], id='array'), pytest.param({}, id='object'),
    pytest.param('', id='empty'), pytest.param('\x00', id='nul'),
])
def test_cli_refuses_invalid_native_results_path(tmp_path, capsys, mode, value):
    run = make_cpu_run(tmp_path) if mode == 'cpu-diagnostic' else make_run(
        tmp_path, 'invalid-input', '/models/RVN.gguf', '1/1/1', native=rows())
    report = json.loads((run / 'report.json').read_text())
    report['native_results'] = value
    (run / 'report.json').write_text(json.dumps(report))
    assert_input_refusal(tmp_path, capsys, run, mode)


@pytest.mark.parametrize('mode', ['write', 'cpu-diagnostic'])
@pytest.mark.parametrize('filename', ['report.json', 'stdout.log'])
@pytest.mark.parametrize('failure', ['invalid-encoding', 'decimal-limit', 'nested-array'])
def test_cli_refuses_malformed_benchmark_json(tmp_path, capsys, mode, filename, failure):
    run = make_cpu_run(tmp_path) if mode == 'cpu-diagnostic' else make_run(
        tmp_path, 'invalid-input', '/models/RVN.gguf', '1/1/1', native=rows())
    if failure == 'invalid-encoding':
        raw = b'\xff'
    elif failure == 'decimal-limit':
        limit = sys.get_int_max_str_digits()
        assert 0 < limit <= 10000, 'fixture requires the bounded interpreter decimal limit'
        raw = b'9' * (limit + 1)
    else:
        # Wrong container shape, not proof that this interpreter exhausts its decoder.
        depth = sys.getrecursionlimit() + 100
        assert depth < 10000, 'fixture requires a bounded interpreter recursion limit'
        raw = b'[' * depth + b'0' + b']' * depth
    (run / filename).write_bytes(raw)
    assert_input_refusal(tmp_path, capsys, run, mode)


@pytest.mark.parametrize('mode', ['write', 'cpu-diagnostic'])
@pytest.mark.parametrize('existing_output', [False, True], ids=['new-output', 'existing-output'])
def test_later_malformed_report_publishes_no_earlier_run(
        tmp_path, capsys, mode, existing_output):
    first_root = tmp_path / 'first'
    second_root = tmp_path / 'second'
    first_root.mkdir()
    second_root.mkdir()
    first = make_cpu_run(first_root) if mode == 'cpu-diagnostic' else make_run(
        first_root, 'healthy', '/models/RVN.gguf', '1/1/1', native=rows(97.5, 18.25))
    second = make_cpu_run(second_root) if mode == 'cpu-diagnostic' else make_run(
        second_root, 'malformed', '/models/RVN.gguf', '1/1/1', native=rows())
    # Establish that the first run really reaches this mode's successful reader.
    loaded = m.load_cpu_diagnostic(first, REGISTRY) if mode == 'cpu-diagnostic' else (
        m.load_run(first, REGISTRY))
    assert loaded['alias'] == 'heretic'
    (second / 'report.json').write_text('null')
    before_inputs = {(str(run), name): (run / name).read_bytes()
                     for run in (first, second) for name in ('report.json', 'stdout.log')}
    out = tmp_path / 'out'
    if existing_output:
        out.mkdir()
        (out / 'owner.md').write_text('Existing owner bytes; do not alter\n')
    before_output = {p.name: p.read_bytes() for p in out.iterdir()} if existing_output else {}
    argv = cpu_write_argv(tmp_path, first, out) if mode == 'cpu-diagnostic' else (
        report_write_argv(tmp_path, first, out))
    argv.insert(2, str(second))
    assert m.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err.startswith('Report refused: ')
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before_output if existing_output else (
        not out.exists())
    assert {(str(run), name): (run / name).read_bytes()
            for run in (first, second) for name in ('report.json', 'stdout.log')} == before_inputs


@pytest.mark.parametrize('mode', ['write', 'cpu-diagnostic'])
@pytest.mark.parametrize('filename', ['report.json', 'stdout.log'])
def test_cli_refuses_json_decoder_recursion_failure(
        tmp_path, capsys, monkeypatch, mode, filename):
    """Inject decoder exhaustion; do not stress the native interpreter's stack."""
    run = make_cpu_run(tmp_path) if mode == 'cpu-diagnostic' else make_run(
        tmp_path, 'decoder-exhaustion', '/models/RVN.gguf', '1/1/1', native=rows())
    selected = (run / filename).read_text()
    original = m.json.loads
    reached = []

    def decoder(text, *args, **kwargs):
        if text == selected:
            reached.append(filename)
            raise RecursionError('fixture JSON decoder recursion limit')
        return original(text, *args, **kwargs)

    monkeypatch.setattr(m.json, 'loads', decoder)
    assert_input_refusal(tmp_path, capsys, run, mode)
    assert reached == [filename]
