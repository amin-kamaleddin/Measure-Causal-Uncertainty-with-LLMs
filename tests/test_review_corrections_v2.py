"""Broader scientific, malformed-batch and log-redaction regressions."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sharecausal.benchmark import bridge_v2
from sharecausal.benchmark.bridge_v2 import analyze_observables
from sharecausal.benchmark.bridge_v2_smoke_data import smoke_tasks
from sharecausal.llm.synthetic_adapter import adapt_record, adapt_response
from sharecausal.llm.synthetic_schema import PUBLIC_CARD

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/llm_bridge_v2'
MARKER = 'HOSTILE_NESTED_FIELD_NAME_MUST_NOT_BE_LOGGED_42'


def measured(name='dummy_measured_x1'):
    record = json.loads((FIXTURES / 'dummy_measured_x1.json').read_text())
    record['provenance']['fixture_id'] = name
    return record


def arrow(source, target):
    item = deepcopy(measured()['response']['edges'][0])
    item.update(source=source, target=target)
    return item


def zero_record(extra=()):
    record = measured('dummy_structural_zero')
    record['response']['edges'] = [e for e in record['response']['edges']
                                    if (e['source'], e['target']) != ('A', 'Y')]
    record['response']['edges'] += [arrow(a, b) for a, b in extra]
    return record


@pytest.mark.parametrize('extra', [(), (('A', 'M'),), (('A', 'C'), ('Y', 'C')),
                                   (('A', 'M'), ('M', 'C'), ('Y', 'C'))])
@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('radius', [0, .25])
def test_structural_zero_survives_strong_association_and_sensitivity(extra, direction, radius):
    frame = smoke_tasks()[0].frame.copy()
    frame['Y'] *= direction * 10
    result = analyze_observables(frame, [zero_record(extra)], card=PUBLIC_CARD, bias_radius=radius)
    assert result['accepted_count'] == 1 and result['rejected_count'] == 0
    ident = result['graph_identification']['dummy_structural_zero']
    assert not ident['has_directed_causal_path'] and ident['structural_zero']
    assert ident['total_effect_status'] == 'graph_entailed_structural_zero'
    for name in ('dummy_structural_zero', 'naive_majority'):
        claim = result['methods'][name]
        assert claim['decision'] == 'abstain' and claim['structural_effect'] == 0.0
        assert claim['estimate'] is claim['ci_lower'] is claim['ci_upper'] is None
        assert claim['model_compatibility'] == 'not_tested'
    robust = result['methods']['robust_screened']
    assert robust['decision'] == 'abstain' and robust['estimate'] is None
    assert robust['envelope']['lower'] <= 0 <= robust['envelope']['upper']
    assert robust['structural_zero_graphs'] == ['dummy_structural_zero']
    # Observable association never changes admission or the structural constraint.
    frame['Y'] = -frame['Y']
    altered = analyze_observables(frame, [zero_record(extra)], card=PUBLIC_CARD, bias_radius=radius)
    assert result['graph_audit'] == altered['graph_audit']
    assert result['graph_identification'] == altered['graph_identification']


def test_structural_zero_vetoes_other_positive_graph_without_being_dropped():
    result = analyze_observables(smoke_tasks()[0].frame, [measured(), zero_record()], card=PUBLIC_CARD)
    assert result['methods']['dummy_measured_x1']['decision'] == 'positive'
    assert result['methods']['robust_screened']['decision'] == 'abstain'
    assert result['accepted_count'] == 2
    assert all(a['status'] == 'accepted' for a in result['graph_audit'])


def test_indirect_causal_path_is_distinct_from_absent_direct_arrow():
    record = zero_record((('A', 'M'), ('M', 'Y')))
    result = analyze_observables(smoke_tasks()[0].frame, [record], card=PUBLIC_CARD)
    ident = result['graph_identification']['dummy_structural_zero']
    assert ident['has_directed_causal_path'] and not ident['structural_zero']
    claim = result['methods']['dummy_structural_zero']
    assert claim['decision'] == 'positive' and claim['adjustment_set'] == ['X1']
    assert claim['model_compatibility'] == 'not_tested'


def test_structural_zero_is_not_conflated_with_backdoor_failure():
    record = zero_record((('U', 'A'), ('U', 'Y')))
    result = analyze_observables(smoke_tasks()[0].frame, [record], card=PUBLIC_CARD)
    ident = result['graph_identification']['dummy_structural_zero']
    assert not ident['identified_by_backdoor']
    assert ident['total_effect_status'] == 'graph_entailed_structural_zero'
    assert result['methods']['dummy_structural_zero']['structural_effect'] == 0
    assert result['methods']['robust_screened']['decision'] == 'abstain'
    assert result['methods']['robust_screened']['envelope'] is not None


def test_estimation_failure_does_not_rewrite_graph_identification(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError(MARKER)
    monkeypatch.setattr(bridge_v2, 'aipw', fail)
    result = analyze_observables(smoke_tasks()[0].frame, [measured()], card=PUBLIC_CARD)
    ident = result['graph_identification']['dummy_measured_x1']
    assert ident['identified_by_backdoor'] and ident['estimation_failed']
    assert result['methods']['dummy_measured_x1']['decision'] == 'abstain'
    robust = result['methods']['robust_screened']
    assert robust['estimation_failure_graphs'] == ['dummy_measured_x1']
    assert 'dummy_measured_x1' not in robust['nonidentified_graphs']
    assert MARKER not in json.dumps(result)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, Decimal('1.2'),
                                   {1, 2}, b'bytes', complex(1, 2), object()])
def test_non_json_response_values_reject_only_that_source(value):
    bad = measured('dummy_bad_serialization')
    bad['response']['edges'][0]['uncertainty'] = value
    adapted = adapt_record(bad, PUBLIC_CARD)
    assert adapted.graph is None and adapted.audit['status'] == 'rejected'
    assert adapted.audit['reason'] == 'response_serialization_failed'
    assert adapted.audit['provenance']['fixture_id'] == 'dummy_bad_serialization'
    result = analyze_observables(smoke_tasks()[0].frame, [bad, measured()], card=PUBLIC_CARD)
    assert result['rejected_count'] == 1 and result['accepted_count'] == 1
    assert result['methods']['dummy_measured_x1']['decision'] == 'positive'
    assert result['methods']['dummy_bad_serialization']['decision'] == 'abstain'


@pytest.mark.parametrize('kind', ['circular', 'mixed_keys', 'deep', 'unicode'])
def test_adversarial_serialization_and_encoding_are_contained(kind):
    bad = measured()
    if kind == 'circular':
        bad['response']['loop'] = bad['response']
    elif kind == 'mixed_keys':
        bad['response'][1] = MARKER
    elif kind == 'deep':
        nested = {}
        bad['response']['nested'] = nested
        for _ in range(1500):
            nested['next'] = {}; nested = nested['next']
    else:
        bad['response'] = '\ud800'
    result = adapt_record(bad, PUBLIC_CARD)
    assert result.graph is None and result.audit['status'] == 'rejected'
    assert MARKER not in json.dumps(result.audit)


def test_malformed_same_identity_cannot_abort_or_overwrite_valid_prediction():
    bad = measured()
    bad['response']['edges'][0]['uncertainty'] = math.nan
    for records in ([bad, measured()], [measured(), bad]):
        result = analyze_observables(smoke_tasks()[0].frame, records, card=PUBLIC_CARD)
        assert result['accepted_count'] == result['rejected_count'] == 1
        assert result['methods']['dummy_measured_x1']['decision'] == 'positive'
        rejected = [v for k, v in result['methods'].items() if k.startswith('rejected_record_')]
        assert len(rejected) == 1 and rejected[0]['decision'] == 'abstain'


@pytest.mark.parametrize('where', ['response', 'arrow', 'provenance', 'nested', 'envelope'])
def test_unknown_field_names_and_values_never_enter_rejection_audits(where):
    record = measured()
    if where == 'response': record['response'][MARKER] = MARKER
    elif where == 'arrow': record['response']['edges'][0][MARKER] = MARKER
    elif where == 'provenance': record['provenance'][MARKER] = MARKER
    elif where == 'nested': record['response']['edges'][0]['uncertainty'] = {MARKER: MARKER}
    else: record[MARKER] = MARKER
    result = adapt_record(record, PUBLIC_CARD)
    assert result.graph is None and result.audit['status'] == 'rejected'
    assert MARKER not in json.dumps(result.audit)


def test_raw_json_nonfinite_and_invalid_utf8_text_are_rejected():
    record = measured()
    record['response']['edges'][0]['uncertainty'] = math.nan
    raw = json.dumps(record['response'])  # deliberately uses nonstandard NaN token
    result = adapt_response(raw, PUBLIC_CARD, record['provenance'])
    assert result.audit['status'] == 'rejected'
    assert result.graph is None


def test_complete_batch_is_committed_before_first_truth_read(monkeypatch, tmp_path):
    from sharecausal.benchmark import bridge_v2_offline as runner
    original_tasks = smoke_tasks()
    reads = []
    def batch_fixed():
        file = tmp_path / 'prediction_batch_manifest.json'
        manifest = json.loads(file.read_text())
        assert len(manifest['predictions']) == 3
        for item in manifest['predictions']:
            raw = (tmp_path / item['prediction_file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item['prediction_sha256']
        assert not any(k in file.read_text() for k in ('true_ate', 'truth', 'label'))
    class GuardedTask:
        def __init__(self, task):
            self.frame, self._task = task.frame, task
        @property
        def true_ate(self):
            batch_fixed(); reads.append('truth'); return self._task.true_ate
        @property
        def identified(self):
            batch_fixed(); return self._task.identified
        @property
        def label(self):
            batch_fixed(); return self._task.label
    monkeypatch.setattr(runner, 'smoke_tasks', lambda: [GuardedTask(t) for t in original_tasks])
    report = runner.run_offline_smoke(FIXTURES, tmp_path)
    assert reads == ['truth'] * 3
    manifest = tmp_path / report['prediction_batch_manifest']
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == report['prediction_batch_sha256']


def test_cli_mixed_valid_zero_malformed_and_hostile_records(tmp_path):
    fixture_dir = tmp_path / 'fixtures'; fixture_dir.mkdir()
    records = [json.loads(p.read_text()) for p in sorted(FIXTURES.glob('*.json'))]
    bad = measured('dummy_nonfinite'); bad['response']['edges'][0]['uncertainty'] = math.nan
    hostile = measured('dummy_hostile_top'); hostile['response'][MARKER] = MARKER
    nested = measured('dummy_hostile_nested'); nested['response']['edges'][0][MARKER] = MARKER
    records += [zero_record(), bad, hostile, nested]
    for index, record in enumerate(records):
        (fixture_dir / f'record_{index}.json').write_text(json.dumps(record))
    env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT / 'src')
    env['OPENROUTER_API_KEY'] = MARKER
    env['SHARECAUSAL_LIVE_LLM_APPROVED'] = '1'
    result = subprocess.run([sys.executable, 'scripts/llm_bridge_v2_offline.py',
          '--fixture-dir', str(fixture_dir), '--results-dir', str(tmp_path / 'output'), '--new-run'],
          cwd=ROOT, env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert MARKER not in result.stdout + result.stderr
    stdout = json.loads(result.stdout); out = Path(stdout['results_dir'])
    for file in out.iterdir():
        assert MARKER not in file.read_text(), file.name
    report = json.loads((out / 'smoke_report.json').read_text())
    assert report['real_llm_calls'] == report['openrouter_cost_usd'] == report['share_records_used'] == 0
    assert report['fixture_mode'] is True
    for audit in report['graph_screen_audit']:
        assert (audit['accepted'], audit['rejected'], audit['source_abstained']) == (3, 4, 1)
        assert len(audit['proposals']) == 8
        by_id = {p['provenance']['fixture_id']: p for p in audit['proposals']}
        assert by_id['dummy_nonfinite']['reason'] == 'response_serialization_failed'
        assert by_id['dummy_structural_zero']['status'] == 'accepted'
        assert by_id['dummy_structural_zero']['identification']['structural_zero']
    assert report['n_method_rows'] == 33
    for item in report['predictions']:
        prediction = json.loads((out / item['prediction_file']).read_text())
        assert prediction['methods']['dummy_measured_x1']['estimate'] is not None
        assert prediction['methods']['dummy_structural_zero']['decision'] == 'abstain'
        assert prediction['methods']['robust_screened']['decision'] == 'abstain'
    assert (out / 'prediction_batch_manifest.json').is_file()


def test_cli_unparseable_deep_and_oversized_integer_records_do_not_abort_valid_batch(tmp_path):
    fixture_dir = tmp_path / 'fixtures'; fixture_dir.mkdir()
    for path in FIXTURES.glob('*.json'):
        (fixture_dir / path.name).write_bytes(path.read_bytes())
    # Exercise the file decoder before adapt_record, including its ValueError
    # and RecursionError paths. No exception text/hostile key reaches output.
    (fixture_dir / 'deep.json').write_text('{"' + MARKER + '":' + '[' * 2000 + 'null' + ']' * 2000 + '}')
    (fixture_dir / 'integer.json').write_text('{"' + MARKER + '":' + '1' * 50000 + '}')
    env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT / 'src')
    completed = subprocess.run([sys.executable, 'scripts/llm_bridge_v2_offline.py',
         '--fixture-dir', str(fixture_dir), '--results-dir', str(tmp_path / 'output')],
         cwd=ROOT, env=env, capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stderr
    assert MARKER not in completed.stdout + completed.stderr
    out = Path(json.loads(completed.stdout)['results_dir'])
    report = json.loads((out / 'smoke_report.json').read_text())
    assert report['n_method_rows'] == 27
    for audit in report['graph_screen_audit']:
        assert (audit['accepted'], audit['rejected'], audit['source_abstained']) == (2, 3, 1)
    for file in out.iterdir():
        assert MARKER not in file.read_text()
    prediction = json.loads((out / 'prediction_0.json').read_text())
    assert prediction['methods']['dummy_measured_x1']['decision'] == 'positive'
