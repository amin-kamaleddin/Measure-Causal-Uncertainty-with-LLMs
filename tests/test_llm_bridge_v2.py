"""Step 6B contract, firewall, privacy, and executable integration tests."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import runpy
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sharecausal.benchmark import bridge_v2
from sharecausal.benchmark.bridge_v2 import analyze_observables, identify_pre_exposure
from sharecausal.benchmark.bridge_v2_offline import (
    load_fixture_records, offline_network_veto, run_offline_smoke, safe_path)
from sharecausal.benchmark.bridge_v2_scoring import score_fixed_predictions
from sharecausal.benchmark.bridge_v2_smoke_data import smoke_tasks
from sharecausal.llm import synthetic_adapter, synthetic_schema
from sharecausal.llm.synthetic_adapter import adapt_record, adapt_response
from sharecausal.llm.synthetic_schema import PUBLIC_CARD, SyntheticGraphAnswer, prompt_hash

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/llm_bridge_v2'


@pytest.fixture
def records():
    return load_fixture_records(FIXTURES)


@pytest.fixture
def valid(records):
    return next(r for r in records if r['provenance']['fixture_id'] == 'dummy_measured_x1')


@pytest.fixture
def frame():
    return smoke_tasks()[0].frame


def analyze(frame, records, **kwargs):
    return analyze_observables(frame, records, card=PUBLIC_CARD, **kwargs)


def same(left, right):
    """Exact categorical comparison; BLAS roundoff only, matching v1 erratum."""
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            same(left[k], right[k])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, float):
        assert math.isfinite(left) and math.isfinite(right)
        assert right == pytest.approx(left, rel=1e-8, abs=1e-8)
    else:
        assert left == right


def test_card_matches_v1_and_share_schema_remains_separate(valid):
    from sharecausal.benchmark.blind import PUBLIC_CARD as V1_CARD
    from sharecausal.llm.schemas import GraphAnswer
    assert PUBLIC_CARD == V1_CARD
    response = valid['response']
    graph = SyntheticGraphAnswer.model_validate(response)
    assert [(e.source, e.target) for e in graph.edges] == [
        (e['source'], e['target']) for e in response['edges']]
    with pytest.raises(ValueError):
        GraphAnswer.model_validate(response)


@pytest.mark.parametrize('mutation', [
    'unknown_node', 'cycle', 'duplicate_edge', 'backward', 'self_loop',
    'wrong_treatment', 'wrong_outcome', 'wrong_estimand', 'wrong_role',
    'post_adjust', 'latent_adjust', 'latent_observed', 'missing_record',
    'missing_edge_evidence', 'blank_rationale', 'bad_uncertainty', 'coerced_abstain',
    'abstain_without_reason', 'abstain_with_edges', 'nonabstain_with_reason', 'empty_graph',
])
def test_hostile_graph_schema_rejected(valid, mutation):
    response = valid['response']
    edge = deepcopy(response['edges'][0])
    if mutation == 'unknown_node': response['edges'][0]['source'] = 'B'
    elif mutation == 'cycle':
        edge.update(source='A', target='X1'); response['edges'].append(edge)
    elif mutation == 'duplicate_edge': response['edges'].append(edge)
    elif mutation == 'backward': response['edges'][0].update(source='C', target='M')
    elif mutation == 'self_loop': response['edges'][0].update(source='X1', target='X1')
    elif mutation == 'wrong_treatment': response['treatment'] = 'M'
    elif mutation == 'wrong_outcome': response['outcome'] = 'C'
    elif mutation == 'wrong_estimand': response['estimand'] = 'controlled_direct_effect'
    elif mutation == 'wrong_role': response['edges'][0].update(source='A', target='Y', causal_role='confounding')
    elif mutation == 'post_adjust': response['suggested_adjustment'] = ['M']
    elif mutation == 'latent_adjust': response['suggested_adjustment'] = ['U']
    elif mutation == 'latent_observed': response['observed_nodes'].append('U')
    elif mutation == 'missing_record': del response['abstain']
    elif mutation == 'missing_edge_evidence': del response['edges'][0]['evidence_reference']
    elif mutation == 'blank_rationale': response['rationale'] = '   '
    elif mutation == 'bad_uncertainty': response['edges'][0]['uncertainty'] = 'certain'
    elif mutation == 'coerced_abstain': response['abstain'] = 'false'
    elif mutation == 'abstain_without_reason': response.update(abstain=True, edges=[], suggested_adjustment=[])
    elif mutation == 'abstain_with_edges': response.update(abstain=True, abstention_reason='decline')
    elif mutation == 'nonabstain_with_reason': response['abstention_reason'] = 'decline'
    elif mutation == 'empty_graph': response['edges'] = []
    with pytest.raises(ValueError):
        SyntheticGraphAnswer.model_validate(response)
    adapted = adapt_record(valid, PUBLIC_CARD)
    assert adapted.graph is None and adapted.audit['status'] == 'rejected'


@pytest.mark.parametrize('mutation', ['missing', 'missing_hash', 'bad_hash', 'live', 'bad_timestamp', 'negative_repeat'])
def test_provenance_fails_closed(valid, frame, mutation):
    if mutation == 'missing': valid['provenance'] = None
    elif mutation == 'missing_hash': del valid['provenance']['prompt_sha256']
    elif mutation == 'bad_hash': valid['provenance']['prompt_sha256'] = '0' * 64
    elif mutation == 'live': valid['provenance']['source_type'] = 'openrouter'
    elif mutation == 'bad_timestamp': valid['provenance']['timestamp'] = '2026-10-16'
    elif mutation == 'negative_repeat': valid['provenance']['repetition_index'] = -1
    result = analyze(frame, [valid])
    assert result['accepted_count'] == 0
    assert result['rejected_count'] == 1
    assert result['methods']['robust_screened']['decision'] == 'abstain'
    assert result['methods']['naive_majority']['decision'] == 'abstain'


def test_invalid_json_and_missing_response_are_explicit_rejections(valid, frame):
    valid['response'] = '{SECRET_SENTINEL not json'
    result = analyze(frame, [valid, None, {}])
    assert result['rejected_count'] == 3
    assert result['accepted_count'] == 0
    assert 'SECRET_SENTINEL' not in json.dumps(result)
    assert result['methods']['robust_screened']['decision'] == 'abstain'
    del valid['response']
    assert adapt_record(valid, PUBLIC_CARD).graph is None


def test_fixtures_have_expected_status_and_adjustment(records, frame):
    result = analyze(frame, records)
    assert (result['accepted_count'], result['rejected_count'], result['source_abstention_count']) == (2, 1, 1)
    assert result['graph_identification']['dummy_measured_x1']['adjustment_sets'] == [['X1']]
    assert result['graph_identification']['no_llm_evidence_graph']['adjustment_sets'] == [['X1', 'X2']]
    assert not result['graph_identification']['dummy_latent_u']['identified_by_backdoor']
    for name in ('dummy_latent_u', 'dummy_invalid_time', 'dummy_explicit_abstention'):
        assert result['methods'][name]['decision'] == 'abstain'
        assert result['methods'][name]['estimate'] is None


def test_one_nonidentified_graph_vetoes_robust_claim(valid, records, frame):
    before = analyze(frame, [valid])
    assert before['methods']['robust_screened']['decision'] == 'positive'
    latent = next(r for r in records if r['provenance']['fixture_id'] == 'dummy_latent_u')
    after = analyze(frame, [valid, latent])
    robust = after['methods']['robust_screened']
    assert robust['decision'] == 'abstain'
    assert robust['envelope'] is None and robust['estimate'] is None
    assert 'dummy_latent_u' in robust['nonidentified_graphs']


def test_all_abstained_and_empty_inputs_cannot_inherit_evidence_vote(frame, records):
    abstained = next(r for r in records if r['response']['abstain'])
    for proposals in ([], [abstained]):
        result = analyze(frame, proposals)
        assert result['methods']['no_llm_evidence_graph']['decision'] == 'positive'
        assert result['methods']['robust_screened']['decision'] == 'abstain'
        assert result['methods']['naive_majority']['decision'] == 'abstain'
        assert not result['edge_frequencies_not_probabilities']


def test_invalid_majority_never_bypasses_checks(records, frame):
    invalid = next(r for r in records if r['provenance']['fixture_id'] == 'dummy_invalid_time')
    invalids = []
    for index in range(3):
        r = deepcopy(invalid)
        r['provenance']['fixture_id'] += f'_{index}'
        invalids.append(r)
    result = analyze(frame, invalids)
    assert result['rejected_count'] == 3
    assert result['methods']['naive_majority']['decision'] == 'abstain'
    assert not result['edge_frequencies_not_probabilities']


def test_majority_latent_confounding_and_selection_still_require_id(valid, frame):
    records = []
    for i in range(3):
        r = deepcopy(valid)
        r['provenance']['fixture_id'] += f'_{i}'
        for a, b in [('U', 'A'), ('U', 'Y')]:
            e = deepcopy(r['response']['edges'][0]); e.update(source=a, target=b)
            r['response']['edges'].append(e)
        records.append(r)
    result = analyze(frame, records)
    assert result['methods']['naive_majority']['decision'] == 'abstain'
    e = deepcopy(valid['response']['edges'][0]); e.update(source='Y', target='observed', causal_role='selection')
    valid['response']['edges'].append(e)
    assert analyze(frame, [valid])['methods']['robust_screened']['decision'] == 'abstain'


def test_post_variables_never_eligible_even_if_not_descendants(valid):
    # M is a post-exposure common cause under this bad scientific hypothesis;
    # temporal arrows M->A are forbidden and M is never a baseline control.
    graph = adapt_record(valid, PUBLIC_CARD).graph
    assert identify_pre_exposure(graph)['adjustment_sets'] == [['X1']]


def test_truth_wrapper_cannot_change_proposals_or_decisions(records, frame):
    class Wrapper:
        def __init__(self, data, truth, label):
            self.frame, self.true_ate, self.mechanism = data, truth, label
        @property
        def true_edges(self):
            raise AssertionError('Decision attempted to read hidden graph')
    a = Wrapper(frame.copy(), 100, 'hidden-one')
    b = Wrapper(frame.copy(), -100, 'hidden-two')
    x, y = analyze(a.frame, records), analyze(b.frame, deepcopy(records))
    same(x, y)
    fixed = json.dumps(x, sort_keys=True, allow_nan=False)
    first = score_fixed_predictions(fixed, truth=a.true_ate, identified=True)
    second = score_fixed_predictions(fixed, truth=b.true_ate, identified=False)
    assert first != second
    assert json.loads(fixed)['methods'] == x['methods']
    with pytest.raises(ValueError):
        analyze(a, records)


def test_physical_import_firewall_and_public_prompt_signature():
    for module in (bridge_v2, synthetic_adapter, synthetic_schema):
        tree = ast.parse(inspect.getsource(module))
        imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not any(any(s in imp for s in ('smoke_data', 'scoring', 'challenge', 'scm', 'openrouter', 'httpx'))
                       for imp in imports)
        for secret in ('true_ate', 'true_edges', 'mechanism', 'potential_outcomes'):
            assert secret not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert list(inspect.signature(synthetic_schema.prompt_payload).parameters) == ['card', 'variant']
    altered = deepcopy(PUBLIC_CARD); altered['true_ate'] = 10
    with pytest.raises(ValueError): synthetic_schema.prompt_payload(altered)
    altered = deepcopy(PUBLIC_CARD); altered['times']['M'] = 0
    with pytest.raises(ValueError): synthetic_schema.prompt_payload(altered)


@pytest.mark.parametrize('extra', ['true_ate', 'true_edges', 'mechanism', 'mergeid'])
def test_nonobservable_columns_and_metadata_rejected(frame, records, extra):
    frame[extra] = 1
    with pytest.raises(ValueError, match='observable'):
        analyze(frame, records)
    clean = frame.drop(columns=extra)
    clean.attrs[extra] = 'hidden'
    with pytest.raises(ValueError): analyze(clean, records)


def test_missing_outcome_and_descendant_leakage(records):
    frame = smoke_tasks()[2].frame
    altered = frame.copy()
    assert (frame.observed == 0).any()
    altered.loc[altered.observed == 0, ['Y', 'C', 'M']] = [np.nan, np.inf, -1e30]
    pd.testing.assert_frame_equal(frame.loc[frame.observed == 1], altered.loc[altered.observed == 1], check_exact=True)
    x, y = analyze(frame, records), analyze(altered, records)
    same(x, y)
    assert all(p['decision'] == 'abstain' for p in x['methods'].values())


@pytest.mark.parametrize('kind', ['none_observed', 'constant_exposure', 'fractional_exposure', 'nonfinite_outcome'])
def test_bad_estimation_input_fails_closed(valid, frame, kind):
    if kind == 'none_observed': frame['observed'] = 0
    elif kind == 'constant_exposure': frame['A'] = 1
    elif kind == 'fractional_exposure':
        frame['A'] = frame['A'].astype(float)
        frame.loc[0, 'A'] = 0.9
    elif kind == 'nonfinite_outcome': frame.loc[0, 'Y'] = np.inf
    result = analyze(frame, [valid])
    assert all(p['decision'] == 'abstain' for p in result['methods'].values())
    assert result['methods']['robust_screened']['envelope'] is None


@pytest.mark.parametrize('radius', [-1, float('nan'), float('inf')])
def test_invalid_sensitivity_configuration(valid, frame, radius):
    with pytest.raises(ValueError): analyze(frame, [valid], bias_radius=radius)


def test_no_network_even_with_key_and_live_environment(monkeypatch, tmp_path, capsys):
    import httpx
    from sharecausal.llm import openrouter
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('attempt')
        raise AssertionError('Live transport path exercised')
    for name in ('available_models', 'complete'):
        monkeypatch.setattr(openrouter, name, forbidden)
    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'request', forbidden)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'TEST_KEY_NEVER_EXPORT_SENTINEL')
    monkeypatch.setenv('SHARECAUSAL_LIVE_LLM_APPROVED', '1')
    entry = runpy.run_path(str(ROOT / 'scripts/llm_bridge_v2_offline.py'))
    out = tmp_path / 'smoke'
    assert entry['main'](['--fixture-dir', str(FIXTURES), '--results-dir', str(out)]) == 0
    assert calls == []
    assert 'TEST_KEY_NEVER_EXPORT_SENTINEL' not in capsys.readouterr().out
    for file in out.iterdir():
        assert 'TEST_KEY_NEVER_EXPORT_SENTINEL' not in file.read_text()
    with offline_network_veto():
        with pytest.raises(RuntimeError, match='forbids all network'):
            socket.create_connection(('example.invalid', 443))
        with socket.socket() as sock:
            with pytest.raises(RuntimeError, match='forbids all network'):
                sock.connect(('127.0.0.1', 1))


def test_offline_outputs_parse_and_predictions_fixed_before_scoring(tmp_path, monkeypatch):
    from sharecausal.benchmark import bridge_v2_offline as runner
    scorer = runner.score_fixed_predictions
    seen = []
    def verify_fixed(serialized, **kwargs):
        file = tmp_path / f'prediction_{len(seen)}.json'
        assert file.read_text() == serialized
        seen.append(hashlib.sha256(serialized.encode()).hexdigest())
        return scorer(serialized, **kwargs)
    monkeypatch.setattr(runner, 'score_fixed_predictions', verify_fixed)
    report = run_offline_smoke(FIXTURES, tmp_path)
    parsed = json.loads((tmp_path / 'smoke_report.json').read_text())
    assert parsed == report
    assert parsed['fixture_mode'] is True
    assert parsed['real_llm_calls'] == parsed['share_records_used'] == parsed['openrouter_cost_usd'] == 0
    assert parsed['n_tasks'] == 3 and parsed['n_method_rows'] == 21
    assert parsed['failures'] == []
    for item in parsed['graph_screen_audit']:
        assert (item['accepted'], item['rejected'], item['source_abstained']) == (2, 1, 1)
    assert seen == [p['prediction_sha256'] for p in report['predictions']]
    assert len(pd.read_csv(tmp_path / 'run_level.csv')) == 21
    assert len(pd.read_csv(tmp_path / 'summary.csv')) == 7
    assert len(json.loads((tmp_path / 'graph_provenance.json').read_text())) == 3
    assert report['summary'][-1]['method'] == 'robust_screened'
    assert report['summary'][-1]['false_sign_risk'] is None


@pytest.mark.parametrize('mode', ['missing_dir', 'empty_dir', 'existing_results', 'private_fixture', 'private_results', 'live_flag'])
def test_command_rejects_bad_configuration(tmp_path, mode):
    fixture, output = FIXTURES, tmp_path / 'output'
    extra = []
    if mode == 'missing_dir': fixture = tmp_path / 'missing'
    elif mode == 'empty_dir':
        fixture = tmp_path / 'empty'; fixture.mkdir()
    elif mode == 'existing_results':
        output.mkdir(); (output / 'retain.txt').write_text('unchanged')
    elif mode == 'private_fixture': fixture = Path.home() / 'share-research/private'
    elif mode == 'private_results': output = tmp_path / 'private/output'
    elif mode == 'live_flag': extra = ['--live']
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/llm_bridge_v2_offline.py'),
          '--fixture-dir', str(fixture), '--results-dir', str(output), *extra],
          capture_output=True, text=True, timeout=30, cwd=ROOT)
    assert result.returncode == 2
    if mode == 'existing_results': assert (output / 'retain.txt').read_text() == 'unchanged'


def test_private_path_veto_precedes_any_read(monkeypatch):
    reads = []
    def forbidden(*args, **kwargs):
        reads.append(args); raise AssertionError('Private path touched')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(Path, 'glob', forbidden)
    with monkeypatch.context() as metadata:
        metadata.setattr(Path, 'resolve', forbidden)
        with pytest.raises(ValueError, match='Private'):
            load_fixture_records(Path.home() / 'share-research/private')
    assert reads == []
    assert safe_path('/tmp/synthetic-output').name == 'synthetic-output'


def test_duplicate_sources_rejected_and_prompt_hash_is_checked(valid, frame):
    with pytest.raises(ValueError, match='Duplicate fixture'):
        analyze(frame, [valid, deepcopy(valid)])
    assert len(prompt_hash(PUBLIC_CARD)) == 64
    assert adapt_response('null', PUBLIC_CARD, valid['provenance']).graph is None


def test_graph_admission_independent_of_outcomes_and_public_card_only(valid, frame):
    altered = frame.copy()
    altered['Y'] = -20 * frame['Y']
    altered['C'] = 1e20
    x, y = analyze(frame, [valid]), analyze(altered, [valid])
    assert x['graph_audit'] == y['graph_audit']
    assert x['graph_identification'] == y['graph_identification']
    assert x['methods']['robust_screened']['decision'] == 'positive'
    assert y['methods']['robust_screened']['decision'] == 'negative'


def test_bias_scenarios_can_suppress_but_not_create_claims(valid, frame):
    result = analyze(frame, [valid], bias_radius=100)
    robust = result['methods']['robust_screened']
    assert robust['decision'] == 'abstain'
    assert robust['envelope']['lower'] < 0 < robust['envelope']['upper']
    assert robust['estimate'] is None


def test_majority_is_validated_even_if_internal_adapter_contract_is_broken(monkeypatch, valid, frame):
    from dataclasses import replace
    adapted = adapt_record(valid, PUBLIC_CARD)
    bad = replace(adapted, graph=replace(adapted.graph, edges=(('M', 'A'), ('M', 'Y'))))
    monkeypatch.setattr(bridge_v2, 'adapt_record', lambda *args: bad)
    result = analyze(frame, [valid])
    assert result['methods']['naive_majority']['decision'] == 'abstain'
    assert result['methods']['robust_screened']['decision'] == 'abstain'


def test_protocol_freeze_checker_unchanged_and_passes():
    result = subprocess.run([sys.executable, 'scripts/check_protocol_freeze.py'], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '14 checked; 1 documented test-only erratum accepted' in result.stdout
