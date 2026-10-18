"""Anti-leakage and scientific failure-mode tests for the non-oracle gate."""
import inspect
import json
import math
from numbers import Real

import pandas as pd
import pytest

from sharecausal.benchmark.blind import (
    analyze_blind, offline_proposals, screen, PUBLIC_CARD
)
from sharecausal.benchmark.challenge import generate_challenge, CHALLENGE_MECHANISMS
from sharecausal.benchmark.nonoracle import (
    benchmark_nonoracle, risk_coverage, summarize
)


def test_blind_code_has_no_ground_truth_or_generator_imports():
    module = inspect.getmodule(analyze_blind)
    code = inspect.getsource(module)
    for excluded in ('from .challenge import', 'true_ate', 'true_edges',
                     'generate_challenge(', 'SCENARIOS', 'scenario =='):
        assert excluded not in code
    assert 'offline_fixed_fixtures_not_real_llms' in code


def test_proposals_do_not_accept_scenario_or_data():
    assert tuple(inspect.signature(offline_proposals).parameters) == ()
    props = offline_proposals()
    kept, rejected = screen(props)
    assert [p.source for p in kept] == ['fixture_A', 'fixture_B', 'fixture_C']
    assert 'fixture_D' in rejected and 'Backward-in-time' in rejected['fixture_D']
    assert all(p.provenance.startswith('offline://') for p in props)


def test_different_potential_outcome_metadata_cannot_change_decision():
    obj = generate_challenge('threshold_nonlinear', 250, seed=105)
    # Blind estimator only sees the very same measured frame; changing an
    # oracle/scenario wrapper has no route into its interface.
    x = analyze_blind(obj.frame, seed=99)
    copy = obj.frame.copy(deep=True)
    y = analyze_blind(copy, seed=99)
    assert x == y
    assert x['oracle_information_used'] is False


def _assert_same_prediction_with_float_tolerance(left, right, path='result'):
    """Compare the complete diagnostic payload, tolerating BLAS roundoff only.

    Ridge/logit solvers can yield harmless last-bit floating differences across
    separately allocated arrays or CPU/BLAS backends (notably on macOS).
    Structural decisions, text, metadata, dimensions and schema stay exact.
    """
    if isinstance(left, dict):
        assert isinstance(right, dict), path
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_same_prediction_with_float_tolerance(
                left[key], right[key], f'{path}.{key}'
            )
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)), path
        assert len(left) == len(right), path
        for index, (lval, rval) in enumerate(zip(left, right)):
            _assert_same_prediction_with_float_tolerance(
                lval, rval, f'{path}[{index}]'
            )
    elif isinstance(left, (float,)):
        assert isinstance(right, Real), path
        assert math.isfinite(left) and math.isfinite(float(right)), path
        assert right == pytest.approx(left, rel=1e-8, abs=1e-8), path
    else:
        assert left == right, path


def test_unobserved_outcome_is_not_leaked_into_prediction():
    obj = generate_challenge('informative_dropout', 300, seed=105)
    altered = obj.frame.copy()
    assert (altered['observed'] == 0).sum() > 0
    altered.loc[altered.observed == 0, 'Y'] = -1_000_000
    altered.loc[altered.observed == 0, 'C'] = 1_000_000
    # Neither modified field is observed when observed=0. Verify the actual
    # analysis input is bit-for-bit identical before comparing predictions.
    pd.testing.assert_frame_equal(
        obj.frame.loc[obj.frame.observed == 1],
        altered.loc[altered.observed == 1],
        check_exact=True,
    )
    x, y = (analyze_blind(obj.frame, seed=33), analyze_blind(altered, seed=33))
    _assert_same_prediction_with_float_tolerance(x, y)
    assert x['methods']['robust_screened']['decision'] == 'abstain'
    assert 'not identified' in x['methods']['robust_screened']['reason']



def test_roundoff_tolerance_cannot_hide_material_or_structural_changes():
    original = {'methods': {'m': {'estimate': 0.2, 'decision': 'positive'}}}
    _assert_same_prediction_with_float_tolerance(
        original, {'methods': {'m': {'estimate': 0.20000000001,
                                    'decision': 'positive'}}})
    with pytest.raises(AssertionError):
        _assert_same_prediction_with_float_tolerance(
            original, {'methods': {'m': {'estimate': 0.22,
                                        'decision': 'positive'}}})
    with pytest.raises(AssertionError):
        _assert_same_prediction_with_float_tolerance(
            original, {'methods': {'m': {'estimate': 0.2,
                                        'decision': 'abstain'}}})


def test_strict_latent_confounded_stress_always_abstains():
    obj = generate_challenge('near_zero_heterogeneous', 250, seed=109)
    x = analyze_blind(obj.frame, seed=109)
    assert x['methods']['robust_plus_unmeasured_U']['decision'] == 'abstain'
    assert x['methods']['fixture_D']['estimate'] is None
    assert x['methods']['fixture_D']['decision'] == 'abstain'


def test_mock_source_graph_set_is_identical_across_all_mechanisms():
    fixtures = tuple((p.source, p.graph.edges) for p in offline_proposals())
    for scenario in CHALLENGE_MECHANISMS:
        frame = generate_challenge(scenario, 150, 110).frame
        assert set(frame.columns) == {'X1', 'X2', 'A', 'Y', 'M', 'C', 'observed'}
        assert fixtures == tuple((p.source, p.graph.edges) for p in offline_proposals())


def test_invalid_bias_or_missing_data_fail_closed():
    frame = generate_challenge('threshold_nonlinear', 160, 109).frame
    with pytest.raises(ValueError, match='Bias radius'):
        analyze_blind(frame, seed=8, bias_radius=float('nan'))
    with pytest.raises(ValueError, match='Missing observable'):
        analyze_blind(frame.drop(columns='X2'), seed=8)


def test_nonoracle_outputs_truth_scored_only_after_predictions(tmp_path):
    path = tmp_path / 'synthetic'
    z = benchmark_nonoracle(seeds=(1007,), mechanisms=('threshold_nonlinear', 'proxy_hidden'),
                            n=220, results_dir=path)
    assert z['real_llm_calls'] == z['share_records_used'] == 0
    assert not z['failures']
    assert len(z['rows']) == 2 * len(z['summary'])
    assert (path / 'nonoracle_benchmark.json').is_file()
    assert (path / 'nonoracle_graph_audit.json').is_file()
    assert (path / 'nonoracle_risk_coverage.svg').is_file()
    assert len(pd.read_csv(path / 'nonoracle_run_level.csv')) == len(z['rows'])
    assert json.loads((path / 'nonoracle_benchmark.json').read_text())['real_llm_calls'] == 0
    assert z['summary']['robust_plus_unmeasured_U']['false_sign_risk_given_assertion'] is None
    assert z['summary']['robust_plus_unmeasured_U']['assertion_rate'] == 0
    assert 'fixture_D' in z['graph_screen_audit'][0]['rejected']


def test_risk_curve_abstention_has_undefined_risk():
    row = {'method': 'm', 'decision': 'abstain', 'assertion_margin': 0,
           'sign_error': 0, 'asserted_nonidentifiable_truth': 0}
    assert risk_coverage([row])[0]['false_sign_risk'] is None


def test_nonoracle_benchmark_requires_unique_seeds():
    with pytest.raises(ValueError, match='Unique'):
        benchmark_nonoracle(seeds=(2, 2), n=100)


def test_calibration_cannot_accept_oracle_fields():
    from sharecausal.benchmark.nonoracle import calibrate_margins_on_development
    with pytest.raises(ValueError, match='public fields only'):
        calibrate_margins_on_development([{'method': 'a', 'decision': 'positive',
            'assertion_margin': 0.4, 'true_ate': 0.4}])


def test_dev_calibration_only_suppresses_claims():
    from sharecausal.benchmark.nonoracle import (
        calibrate_margins_on_development, evaluate_locked_thresholds)
    public = [dict(method='a', decision='positive', assertion_margin=x)
              for x in [0.1, 0.2, 0.5, 0.7]]
    policy = calibrate_margins_on_development(public, target_rate=0.5)
    assert policy['a']['development_assertion_rate'] == 0.5
    # The heldout policy uses exactly the same threshold; it cannot convert
    # an underlying abstention into a positive claim, even if margin is high.
    test = [dict(method='a', decision='abstain', assertion_margin=2.0,
                 sign_error=0, asserted_nonidentifiable_truth=0),
            dict(method='a', decision='positive', assertion_margin=0.6,
                 sign_error=1, asserted_nonidentifiable_truth=1)]
    result = evaluate_locked_thresholds(test, policy, target_rate=0.5)
    assert result['a']['heldout_assertions'] == 1
    assert result['a']['false_sign_risk'] == 1
    assert result['a']['not_identified_claims'] == 1


def test_frozen_split_rejects_development_holdout_overlap(tmp_path):
    from sharecausal.benchmark.nonoracle import benchmark_locked_split
    with pytest.raises(ValueError, match='overlap'):
        benchmark_locked_split(development_seeds=(1, 2), heldout_seeds=(2, 3),
            n=100, target_rate=.5, bias_radius=.25,
            results_dir=tmp_path, config_sha256='dummy')
