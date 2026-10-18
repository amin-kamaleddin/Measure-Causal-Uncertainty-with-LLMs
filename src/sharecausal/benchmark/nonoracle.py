"""Synthetic scoring boundary: only this module can see SCM ground truth.

Graph proposals and the analysis decision are called through the independent
blind.py module. Real SHARE data are not loaded here; no API calls occur.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from .blind import analyze_blind
from .challenge import (CHALLENGE_MECHANISMS, NONIDENTIFIABLE_GROUND_TRUTH,
                        generate_challenge)


def score_predictions(predictions: dict, *, truth: float, true_identified: bool,
                      scenario: str, seed: int, n: int, tolerance: float = 0.10) -> list[dict]:
    """The ONLY scoring boundary that receives known generating truth."""
    scored = []
    for method, p in predictions['methods'].items():
        direction = p['decision']
        asserted = direction in ('positive', 'negative')
        false_sign = (int(abs(truth) <= tolerance or
                       (direction == 'positive' and truth < -tolerance) or
                       (direction == 'negative' and truth > tolerance))
                      if asserted else 0)
        estimate = p['estimate']
        scored.append({'method': method, 'scenario': scenario, 'seed': int(seed),
            'n': int(n), 'n_observed': predictions['n_observed'],
            'truth': float(truth), 'truth_identifiable_by_backdoor': true_identified,
            'estimate': estimate, 'ci_lower': p['ci_lower'], 'ci_upper': p['ci_upper'],
            'cover': (p['ci_lower'] <= truth <= p['ci_upper'] if estimate is not None else None),
            'decision': direction, 'assertion_margin': float(p['assertion_margin']),
            'sign_error': false_sign,
            'asserted_nonidentifiable_truth': int(asserted and not true_identified),
            'reason': p.get('reason'), 'adjustment_set': p.get('adjustment_set')})
    return scored


def summarize(rows: list[dict]) -> dict:
    result = {}
    for method in sorted({r['method'] for r in rows}):
        cases = [r for r in rows if r['method'] == method]
        estimated = [r for r in cases if r['estimate'] is not None]
        claimed = [r for r in cases if r['decision'] != 'abstain']
        known_id = [r for r in estimated if r['truth_identifiable_by_backdoor']]
        result[method] = {
            'n_cases': len(cases), 'n_estimated': len(estimated),
            'n_asserted': len(claimed),
            'assertion_rate': len(claimed) / len(cases),
            'false_sign_count': sum(r['sign_error'] for r in claimed),
            'false_sign_risk_given_assertion': (sum(r['sign_error'] for r in claimed) / len(claimed)
                                                if claimed else None),
            'asserted_nonidentifiable_count': sum(r['asserted_nonidentifiable_truth'] for r in claimed),
            'rmse_descriptive_estimate': (math.sqrt(np.mean([(r['estimate'] - r['truth'])**2
                                          for r in estimated])) if estimated else None),
            'interval_coverage_on_identifiable_truth': (float(np.mean([r['cover'] for r in known_id]))
                                                     if known_id else None),
        }
    return result


def risk_coverage(rows: list[dict], thresholds=(0.0, 0.1, 0.25, 0.5)) -> list[dict]:
    """Fixed, precommitted thresholds applied only to observable margins.

    The truth is read only to SCORE retained assertions, never to choose one.
    This is not a calibrated prediction-set guarantee.
    """
    curve = []
    for method in sorted({r['method'] for r in rows}):
        subset = [r for r in rows if r['method'] == method]
        for threshold in thresholds:
            selected = [r for r in subset if r['decision'] != 'abstain'
                        and r['assertion_margin'] > threshold]
            curve.append({'method': method, 'margin_threshold': threshold,
                          'n_total': len(subset), 'n_asserted': len(selected),
                          'assertion_rate': len(selected)/len(subset),
                          'false_sign_risk': (sum(r['sign_error'] for r in selected)/len(selected)
                                              if selected else None),
                          'unidentified_claim_rate': (sum(r['asserted_nonidentifiable_truth']
                                                    for r in selected)/len(selected)
                                                    if selected else None)})
    return curve


def benchmark_nonoracle(*, seeds=(1001, 1002, 1003), n=500, results_dir=None,
                       mechanisms=CHALLENGE_MECHANISMS, bias_radius=0.25) -> dict:
    """Held-out *mechanism-family* check; labels are absent from analyze_blind().

    This is a post-pilot, internally frozen challenge, not a prospectively
    externally registered or yet-independent replication.  Candidates do not
    adapt to any synthetic truth, but generator source is inspectable.
    """
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Unique, non-empty seed list required')
    rows, failures, audit = [], [], []
    for mechanism in mechanisms:
        for seed in seeds:
            dgp = generate_challenge(mechanism, n, int(seed))
            try:
                # **Information firewall:** only a DataFrame is supplied. The
                # blind analyzer cannot access `true_ate`, `true_edges`, or
                # the generator's scenario label.
                prediction = analyze_blind(dgp.frame, seed=int(seed), bias_radius=bias_radius)
            except Exception as exc:
                failures.append({'mechanism': mechanism, 'seed': int(seed),
                                 'error_type': type(exc).__name__, 'detail': str(exc)})
                continue
            rows.extend(score_predictions(prediction, truth=dgp.true_ate,
                        true_identified=mechanism not in NONIDENTIFIABLE_GROUND_TRUTH,
                        scenario=mechanism, seed=int(seed), n=n))
            audit.append({'mechanism': mechanism, 'seed': int(seed),
                          'n_observed': prediction['n_observed'],
                          'proposal_type': prediction['proposal_source_type'],
                          'accepted': [p['source'] for p in prediction['screened_sources']],
                          'rejected': prediction['rejected_sources'],
                          'true_edges_only_used_in_scoring': True})
    summary = summarize(rows)
    curve = risk_coverage(rows)
    ret = {'kind': 'nonoracle_offline_fixture_benchmark',
           'publication_status': 'exploratory internally frozen challenge, not externally preregistered',
           'real_llm_calls': 0, 'share_records_used': 0,
           'mechanisms': list(mechanisms), 'seeds': list(seeds), 'n_per_replication': n,
           'bias_radius': bias_radius, 'false_sign_tolerance': 0.10,
           'summary': summary, 'risk_coverage': curve, 'failures': failures,
           'graph_screen_audit': audit, 'rows': rows,
           'warning': 'Fixed offline graph fixtures, NOT real LLMs. No oracle labels or effects are used for proposals/decisions. '
                      'True mechanism labels are used ONLY to score answers. Selection/hidden confounding '
                      'remain unidentified without further assumptions; interval coverage here is not structural coverage.'}
    if results_dir is not None:
        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)
        from sharecausal.reporting.provenance import environment
        ret['environment'] = environment()
        (path / 'nonoracle_benchmark.json').write_text(json.dumps(ret, indent=2, allow_nan=False))
        pd.DataFrame(rows).to_csv(path / 'nonoracle_run_level.csv', index=False)
        pd.DataFrame([{'method': k, **v} for k, v in summary.items()]).to_csv(
            path / 'nonoracle_summary.csv', index=False)
        (path / 'nonoracle_graph_audit.json').write_text(json.dumps(audit, indent=2))
        pd.DataFrame(curve).to_csv(path / 'nonoracle_risk_coverage.csv', index=False)
        from sharecausal.reporting.plots import nonoracle_risk_plot
        nonoracle_risk_plot(curve, path)
    return ret


def calibrate_margins_on_development(public_rows: list[dict], target_rate=0.50) -> dict:
    """Pre-specified, *outcome-label-free* assertion-margin calibration.

    Caller must strip `truth`, `scenario`, `true_edges`, `sign_error` etc.
    This calibration is allowed to suppress assertions, never create them.
    An arm that inherently abstains more than target is marked infeasible.
    """
    if not 0 < target_rate < 1:
        raise ValueError('Target coverage must lie strictly between 0 and 1')
    if any(set(r) != {'method', 'assertion_margin', 'decision'} for r in public_rows):
        raise ValueError('Development calibration must receive public fields only')
    out = {}
    for method in sorted({r['method'] for r in public_rows}):
        group = [r for r in public_rows if r['method'] == method]
        available = sorted((r['assertion_margin'] for r in group
                            if r['decision'] != 'abstain' and r['assertion_margin'] > 0), reverse=True)
        goal = max(1, round(target_rate * len(group)))
        if len(available) < goal:
            cutoff = 0.0
            feasible = False
        elif len(available) == goal:
            cutoff = 0.0
            feasible = True
        else:
            a, b = available[goal - 1], available[goal]
            cutoff = (a + b) / 2 if a > b else a
            feasible = True
        achieved = sum(r['decision'] != 'abstain' and r['assertion_margin'] > cutoff
                       for r in group) / len(group)
        out[method] = {'threshold': float(cutoff), 'development_assertion_rate': achieved,
                       'target_rate': target_rate, 'feasible_in_development': feasible}
    return out


def evaluate_locked_thresholds(rows: list[dict], policy: dict, target_rate=0.5,
                               tolerance=0.1) -> dict:
    """Apply dev-only thresholds to held-out rows; no threshold adjustment."""
    results = {}
    for method in sorted(policy):
        group = [r for r in rows if r['method'] == method]
        if not group:
            raise ValueError(f'No test cases for {method}')
        thr = policy[method]['threshold']
        selected = [r for r in group if r['decision'] != 'abstain' and r['assertion_margin'] > thr]
        rate = len(selected)/len(group)
        risk = sum(r['sign_error'] for r in selected)/len(selected) if selected else None
        results[method] = {'target_rate': target_rate, 'threshold_from_development': thr,
            'development_feasible': policy[method]['feasible_in_development'],
            'heldout_assertion_rate': rate, 'n_heldout': len(group),
            'heldout_assertions': len(selected),
            'wrong_sign_count': sum(r['sign_error'] for r in selected),
            'false_sign_risk': risk,
            'not_identified_claims': sum(r['asserted_nonidentifiable_truth'] for r in selected),
            'comparable_at_target_coverage': (policy[method]['feasible_in_development']
                                               and abs(rate-target_rate) <= tolerance),
            'risk_undefined_reason': ('No assertions' if risk is None else None)}
    return results


def benchmark_locked_split(*, development_seeds: tuple[int, ...], heldout_seeds: tuple[int, ...],
                           n: int, target_rate: float, bias_radius: float,
                           results_dir: str | Path, config_sha256: str) -> dict:
    """Execute the frozen, non-oracle development and holdout evaluation."""
    if set(development_seeds) & set(heldout_seeds):
        raise ValueError('Development and held-out seeds overlap')
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    development = benchmark_nonoracle(seeds=development_seeds, n=n, bias_radius=bias_radius,
                                     results_dir=path / 'development')
    public_calibration = [dict(method=r['method'], assertion_margin=r['assertion_margin'],
                               decision=r['decision']) for r in development['rows']]
    policy = calibrate_margins_on_development(public_calibration, target_rate)
    heldout = benchmark_nonoracle(seeds=heldout_seeds, n=n, bias_radius=bias_radius,
                                 results_dir=path / 'heldout')
    calibrated = evaluate_locked_thresholds(heldout['rows'], policy, target_rate)
    report = {'kind':'frozen_nonoracle_split_v1',
              'config_sha256':config_sha256,
              'generator_families':list(CHALLENGE_MECHANISMS),
              'development_seeds':list(development_seeds),
              'heldout_seeds':list(heldout_seeds),
              'n_per_replication':n, 'target_rate':target_rate, 'bias_radius':bias_radius,
              'real_llm_calls':0, 'share_records_used':0,
              'development_failures':development['failures'],
              'heldout_failures':heldout['failures'],
              'calibration_does_not_use_truth_labels':True,
              'calibration':policy,'heldout_target_coverage_comparison':calibrated,
              'heldout_unthresholded_summary':heldout['summary'],
              'warning': 'Offline synthetic fixtures are not real LLM outputs. '
                 'This is an internally frozen, single-codebase evaluation, not an externally '
                 'preregistered confirmatory replication. Matched-coverage comparisons are '
                 'valid only where held-out assertion rates are close. Calibration selects '
                 'margin thresholds from *development observable diagnostics only*; '
                 'known structural truth is used strictly for scoring.'}
    (path/'locked_split_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
