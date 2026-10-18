"""Blinded graph/decision engine with offline candidate fixtures.

Scientific scope: this tests the *pipeline*, not real LLM ability or a causal
identification guarantee.  Candidate DAGs are fixed across all mechanisms and
seeds, with mock sources, public variable timing and no knowledge of truth.
This module does not import SCM generators, truth labels or scoring functions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd

from sharecausal.causal.estimators import aipw, unadjusted
from sharecausal.causal.graphs import GraphSpec, validate_temporal_graph
from sharecausal.causal.identification import identify
from sharecausal.causal.robust_conclusions import ConditionalInterval, assess
from .ablations import majority_graph

# Intentionally not loaded from any data-generating mechanism.  These are
# visible questionnaire-style metadata given to all mock graph proposers.
# C is measured after the outcome; M after A; neither is a baseline control.
PUBLIC_CARD = {
    'treatment': 'A', 'outcome': 'Y',
    'description': 'Hypothetical exposure A and continuous subsequent outcome Y',
    'variables': {'X1': 'pre-exposure factor', 'X2': 'pre-exposure factor',
                  'M': 'post-exposure measure', 'C': 'post-outcome measure',
                  'observed': 'indicator of whether Y was collected'},
    'times': {'X1': 0, 'X2': 0, 'A': 1, 'M': 2, 'Y': 3, 'C': 4},
}
_OBSERVED = frozenset({'X1', 'X2', 'A', 'Y', 'M', 'C'})
_BASE = (('X1', 'A'), ('X2', 'A'), ('X1', 'Y'), ('X2', 'Y'), ('A', 'Y'))
_POST = (('A', 'M'), ('A', 'C'), ('Y', 'C'))


@dataclass(frozen=True)
class Proposal:
    source: str
    graph: GraphSpec
    provenance: str


def offline_proposals() -> tuple[Proposal, ...]:
    """Four fixed *synthetic source fixtures*, NEVER real LLM responses.

    A and C assume both X variables are confounders; B disputes X2->Y.
    D asserts a temporally impossible M->A.  No generator-label argument,
    observed outcome statistics or ground truth are used in these proposals.
    """
    return (
        Proposal('fixture_A', GraphSpec('fixture_A', _BASE + _POST, _OBSERVED,
                                       'Two pre-exposure confounders'),
                 'offline://fixture_A/v1'),
        Proposal('fixture_B', GraphSpec('fixture_B',
                 tuple(e for e in _BASE if e != ('X2', 'Y')) + _POST,
                 _OBSERVED, 'X2 assumed to affect participation but not Y'),
                 'offline://fixture_B/v1'),
        Proposal('fixture_C', GraphSpec('fixture_C', _BASE + _POST + (('M', 'Y'),),
                                       _OBSERVED, 'M may mediate part of total effect'),
                 'offline://fixture_C/v1'),
        Proposal('fixture_D', GraphSpec('fixture_D', _BASE + (('M', 'A'), ('M', 'Y')),
                                       _OBSERVED, 'Deliberate temporal error'),
                 'offline://fixture_D/v1'),
    )


def screen(proposals: tuple[Proposal, ...], card: Mapping = PUBLIC_CARD) -> tuple[list[Proposal], dict[str, str]]:
    """Source-agnostic validity filter: DAG + temporal and provenance checks.

    Validity is NOT evidence that any accepted graph is causally correct. A
    scientific PI must review source support before real-world application.
    """
    kept, rejected = [], {}
    if not card.get('times'):
        raise ValueError('Required temporal metadata missing')
    for p in proposals:
        try:
            if not p.provenance.startswith('offline://'):
                raise ValueError('Missing auditable offline provenance')
            validate_temporal_graph(p.graph, dict(card['times']))
            if not {'A', 'Y'} <= p.graph.observed:
                raise ValueError('Treatment/outcome absent')
        except ValueError as exc:
            rejected[p.source] = str(exc)
        else:
            kept.append(p)
    return kept, rejected


def _decision(fit: dict) -> str:
    if fit['ci_upper'] < 0:
        return 'negative'
    if fit['ci_lower'] > 0:
        return 'positive'
    return 'abstain'


def _claim(fit: dict | None, decision: str | None = None, **extras) -> dict:
    if fit is None:
        return dict(estimate=None, ci_lower=None, ci_upper=None,
                    decision='abstain', assertion_margin=0.0, **extras)
    resolved = decision if decision is not None else _decision(fit)
    margin = (max(float(fit['ci_lower']), -float(fit['ci_upper']), 0.0)
              if resolved != 'abstain' else 0.0)
    return dict(estimate=float(fit['effect']), ci_lower=float(fit['ci_lower']),
                ci_upper=float(fit['ci_upper']), decision=resolved,
                assertion_margin=margin, **extras)


def analyze_blind(frame: pd.DataFrame, *, seed: int, bias_radius: float = 0.25,
                  card: Mapping = PUBLIC_CARD) -> dict:
    """Construct and assess graph claims using observable data alone.

    Interface deliberately accepts *only* a DataFrame, public metadata, seed,
    and a precommitted confounding shift.  Never pass SCMData, scenario labels,
    DAG truth or true effects. Missing outcomes are entirely excluded before
    fitting; their existence is used for a conservative selection warning.
    """
    if bias_radius < 0 or not math.isfinite(bias_radius):
        raise ValueError('Bias radius must be finite and nonnegative')
    required = {'X1', 'X2', 'A', 'Y', 'M', 'C', 'observed'}
    if not required <= set(frame):
        raise ValueError(f'Missing observable columns: {sorted(required - set(frame))}')
    if not frame['observed'].isin([0, 1]).all():
        raise ValueError('Invalid follow-up indicator')
    incomplete_followup = bool((frame['observed'] == 0).any())
    df = frame.loc[frame['observed'] == 1].copy()
    if df.empty:
        raise ValueError('No observed outcomes')
    if not np.isfinite(df[['A', 'Y', 'X1', 'X2', 'M', 'C']].to_numpy()).all():
        raise ValueError('Non-finite measured values')
    a = df['A'].to_numpy(dtype=int)
    y = df['Y'].to_numpy(dtype=float)
    fit_cache: dict[tuple[str, ...], dict] = {}

    def estimate(covariates: tuple[str, ...]):
        covariates = tuple(sorted(covariates))
        if covariates not in fit_cache:
            if not covariates:
                fit_cache[covariates] = unadjusted(y, a)
            else:
                fit_cache[covariates] = aipw(
                    y, a, df[list(covariates)].to_numpy(), folds=3, seed=seed
                ).as_dict()
        return fit_cache[covariates]

    props = offline_proposals()
    supported, rejected = screen(props, card)
    results = {}
    results['unadjusted'] = _claim(estimate(()))
    results['no_llm_evidence_graph'] = _claim(estimate(('X1', 'X2')))
    results['naive_all_adjust'] = _claim(estimate(('X1', 'X2', 'M', 'C')),
                                          warning='Conditions on post-exposure variables')

    for p in props:
        if p.source in rejected:
            results[p.source] = _claim(None, reason='Invalid candidate graph: ' + rejected[p.source])
            continue
        identification = identify(p.graph)
        if not identification['identified_by_backdoor']:
            results[p.source] = _claim(None, reason='No observed backdoor set')
        else:
            adjustment = tuple(identification['adjustment_sets'][0])
            results[p.source] = _claim(estimate(adjustment), adjustment_set=list(adjustment))

    # Majority uses all four candidate fixtures *without* screening,
    # modelling a naive directed-edge vote; fail closed on cycles/temporal error.
    majority = majority_graph([p.graph for p in props])
    try:
        validate_temporal_graph(majority, dict(card['times']))
        mid = identify(majority)
        if not mid['identified_by_backdoor']:
            results['naive_majority'] = _claim(None, reason='No observed backdoor set')
        else:
            adjusted = tuple(mid['adjustment_sets'][0])
            results['naive_majority'] = _claim(estimate(adjusted), adjustment_set=list(adjusted))
    except ValueError as exc:
        results['naive_majority'] = _claim(None, reason=str(exc))

    intervals = []
    accepted = []
    for p in supported:
        id_result = identify(p.graph)
        if not id_result['identified_by_backdoor']:
            intervals.append(ConditionalInterval(p.source, 0, 0, 0, False))
            continue
        covariates = tuple(id_result['adjustment_sets'][0])
        fitted = estimate(covariates)
        intervals.append(ConditionalInterval(p.source, fitted['effect'],
                        fitted['ci_lower'], fitted['ci_upper'], True))
        accepted.append({'source': p.source, 'provenance': p.provenance,
                         'adjustment_set': list(covariates)})
    if incomplete_followup:
        intervals.append(ConditionalInterval('unresolved_outcome_selection', 0, 0, 0, False))
    verdict = assess(intervals, shifts=(-bias_radius, 0, bias_radius),
                     expected_graphs=[x.graph for x in intervals])
    # Robust methods change a conclusion, not a point estimator.  Report the
    # fixed single-graph estimate for diagnostic RMSE ONLY, never as a new ATE.
    representative = estimate(('X1', 'X2'))
    results['robust_screened'] = _claim(representative, verdict['decision'],
        reason=verdict['reason'], envelope=verdict['envelope'],
        meaning='Conditional robustness rule, not a structural confidence interval')
    if verdict['envelope'] is not None and verdict['decision'] != 'abstain':
        bounds = verdict['envelope']
        results['robust_screened']['assertion_margin'] = max(
            float(bounds['lower']), -float(bounds['upper']), 0.0)
    else:
        results['robust_screened']['assertion_margin'] = 0.0

    strict_graph = GraphSpec('latent_U_stress', _BASE + (('U', 'A'), ('U', 'Y')),
                             _OBSERVED, 'Admitted unmeasured common cause')
    strict_id = identify(strict_graph)
    if strict_id['identified_by_backdoor']:
        raise AssertionError('Latent confounded stress DAG unexpectedly identified')
    strict_verdict = assess(intervals + [ConditionalInterval('latent_U_stress', 0, 0, 0, False)],
                            shifts=(-bias_radius, 0, bias_radius))
    results['robust_plus_unmeasured_U'] = _claim(
        representative, strict_verdict['decision'], reason=strict_verdict['reason'])
    return {'methods': results, 'screened_sources': accepted,
            'rejected_sources': rejected, 'missing_outcome_detected': incomplete_followup,
            'n_observed': len(df), 'n_total': len(frame),
            'oracle_information_used': False,
            'proposal_source_type': 'offline_fixed_fixtures_not_real_llms',
            'bias_radius': bias_radius}

