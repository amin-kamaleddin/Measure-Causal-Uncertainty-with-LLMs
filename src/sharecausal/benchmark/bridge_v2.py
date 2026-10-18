"""V2 observable-only decision engine. No generators, scorers, or LLM clients.

Admission means schema/temporal validity under public assumptions, not proof of
causal truth. All accepted graphs constrain the robust decision, including U.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math

import networkx as nx
import numpy as np
import pandas as pd

from sharecausal.causal.estimators import aipw, unadjusted
from sharecausal.causal.graphs import GraphSpec, validate_temporal_graph
from sharecausal.causal.identification import identify
from sharecausal.causal.robust_conclusions import ConditionalInterval, assess
from sharecausal.llm.synthetic_adapter import adapt_record
from sharecausal.llm.synthetic_schema import MEASURED, PublicCard

_BASE = (('X1', 'A'), ('X2', 'A'), ('X1', 'Y'), ('X2', 'Y'), ('A', 'Y'))
IMPLEMENTATION_VERSION = 'v2.1-independent-review'


def identify_pre_exposure(graph: GraphSpec) -> dict:
    # Keep every edge/latent node; restrict conditioning to the public baseline
    # variables. Frozen identify() otherwise may admit disconnected M or C.
    result = identify(replace(graph, observed=frozenset({'A', 'Y', 'X1', 'X2'})))
    has_path = nx.has_path(graph.networkx(), 'A', 'Y')
    result.update(has_directed_causal_path=has_path, structural_zero=not has_path,
                  identification_scope='Observed-variable pre-exposure backdoor criterion only',
                  total_effect_status=('graph_entailed_structural_zero' if not has_path else
                      'identified_by_pre_exposure_backdoor' if result['identified_by_backdoor'] else
                      'not_identified_by_backdoor'))
    return result


def _abstain(reason: str) -> dict:
    return {'decision': 'abstain', 'reason': reason, 'estimate': None,
            'ci_lower': None, 'ci_upper': None, 'assertion_margin': 0.0,
            'envelope': None}


def analyze_observables(frame: pd.DataFrame, records: list[dict], *, card: dict,
                        seed: int = 601, bias_radius: float = 0.25) -> dict:
    """Accept only observables and public records, never a generating wrapper.

    Extra columns and DataFrame.attrs are rejected to make leakage conspicuous.
    No outcome-dependent screening of candidate graphs is permitted.
    """
    public = PublicCard.model_validate(card)
    if not isinstance(records, list):
        raise ValueError('Records must be a list')
    if not math.isfinite(bias_radius) or bias_radius < 0:
        raise ValueError('Bias radius must be finite and nonnegative')
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != MEASURED or frame.attrs:
        raise ValueError('Exactly observable columns and no metadata required')
    if not frame['observed'].isin([0, 1]).all():
        raise ValueError('Invalid observation indicator')

    # Proposal admission occurs before any examination of outcome statistics.
    proposals = [adapt_record(r, card) for r in records]
    ids = [p.audit['provenance']['fixture_id'] for p in proposals
           if p.audit['provenance'] and p.audit['status'] != 'rejected']
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate fixture identity; repetitions require separate fixture IDs')
    accepted = [p for p in proposals if p.graph is not None]
    audit = [p.audit for p in proposals]
    missing = bool((frame['observed'] == 0).any())
    # Discard Y AND all descendants for R=0 before validating numeric input.
    df = frame.loc[frame['observed'] == 1, sorted(MEASURED - {'observed'})].copy()
    methods, graph_results, fit_cache = {}, {}, {}

    def fit(adjustment):
        key = tuple(adjustment)
        if key not in fit_cache:
            values = df[['A', 'Y', *key]].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not df['A'].isin([0, 1]).all():
                raise ValueError('Nonfinite observed input or nonbinary exposure')
            if len(df) < 30:
                raise ValueError('Insufficient observed outcomes')
            result = (aipw(df.Y, df.A, df[list(key)].to_numpy(), folds=3, seed=seed).as_dict()
                      if key else unadjusted(df.Y, df.A))
            if not all(math.isfinite(float(result[k])) for k in ('effect', 'ci_lower', 'ci_upper')):
                raise ValueError('Nonfinite fit')
            fit_cache[key] = result
        return fit_cache[key]

    def graph_claim(graph):
        identification = None
        try:
            validate_temporal_graph(graph, public.times)
            if not graph.supported:
                return _abstain('Unsupported graph'), {'identified_by_backdoor': False}
            identification = identify_pre_exposure(graph)
            if identification['structural_zero']:
                claim = _abstain('Graph entails zero total effect: no directed A-to-Y causal path')
                claim.update(structural_effect=0.0,
                             estimand='population_total_effect',
                             total_effect_status=identification['total_effect_status'],
                             model_compatibility='not_tested',
                             compatibility_warning='Structural exclusion is assumed, not empirically verified')
                return claim, identification
            if not identification['identified_by_backdoor']:
                claim = _abstain('Not identified by observed pre-exposure backdoor adjustment or unresolved selection')
                claim.update(total_effect_status=identification['total_effect_status'],
                             model_compatibility='not_tested')
                return claim, identification
            adjustment = identification['adjustment_sets'][0]
            fitted = fit(adjustment)
            verdict = assess([ConditionalInterval(graph.name, fitted['effect'],
                             fitted['ci_lower'], fitted['ci_upper'], True)])
            decision = verdict['decision']
            reason = verdict['reason']
            if missing:
                decision, reason = 'abstain', 'Unresolved outcome selection: some outcomes unobserved'
            claim = {'estimate': float(fitted['effect']), 'ci_lower': float(fitted['ci_lower']),
                     'ci_upper': float(fitted['ci_upper']), 'decision': decision, 'reason': reason,
                     'adjustment_set': adjustment, 'estimand': 'population_total_effect',
                     'total_effect_status': identification['total_effect_status'],
                     'model_compatibility': 'not_tested',
                     'assertion_margin': (max(fitted['ci_lower'], -fitted['ci_upper'], 0.0)
                                          if decision != 'abstain' else 0.0),
                     'interval_meaning': 'Model-conditional sampling interval only',
                     'estimator_diagnostics': {k: v for k, v in fitted.items()
                                               if k not in ('effect', 'ci_lower', 'ci_upper')}}
            return claim, identification
        except (ValueError, TypeError, FloatingPointError):
            # No input values or raw exception text in public result logs.
            status = ({'identified_by_backdoor': False, 'graph_validation_failed': True}
                      if identification is None else {**identification, 'estimation_failed': True})
            claim = _abstain('Invalid graph or insufficient/invalid observable estimation input')
            claim.update(model_compatibility='not_tested',
                         total_effect_status=status.get('total_effect_status', 'invalid_graph'))
            return claim, status

    evidence = GraphSpec('no_llm_evidence_graph', _BASE, MEASURED,
                         'Fixed public evidence assumptions; never an imputed source vote')
    methods['no_llm_evidence_graph'], graph_results[evidence.name] = graph_claim(evidence)
    intervals = []
    identity_counts = Counter(p.audit['provenance']['fixture_id'] for p in proposals
                              if p.audit['provenance'])
    structural_zeros = []
    estimation_failures = []
    for index, p in enumerate(proposals):
        name = (p.audit['provenance']['fixture_id'] if p.audit['provenance']
                else f'rejected_record_{index}')
        p.audit['record_index'] = index
        if p.audit['status'] == 'rejected' and identity_counts[name] > 1:
            name = f'rejected_record_{index}'
        if p.graph is None:
            methods[name] = _abstain(p.audit['reason'])
            continue
        claim, identification = graph_claim(p.graph)
        methods[name] = claim
        graph_results[name] = identification
        p.audit['identification'] = identification
        # Estimation failure blocks assertions just as missing identification.
        zero = identification.get('structural_zero', False)
        if zero:
            structural_zeros.append(name)
        if identification.get('estimation_failed'):
            estimation_failures.append(name)
        good = zero or claim['estimate'] is not None
        intervals.append(ConditionalInterval(name, claim['estimate'] or 0.0,
                         claim['ci_lower'] or 0.0, claim['ci_upper'] or 0.0, good))

    # Vote over accepted nonabstaining source records only; never substitute the
    # evidence graph for a rejected response. Votes are not edge probabilities.
    counts = Counter(e for p in accepted for e in p.graph.edges)
    majority_edges = tuple(sorted(e for e, count in counts.items() if count > len(accepted) / 2))
    if not accepted or not majority_edges:
        methods['naive_majority'] = _abstain('No accepted majority graph')
    else:
        majority = GraphSpec('naive_majority', majority_edges, MEASURED,
                             'Strict directed-edge majority, followed by identification',
                             selection_warning=any(b == 'observed' for _, b in majority_edges))
        methods['naive_majority'], graph_results[majority.name] = graph_claim(majority)

    if missing:
        intervals.append(ConditionalInterval('unresolved_outcome_selection', 0, 0, 0, False))
    verdict = assess(intervals, shifts=(-bias_radius, 0, bias_radius),
                     expected_graphs=[p.graph.name for p in accepted]
                     + (['unresolved_outcome_selection'] if missing else []))
    # Empty accepted set must remain an abstention even if selection sentinel exists.
    if not accepted:
        verdict = {'decision': 'abstain', 'reason': 'No accepted source graphs', 'envelope': None}
    robust = _abstain(verdict['reason'])
    robust.update(verdict)
    robust['structural_zero_graphs'] = structural_zeros
    robust['estimation_failure_graphs'] = estimation_failures
    if 'nonidentified_graphs' in robust:
        robust['nonidentified_graphs'] = [name for name in robust['nonidentified_graphs']
                                         if name not in estimation_failures]
    if estimation_failures and verdict['envelope'] is None:
        robust['reason'] = 'At least one supported graph lacks a valid conditional estimate; assertion withheld'
    if structural_zeros and verdict['envelope'] is not None:
        robust['reason'] = 'A supported graph entails structural zero; directional total-effect assertion withheld'
    robust['meaning'] = 'Assumption-robustness envelope; not a simultaneous 95% confidence set'
    robust['bias_shifts'] = [-bias_radius, 0.0, bias_radius]
    if robust['decision'] != 'abstain':
        bounds = robust['envelope']
        robust['assertion_margin'] = max(bounds['lower'], -bounds['upper'], 0.0)
    # No representative point estimate masquerading as a robust causal effect.
    methods['robust_screened'] = robust
    return {'implementation_version': IMPLEMENTATION_VERSION,
            'methods': methods, 'graph_audit': audit, 'graph_identification': graph_results,
            'accepted_count': len(accepted),
            'rejected_count': sum(a['status'] == 'rejected' for a in audit),
            'source_abstention_count': sum(a['status'] == 'source_abstained' for a in audit),
            'edge_frequencies_not_probabilities': {f'{a}->{b}': n / len(accepted)
                                                   for (a, b), n in sorted(counts.items())},
            'n_total': len(frame), 'n_observed': len(df), 'missing_outcome_detected': missing,
            'bias_radius': bias_radius, 'oracle_information_used': False}
