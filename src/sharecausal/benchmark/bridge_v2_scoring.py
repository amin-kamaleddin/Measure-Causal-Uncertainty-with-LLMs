"""Post-decision scorer. This module alone reads synthetic truth for scoring."""
from __future__ import annotations

import json
import math


def score_fixed_predictions(prediction_json: str, *, truth: float,
                            identified: bool, tolerance: float = 0.10) -> list[dict]:
    """Deserialize an already fixed prediction; never return updated decisions."""
    if not math.isfinite(truth) or tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError('Invalid scoring truth/tolerance')
    predictions = json.loads(prediction_json)
    rows = []
    for method, p in predictions['methods'].items():
        asserted = p['decision'] in ('positive', 'negative')
        wrong = asserted and (abs(truth) <= tolerance or
                              (p['decision'] == 'positive' and truth < 0) or
                              (p['decision'] == 'negative' and truth > 0))
        rows.append({'method': method, 'decision': p['decision'], 'asserted': asserted,
                     'estimate': p['estimate'], 'ci_lower': p['ci_lower'], 'ci_upper': p['ci_upper'],
                     'assertion_margin': p['assertion_margin'], 'reason': p['reason'],
                     'graph_total_effect_status': p.get('total_effect_status'),
                     'model_compatibility': p.get('model_compatibility'),
                     'graph_entailed_structural_effect': p.get('structural_effect'),
                     'truth': truth, 'truth_identified': identified, 'sign_error': int(wrong),
                     'asserted_nonidentified': int(asserted and not identified),
                     'squared_error': ((p['estimate'] - truth) ** 2 if p['estimate'] is not None else None),
                     'conditional_interval_covers': (p['ci_lower'] <= truth <= p['ci_upper']
                         if identified and p['estimate'] is not None else None)})
    return rows


def summarize_scores(rows: list[dict]) -> list[dict]:
    summary = []
    for method in sorted({r['method'] for r in rows}):
        cases = [r for r in rows if r['method'] == method]
        asserted = [r for r in cases if r['asserted']]
        errors = sum(r['sign_error'] for r in asserted)
        mse = [r['squared_error'] for r in cases if r['squared_error'] is not None]
        coverage = [r['conditional_interval_covers'] for r in cases
                    if r['conditional_interval_covers'] is not None]
        summary.append({'method': method, 'n_tasks': len(cases), 'n_asserted': len(asserted),
                        'assertion_rate': len(asserted) / len(cases), 'false_sign_count': errors,
                        'false_sign_risk': errors / len(asserted) if asserted else None,
                        'asserted_nonidentified_count': sum(r['asserted_nonidentified'] for r in cases),
                        'rmse_conditional_point_estimates': math.sqrt(sum(mse) / len(mse)) if mse else None,
                        'conditional_interval_coverage_identifiable_truth': (
                            sum(coverage) / len(coverage) if coverage else None)})
    return summary
