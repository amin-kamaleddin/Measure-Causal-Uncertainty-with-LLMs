"""Benchmarks: separate conditional interval coverage from selective risk."""
import math
import numpy as np

def aggregate(rows):
    keys=sorted(set(x['method'] for x in rows))
    out={}
    for k in keys:
        ar=[x for x in rows if x['method']==k]
        numeric=[x for x in ar if math.isfinite(x['estimate'])]
        asserted=[x for x in ar if x['decision'] in ('positive','negative')]
        out[k]={'scenarios':sorted(set(x['scenario'] for x in ar)),
                'replicates':len(ar),'computed_estimates':len(numeric),
                'mean_bias':float(np.mean([x['estimate']-x['truth'] for x in numeric])) if numeric else None,
                'rmse':float(np.sqrt(np.mean([(x['estimate']-x['truth'])**2 for x in numeric]))) if numeric else None,
                'all_scenario_empirical_interval_coverage_not_guaranteed':float(np.mean([x['cover'] for x in numeric])) if numeric else None,
                'conditionally_identifiable_scenario_coverage':float(np.mean([x['cover'] for x in numeric
                    if x['scenario'] not in ('hidden','selection')])) if any(x['scenario'] not in ('hidden','selection') for x in numeric) else None,
                'assertion_rate':len(asserted)/len(ar) if ar else None,
                'selective_false_sign_risk':(sum(x['sign_error'] for x in asserted)/len(asserted)) if asserted else None,
                'n_assertions':len(asserted),
                'n_false_sign_assertions':sum(x['sign_error'] for x in asserted)}
    return out
