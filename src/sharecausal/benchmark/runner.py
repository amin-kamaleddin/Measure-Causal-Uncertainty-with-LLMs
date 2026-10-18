"""A runnable multi-SCM benchmark: no models or SHARE data are called."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .scm import generate, SCENARIOS
from .metrics import aggregate
from .ablations import graph_accuracy
from sharecausal.causal.estimators import aipw,unadjusted
from sharecausal.causal.robust_conclusions import assess, ConditionalInterval


def _record(method, scenario, seed, n, truth, fit, verdict=None):
    est=float(fit['effect'])
    lo=float(fit['ci_lower']);hi=float(fit['ci_upper'])
    verdict=verdict or ('negative' if hi<0 else 'positive' if lo>0 else 'abstain')
    return {'method':method,'scenario':scenario,'seed':seed,'n':n,
            'truth':truth,'estimate':est,'ci_lower':lo,'ci_upper':hi,
            'cover':bool(lo<=truth<=hi),'decision':verdict,
            'sign_error':int((verdict=='positive' and truth<=0) or (verdict=='negative' and truth>=0)),
            'scenario_is_oracle_labeled':True}


def benchmark(quick=True, seeds=None, n=700, results_dir=None):
    """Conservative DAG inclusion uses synthetic truth for *validation*, not deployable graph selection.

    Oracle graph-set informed robust arm is intentionally advantaged; outcomes
    cannot establish improvement from actual LLM candidate generation.
    """
    scenarios=SCENARIOS if not quick else ('measured','hidden','mediator','collider','weak_positivity','selection')
    seeds=seeds if seeds is not None else ([11,29,47,67] if quick else list(range(100,140)))
    rows=[];failures=[];graph_qualities=[]
    for scenario in scenarios:
        for seed in seeds:
            obj=generate(scenario,n,seed)
            if seed==seeds[0]:
                assumed=(('X1','A'),('X2','A'),('X1','Y'),('X2','Y'),('A','Y'))
                quality=graph_accuracy(assumed,obj.true_edges)
                graph_qualities.append({'scenario':scenario,'directed_recall':quality['recall'],
                    'directed_precision':quality['precision'],'directed_shd':quality['directed_shd']})
            d=obj.frame; d=d.loc[d.observed==1].copy()
            a=d.A.to_numpy(dtype=int);y=d.Y.to_numpy(dtype=float)
            if min(np.bincount(a,minlength=2))<15:
                failures.append({'scenario':scenario,'seed':seed,'error':'Too few exposed/unexposed'});continue
            try:
                raw=unadjusted(y,a)
                single=aipw(y,a,d[['X1','X2']].to_numpy(),folds=3,seed=seed).as_dict()
                all_adjust=aipw(y,a,d[['X1','X2','M','C']].to_numpy(),folds=3,seed=seed).as_dict()
            except Exception as exc:
                failures.append({'scenario':scenario,'seed':seed,'error':str(exc)});continue
            for method,estimate in [('unadjusted',raw),('single_graph_pre',single),('naive_all_adjust',all_adjust)]:
                rows.append(_record(method,scenario,seed,n,obj.true_ate,estimate))
            # Graph set contains acknowledged hidden or selection mechanism; conservative oracle ablation.
            is_nonid=scenario in ('hidden','selection')
            # For collider/mediator ambiguity include post-treatment-adjusted alternate estimate,
            # but this is a deliberately conservative comparison, not legitimate total-effect pooling.
            candidate=[ConditionalInterval('pre_adjusted',single['effect'],single['ci_lower'],single['ci_upper'],True)]
            if is_nonid:
                candidate.append(ConditionalInterval('unmeasured_or_selection',0.,0.,0.,False))
            if scenario in ('collider','mediator'):
                candidate.append(ConditionalInterval('post_treatment_assumed_pre',all_adjust['effect'],
                          all_adjust['ci_lower'],all_adjust['ci_upper'],True))
            verdict=assess(candidate,shifts=(-.25,0,.25))
            row=_record('oracle_graph_set_robust',scenario,seed,n,obj.true_ate,single,verdict['decision'])
            row['why_abstain']=verdict['reason'];row['robust_envelope']=verdict['envelope']
            rows.append(row)
    summary=aggregate(rows)
    result={'kind':'independent_synthetic','scm_version':'0.2_validated_colliders','quick':quick,'scenarios':list(scenarios),
            'seeds':list(seeds),'n_per_simulation':n,'summary':summary,
            'failed_runs':failures,'rows':rows,'fixed_graph_baseline_quality':graph_qualities,
            'warning':'Oracle graph-set membership derived from generating mechanisms. '
                'NOT performance of LLM inference or calibrated full structural intervals; '
                'coverage in hidden/selection scenarios is not identifiable 95% ATE inference.'}
    if results_dir is not None:
        import pandas as pd
        from sharecausal.reporting.provenance import environment
        path=Path(results_dir);path.mkdir(parents=True,exist_ok=True)
        result['environment']=environment()
        (path/'benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        pd.DataFrame(rows).to_csv(path/'benchmark_run_level.csv',index=False)
        pd.DataFrame([{'method':k,**v} for k,v in summary.items()]).to_csv(path/'benchmark_summary.csv',index=False)
        pd.DataFrame(graph_qualities).to_csv(path/'fixed_graph_quality.csv',index=False)
    return result
