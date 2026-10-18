"""Observable-only v2.4 graph-to-estimator bridge. No generator/certificate/scorer imports."""
import math
import numpy as np
from sharecausal.causal.estimators import aipw,unadjusted
from sharecausal.causal.robust_conclusions import ConditionalInterval,assess
from sharecausal.preflight.contracts import sha
from sharecausal.preflight_v23.admission import audit,no_llm,supported_exclusion
from sharecausal.preflight_v23.packets import task_hash,POLICIES
from .graphs import identify_graph,public_direction_eligible
from .observations import clean_frame,frame_hash
from .capture import vote

from .constants import METHODS


def abstain(reason):
    return {'decision':'abstain','reason':reason,'estimate':None,'ci_lower':None,'ci_upper':None,
            'envelope':None,'assertion_margin':0.,'scope':'no_directional_causal_claim'}


def analyze(task,frame,selected,models,settings):
    if set(selected)-set(models):raise ValueError('unexpected_source_identity')
    base=[p.model_dump() for p in no_llm(task)]
    majority,majority_audit=vote(task,selected,models)
    candidates=base+[raw for value in selected.values() for raw in value['proposals']]+([majority] if majority else [])
    public_checks={}
    # Fix graph admission AND estimator/adjustment choices before examining any sample value.
    for raw in candidates:
        for policy in POLICIES:
            admission,p=audit(task,raw,policy)
            identity=identify_graph(task,[(e.source,e.target) for e in p.graph.edges]) if admission['admitted'] else None
            public_checks[sha([raw,policy])]=(admission,identity)
    clean=clean_frame(task,frame);recorded=clean.loc[clean.observed==1].copy()
    sample_digest=frame_hash(task,clean);missing=bool((clean.observed==0).any())
    fit_cache={};graph_cache={};fits=[]

    def fit(adjustment):
        key=tuple(adjustment)
        if key not in fit_cache:
            try:
                if len(recorded)<30:raise ValueError('too_few_recorded_cases')
                result=(aipw(recorded.Y,recorded.A,recorded[list(key)].to_numpy(dtype=float),
                        folds=settings['folds'],seed=settings['estimator_seed'],trim=settings['propensity_clip']).as_dict()
                        if key else unadjusted(recorded.Y,recorded.A))
                if any(not math.isfinite(float(result[k])) for k in ('effect','se','ci_lower','ci_upper')):
                    raise ValueError('nonfinite_estimator')
                if key and result['overlap_fraction_below_005_or_above_095']>settings['maximum_extreme_propensity_fraction']:
                    raise ValueError('positivity_diagnostic_failed')
                fit_cache[key]={'ok':True,'estimate':float(result['effect']),
                    'ci_lower':float(result['ci_lower']),'ci_upper':float(result['ci_upper']),
                    'estimator':'cross_fitted_AIPW' if key else 'difference_in_means',
                    'adjustment':list(key),'diagnostics':result,
                    'interval_meaning':'approximate_model_conditional_sampling_95_interval_not_structural_uncertainty'}
            except (ValueError,TypeError,FloatingPointError,ArithmeticError):
                fit_cache[key]={'ok':False,'reason':'estimator_or_positivity_failure','adjustment':list(key)}
            fits.append(fit_cache[key])
        return fit_cache[key]

    def conditional(raw,policy):
        key=sha([raw,policy])
        if key in graph_cache:return graph_cache[key]
        admission,identified=public_checks[key]
        row={'admission':admission,'identification':None,'fit':None,'conditional_decision':'abstain'}
        if admission['admitted']:
            row['identification']=identified
            if identified['status']=='graph_conditional_structural_zero':
                row.update(structural_effect=0.,conditional_zero=True)
            elif identified['status']=='identified_by_baseline_backdoor':
                fitted=fit(identified['adjustment']);row['fit']=fitted
                if fitted['ok']:
                    row['conditional_decision']=('positive' if fitted['ci_lower']>0 else
                        'negative' if fitted['ci_upper']<0 else 'abstain')
        graph_cache[key]=row;return row

    def combine(raws,policy,shifts):
        rows=[conditional(raw,policy) for raw in raws]
        accepted=[r for r in rows if r['admission']['admitted']]
        out=abstain('no_admitted_graph');out['graphs']=rows
        if not accepted:return out
        if supported_exclusion(task):
            # Public global exclusion dominates observational associations for every causal arm.
            out.update(reason='public_supported_structural_zero',structural_effect=0.,scope='public_identified_zero');return out
        # Rejected/absent sources remain in audits/denominators but are not imputed graphs.
        if any(r['identification']['status']=='graph_conditional_structural_zero' for r in accepted):
            out.update(reason='candidate_graph_entailed_zero_blocks_direction');return out
        if any(r['identification']['status']!='identified_by_baseline_backdoor' for r in accepted):
            out.update(reason='graph_identification_or_selection_unresolved');return out
        if any(not r['fit']['ok'] for r in accepted):
            out.update(reason='admitted_graph_estimator_failed');return out
        intervals=[ConditionalInterval(str(i),r['fit']['estimate'],r['fit']['ci_lower'],r['fit']['ci_upper'],True)
                   for i,r in enumerate(accepted)]
        verdict=assess(intervals,shifts=shifts,expected_graphs=[str(i) for i in range(len(accepted))])
        out.update(envelope=verdict['envelope'],conditional_decision=verdict['decision'],
            scope='graph_conditional_estimates_public_claim_gate',bias_shifts=list(shifts),
            uncertainty_meaning='assumption_robustness_envelope_not_simultaneous_95_confidence_set')
        if len(accepted)==1:
            f=accepted[0]['fit'];out.update(estimate=f['estimate'],ci_lower=f['ci_lower'],ci_upper=f['ci_upper'])
        if missing or task.card.selection!='complete_outcomes':out['reason']='outcome_recording_unresolved'
        elif not public_direction_eligible(task):out['reason']='public_identification_not_established_no_unconditional_direction'
        else:
            out.update(decision=verdict['decision'],reason=verdict['reason'],scope='public_randomization_plus_graph_conditional_sampling')
            if out['decision']!='abstain':
                env=out['envelope'];out['assertion_margin']=max(env['lower'],-env['upper'],0.)
        return out

    methods={};association=fit(())
    methods['association_unadjusted']=abstain('association_only_not_causal_identification')
    if association['ok']:
        methods['association_unadjusted'].update(estimate=association['estimate'],ci_lower=association['ci_lower'],
            ci_upper=association['ci_upper'],scope='observed_case_association_only',
            association_direction='positive' if association['ci_lower']>0 else 'negative' if association['ci_upper']<0 else 'abstain')
    else:methods['association_unadjusted']['reason']='association_estimator_failed'
    methods['no_llm_evidence_set']=combine(base,'credible_alternatives',settings['bias_shifts'])
    source_audits={};all_raw=[]
    for model in models:
        parsed=selected.get(model,{'status':'absent','proposals':[]})
        raws=parsed['proposals'];all_raw.extend(raws)
        arm='source_'+model.split('/')[0][-1]
        methods[arm]=combine(raws,'credible_alternatives',[0.]) if raws else abstain(parsed['status'])
        methods[arm]['source_response_status']=parsed['status']
        source_audits[model]={'response_status':parsed['status'],
            'policy_audits':{policy:[audit(task,raw,policy)[0] for raw in raws] for policy in POLICIES}}
    methods['single_preselected_source']=dict(methods['source_'+models[0].split('/')[0][-1]])
    methods['naive_strict_majority']=(combine([majority],'permissive_sensitivity',[0.])
        if majority else abstain(majority_audit['reason']))
    methods['naive_strict_majority']['majority_audit']=majority_audit
    for policy in POLICIES:methods['robust_'+policy]=combine(all_raw,policy,settings['bias_shifts'])
    methods['always_abstain']=abstain('prespecified_abstention_reference_not_a_success')
    if set(methods)!=set(METHODS):raise ValueError('method_inventory_mismatch')
    return {'schema_version':'pilot-prediction-2.4','public_task_hash':task_hash(task),
            'analysis_sample_sha256':sample_digest,'n_eligible_sample_units':len(clean),'n_recorded':len(recorded),
            'missing_outcome':missing,'methods':methods,'source_audit':source_audits,
            'planned_source_denominator':len(models),'estimator_fits':fits,
            'estimator_failure_count':sum(not f['ok'] for f in fits),
            'graph_results':list(graph_cache.values()),'oracle_information_used':False}
