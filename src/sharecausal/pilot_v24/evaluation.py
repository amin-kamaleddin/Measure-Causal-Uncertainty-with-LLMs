"""Independent scorer process: consumes a verified sealed batch, never edits predictions."""
import math
from fractions import Fraction as F
import numpy as np
from scipy.stats import t
from sharecausal.preflight_v23.certificates import Truth,certify
from sharecausal.preflight_v23.scoring import score
from sharecausal.preflight_v23.precision import bounded_half_width
from sharecausal.preflight_v23.packets import task_hash
from .constants import METHODS


def paired_families(differences,families):
    if set(differences)!=set(families) or not differences:raise ValueError('paired_card_inventory_invalid')
    n=len(differences);labels=sorted(set(families.values()));g=len(labels)
    if g<3:raise ValueError('at_least_three_declared_families_required')
    mean=sum(differences.values())/n
    sums=np.array([sum(differences[h]-mean for h in differences if families[h]==f) for f in labels])/n
    se=math.sqrt(g/(g-1)*float(np.square(sums).sum()))
    half_t=float(t.ppf(.975,g-1))*se
    weights=[sum(families[h]==f for h in families)/n for f in labels]
    half_b=bounded_half_width(weights)
    return {'eligible_cards':n,'declared_families':g,'paired_mean_difference':mean,
            'family_cluster_se_exploratory':se,'family_t_lower_UNVALIDATED':max(-1.,mean-half_t),
            'family_t_upper_UNVALIDATED':min(1.,mean+half_t),
            'family_bound_lower_conditional_independence':max(-1.,mean-half_b),
            'family_bound_upper_conditional_independence':min(1.,mean+half_b),
            'family_bound_unclipped_half_width':half_b,
            'meaning':'conditional_on_exposed_cards_and_assumed_independent_declared_families_not_confirmatory'}


def evaluate(batch,raw_worlds,tasks,family_assignments,config):
    worlds=[Truth.model_validate(w) for w in raw_worlds]
    if len({w.world_id for w in worlds})!=len(worlds) or {w.public_task_hash for w in worlds}!=set(tasks):
        raise ValueError('truth_card_inventory_mismatch')
    family={x['task_hash']:x['family_id'] for x in family_assignments}
    if set(family)!=set(tasks) or any(family[w.public_task_hash]!=w.family_id for w in worlds):raise ValueError('truth_family_binding_invalid')
    certifications={w.world_id:certify(w,tasks[w.public_task_hash],worlds) for w in worlds}
    expected={(h,i) for h in tasks for i in range(config['sample_repetitions'])}
    indexed={(p['public_task_hash'],p['sample_repetition']):p for p in batch}
    if len(indexed)!=len(batch) or set(indexed)!=expected:raise ValueError('prediction_card_repetition_inventory_mismatch')
    if any(set(p['methods'])!=set(METHODS) for p in batch):raise ValueError('method_inventory_mismatch')
    main=[];curves=[];card_rows=[];replicate=[];estimates=[]
    for i in range(config['sample_repetitions']):
        predictions={h:indexed[h,i] for h in tasks}
        for method in METHODS:
            claims={h:p['methods'][method]['decision'] for h,p in predictions.items()}
            result=score(claims,worlds,certifications,tasks,.1)
            summary={k:v for k,v in result.items() if k not in ('cards','component_counts','component_rates_per_all_cards')}
            summary.update(method=method,sample_repetition=i,**result['component_counts'],
                estimator_failure_cards=sum(any(g.get('fit') and not g['fit']['ok'] for g in p['methods'][method].get('graphs',[]))
                                             for p in predictions.values()))
            summary['coverage_matched']=abs(result['assertion_rate']-config['assertion_target'])<=config['matching_tolerance']
            summary['precision_status']='exploratory_no_confirmatory_claim'
            replicate.append(summary)
            if i==config['primary_sample_repetition']:
                main.append(summary)
                card_rows.extend(dict(method=method,family_id=family[r['task_hash']],**r) for r in result['cards'])
                for threshold in config['fixed_margin_thresholds']:
                    filtered={h:p['methods'][method]['decision'] if p['methods'][method]['assertion_margin']>threshold
                              else 'abstain' for h,p in predictions.items()}
                    for practical in config['practical_zero_sensitivity']:
                        s=score(filtered,worlds,certifications,tasks,practical)
                        curves.append({'method':method,'margin_threshold':threshold,'practical_zero':practical,
                            **{k:v for k,v in s.items() if k not in ('cards','component_counts','component_rates_per_all_cards')},
                            **s['component_counts'],'coverage_matched':abs(s['assertion_rate']-config['assertion_target'])<=config['matching_tolerance'],
                            'threshold_source':'fixed_before_truth_no_retuning'})
            for h,p in predictions.items():
                pred=p['methods'][method]
                for w in (w for w in worlds if w.public_task_hash==h):
                    ate=float(F(w.population_effect_exact))
                    point=pred.get('estimate');lower=pred.get('ci_lower');upper=pred.get('ci_upper')
                    conditional=[g.get('conditional_decision') for g in pred.get('graphs',[])]
                    estimates.append({'method':method,'task_hash':h,'sample_repetition':i,'world_id':w.world_id,
                        'certification':certifications[w.world_id]['status'],'truth_population_effect':ate,
                        'point_estimate':point,'ci_lower':lower,'ci_upper':upper,
                        'prediction_scope':pred['scope'],
                        'estimate_target':'observed_case_association' if method=='association_unadjusted' else 'graph_conditional_population_total_effect',
                        'squared_error_descriptive_world_probe':(point-ate)**2 if point is not None else None,
                        'interval_covers_identified_truth_ONLY':bool(lower<=ate<=upper) if method!='association_unadjusted' and lower is not None and certifications[w.world_id]['status']=='identified' else None,
                        'unconditional_decision':pred['decision'],
                        'conditional_direction_without_unconditional_identification':bool(any(c in ('positive','negative') for c in conditional)
                            and certifications[w.world_id]['status']!='identified'),
                        'diagnostic_unit':'nested_world_sample_probe_not_independent_card'})
    ref={r['task_hash']:r for r in card_rows if r['method']=='no_llm_evidence_set'}
    pairs=[]
    for method in METHODS:
        rows={r['task_hash']:r for r in card_rows if r['method']==method}
        for metric in ('asserted','known_unjustified'):
            differences={h:float(rows[h][metric])-float(ref[h][metric]) for h in tasks}
            pairs.append({'method':method,'reference':'no_llm_evidence_set','metric':metric,
                **paired_families(differences,family),
                'both_arms_coverage_matched':next(r for r in main if r['method']==method)['coverage_matched']
                    and next(r for r in main if r['method']=='no_llm_evidence_set')['coverage_matched']})
    return {'main_summary':main,'per_card':card_rows,'nested_repetition_summary':replicate,'risk_assertion_curves':curves,
            'paired_family_differences':pairs,'world_estimation_diagnostics':estimates,'certifications':certifications,
            'eligible_cards':len(tasks),'declared_families':len(set(family.values())),
            'world_probes':len(worlds),'primary_sample_repetition':config['primary_sample_repetition'],
            'independent_truth_custody':'DECISION_REQUIRED_separate_process_single_team_exposed_design'}
