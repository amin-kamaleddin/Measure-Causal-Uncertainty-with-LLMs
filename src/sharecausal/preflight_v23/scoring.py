"""Post-seal per-card primary endpoints and fixed risk/assertion curves."""
from fractions import Fraction as F

COMPONENTS=('wrong_sign_identified_nonnull','assertion_on_proven_nonidentification',
            'supported_structural_zero_violation')


def score(claims,worlds,certificates,tasks,practical_zero):
    if practical_zero<0 or set(claims)!=set(tasks):raise ValueError('eligible_card_claim_coverage_required')
    if len({w.world_id for w in worlds})!=len(worlds):raise ValueError('duplicate_world_ID')
    if set(w.public_task_hash for w in worlds)!=set(tasks):raise ValueError('world_task_coverage_required')
    if any(c not in ('positive','negative','abstain') for c in claims.values()):raise ValueError('invalid_fixed_claim')
    rows=[]
    for h in sorted(tasks):
        group=[w for w in worlds if w.public_task_hash==h];claim=claims[h];asserted=claim!='abstain'
        components={k:False for k in COMPONENTS};unknown=False;compat=[]
        for w in group:
            cert=certificates[w.world_id];ate=F(w.population_effect_exact)
            nonnull=abs(ate)>F(str(practical_zero));same=nonnull and ((ate>0)==(claim=='positive'))
            components[COMPONENTS[0]]|=asserted and cert['status']=='identified' and nonnull and not same
            components[COMPONENTS[1]]|=asserted and cert['status']=='provably_nonidentified'
            components[COMPONENTS[2]]|=asserted and cert['supported_structural_zero']
            unknown|=asserted and cert['status']=='uncertified';compat.append(same)
        rows.append({'task_hash':h,'decision':claim,'asserted':asserted,**components,
            'known_unjustified':any(components.values()),'unresolved_assertion':unknown,
            'all_prespecified_world_signs_compatible':all(compat) if asserted else None,
            'world_probes_not_independent':len(group)})
    n=len(rows);assertions=sum(r['asserted'] for r in rows);known=sum(r['known_unjustified'] for r in rows)
    unknown=sum(r['unresolved_assertion'] and not r['known_unjustified'] for r in rows)
    counts={k:sum(r[k] for r in rows) for k in COMPONENTS}
    return {'eligible_cards':n,'assertions':assertions,'assertion_rate':assertions/n,
            'abstention_rate':1-assertions/n,'known_unjustified_count':known,
            'unresolved_additional_count':unknown,'unjustified_rate_lower':known/n,
            'unjustified_rate_upper':(known+unknown)/n,'component_counts':counts,
            'component_rates_per_all_cards':{k:v/n for k,v in counts.items()},
            'unjustified_risk_among_assertions_lower':known/assertions if assertions else None,
            'unjustified_risk_among_assertions_upper':(known+unknown)/assertions if assertions else None,
            'wrong_sign_identified_nonnull_per_all_assertions':counts[COMPONENTS[0]]/assertions if assertions else None,
            'risk_denominator_asserted_cards':assertions,'cards':rows}


def fixed_curves(probes,thresholds,practical_zeros,worlds,certificates,tasks,target,tolerance):
    rows=[]
    for name,decisions in probes.items():
        if set(decisions)!=set(tasks):raise ValueError('probe_eligible_card_mismatch')
        for threshold in thresholds:
            claims={h:r['decision'] if r['margin']>threshold else 'abstain' for h,r in decisions.items()}
            for practical_zero in practical_zeros:
                s=score(claims,worlds,certificates,tasks,practical_zero)
                rows.append({'probe':name,'threshold_fixed_before_truth':threshold,'practical_zero':practical_zero,
                    **{k:v for k,v in s.items() if k not in ('cards','component_counts','component_rates_per_all_cards')},
                    **s['component_counts'],'coverage_matched':abs(s['assertion_rate']-target)<=tolerance,
                    'status':'scoring_contract_probe_NOT_model_performance'})
    return rows
