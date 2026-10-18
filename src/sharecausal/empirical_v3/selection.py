"""Truncation by death is never treated as ordinary missing depression."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sharecausal.causal.estimators import frame_matrix, _propensity
from .statistics import influence_interval

STATUSES = ['known_dead_undefined_Y', 'recorded_alive_observed_Y',
            'recorded_alive_noninterview', 'recorded_alive_interview_missing_Y',
            'unknown_vital_status']


def classify(d):
    dead=d.deceased9.eq(1) | d.interview9.eq(2)
    alive=d.deceased9.eq(0) | d.interview9.eq(1)
    if (dead & (alive | d.Y.notna())).any():
        raise ValueError('Contradictory death/interview/outcome status; manual review required')
    status=pd.Series('unknown_vital_status',index=d.index)
    status.loc[dead]='known_dead_undefined_Y'
    status.loc[alive & d.Y.notna()]='recorded_alive_observed_Y'
    status.loc[alive & d.Y.isna() & ~d.interview9.eq(1)]='recorded_alive_noninterview'
    status.loc[alive & d.Y.isna() & d.interview9.eq(1)]='recorded_alive_interview_missing_Y'
    if (d.Y.notna() & ~alive).any():
        raise ValueError('Observed Y without survival evidence')
    return status


def counts(d):
    d=d.copy();d['status9']=classify(d)
    rows=[]
    for country,g in [('ALL',d),*[(str(int(k)),v) for k,v in d.groupby('country')]]:
        for arm,s in g.groupby('A'):
            record=dict(country=country,A=int(arm),n_eligible=len(s))
            record.update({name:int(s.status9.eq(name).sum()) for name in STATUSES})
            if sum(record[k] for k in STATUSES)!=len(s):raise ValueError('Status denominator mismatch')
            rows.append(record)
    return rows


def mnar(d, shifts):
    status=classify(d)
    alive=d.loc[status.str.startswith('recorded_alive')].copy()
    summaries={}
    for arm in [0,1]:
        group=alive.loc[alive.A.eq(arm)]
        seen=group.Y.dropna()
        if not len(group) or not len(seen):raise ValueError('Insufficient survivor cells')
        summaries[arm]=dict(n=len(group),observed=len(seen),missing=len(group)-len(seen),mean=float(seen.mean()))
    def completed(arm, delta):
        s=summaries[arm]
        imputed=float(np.clip(s['mean']+delta,0,12))
        return (s['observed']*s['mean']+s['missing']*imputed)/s['n']
    rows=[dict(delta_unexposed=float(d0),delta_exposed=float(d1),
               mean_difference=completed(1,d1)-completed(0,d0),
               n_recorded_alive=len(alive),n_missing=sum(s['missing'] for s in summaries.values()),
               scope='unadjusted_recorded_alive_pattern_mixture_not_causal') for d0 in shifts for d1 in shifts]
    low={a:s['observed']*s['mean']/s['n'] for a,s in summaries.items()}
    high={a:(s['observed']*s['mean']+12*s['missing'])/s['n'] for a,s in summaries.items()}
    return rows,dict(by_arm={str(k):v for k,v in summaries.items()},
                     worst_case_difference_lower=low[1]-high[0],worst_case_difference_upper=high[1]-low[0],
                     excluded_deaths=int(status.eq(STATUSES[0]).sum()),
                     excluded_unknown_vital=int(status.eq(STATUSES[-1]).sum()),
                     shifts='arbitrary EURO-D-unit sensitivity parameters, not calibrated bounds',
                     target='recorded_alive_association_not_principal_stratum_or_population_effect')


def response_weighting(d,config):
    status=classify(d)
    sample=d.loc[status.str.startswith('recorded_alive')].dropna(subset=['age','female','education','country']).copy()
    observed=sample.Y.notna().astype(int).to_numpy()
    sizes=np.bincount(observed,minlength=2)
    if min(sizes)<config['folds']*3 or sample.country.nunique()<3:
        return dict(status='NOT_ESTIMABLE',reason='insufficient_survivor_response_cells')
    # Exposure is a predictor of response, not a claim of causal adjustment.
    x=frame_matrix(sample,continuous=('age','female','education','A'))
    p=np.empty(len(sample))
    for tr,te in StratifiedKFold(config['folds'],shuffle=True,random_state=config['seed']).split(x,observed):
        p[te]=_propensity().fit(x[tr],observed[tr]).predict_proba(x[te])[:,1]
    clipped=np.clip(p,config['response_probability_floor'],1)
    w=observed/clipped
    means={a:float(np.sum(w[sample.A.eq(a)]*sample.Y.fillna(0).to_numpy()[sample.A.eq(a)])/w[sample.A.eq(a)].sum()) for a in [0,1]}
    psi=np.zeros(len(sample))
    for a,sign in [(1,1),(0,-1)]:
        arm=sample.A.eq(a).to_numpy()
        psi+=sign*arm*w*(sample.Y.fillna(0).to_numpy()-means[a])/(w[arm].sum()/len(sample))
    # Weighted-mean estimating-equation influence approximation, conditional on fitted weights.
    interval=influence_interval(psi+means[1]-means[0],sample.country)
    return dict(status='EXPLORATORY_MAR_DIAGNOSTIC',n_recorded_alive_complete_covariates=len(sample),
                n_observed=int(observed.sum()),n_missing=int((1-observed).sum()),
                death_records_used=0,unknown_vital_records_used=0,
                response_probability_min=float(p.min()),max_weight=float(w.max()),
                clipped_count=int(np.sum(p!=clipped)),**interval,
                limitations='Recorded-alive sample only; assumes response MAR given A and demographics/country. '
                            'Nuisance uncertainty omitted in weighting-only interval; no survival correction or causal effect.')
