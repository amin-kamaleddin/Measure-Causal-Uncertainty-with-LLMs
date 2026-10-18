"""Matched observational standardization with paired country influence scores."""
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import StratifiedKFold
from sharecausal.causal.estimators import aipw, frame_matrix, unadjusted, _propensity
from sharecausal.causal.diagnostics import standardized_mean_difference

BASE = ('age', 'female', 'education')


def influence_interval(scores, clusters=None):
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all() or len(scores) < 3:
        raise ValueError('Finite influence scores required')
    center = scores - scores.mean()
    if clusters is None:
        groups = len(scores)
        se = scores.std(ddof=1) / np.sqrt(groups)
        scope = 'individual_iid_exploratory'
    else:
        cl = np.asarray(clusters)
        if len(cl) != len(scores) or pd.isna(cl).any():
            raise ValueError('Invalid cluster vector')
        groups = len(np.unique(cl))
        if groups < 3:
            raise ValueError('At least three independent clusters required')
        sums = np.array([center[cl == k].sum() for k in np.unique(cl)])
        se = np.sqrt(groups/(groups-1) * np.sum(sums*sums)) / len(scores)
        scope = 'cluster_influence_t_approximation'
    effect = float(scores.mean())
    q = t.ppf(.975, groups-1)
    return dict(effect=effect, se=float(se), ci_lower=float(effect-q*se),
                ci_upper=float(effect+q*se), clusters=groups, interval_scope=scope)


def fit(d, config, covariates=BASE, cluster='country'):
    if d[list(covariates)+['country','A','Y']].isna().any().any():
        raise ValueError('Caller must fix a complete analytic sample')
    levels = d[cluster] if cluster else None
    categories = ('country',) if cluster == 'country' else ()
    est, psi = aipw(d.Y, d.A, frame_matrix(d, covariates, categories),
                    folds=config['folds'], seed=config['seed'], cluster=levels,
                    trim=config['propensity_clip'], return_scores=True)
    result = est.as_dict()
    # Preserve the original normal interval and add a few-country t approximation.
    result.update({f't_{k}':v for k,v in influence_interval(psi, levels).items()
                   if k in ['ci_lower','ci_upper','clusters','interval_scope']})
    result['interpretation'] = 'observed_outcome_standardized_association_not_identified_effect'
    return result, psi


def paired(d, config, extra):
    sample = d.dropna(subset=[*BASE, 'country','A','Y',*extra]).copy()
    first, p0 = fit(sample, config)
    second, p1 = fit(sample, config, (*BASE, *extra))
    diff = influence_interval(p1-p0, sample.country)
    diff.update(n=len(sample), contrast='with_minus_without_'+'_'.join(extra),
                same_rows=True, same_crossfit_folds=True,
                limitation='paired nuisance-fit influence approximation; no structural or survey-design coverage')
    return first, second, diff


def balance(d, config, covariates=BASE):
    x = frame_matrix(d, covariates)
    a = np.asarray(d.A, int)
    probabilities = np.empty(len(d))
    for train, test in StratifiedKFold(config['folds'], shuffle=True,
                                       random_state=config['seed']).split(x,a):
        probabilities[test] = _propensity().fit(x[train],a[train]).predict_proba(x[test])[:,1]
    clipped = np.clip(probabilities, config['propensity_clip'], 1-config['propensity_clip'])
    weights = a/clipped + (1-a)/(1-clipped)
    variables = {v:d[v].to_numpy() for v in covariates}
    variables.update({f'country_{int(k)}':d.country.eq(k).to_numpy() for k in sorted(d.country.unique())})
    rows=[]
    for name, values in variables.items():
        raw = standardized_mean_difference(values, a)
        weighted = standardized_mean_difference(values, a, weights)
        rows.append(dict(variable=name, smd_raw=float(raw) if np.isfinite(raw) else None,
                         smd_weighted=float(weighted) if np.isfinite(weighted) else None))
    diag=dict(n=len(d), propensity_quantiles={str(q):float(np.quantile(probabilities,q))
              for q in [0,.01,.05,.5,.95,.99,1]},
              fraction_extreme=float(np.mean((probabilities<.05)|(probabilities>.95))),
              clipped_count=int(np.sum(probabilities != clipped)),
              effective_sample_size_by_arm={str(k):float(weights[a==k].sum()**2/np.sum(weights[a==k]**2)) for k in [0,1]},
              max_abs_weighted_smd=max(abs(r['smd_weighted']) for r in rows if r['smd_weighted'] is not None),
              limitation='Measured balance cannot validate exchangeability; no longitudinal survey weights')
    return rows, diag


def country_estimates(d, config):
    rows=[]
    for country, group in d.groupby('country'):
        sizes=group.A.value_counts().reindex([0,1],fill_value=0)
        row=dict(country=int(country), n=len(group), exposed=int(sizes[1]), unexposed=int(sizes[0]),
                 status='INSUFFICIENT_SUPPORT', effect=None, ci_lower=None, ci_upper=None,
                 propensity_extreme=None, clusters=None, uncertainty=None)
        if len(group)>=config['country_min_n'] and min(sizes)>=config['country_min_arm']:
            # Available PSU identifiers are only used within a country, never pooled un-nested.
            # Release PSU identifiers may be strings, not numeric survey values.
            psu_text=group.psu.astype(str).str.strip()
            valid_psu=group.psu.notna() & psu_text.ne('') & ~psu_text.str.fullmatch(r'-\d+(?:\.0)?')
            clustered=valid_psu.all() and group.psu.nunique()>=3
            est,_=fit(group,config,cluster='psu' if clustered else None)
            row.update(status='EXPLORATORY_ASSOCIATION', effect=est['effect'],
                       ci_lower=est['t_ci_lower'],ci_upper=est['t_ci_upper'],
                       propensity_extreme=est['overlap_fraction_below_005_or_above_095'],
                       clusters=est['t_clusters'], uncertainty='PSU_cluster_t' if clustered else 'iid_t_no_valid_PSU')
        rows.append(row)
    return rows


def subgroup_estimates(d, config):
    # cc733_ 1/2 vs 3, restricted to actual SHARELIFE respondents; 4/5 excluded.
    sample=d.loc[d.SHARELIFE.eq(1)].dropna(subset=['poor']).copy()
    rows=[]
    for name, subset, covariates in [
            ('childhood_eligible_without_childhood_adjustment',sample,BASE),
            ('childhood_eligible_with_childhood_adjustment',sample,(*BASE,'poor')),
            ('childhood_nonpoor_categories_1_2',sample.loc[sample.poor.eq(0)],BASE),
            ('childhood_poor_category_3',sample.loc[sample.poor.eq(1)],BASE)]:
        sizes=subset.A.value_counts().reindex([0,1],fill_value=0)
        row=dict(analysis=name,n=len(subset),exposed=int(sizes[1]),unexposed=int(sizes[0]),
                 effect=None,ci_lower=None,ci_upper=None,status='INSUFFICIENT_SUPPORT')
        if len(subset)>=config['subgroup_min_n'] and min(sizes)>=config['subgroup_min_arm'] and subset.country.nunique()>=3:
            est,_=fit(subset,config,covariates)
            row.update(effect=est['effect'],ci_lower=est['t_ci_lower'],ci_upper=est['t_ci_upper'],
                       status='EXPLORATORY_ASSOCIATION_not_effect_modification')
        rows.append(row)
    difference=None
    if len(sample)>=config['subgroup_min_n'] and min(sample.A.value_counts())>=config['subgroup_min_arm'] and sample.country.nunique()>=3:
        _,_,difference=paired(sample,config,('poor',))
    return rows,difference
