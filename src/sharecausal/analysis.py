"""Reproducible real SHARE analysis; all products stay under private results dir."""
from __future__ import annotations
import json
from pathlib import Path
import yaml
import numpy as np
from .data.archives import locate_archives
from .data.panel import construct
from .causal.estimators import aipw,unadjusted,frame_matrix
from .causal.heterogeneity import estimate_subgroups
from .causal.attrition import observation_summary,illustrative_mar_weights
from .causal.sensitivity import default_structural_verdict,shifted_intervals
from .causal.graphs import SPECS
from .causal.identification import identify
from .data.weights import weighted_mean
from .reporting.tables import baseline
from .reporting.plots import flow_plot,effect_plot,graph_plot,subgroup_plot,attrition_plot,sensitivity_plot
from .reporting.provenance import manifest
from .reporting.manuscript import manuscript


def _safe_json(data):
    def convert(obj):
        if isinstance(obj,dict):return {str(k):convert(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)):return [convert(v) for v in obj]
        if isinstance(obj,(np.integer,np.floating)):return obj.item()
        if isinstance(obj,(np.bool_,)):return bool(obj)
        return obj
    return json.dumps(convert(data),indent=2,allow_nan=False)


def run(data_dir,results_dir,config_path=None):
    config={'seed':20261015,'folds':3,'confounding_shifts':[-1.0,0.0,1.0],
            'processing_mode':'local_only','cluster':'country','minimum_subgroup':50}
    if config_path:
        with open(config_path,encoding='utf-8') as f:config.update(yaml.safe_load(f) or {})
    if config['processing_mode']!='local_only':
        raise ValueError('External SHARE processing unimplemented on purpose; local_only required')
    dest=Path(results_dir).expanduser().resolve()
    # Fail closed if trying to write under source repository or an exposed source-control path.
    if 'private' not in str(dest).lower() or any(part in ('.git','src','tests','.github') for part in dest.parts):
        raise ValueError('Real-data outputs must be under a private-named location outside source/CI')
    dest.mkdir(parents=True,exist_ok=True)
    paths=locate_archives(data_dir)
    panel=construct(paths);d=panel.data
    sub=d.loc[d.Y.notna()].copy()
    n_country=int(sub.country.nunique())
    if n_country<3:raise ValueError('Insufficient countries for cluster inference')
    raw=unadjusted(sub.Y,sub.A,cluster=sub.country)
    subset=sub.dropna(subset=['age','female','education','country']).copy()
    if not len(subset):raise ValueError('No complete cases')
    x=frame_matrix(subset)
    est=aipw(subset.Y,subset.A,x,config['folds'],config['seed'],cluster=subset.country)
    # OLS adjustment baseline; still conditional association, not causal identification.
    import statsmodels.formula.api as smf
    fit_ols=smf.ols('Y ~ A + age + female + education + C(country)',data=subset).fit(
          cov_type='cluster',cov_kwds={'groups':subset.country})
    from scipy.stats import norm
    z=norm.ppf(.975);beta=float(fit_ols.params['A']);se=float(fit_ols.bse['A'])
    ols={'effect':beta,'se':se,'ci_lower':beta-z*se,'ci_upper':beta+z*se,
         'n':len(subset),'method':'OLS linear adjustment, country-cluster sandwich'}
    result={'flow':panel.flow,'diagnostics':panel.diagnostics,
            'identification':{g.name:identify(g) for g in SPECS},
            'estimates':{'unadjusted':raw,'ols_age_sex_education_country':ols,
                         'aipw_demographic_country':est.as_dict()},
            'subgroups':{},'attrition':observation_summary(d),
            'protocol_status':'Exploratory observational, not identified causal trial',
            'privacy_review_required':True}
    both=subset.loc[subset.D8.notna()].copy()
    if len(both)>50:
        comparable=aipw(both.Y,both.A,frame_matrix(both),config['folds'],config['seed'],cluster=both.country)
        ext_x=frame_matrix(both,continuous=('age','female','education','D8'))
        alt=aipw(both.Y,both.A,ext_x,config['folds'],config['seed'],cluster=both.country)
        result['estimates']['same_sample_without_w8_depression']=comparable.as_dict()
        result['estimates']['aipw_with_w8_depression']=alt.as_dict()
    # W7 retrospective subset does not support W7 regular health adjustment.
    result['subgroups']=estimate_subgroups(subset,min_n=config['minimum_subgroup'],
                                          folds=config['folds'],seed=config['seed'])
    result['structural_robustness']=default_structural_verdict(est.as_dict(),
        result['estimates'].get('aipw_with_w8_depression'),config['confounding_shifts'])
    result['sensitivity_shifts']=shifted_intervals(est.as_dict(),config['confounding_shifts'])
    # Separate optional MAR illustrative weighted descriptive outcome contrast;
    # do not treat death as simply missing at random or weights as valid survival adjustment.
    observed=d.dropna(subset=['age','female','education','country']).copy()
    if len(observed)>100 and observed.observed9.nunique()>1:
        Xm=frame_matrix(observed)
        selection_weights=illustrative_mar_weights(Xm,observed.observed9)
        observed['iw']=selection_weights
        seen=observed.loc[observed.Y.notna()]
        if seen.A.nunique()==2:
            result['attrition']['illustrative_mar_weighted_observed_difference']=(
                weighted_mean(seen.loc[seen.A==1,'Y'],seen.loc[seen.A==1,'iw'])-
                weighted_mean(seen.loc[seen.A==0,'Y'],seen.loc[seen.A==0,'iw']))
            result['attrition']['mar_warning']='Not a death correction, not cross-fitted selection model, not causal'
    result['baseline_table']=baseline(d).to_dict(orient='records')
    result['provenance']=manifest(paths,config,config['seed'],'3-fold AIPW country-cluster + descriptive + DAG sensitivity')
    (dest/'results.json').write_text(_safe_json(result),encoding='utf-8')
    baseline(d).to_csv(dest/'baseline_table1_private.csv',index=False)
    flow_plot(panel.flow,dest);effect_plot(result['estimates'],dest);graph_plot(dest)
    subgroup_plot(result['subgroups'],dest)
    attrition_plot(result['attrition'],dest)
    sensitivity_plot(result['sensitivity_shifts'],dest)
    manuscript(result,dest)
    return result
