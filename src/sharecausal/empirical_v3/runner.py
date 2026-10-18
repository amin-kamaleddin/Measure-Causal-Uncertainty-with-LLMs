"""Offline, memory-only real SHARE analysis in a separately versioned output tree."""
import importlib.metadata
import csv
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import yaml
import statsmodels.formula.api as smf
from sharecausal.data.archives import locate_archives, metadata
from sharecausal.data.panel import construct
from sharecausal.causal.estimators import unadjusted
from sharecausal.reporting.tables import baseline
from . import VERSION
from .graphs import audit_records, write_audit
from .io import digest, private_destination, write_json, write_csv
from .statistics import BASE, fit, paired, balance, country_estimates, subgroup_estimates
from .selection import counts, classify, mnar, response_weighting
from .report import graph_figures, figures, appendix


def load_config(path):
    data=yaml.safe_load(Path(path).read_text())
    expected={'version','processing_mode','seed','folds','propensity_clip','country_min_n',
              'country_min_arm','subgroup_min_n','subgroup_min_arm','response_probability_floor',
              'mnar_shifts','confounding_shifts'}
    if not isinstance(data,dict) or set(data)!=expected or data['version']!=VERSION or data['processing_mode']!='local_only':
        raise ValueError('Invalid version/configuration; local_only required')
    for field in ['seed','folds','country_min_n','country_min_arm','subgroup_min_n','subgroup_min_arm']:
        if type(data[field]) is not int or data[field]<1:raise ValueError('Invalid positive integer configuration')
    if data['folds']<2 or not 0<data['propensity_clip']<.5 or not 0<data['response_probability_floor']<1:
        raise ValueError('Invalid probability/folds configuration')
    for field in ['mnar_shifts','confounding_shifts']:
        if not isinstance(data[field],list) or not data[field] or not all(type(x) in [int,float] and np.isfinite(x) for x in data[field]):
            raise ValueError('Invalid sensitivity grid')
    return data


def status_metadata(paths):
    output={}
    for wave in [8,9]:
        m=metadata(paths[wave],wave,'cv_r')
        fields=['interview','deceased','int_year','int_month']
        labels={v:m['value_labels'][v] for v in fields}
        if labels['interview'].get('1')!='Main interview' or labels['interview'].get('2')!='End-of-Life interview' or labels['deceased'].get('0')!='Alive' or labels['deceased'].get('1')!='Deceased':
            raise ValueError('Status coding differs from verified release; manual review required')
        output[str(wave)]=dict(labels=labels,variable_labels={v:m['variable_labels'][v] for v in fields})
    for wave,module,variables in [(8,'ac',['ac035d5']),(7,'cc',['cc733_']),(9,'gv_health',['eurod'])]:
        m=metadata(paths[wave],wave,module)
        output[f'{wave}_{module}']={v:{'label':m['variable_labels'][v],'values':m['value_labels'][v]} for v in variables}
    return output


def eligibility(d):
    rows=[]
    for name,mask in [('all',pd.Series(True,index=d.index)),('observed_Y',d.Y.notna()),
                       ('complete_main',d[['Y',*BASE,'country']].notna().all(axis=1)),
                       ('childhood_eligible',d.SHARELIFE.eq(1)&d.poor.notna()),
                       ('childhood_ineligible_or_missing',~(d.SHARELIFE.eq(1)&d.poor.notna()))]:
        g=d.loc[mask]
        rows.append(dict(sample=name,n=len(g),exposed=int(g.A.eq(1).sum()),
                         countries=int(g.country.nunique()),observed_Y=int(g.Y.notna().sum()),
                         childhood_eligible=int((g.SHARELIFE.eq(1)&g.poor.notna()).sum()),
                         D7_observed=int(g.D7.notna().sum()),
                         jointly_D7_childhood=int((g.D7.notna()&g.poor.notna()&g.SHARELIFE.eq(1)).sum()),
                         age_mean=float(g.age.mean()) if g.age.notna().any() else None,
                         female_fraction=float(g.female.mean()) if g.female.notna().any() else None,
                         education_mean=float(g.education.mean()) if g.education.notna().any() else None))
    return rows


def run(args, stage):
    root=private_destination(args.results_dir)
    stage[0]='configuration'
    config=load_config(args.config)
    stage[0]='saved_graph_audit'
    graph=audit_records(args.record)
    theory=write_audit(root,graph)
    graph_figures(root,theory)  # Written before any respondent reading/estimation.
    inputs={str(Path(p).resolve()):digest(p) for p in args.record}
    for name in ['comparison_summary.json','edge_proposals.csv','edge_provenance_PRIVATE.csv']:
        file=Path(args.comparison_dir)/name
        inputs[str(file.resolve())]=digest(file)
    comparison=json.loads((Path(args.comparison_dir)/'comparison_summary.json').read_text())
    if comparison['real_llm_response_records']!=graph['n_models'] or comparison['n_unique_proposed_edges']!=graph['unique_arrows']:
        raise ValueError('Comparison export and saved responses disagree')
    source_models={m['model']:m['source_sha256'] for m in comparison['models']}
    for p in args.record:
        item=json.loads(Path(p).read_text())
        if source_models.get(item['model_requested'])!=digest(p):raise ValueError('Response source hash differs from comparison')
    with (Path(args.comparison_dir)/'edge_proposals.csv').open() as stream:
        exported_edges=list(csv.DictReader(stream))
    with (Path(args.comparison_dir)/'edge_provenance_PRIVATE.csv').open() as stream:
        exported_provenance=list(csv.DictReader(stream))
    expected_arrows={r['arrow'] for r in graph['edge_provenance']}
    if len(exported_edges)!=len(expected_arrows) or {r['arrow'] for r in exported_edges}!=expected_arrows:
        raise ValueError('Comparison edge export mismatch')
    if sorted((r['model'],r['arrow'],r['evidence_reference']) for r in exported_provenance)!=sorted(
            (r['model'],r['arrow'],r['evidence']) for r in graph['edge_provenance']):
        raise ValueError('Comparison provenance export mismatch')
    independent='NOT_LOCATED: original independent audit files were not supplied at execution. '
    independent+='This run computes a separate reproducible audit; it does not claim independent review.'
    if args.independent_audit_dir:
        for name in ['scientific_audit_PRIVATE.md','graph_conditional_backdoor_audit.csv']:
            file=Path(args.independent_audit_dir)/name
            inputs[str(file.resolve())]=digest(file)
        # Read completely, retain private source snapshots without pretending adjudication is automated.
        independent=(Path(args.independent_audit_dir)/'scientific_audit_PRIVATE.md').read_text()
        original=pd.read_csv(Path(args.independent_audit_dir)/'graph_conditional_backdoor_audit.csv')
        if len(original)!=4:raise ValueError('Independent audit model count mismatch')
        (root/'independent_audit_input_PRIVATE.md').write_text(independent)
        original.to_csv(root/'independent_backdoor_input_PRIVATE.csv',index=False)
        independent='READ: independent audit and four-model backdoor table supplied; private hashes recorded.'
    stage[0]='archive_metadata'
    paths=locate_archives(args.data_dir)
    inputs.update({str(p.resolve()):digest(p) for p in paths.values()})
    labels=status_metadata(paths)
    write_json(root/'verified_metadata.json',labels)
    stage[0]='panel_construction'
    panel=construct(paths)
    d=panel.data  # Never serialized or cached. Identifiers remain in memory only.
    classify(d)  # Contradictory death/outcome records fail before fitting.
    if not d.deceased8.eq(0).all():raise ValueError('W8 interview/death status conflict')
    observed=d.loc[d.Y.notna()].copy()
    main=observed.dropna(subset=[*BASE,'country']).copy()
    if main.country.nunique()<3:raise ValueError('Insufficient country clusters')
    stage[0]='matched_estimation'
    primary,_=fit(main,config)
    without,with_d8,delta=paired(main,config,('D8',))
    estimates=dict(unadjusted=unadjusted(observed.Y,observed.A,cluster=observed.country),
                   aipw_demographic_country=primary,
                   same_sample_without_w8_depression=without,aipw_with_w8_depression=with_d8)
    ols=smf.ols('Y ~ A + age + female + education + C(country)',data=main).fit(
        cov_type='cluster',cov_kwds={'groups':main.country})
    from scipy.stats import norm
    b,se=float(ols.params['A']),float(ols.bse['A'])
    estimates['ols_age_sex_education_country']=dict(effect=b,se=se,ci_lower=b-norm.ppf(.975)*se,
        ci_upper=b+norm.ppf(.975)*se,n=len(main))
    d7=main.dropna(subset=['D7'])
    d7_result=dict(status='NOT_ESTIMABLE',n=len(d7))
    if len(d7)>=config['subgroup_min_n'] and min(d7.A.value_counts().reindex([0,1],fill_value=0))>=config['subgroup_min_arm'] and d7.country.nunique()>=3:
        x0,x1,change=paired(d7,config,('D7',))
        d7_result=dict(status='EXPLORATORY_PRIOR_WAVE_DEPRESSION',without=x0,with_D7=x1,paired_change=change,
                       limitation='Different W7 regular-interview universe; cannot transfer to childhood sample')
    stage[0]='selection_diagnostics'
    selection_counts=counts(d)
    mnar_grid,mnar_summary=mnar(d,config['mnar_shifts'])
    response=response_weighting(d,config)
    stage[0]='country_childhood_overlap'
    balances,overlap=balance(main,config)
    balances_d8,overlap_d8=balance(main.dropna(subset=['D8']),config,(*BASE,'D8'))
    childhood,childhood_change=subgroup_estimates(main,config)
    childhood_selection=[]
    for k,g in [('ALL',d),*[(str(int(k)),g) for k,g in d.groupby('country')]]:
        for a,arm in g.groupby('A'):
            mask=arm.SHARELIFE.eq(1)&arm.poor.notna()
            childhood_selection.append(dict(country=k,A=int(a),n_eligible=len(arm),
                childhood_eligible=int(mask.sum()),childhood_with_observed_Y=int((mask&arm.Y.notna()).sum()),
                childhood_with_complete_main=int((mask&arm[['Y',*BASE,'country']].notna().all(axis=1)).sum()),
                childhood_missing_or_ineligible=int((~mask).sum())))
    country=country_estimates(main,config)
    country_d8=[]
    for k,g in main.dropna(subset=['D8']).groupby('country'):
        if len(g)>=config['country_min_n'] and min(g.A.value_counts().reindex([0,1],fill_value=0))>=config['country_min_arm']:
            # No country-cluster interval is possible within a single country; explicitly omit paired CI here.
            e0,_=fit(g,config,cluster=None);e1,_=fit(g,config,(*BASE,'D8'),cluster=None)
            country_d8.append(dict(country=int(k),n=len(g),without_D8=e0['effect'],with_D8=e1['effect'],
                                   change=e1['effect']-e0['effect'],paired_interval='OMITTED_country_cluster_not_available'))
    # Compare to immutable prior aggregates without overwriting them.
    previous=json.loads(Path(args.prior_results).read_text());inputs[str(Path(args.prior_results).resolve())]=digest(args.prior_results)
    reproduction=[]
    for name,r in estimates.items():
        if name in previous['estimates']:
            old=previous['estimates'][name]
            reproduction.append(dict(analysis=name,n_new=r['n'],n_prior=old['n'],effect_new=r['effect'],
                                     effect_prior=old['effect'],difference=r['effect']-old['effect'],
                                     matched=(r['n']==old['n'] and abs(r['effect']-old['effect'])<1e-8)))
    shifts=[dict(shift=float(s),effect=primary['effect']+s,lower=primary['ci_lower']+s,
                 upper=primary['ci_upper']+s,interpretation='arbitrary_bias_stress_not_validated_bound')
            for s in config['confounding_shifts']]
    result=dict(version=VERSION,status='exploratory_not_frozen',causal_decision='ABSTAIN',
                real_llm_calls=0,openrouter_cost_usd=0,network_calls=0,respondent_uploads=0,
                saved_real_response_records=4,fixture_mode=False,share_records_used=len(d),
                graph_audit=graph,flow=panel.flow,panel_diagnostics=panel.diagnostics,estimates=estimates,
                paired_D8=delta,prior_wave_depression=d7_result,selection_counts=selection_counts,
                mnar_grid=mnar_grid,mnar_summary=mnar_summary,survivor_response_weighting=response,
                balance=balances,overlap=overlap,overlap_D8=overlap_d8,countries=country,
                country_D8=country_d8,childhood=childhood,childhood_adjustment_change=childhood_change,
                childhood_selection=childhood_selection,
                eligibility=eligibility(d),confounding_stress=shifts,prior_reproduction=reproduction,
                independent_audit_status=independent,privacy_review_required=True)
    stage[0]='private_reporting'
    tables={'selection_counts':selection_counts,'measured_balance':balances,'measured_balance_D8':balances_d8,
            'country_associations':country,'country_D8_sensitivity':country_d8,
            'childhood_associations':childhood,'eligibility':result['eligibility'],
            'childhood_selection':childhood_selection,
            'survivor_mnar':mnar_grid,'prior_reproduction':reproduction,'confounding_stress':shifts,
            'estimates':[dict(analysis=n,n=r['n'],effect=r['effect'],ci_lower=r['ci_lower'],ci_upper=r['ci_upper']) for n,r in estimates.items()]}
    changes=[delta]
    if childhood_change:changes.append(childhood_change)
    if d7_result['status']=='EXPLORATORY_PRIOR_WAVE_DEPRESSION':
        changes.append(d7_result['paired_change'])
        tables['prior_depression_associations']=[dict(analysis=n,**{k:r[k] for k in ['n','effect','ci_lower','ci_upper']})
            for n,r in [('without_D7',d7_result['without']),('with_D7',d7_result['with_D7'])]]
    tables['paired_model_changes']=[{k:r[k] for k in ['contrast','n','effect','se','ci_lower','ci_upper','clusters','interval_scope']} for r in changes]
    for name,rows in tables.items():
        if rows:write_csv(root/f'{name}.csv',rows)
    baseline(d).to_csv(root/'baseline_table1_PRIVATE.csv',index=False)
    figures(root,result);appendix(root,result)
    repo=Path(__file__).resolve().parents[3]
    sources=[p for folder in ['src','scripts','configs'] for p in (repo/folder).rglob('*')
             if p.is_file() and p.suffix in {'.py','.yaml'}]
    provenance=dict(source_archive_and_record_hashes=inputs,config=config,config_sha256=digest(args.config),
                    source_hashes={str(p.relative_to(repo)):digest(p) for p in sources},
                    python=sys.version,platform=platform.platform(),
                    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn',
                              'networkx','matplotlib','statsmodels','pydantic','PyYAML']},
                    seed=config['seed'],folds=config['folds'],row_level_storage='none_memory_only',
                    processing_mode='local_only',guard='network_credentials_subprocess_disabled')
    result['provenance']=provenance
    write_json(root/'results.json',result)
    write_json(root/'provenance.json',provenance)
    # Hash all products, including images and config/source/input hashes carried in provenance.
    for p in root.iterdir():
        if p.is_file():p.chmod(0o600)
    write_json(root/'artifact_manifest.json',{p.name:digest(p) for p in sorted(root.iterdir()) if p.is_file()})
    stage[0]='complete'
    return result
