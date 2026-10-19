"""Only missing SHARE clinical diagnostics, offline and memory-only, private."""
import argparse,json,os,hashlib,csv
from pathlib import Path
import numpy as np
import pandas as pd
from sharecausal.empirical_v3.io import install_offline_guard,write_json,write_csv,digest
from sharecausal.empirical_v3.statistics import BASE,fit
from sharecausal.empirical_v3.runner import load_config
from sharecausal.empirical_v3.selection import classify,STATUSES
from sharecausal.data.archives import locate_archives,metadata
from sharecausal.data.panel import construct
from sharecausal.data.clean import survey_numeric

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
 os.umask(0o077);os.environ.pop('OPENROUTER_API_KEY',None);install_offline_guard()
 out=args.output.resolve()
 if 'private' not in out.parts or out.is_relative_to(Path.cwd()):raise ValueError('outside_repo_explicit_private_component_required')
 out.mkdir(mode=0o700,parents=True,exist_ok=True)
 if (out/'summary.json').exists():print('Completed private SHARE update reused; no respondent reread.');return
 paths=locate_archives(args.data_dir);original=Path('outputs/private/share_llm_followup_v3_release_review');saved=json.loads((original/'results.json').read_text());manifest=json.loads((original/'artifact_manifest.json').read_text())
 for name,expected in manifest.items():
  if digest(original/name)!=expected:raise ValueError('original_empirical_artifact_changed')
 meta=metadata(paths[9],9,'gv_health');labels=meta['value_labels']['eurodcat'];description=meta['variable_labels']['eurodcat']
 write_json(out/'verified_binary_metadata.json',dict(label=description,codes=labels))
 # Fail closed if local code labels don't substantiate which code denotes
 # clinically elevated symptoms. No presumed threshold from variable name.
 depressed=[];not_depressed=[]
 for code,label in labels.items():
  text=label.lower()
  if 'not' in text or 'no ' in text or 'less' in text or '0-3' in text or '0 - 3' in text:not_depressed.append(float(code))
  elif 'depress' in text or '4' in text:depressed.append(float(code))
 panel=construct(paths);d=panel.data;status=classify(d)
 if panel.flow!=saved['flow']:raise ValueError('original_cohort_reproduction_mismatch')
 main=d.loc[d.Y.notna()].dropna(subset=[*BASE,'country']).copy()
 sd=float(main.Y.std(ddof=1));old=saved['estimates']['aipw_demographic_country'];standardized=dict(observed_sd=sd,n=len(main),standardized_association=old['effect']/sd,ci_lower=old['ci_lower']/sd,ci_upper=old['ci_upper']/sd,meaning='observed_outcome_SD_scaling_not_causal_effect')
 table=[]
 for (a,s),group in d.assign(status9=status).groupby(['A','status9']):
  for variable in ('age','female','education','D8','D7','poor','Y'):
   values=group[variable].dropna();table.append(dict(exposure=int(a),followup_status=s,group_n=len(group),variable=variable,available_n=len(values),missing_n=len(group)-len(values),mean=float(values.mean()) if len(values) else None,sd=float(values.std(ddof=1)) if len(values)>1 else None))
 write_csv(out/'table1_with_denominators.csv',table)
 # Aggregate binary coding crosscheck only, no individual records exported.
 coding=[]
 for code,g in d.loc[d.interview9.eq(1)&d.Y.notna()].groupby('eurodcat9'):
  coding.append(dict(code=float(code),n=len(g),min_eurod=float(g.Y.min()),max_eurod=float(g.Y.max()),label=labels.get(str(int(code)),labels.get(str(code),''))))
 write_csv(out/'binary_coding_crosscheck.csv',coding)
 config=load_config('configs/real_share_followup_v3.yaml');binary=dict(status='CODING_REQUIRES_REVIEW',variable_label=description,codes=labels)
 if len(depressed)==1 and len(not_depressed)==1 and set(depressed+not_depressed)==set(float(x) for x in labels if float(x)>=0):
  b=d.copy();valid=survey_numeric(b.eurodcat9,set(depressed+not_depressed));b['Y']=valid.map({depressed[0]:1.,not_depressed[0]:0.}).where(b.interview9.eq(1)&~b.invalid_time_order)
  # Verify stated binary classification against continuous score row-by-row in
  # memory; release only aggregate agreement counts.
  same=b.Y.notna()&d.Y.notna();agreement=int((b.loc[same,'Y']==d.loc[same,'Y'].ge(4).astype(float)).sum())
  if agreement!=int(same.sum()):raise ValueError('binary_threshold_crosscheck_failed')
  sample=b.loc[b.Y.notna()].dropna(subset=[*BASE,'country']);estimate,_=fit(sample,config)
  prevalence=[dict(exposure=int(a),observed_n=len(g),cases=int(g.Y.sum()),prevalence=float(g.Y.mean())) for a,g in b.loc[b.Y.notna()].groupby('A')]
  binary=dict(status='VERIFIED_SECONDARY_OBSERVATIONAL',positive_code=depressed[0],negative_code=not_depressed[0],threshold='EURO-D>=4',crosschecked_n=int(same.sum()),agreements=agreement,estimate=estimate,prevalence=prevalence)
  write_csv(out/'binary_observed_prevalence.csv',prevalence)
 write_json(out/'binary_outcome.json',binary)
 # Exact missing leave-one-country-out refits, checkpoint each exclusion. No
 # re-estimation of previously completed whole-sample continuous analysis.
 leave=[]
 for country in sorted(main.country.unique()):
  cache=out/('leave_country_'+str(int(country))+'.json')
  if cache.exists():row=json.loads(cache.read_text())
  else:
   subset=main.loc[main.country.ne(country)]
   if min(subset.A.value_counts())<20:raise ValueError('leave_country_support_failure')
   result,_=fit(subset,config);row=dict(omitted_country=int(country),n=len(subset),exposed=int(subset.A.sum()),effect=result['effect'],ci_lower=result['t_ci_lower'],ci_upper=result['t_ci_upper'],scope='refitted_observed_association_not_causal');write_json(cache,row)
  leave.append(row)
 write_csv(out/'leave_one_country_out.csv',leave)
 # Explicit unknown-survival scenarios: hypothetical fractions alive are
 # sensitivity parameters, not estimated vital status. Known dead never imputed.
 byarm={}
 for a in (0,1):
  g=d.loc[d.A.eq(a)];st=status.loc[g.index];seen=g.Y.dropna();byarm[a]=dict(observed=len(seen),mean=float(seen.mean()),known_alive=int(st.str.startswith('recorded_alive').sum()),missing_alive=int((st.str.startswith('recorded_alive')&g.Y.isna()).sum()),unknown=int(st.eq('unknown_vital_status').sum()))
 tipping=[]
 for fraction in (0.,.25,.5,.75,1.):
  for delta0 in (-4.,-2.,0.,2.,4.):
   for delta1 in (-4.,-2.,0.,2.,4.):
    means={}
    for a,delta in ((0,delta0),(1,delta1)):
     r=byarm[a];unknown_alive=fraction*r['unknown'];missing=r['missing_alive']+unknown_alive
     means[a]=(r['observed']*r['mean']+missing*float(np.clip(r['mean']+delta,0,12)))/(r['known_alive']+unknown_alive)
    tipping.append(dict(hypothetical_unknown_alive_fraction=fraction,delta_unexposed=delta0,delta_exposed=delta1,difference=means[1]-means[0],scope='hypothetical_survivor_composition_and_nonresponse_association_not_causal',known_deaths_imputed=0))
 write_csv(out/'unknown_survival_tipping.csv',tipping)
 summary=dict(status='COMPLETED_LOCAL_ONLY',standardized=standardized,binary=binary,leave_country_refits=len(leave),leave_country_min=min(r['effect'] for r in leave),leave_country_max=max(r['effect'] for r in leave),tipping_min=min(r['difference'] for r in tipping),tipping_max=max(r['difference'] for r in tipping),prior_artifacts_verified=len(manifest),prior_main_not_refitted=True,row_level_exports=0,network_calls=0,provider_calls=0,causal_conclusion='ABSTAIN',archive_sha256={str(w):digest(path) for w,path in paths.items()},source_sha256=digest(Path(__file__)),original_results_sha256=digest(original/'results.json'))
 write_json(out/'summary.json',summary);print('Missing SHARE diagnostics completed privately; no network or respondent export.')
if __name__=='__main__':main()
