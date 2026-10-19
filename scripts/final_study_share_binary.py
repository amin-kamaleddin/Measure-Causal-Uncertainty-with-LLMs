"""Missing binary outcome only; verified No/Yes labels and in-memory threshold check."""
import argparse,os,json
from pathlib import Path
from sharecausal.empirical_v3.io import install_offline_guard,write_json,write_csv,digest
from sharecausal.empirical_v3.runner import load_config
from sharecausal.empirical_v3.statistics import BASE,fit
from sharecausal.data.archives import locate_archives,metadata
from sharecausal.data.panel import construct
from sharecausal.data.clean import survey_numeric

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 os.umask(0o077);os.environ.pop('OPENROUTER_API_KEY',None);install_offline_guard();out=a.output.resolve()
 if 'private' not in out.parts:raise ValueError('private_output_required')
 out.mkdir(mode=0o700,parents=True,exist_ok=True)
 if (out/'summary.json').exists():print('Completed binary result reused.');return
 paths=locate_archives(a.data_dir);meta=metadata(paths[9],9,'gv_health');labels=meta['value_labels']['eurodcat']
 if meta['variable_labels']['eurodcat']!='EURO-D caseness' or labels.get('0')!='No' or labels.get('1')!='Yes':raise ValueError('binary_metadata_mismatch')
 panel=construct(paths);d=panel.data;binary=survey_numeric(d.eurodcat9,{0,1}).where(d.interview9.eq(1)&~d.invalid_time_order);same=binary.notna()&d.Y.notna()
 if not (binary.loc[same]==d.loc[same,'Y'].ge(4).astype(float)).all():raise ValueError('binary_threshold_mismatch')
 if (binary.notna()&d.Y.isna()).any():raise ValueError('binary_without_continuous_universe_requires_review')
 sample=d.copy();sample['Y']=binary;sample=sample.loc[sample.Y.notna()].dropna(subset=[*BASE,'country'])
 config=load_config('configs/real_share_followup_v3.yaml');result,_=fit(sample,config)
 prevalence=[]
 for arm,group in d.assign(binary=binary).loc[binary.notna()].groupby('A'):
  prevalence.append(dict(exposure=int(arm),n=len(group),cases=int(group.binary.sum()),prevalence=float(group.binary.mean())))
 write_csv(out/'binary_prevalence.csv',prevalence)
 write_json(out/'summary.json',dict(status='VERIFIED_SECONDARY_OBSERVATIONAL',metadata=dict(label=meta['variable_labels']['eurodcat'],codes=labels),threshold='EURO-D>=4',crosschecked_n=int(same.sum()),threshold_agreements=int(same.sum()),estimate=result,prevalence=prevalence,continuous_primary_recomputed=False,leave_country_recomputed=False,respondent_exports=0,network_calls=0,causal_decision='ABSTAIN',source_sha256=digest(Path(__file__))))
 print('Verified binary observational association completed privately; no network.')
if __name__=='__main__':main()
