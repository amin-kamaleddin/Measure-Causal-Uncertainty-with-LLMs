"""Only scored protected synthetic aggregates, no model calls or truth reads."""
import argparse,json,os,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--package',required=True,type=Path);a=p.parse_args();run=a.run.resolve();package=a.package.resolve();out=package/'figures';out.mkdir(parents=True,exist_ok=True);os.umask(0o077);os.environ['MPLCONFIGDIR']=str(package/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})
if not (run/'scoring_integrity.json').exists():raise ValueError('scoring_not_complete')
manifest=[]
def save(fig,name,caption):
 for ext in ('pdf','svg','png'):
  file=out/(name+'.'+ext);fig.savefig(file,dpi=300,facecolor='white');file.chmod(0o600);manifest.append(dict(file=str(file.relative_to(package)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),caption=caption,source='sealed real-provider study, post-score aggregates only'))
 plt.close(fig)
edges=pd.read_csv(run/'evidence_edge_accuracy.csv');labels=['Gemini','Luna','Llama','DeepSeek'];fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,4.8));x=np.arange(4)
ax.bar(x-.18,edges.edge_support_precision,.34,label='Exact evidence precision',color='#236B8E');ax.bar(x+.18,edges.entailed_edge_recall,.34,label='Required-arrow recall',color='#C27C45');ax.set(xticks=x,xticklabels=labels,ylim=(0,1.25),ylabel='Fraction',title='A. Grounded edge extraction');ax.set_yticks(np.arange(0,1.01,.2));ax.legend(fontsize=9,loc='upper center')
ax2.bar(x,edges.total_graphs,color='#DFE6EC',label='All proposed graphs');ax2.bar(x,edges.admitted_graphs,color='#4A8D71',label='Compiler admitted');ax2.set(xticks=x,xticklabels=labels,ylabel='Graph hypotheses (nested in 84 tasks)',title='B. Whole-graph support and chronology');ax2.legend(fontsize=9);fig.tight_layout()
save(fig,'figure3_evidence_graph_support','Evidence precision/required-arrow recall and admitted graph counts. Graphs are nested in tasks; edge frequencies are not causal probabilities. Exact-span compiler has no independent human adjudication.')
providers=pd.read_csv(package/'tables/actual_unique_provider_requests.csv') if (package/'tables/actual_unique_provider_requests.csv').exists() else pd.read_csv(run/'provider_comparison.csv')
fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,4.8));bottom=np.zeros(4)
for column,label,color in [('valid_graph_or_critic','Typed valid','#4A8D71'),('explicit_abstention','Explicit source abstention','#C5A34B'),('technical_failures','Technical / pipeline failure','#B75A55')]:
 values=providers[column].to_numpy();ax.bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
ax.set(xticks=x,xticklabels=labels,ylabel='Actual unique request attempts',title='A. All operational outcomes retained');ax.legend(fontsize=8)
ax2.bar(x,providers.cost_usd,color='#236B8E');ax2.set(xticks=x,xticklabels=labels,ylabel='Provider-reported generation charges (USD)',title='B. Actual cost, including failed generations');fig.tight_layout()
save(fig,'figure4_provider_outcomes_cost','All unique attempts, including initial wrong-slug rejections and corrected requests; no completed paid response was resent. Typed-valid does not mean admitted causal graph. Original16-worker and repair64-worker schedules differ operationally.')
methods=pd.read_csv(run/'method_comparison.csv');curves=pd.read_csv(run/'risk_coverage.csv');paired=pd.read_csv(run/'family_paired_contrasts.csv');fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,5.7),gridspec_kw={'width_ratios':[1.2,1]})
colors={'no_llm_evidence':'#222222','full_policy':'#1B8191','no_llm_all_baseline':'#B75A55','raw_pool':'#A87532','screened_pool':'#72568D','google':'#2E73B7','openai':'#4D975C','llama':'#A75C96','deepseek':'#777777','strict_majority':'#C28355','screened_critic':'#607D65'}
for method in colors:
 g=curves.loc[curves.method.eq(method)&curves.practical_null.eq(.1)&curves.assertions.gt(0)].sort_values('assertion_rate')
 if len(g):ax.plot(g.assertion_rate,g.selective_risk,marker='o',markersize=3,lw=1,alpha=.8,c=colors[method],label=method.replace('_',' '))
ax.axvspan(.3,.5,alpha=.08,color='green');ax.set(xlabel='Heldout assertion fraction (84 cards / 14 families)',ylabel='Recorded unjustified risk among assertions',title='A. Fixed-threshold risk–assertion trade-off');ax.set_ylim(-.03,1.03);ax.set_xlim(-.02,1.02);ax.legend(fontsize=7,loc='upper left');ax.text(.98,.02,'Zero assertions: risk undefined\nUncertified assertions have risk bounds',transform=ax.transAxes,ha='right',fontsize=8)
chosen=paired.loc[paired.method.isin(['full_policy','google','openai','llama','deepseek','raw_pool','strict_majority'])].copy();chosen=chosen.set_index('method').reindex(['full_policy','google','openai','llama','deepseek','raw_pool','strict_majority']).reset_index()
for i,row in chosen.iterrows():
 lo=row.risk_difference_lower;hi=row.risk_difference_upper
 if pd.notna(lo) and pd.notna(hi):ax2.plot([lo,hi],[i,i],color=colors[row['method']],lw=2);point=float(methods.loc[methods.method.eq(row['method']),'selective_risk'].iloc[0])-float(methods.loc[methods.method.eq('no_llm_evidence'),'selective_risk'].iloc[0]);ax2.plot(point,i,'o',c=colors[row['method']])
 else:ax2.text(0,i,'undefined',va='center',fontsize=9)
ax2.axvline(0,c='.6',ls='--');ax2.set_yticks(range(len(chosen)),chosen.method.str.replace('_',' '));ax2.invert_yaxis();ax2.set_xlabel('Risk difference versus strong no-LLM baseline');ax2.set_title('B. Descriptive paired family bootstrap');ax2.text(.5,-.18,'5,000 family resamples; sparse-event bounds unvalidated.\nNo zero-event precision or superiority guarantee.',transform=ax2.transAxes,ha='center',fontsize=8);fig.tight_layout()
save(fig,'figure5_risk_coverage_uncertainty','Prespecified threshold curves and paired14-family bootstrap contrasts. Whole6card families move together; repeats are not independent. Undefined zero-assertion risks excluded. Bootstrap intervals are descriptive and may be degenerate under sparse errors; no confirmatory superiority.')
(package/'synthetic_figure_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(figures=3,files=len(manifest))))
