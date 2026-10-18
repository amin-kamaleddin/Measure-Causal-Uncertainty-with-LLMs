"""Render publication figures solely from post-score synthetic aggregates."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
NAMES={'no_llm_evidence_set':'No LLM','google/gemini-3.5-flash-lite':'Gemini','openai/gpt-6-luna':'Luna','meta-llama/llama-4-maverick':'Llama','deepseek/deepseek-v4.1-flash':'DeepSeek','naive_strict_majority':'Strict majority','robust_credible_alternatives':'Robust: credible','robust_supported_only':'Robust: supported','robust_permissive_sensitivity':'Robust: permissive','strict_unmeasured_or_selection_stress':'Abstention control'}
def rows(p):return list(csv.DictReader(p.open()))
def save(fig,out,name):
 fig.tight_layout()
 for suffix in ('pdf','svg','png'):fig.savefig(out/(name+'.'+suffix),dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 receipt=a.results/('scorer_receipt.json' if (a.results/'scorer_receipt.json').exists() else 'completed_run.json')
 assert json.loads(receipt.read_text())['seal_verified_before_truth_open']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 methods={r['method']:r for r in rows(a.results/'method_results.csv')};shown=[m for m in NAMES if m in methods];ys=np.arange(len(shown))
 fig,(ax,err)=plt.subplots(1,2,figsize=(10.2,5.5),gridspec_kw={'width_ratios':[3,1.2]})
 ax.barh(ys,[float(methods[m]['assertion_rate']) for m in shown],color=['#366b92' if not m.startswith('robust') else '#427962' for m in shown])
 for y,m in zip(ys,shown):ax.text(float(methods[m]['assertion_rate'])+.015,y,f"{methods[m]['assertions']}/24",va='center',fontsize=9)
 ax.set(yticks=ys,yticklabels=[NAMES[m] for m in shown],xlim=(0,1),xlabel='Directional assertion rate',title='A  All primary tasks remain in the denominator');ax.invert_yaxis();ax.axvspan(.4,.6,color='#ece5d4',alpha=.5,zorder=0)
 err.scatter([int(methods[m]['composite_unjustified_assertion_cards']) for m in shown],ys,color='#8b3a3a',s=28)
 err.set(yticks=ys,yticklabels=[],xlim=(-.2,max(1,max(int(methods[m]['composite_unjustified_assertion_cards']) for m in shown))+.5),xlabel='Known unjustified\nassertion cards',title='B  Observed errors');err.invert_yaxis()
 save(fig,a.output,'synthetic_assertions_and_errors')
 model=rows(a.results/'model_results.csv');fig,ax=plt.subplots(figsize=(7,3.7));success=np.array([int(r.get('provider_schema_responses',r.get('outer_schema_responses'))) for r in model]);tot=np.array([int(r.get('actual_posts',r.get('attempted'))) for r in model]);x=np.arange(4)
 ax.bar(x,success,color='#366b92',label='Outer schema response');ax.bar(x,tot-success,bottom=success,color='#b55f51',label='Technical failure');ax.set(xticks=x,xticklabels=['Gemini','Luna','Llama','DeepSeek'],ylim=(0,35),ylabel='Requests (base + counterargument)',title='Recorded provider outcomes; semantic validity checked separately');ax.legend(frameon=False,fontsize=9)
 for i,n in enumerate(tot):ax.text(i,n+.4,str(n),ha='center')
 save(fig,a.output,'synthetic_provider_outcomes')
 risk=rows(a.results/'risk_abstention.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.6))
 for m in ('no_llm_evidence_set','naive_strict_majority','robust_credible_alternatives'):
  r=[q for q in risk if q['method']==m and float(q['practical_zero'])==.1];r.sort(key=lambda q:float(q['margin_threshold']))
  axes[0].plot([float(q['margin_threshold']) for q in r],[float(q['assertion_rate']) for q in r],marker='o',label=NAMES[m])
  valid=[q for q in r if int(q['assertions'])]
  axes[1].plot([float(q['assertion_rate']) for q in valid],[float(q['unjustified_risk_among_assertions_lower']) for q in valid],marker='o',alpha=.75,label=NAMES[m])
 axes[0].set(xlabel='Fixed envelope margin',ylabel='Assertion rate',ylim=(-.02,1));axes[1].set(xlabel='Assertion rate',ylabel='Known unjustified selective risk',ylim=(-.04,1),xlim=(0,1));axes[0].legend(frameon=False,fontsize=8);axes[1].set_title('Undefined at zero assertions; coincident points possible',fontsize=9)
 save(fig,a.output,'synthetic_risk_assertion')
 print(json.dumps({'figures':9,'data':'post-score synthetic aggregates only','output':str(a.output.resolve())}))
if __name__=='__main__':main()
