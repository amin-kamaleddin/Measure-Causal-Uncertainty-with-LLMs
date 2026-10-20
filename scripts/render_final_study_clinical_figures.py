"""Vector publication figures from protected aggregate outputs only."""
import os,json,argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package',required=True,type=Path,help='Private source package with computed clinical summaries')
p.add_argument('--empirical-results',type=Path,default=Path('outputs/private/share_llm_followup_v3_release_review'),help='Directory with results.json from the executed empirical analysis')
p.add_argument('--output-package',type=Path,help='Separate private output package; defaults to --package')
a=p.parse_args();os.umask(0o077);inputs=a.package.resolve();root=(a.output_package or a.package).resolve()
repo=Path(__file__).resolve().parents[1]
if root.is_relative_to(repo) or root==root.parent:raise ValueError('clinical_output_must_be_outside_repository')
out=root/'figures';out.mkdir(mode=0o700,parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(root/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import pandas as pd
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})
old=a.empirical_results.resolve();r=json.loads((old/'results.json').read_text());binary=json.loads((inputs/'share_binary/summary.json').read_text())
# Counts are read from saved disjoint aggregate states, never embedded findings.
all_rows=[row for row in r['selection_counts'] if str(row['country'])=='ALL']
if len(all_rows)!=2 or {int(row['A']) for row in all_rows}!={0,1}:raise ValueError('two_disjoint_exposure_arms_required')
fields=('n_eligible','known_dead_undefined_Y','recorded_alive_observed_Y','recorded_alive_noninterview','recorded_alive_interview_missing_Y','unknown_vital_status')
for row in all_rows:
 if any(type(row[k]) is not int or row[k]<0 for k in fields):raise ValueError('nonnegative_integer_flow_counts_required')
 if row['n_eligible']!=sum(row[k] for k in fields[1:]):raise ValueError('followup_states_not_exhaustive')
flow={k:sum(row[k] for row in all_rows) for k in fields}
if flow['n_eligible']!=r['flow']['valid_wave8_activity']:raise ValueError('eligible_flow_binding_mismatch')
flow['alive_missing']=flow['recorded_alive_noninterview']+flow['recorded_alive_interview_missing_Y']
flow['known_alive']=flow['recorded_alive_observed_Y']+flow['alive_missing']
flow['main_complete']=r['estimates']['aipw_demographic_country']['n']
if not 0<flow['main_complete']<=flow['recorded_alive_observed_Y']:raise ValueError('invalid_complete_sample_size')
manifest=[]
def save(fig,name,caption):
 for ext in ('pdf','svg','png'):
  file=out/(name+'.'+ext);fig.savefig(file,dpi=300,facecolor='white');file.chmod(0o600);manifest.append(dict(file=str(file.relative_to(root)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),caption=caption,source='verified private aggregate tables, no respondent records'))
 plt.close(fig)
def box(ax,xy,w,h,text,color):
 ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=.015',facecolor=color,edgecolor='#32465A',lw=1,clip_on=False));ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=10)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#32465A',lw=1.2))
fig,ax=plt.subplots(figsize=(11,4.6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
for x,text,c in [(0.01,'Public evidence\n& temporal constraints','#DDECF3'),(.27,'Four model graph sets\n+ exact quote audit','#E7E2F2'),(.53,'Formal identification\n+ conditional estimates','#E1EFDF'),(.79,'Robust supported sign\nor explicit ABSTAIN','#FAE8D2')]:box(ax,(x,.58),.19,.27,text,c)
for x in (.20,.46,.72):arrow(ax,(x,.715),(x+.065,.715))
box(ax,(.08,.10),.36,.24,'Synthetic validation\nSeal predictions → open known truth\nFailures remain in denominators','#F1F4F7');box(ax,(.56,.10),.36,.24,'SHARE clinical application\nUnknown causal truth\nAssociation ≠ intervention benefit','#F1F4F7')
arrow(ax,(.37,.58),(.26,.35));arrow(ax,(.63,.58),(.74,.35));ax.set_title('Evidence before effects: a reproducible causal-assertion workflow',loc='left',pad=14,fontweight='bold')
save(fig,'figure1_workflow','Evidence-to-graph-to-decision workflow, separating truth-known synthetic evaluation from observational SHARE interpretation.')
fig,axes=plt.subplots(2,1,figsize=(10,7.4),gridspec_kw={'height_ratios':[1,2]})
ax=axes[0];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
for x,t in [(.02,'Wave 7\nChildhood history OR\nregular depression cohort'),(.36,'Wave 8\nPrior-year club participation\nD8 overlaps exposure recall'),(.70,'Wave 9\nEURO-D and caseness\nDeath / observation status')]:box(ax,(x,.15),.27,.60,t,'#E2ECF3')
arrow(ax,(.29,.46),(.35,.46));arrow(ax,(.63,.46),(.69,.46));ax.set_title('A. Three-wave timing and measurement ambiguity',loc='left')
ax=axes[1];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));box(ax,(.31,.78),.38,.17,f"Valid Wave 8 participation: {flow['n_eligible']:,}",'#DDECF3')
labels=[(f"Known death\n{flow['known_dead_undefined_Y']:,}",.01,'#F4DEDE'),(f"Known alive\n{flow['known_alive']:,}",.35,'#E1EFDF'),(f"Unknown survival\n{flow['unknown_vital_status']:,}",.69,'#FAE8D2')]
for text,x,c in labels:box(ax,(x,.43),.29,.17,text,c);arrow(ax,(.50,.78),(x+.145,.60))
box(ax,(.16,.09),.29,.18,f"Observed EURO-D\n{flow['recorded_alive_observed_Y']:,}",'#E1EFDF');box(ax,(.53,.09),.29,.18,f"Alive, outcome missing\n{flow['alive_missing']:,}",'#FAE8D2');arrow(ax,(.495,.43),(.305,.27));arrow(ax,(.495,.43),(.675,.27));ax.text(.50,.01,f"Main complete covariate sample: {flow['main_complete']:,}. Death is not ordinary missing depression.",ha='center',fontsize=10);ax.set_title('B. Disjoint follow-up states in the eligible panel',loc='left');fig.tight_layout(h_pad=2)
save(fig,'figure2_share_timing_flow','SHARE timing and disjoint cohort flow. Wave7 childhood and earlier depression universes do not jointly define a common baseline here.')
models=r['estimates'];d7=r['prior_wave_depression'];rows=[('Main demographic / country',models['aipw_demographic_country']),('Matched D8 sample: without D8',models['same_sample_without_w8_depression']),('Matched D8 sample: with D8',models['aipw_with_w8_depression']),('Earlier W7 cohort: without D7',d7['without']),('Earlier W7 cohort: with D7',d7['with_D7'])]
fig,(ax,ax2)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[2.4,1]})
for i,(label,item) in enumerate(rows):
 lo=item.get('t_ci_lower',item['ci_lower']);hi=item.get('t_ci_upper',item['ci_upper']);ax.errorbar(item['effect'],i,xerr=[[item['effect']-lo],[hi-item['effect']]],fmt='o',color='#236B8E',capsize=3);ax.text(.04,i,f"n={item['n']:,}",va='center',fontsize=9)
ax.set_yticks(range(len(rows)),[x[0] for x in rows]);ax.invert_yaxis();ax.axvline(0,c='.5',ls='--');ax.set_xlabel('Observed-outcome EURO-D association (points)');ax.set_title('Conditional associations and timing sensitivity',loc='left');ax.set_xlim(-.8,.25)
b=binary['estimate'];ax2.errorbar(b['effect']*100,0,xerr=[[(b['effect']-b['t_ci_lower'])*100],[(b['t_ci_upper']-b['effect'])*100]],fmt='s',color='#924D2D',capsize=3);ax2.axvline(0,c='.5',ls='--');ax2.set_yticks([0],['Caseness\nEURO-D ≥4']);ax2.set_ylim(-1,1);ax2.set_xlabel('Adjusted prevalence difference\n(percentage points)');ax2.set_title('Secondary observational endpoint',loc='left',fontsize=10);fig.tight_layout()
save(fig,'figure6_share_conditional_estimates','Conditional sampling intervals: country-influence t approximation where available. D8 timing ambiguous; earlier W7 cohort differs. Binary caseness is a screening endpoint, not a clinical diagnosis or causal effect.')
tipping=pd.read_csv(inputs/'share_update/unknown_survival_tipping.csv');fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,4.7))
main=models['aipw_demographic_country'];shifts=np.linspace(-1,1,201);ax.fill_between(shifts,main['ci_lower']+shifts,main['ci_upper']+shifts,color='#DDECF3',label='Shifted conditional sampling interval');ax.plot(shifts,main['effect']+shifts,c='#236B8E');ax.axhline(0,c='.5',ls='--');ax.axvline(-main['effect'],c='#924D2D',ls=':');ax.set(xlabel='Arbitrary confounding bias shift (EURO-D points)',ylabel='Shifted observed association',title='A. Confounding assumptions can reverse interpretation');ax.legend(fontsize=8,loc='upper left')
for f,g in tipping.groupby('hypothetical_unknown_alive_fraction'):
 # Hold unexposed missing-outcome shift at zero; vary exposed missing outcome.
 g=g.loc[g.delta_unexposed.eq(0)].sort_values('delta_exposed');ax2.plot(g.delta_exposed,g.difference,marker='o',label=f'{f:.0%} unknown assumed alive')
ax2.axhline(0,c='.5',ls='--');ax2.set(xlabel='Exposed missing-outcome shift (unexposed shift=0)',ylabel='Hypothetical survivor mean difference',title='B. Unknown survival and nonresponse sensitivity');ax2.legend(fontsize=8);fig.tight_layout()
save(fig,'figure7_sensitivity_tipping','Arbitrary bias shifts and hypothetical unknown-survivor composition are assumptions, not calibrated bounds or sampling intervals. Known deaths are never imputed. A sign reversal is possible in the full two-arm shift grid.')
# Supplementary conceptual DAG alternatives; arrows are hypotheses,not facts.
import networkx as nx
fig,axes=plt.subplots(1,2,figsize=(12,5.6))
positions={'B':(.05,.85),'H':(.05,.15),'U':(.55,1.),'A':(.40,.55),'D8':(.45,.12),'Y':(.90,.55),'S':(1.15,.15)}
common=[('B','A'),('B','Y'),('H','A'),('H','Y'),('U','A'),('U','Y'),('A','Y'),('A','S'),('Y','S')]
for ax,variant in zip(axes,('earlier','overlap')):
 edges=common+([('D8','A'),('D8','Y')] if variant=='earlier' else [('A','D8'),('D8','Y')])
 graph=nx.DiGraph(edges)
 nx.draw_networkx_nodes(graph,positions,ax=ax,node_color=['#F4DEDE' if n=='U' else '#E2ECF3' for n in graph],node_size=1050,edgecolors='#32465A')
 nx.draw_networkx_labels(graph,positions,ax=ax,font_size=11)
 nx.draw_networkx_edges(graph,positions,ax=ax,edgelist=[e for e in edges if e[0]!='U'],node_size=1050,arrowsize=16,width=1.15,edge_color='#60788A',connectionstyle='arc3,rad=.06')
 nx.draw_networkx_edges(graph,positions,ax=ax,edgelist=[e for e in edges if e[0]=='U'],node_size=1050,arrowsize=16,width=1.4,style='dashed',edge_color='#AA4B49',connectionstyle='arc3,rad=-.08')
 ax.set_title('A. D8 as an earlier common cause' if variant=='earlier' else 'B. D8 within the participation pathway',loc='left',fontsize=11);ax.set_xlim(-.12,1.31);ax.set_ylim(-.12,1.18);ax.axis('off')
fig.text(.5,.04,'B:demographics/country   H:childhood history   U:unmeasured health/social history\nS:outcome recording;death-defined depression needs a separate estimand. Hypothetical arrows require scientific adjudication.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.10,1,1])
save(fig,'figureS1_alternative_causal_graphs','Conceptual D8 alternatives with latent common causes and recording selection. Arrows are hypotheses,not validated causal facts;recording selection is not a model that imputes depression after death.')

(root/'clinical_figure_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(figures=len(manifest)//3,files=len(manifest),output=str(out))))

(root/'clinical_figure_provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('empirical_results',old/'results.json'),('binary_summary',inputs/'share_binary/summary.json'),('tipping_grid',inputs/'share_update/unknown_survival_tipping.csv')]},flow_counts=flow,no_raw_respondent_input=True,provider_calls=0),indent=2))
