"""Vector-friendly plots (svg/png) from computed aggregate outputs only."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def flow_plot(flow,folder):
    choices=[('Wave 8 interviewed',flow['wave8_main_interview']),
             ('Valid W8 activity',flow['valid_wave8_activity']),
             ('W9 linked coverscreen',flow['w9_cv_record_linked']),
             ('W9 main interview',flow['w9_main_interview']),
             ('W9 valid EURO-D',flow['w9_valid_eurod_forward_order'])]
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.barh([x[0] for x in choices][::-1],[x[1] for x in choices][::-1])
    for i,(_,v) in enumerate(choices[::-1]):ax.text(v+250,i,f'{v:,}',va='center',fontsize=10)
    ax.set_xlim(0,max(x[1] for x in choices)*1.12)
    ax.set_xlabel('Respondents');ax.set_title('SHARE Waves 8–9: recorded observation flow')
    ax.spines[['right','top']].set_visible(False)
    _save(fig,folder,'participant_flow')


def effect_plot(estimates,folder):
    rows=[(k,v) for k,v in estimates.items() if isinstance(v,dict) and v.get('effect') is not None]
    labels=[x[0].replace('_',' ') for x in rows]
    fig,ax=plt.subplots(figsize=(9,max(3.5,.8*len(rows))),layout='constrained')
    centers=[v['effect'] for _,v in rows]
    left=[v['effect']-v['ci_lower'] for _,v in rows]
    right=[v['ci_upper']-v['effect'] for _,v in rows]
    ax.errorbar(centers,list(range(len(rows))),xerr=[left,right],fmt='o',capsize=4)
    ax.axvline(0,ls='--',linewidth=1)
    ax.set_yticks(list(range(len(rows))),labels);ax.invert_yaxis()
    ax.set_xlabel('Wave 9 EURO-D difference (points); graph-conditional intervals')
    ax.set_title('Observed-outcome modeling sensitivity — NOT identified causal effects')
    ax.spines[['right','top']].set_visible(False)
    _save(fig,folder,'model_comparison')


def benchmark_plots(benchmark,folder):
    methods=sorted(benchmark['summary'])
    m=[benchmark['summary'][k] for k in methods]
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    x=list(range(len(methods)))
    ax.bar(x,[a['rmse'] for a in m]);ax.set_xticks(x,[s.replace('_','\n') for s in methods])
    ax.set_ylabel('RMSE vs known structural ATE');ax.set_title('Synthetic benchmark: estimation error')
    _save(fig,folder,'benchmark_rmse')
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    ax.scatter([a['assertion_rate'] for a in m],
               [a['selective_false_sign_risk'] if a['selective_false_sign_risk'] is not None else float('nan') for a in m],
               s=80)
    for k,a in zip(methods,m):
        if a['selective_false_sign_risk'] is not None:
            ax.annotate(k.replace('_',' '),(a['assertion_rate'],a['selective_false_sign_risk']),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.set_xlim(-.05,1.05);ax.set_ylim(-.05,1.05)
    ax.set_xlabel('Assertion rate');ax.set_ylabel('False sign among asserted results')
    ax.set_title('Synthetic selective risk vs abstention (oracle graph-set warning)')
    _save(fig,folder,'benchmark_selective_risk')


def graph_plot(folder):
    import networkx as nx
    from sharecausal.causal.graphs import SPECS
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,s in zip(axs,SPECS[:2]):
        g=s.networkx()
        pos={'B':(0,.8),'D8':(1,0),'A':(1,1.5),'Y':(2,.8)}
        nx.draw_networkx(g,pos=pos,ax=ax,arrows=True,node_size=1500,font_size=11,
                         arrowsize=22,edge_color='#596275',node_color='#e5edf2')
        ax.set_title(s.name.replace('_',' '));ax.axis('off')
    _save(fig,folder,'alternative_dags')


def _save(fig,folder,name):
    path=Path(folder);path.mkdir(parents=True,exist_ok=True)
    for suff in ('.png','.svg'):
        fig.savefig(path/(name+suff),dpi=210,bbox_inches='tight')
    plt.close(fig)


def subgroup_plot(subgroups,folder):
    rows=[]
    for k in ('0','1'):
        v=subgroups.get(k,{})
        if v.get('status')=='estimated':rows.append(('Childhood poor' if k=='1' else 'Average/well-off childhood',v))
    if not rows:return
    fig,ax=plt.subplots(figsize=(8,3.2),layout='constrained')
    for i,(name,est) in enumerate(rows):
        ax.errorbar([est['effect']],[i],xerr=[[est['effect']-est['ci_lower']],[est['ci_upper']-est['effect']]],fmt='o',capsize=5)
    ax.axvline(0,ls='--',linewidth=1)
    ax.set_yticks(range(len(rows)),[a for a,_ in rows]);ax.invert_yaxis()
    ax.set_xlabel('Conditional AIPW association (EURO-D points)')
    ax.set_title('Childhood financial categories: exploratory subgroup associations')
    _save(fig,folder,'childhood_subgroups')


def attrition_plot(attrition,folder):
    a=[attrition[str(k)]['proportion_observed']*100 for k in (0,1)]
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    ax.bar(['No club reported','Club reported'],a)
    ax.set_ylim(0,100);ax.set_ylabel('Valid Wave 9 EURO-D (%)')
    ax.set_title('Wave 9 outcome observation by Wave 8 activity')
    for i,x in enumerate(a):ax.text(i,x+1,f'{x:.1f}%',ha='center')
    _save(fig,folder,'outcome_observation')


def sensitivity_plot(shifts,folder):
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    x=[x['bias_shift'] for x in shifts];y=[x['effect'] for x in shifts]
    ax.plot(x,y,marker='o')
    ax.fill_between(x,[x['lower'] for x in shifts],[x['upper'] for x in shifts],alpha=.2)
    ax.axhline(0,ls='--',linewidth=1)
    ax.set_xlabel('Assumed additive outcome-scale confounding bias (points)')
    ax.set_ylabel('Scenario-shifted model difference (EURO-D points)')
    ax.set_title('Illustrative bias-shift sensitivity, NOT identified bounds')
    _save(fig,folder,'confounding_sensitivity')


def graph_quality_plot(graph_qualities,folder):
    if not graph_qualities:return
    labels=[x['scenario'] for x in graph_qualities]
    vals=[x['directed_recall'] for x in graph_qualities]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    ax.bar(labels,vals)
    ax.set_ylim(0,1);ax.set_ylabel('Known-DAG edge recall of fixed baseline graph')
    ax.set_title('Synthetic SCM graph misspecification — NOT LLM performance')
    ax.tick_params(axis='x',rotation=30)
    _save(fig,folder,'synthetic_graph_baseline_recall')


def nonoracle_risk_plot(curve, result_dir):
    """Real execution: error versus assertion rate, no oracle selector.

    Zero-assertion methods have undefined risk and are excluded from the
    plotted risk axis. No model-calibration interpretation is warranted.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    path = Path(result_dir)
    fig, ax = plt.subplots(figsize=(8, 5.3), layout='constrained')
    for method in sorted({r['method'] for r in curve}):
        pts = sorted((r for r in curve if r['method'] == method
                      and r['false_sign_risk'] is not None),
                     key=lambda x:x['assertion_rate'])
        if not pts:
            continue
        ax.plot([x['assertion_rate'] for x in pts],
                [x['false_sign_risk'] for x in pts], 'o-',
                label=method.replace('_', ' '), markersize=4)
    ax.set(xlabel='Fraction of scenarios with a directional assertion',
           ylabel='Wrong-sign risk among asserted scenarios',
           title='Offline fixture evaluation — no live LLMs; no oracle graph selection',
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc='best', ncol=2)
    fig.savefig(path / 'nonoracle_risk_coverage.png', dpi=185)
    fig.savefig(path / 'nonoracle_risk_coverage.svg')
    plt.close(fig)
