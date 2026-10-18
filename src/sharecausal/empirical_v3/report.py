"""All empirical figures and text remain private pending disclosure review."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .selection import STATUSES


def save(fig, root, name):
    for suffix in ['pdf','png']:
        path=root/f'{name}.{suffix}'
        fig.savefig(path,dpi=300,bbox_inches='tight')
        path.chmod(0o600)
    plt.close(fig)


def graph_figures(root,theory):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    for timing in ['pre','post']:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        candidates=[s for s in theory if s['timing'].endswith(timing)]
        pos={'B':(0,1.1),'H':(0,-.8),'A':(1,.1),'D8':(1,1.2),'Y':(2,.4),
             'U':(1,-1.6),'V':(2,-.9),'R':(3,.15)}
        for ax,s in zip(axes.flat,candidates):
            g=nx.DiGraph(s['edges'])
            nx.draw_networkx_nodes(g,pos,ax=ax,node_size=900,node_color='#dceaf5')
            nx.draw_networkx_labels(g,pos,ax=ax,font_size=9)
            # Curves prevent long edges passing through unrelated node centers
            # (e.g. H->V crossing U or A->R crossing Y).
            nx.draw_networkx_edges(g,pos,ax=ax,node_size=900,arrowsize=18,
                                   connectionstyle='arc3,rad=0.16')
            ax.set_title(s['scenario']+' — hypothesis only');ax.axis('off')
        fig.suptitle('D8 timing alternatives; U latent; V survival; R response among V=1\n'
                     'Y defined only in survivors; drawings do not identify a survivor causal effect',fontsize=11)
        save(fig,root,'dag_D8_'+timing)


def figures(root,result):
    est=result['estimates']
    labels={'unadjusted':'Unadjusted observed outcomes',
            'aipw_demographic_country':'AIPW: demographics and country',
            'same_sample_without_w8_depression':'Matched sample: without W8 depression',
            'aipw_with_w8_depression':'Matched sample: with W8 depression',
            'ols_age_sex_education_country':'OLS: demographics and country'}
    fig,ax=plt.subplots(figsize=(8,5))
    for i,(name,r) in enumerate(est.items()):
        ax.errorbar(r['effect'],i,xerr=[[r['effect']-r['ci_lower']],[r['ci_upper']-r['effect']]],fmt='o',color='#155f82')
    ax.set_yticks(range(len(est)),[labels.get(n,n) for n in est]);ax.axvline(0,color='gray',ls=':')
    ax.set_xlabel('Observed-outcome EURO-D contrast (conditional normal 95% interval)')
    ax.set_title('Exploratory standardized associations — causal decision ABSTAIN')
    save(fig,root,'association_forest')
    statuses=[r for r in result['selection_counts'] if r['country']=='ALL']
    fig,ax=plt.subplots(figsize=(8,4));bottom=np.zeros(len(statuses))
    for status in STATUSES:
        values=np.array([r[status]/r['n_eligible'] for r in statuses])
        ax.bar([str(r['A']) for r in statuses],values,bottom=bottom,label=status);bottom+=values
    ax.set_xlabel('W8 participation A');ax.set_ylabel('Fraction of W8 eligible cohort')
    ax.legend(bbox_to_anchor=(1.02,1),fontsize=8);save(fig,root,'selection_status')
    countries=[r for r in result['countries'] if r['effect'] is not None]
    fig,ax=plt.subplots(figsize=(7,8))
    for i,r in enumerate(countries):
        ax.errorbar(r['effect'],i,xerr=[[r['effect']-r['ci_lower']],[r['ci_upper']-r['effect']]],fmt='o',color='#155f82')
    ax.set_yticks(range(len(countries)),[f"{r['country']} (n={r['n']})" for r in countries])
    ax.axvline(0,color='gray',ls=':');ax.set_xlabel('Country-specific observed association, conditional t interval')
    save(fig,root,'country_associations')
    rows=result['balance']
    fig,ax=plt.subplots(figsize=(7,8))
    for field,label in [('smd_raw','Unweighted'),('smd_weighted','Propensity weighted')]:
        ax.scatter([r[field] for r in rows],range(len(rows)),label=label,s=16)
    ax.set_yticks(range(len(rows)),[r['variable'] for r in rows]);ax.axvline(0,color='gray',ls=':')
    ax.set_xlabel('Standardized mean difference');ax.legend();save(fig,root,'measured_balance')
    rows=result['mnar_grid'];shifts=sorted({r['delta_exposed'] for r in rows})
    matrix=np.array([[next(r['mean_difference'] for r in rows if r['delta_unexposed']==a and r['delta_exposed']==b)
                      for b in shifts] for a in shifts])
    fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(matrix,origin='lower',cmap='RdBu_r')
    ax.set_xticks(range(len(shifts)),shifts);ax.set_yticks(range(len(shifts)),shifts)
    ax.set_xlabel('Missing-survivor shift, exposed');ax.set_ylabel('Missing-survivor shift, unexposed')
    ax.set_title('Recorded-alive pattern-mixture association\nArbitrary stress parameters; deaths and unknown vital status excluded')
    fig.colorbar(im,ax=ax,label='EURO-D mean difference');save(fig,root,'survivor_mnar')
    keys=['valid_wave8_activity','w9_cv_record_linked','w9_valid_eurod_forward_order']
    fig,ax=plt.subplots(figsize=(7,4));ax.barh(keys,[result['flow'][k] for k in keys],color='#155f82')
    ax.set_xlabel('Participants (W8 anchored; linked status includes deaths)');save(fig,root,'participant_flow')
    groups=[r for r in result['childhood'] if r['effect'] is not None]
    fig,ax=plt.subplots(figsize=(8,4))
    for i,r in enumerate(groups):
        ax.errorbar(r['effect'],i,xerr=[[r['effect']-r['ci_lower']],[r['ci_upper']-r['effect']]],fmt='o')
    ax.set_yticks(range(len(groups)),[r['analysis'] for r in groups]);ax.axvline(0,color='gray',ls=':')
    ax.set_xlabel('Exploratory childhood-subset association (conditional t interval)');save(fig,root,'childhood_subsets')


def appendix(root,result):
    lines=['# PRIVATE — exploratory SHARE–LLM follow-up', '',
           'Disclosure review required before any release. No causal effect or methodological superiority is established.', '',
           'Four previously collected graph records were audited offline before estimation. All three nonempty proposals '
           'include U→A and U→Y; no measured backdoor set exists under these proposals. This is a backdoor audit, '
           'not a general nonidentification proof. One no-graph abstention is missing graph evidence.', '',
           '| Analysis | n | Association | Conditional normal 95% interval |',
           '|---|---:|---:|---:|']
    for name,r in result['estimates'].items():
        lines.append(f"| {name} | {r['n']} | {r['effect']:.6f} | [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}] |")
    p=result['paired_D8']
    lines+=['',f"D8 inclusion changes the matched-sample association by {p['effect']:.6f} "
            f"(paired country-influence t interval [{p['ci_lower']:.6f}, {p['ci_upper']:.6f}], "
            f"n={p['n']}, {p['clusters']} countries). Folds and rows are shared; intervals are approximate. "
            'D8 overlaps the exposure reference year: neither confounding nor mediation is established.', '',
            '## Selection and truncation', '',
            'Counts distinguish recorded deaths (Y undefined), recorded-alive interview/item nonresponse, '
            'and unlinked/unknown vital status. Recorded alive does not independently validate vital status for every person. '
            'Pattern-mixture shifts and bounded-outcome ranges apply only within recorded survivors. '
            'Unknown survival is excluded, not relabeled alive. Response weighting assumes MAR in that restricted domain; '
            'it is not a death correction, population effect or principal-stratum effect.', '',
            '## Scientific limits and unresolved decisions', '',
            'DAG arrows remain hypotheses. B/H are conceptual blocks: demographic proxies do not exhaust health confounding. '
            'Hypothetical deletion of U is not empirical identification. Depression comparisons condition on observed outcomes. '
            'Childhood categories 1/2 versus 3 require SHARELIFE eligibility; categories 4/5 and missing histories are excluded. '
            'W7 depression and childhood history have disjoint observation universes here; no fabricated joint adjustment is made. '
            'Country clustering is a dependence sensitivity, not complete complex-survey inference. Cross-fitting splits people, '
            'not countries. No valid supplied longitudinal survey weight has been established. Country/subgroup intervals are '
            'exploratory, uncorrected for multiplicity; PSU clustering where available does not by itself recover survey design. '
            'Fixed regularized nuisance models may be misspecified. Normal legacy intervals are reproduced; t intervals are '
            'additional approximations, not validated coverage guarantees. Bias shifts are uncalibrated stress tests.', '',
            'DECISION_REQUIRED: intervention/target population; D8 temporal adjudication; vital-status ascertainment and '
            'unknown-survival treatment; substantive confounder measurement; plausible MNAR/confounding parameters; '
            'survey design and subgroup interpretation; publication disclosure review.', '',
            '## Independent audit availability', '',result['independent_audit_status'], '',
            '## Real-provider synthetic validation', '',
            'A separate offline recorded-response replay adapter is implemented and tested on synthetic records. '
            'The four SHARE-domain prompts do not match synthetic public packets and are ineligible for synthetic accuracy '
            'scoring. No real-provider synthetic validation run is possible with these records; no new calls are authorized.', '']
    (root/'results_appendix_PRIVATE.md').write_text('\n'.join(lines));(root/'results_appendix_PRIVATE.md').chmod(0o600)
