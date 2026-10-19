"""Evidence-to-graph compiler and estimation. Observable inputs only."""
from itertools import combinations,product
from types import SimpleNamespace as NS
import re
import math
import numpy as np
from sharecausal.preflight.contracts import sha
from sharecausal.preflight.graph_utils import is_acyclic,has_path
from sharecausal.pilot_v24.graphs import identify_graph
from sharecausal.causal.estimators import aipw,unadjusted

METHODS=('descriptive_association','no_llm_evidence','no_llm_all_baseline',
         'google','openai','llama','deepseek','strict_majority','raw_pool',
         'screened_pool','screened_critic','full_policy','always_abstain')

def edge_set(graph):return sorted(set((e['source'],e['target']) for e in graph['edges']))

def baseline(card):
 """Fixed parser on exact public prose, with all six ex-ante document styles."""
 patterns=[r'\b([UXAMYC]) directly changes ([UXAMYC])',r'\b([UXAMYC]) is causally responsive to ([UXAMYC])',
  r'direct influence of ([UXAMYC]) on ([UXAMYC])',r'input to ([UXAMYC]) is ([UXAMYC])',
  r'Changing ([UXAMYC]) changes the mechanism for ([UXAMYC])',r'identifies ([UXAMYC]) as a direct cause of ([UXAMYC])']
 optional=[r'([UXAMYC]) might directly influence ([UXAMYC])',r'channel from ([UXAMYC]) to ([UXAMYC])',
  r'allows ([UXAMYC]) to act on ([UXAMYC])',r'whether ([UXAMYC]) directly changes ([UXAMYC])',
  r'contribution of ([UXAMYC]) to ([UXAMYC])',r'arrow from ([UXAMYC]) to ([UXAMYC])']
 required=[];ambiguous=[]
 for doc in card.documents:
  for j,pat in enumerate(patterns):
   if any(re.search(option,doc.text) for option in optional):continue
   match=re.search(pat,doc.text)
   if match:
    a,b=match.groups()
    if j in (1,3):a,b=b,a
    required.append(dict(source=a,target=b,evidence_id=doc.evidence_id,quote=doc.text,support='entailed',role='direct'))
  for pat in optional:
   match=re.search(pat,doc.text)
   if match:
    a,b=match.groups();ambiguous.append(dict(source=a,target=b,evidence_id=doc.evidence_id,quote=doc.text,support='credible_ambiguous',role='uncertain'))
 return [dict(edges=required+[e for e,yes in zip(ambiguous,bits) if yes],explanation='Fixed exhaustive evidence parser; no model vote.') for bits in product((False,True),repeat=len(ambiguous))]

def compile_graph(card,rubric,graph):
 required={tuple(x['edge']) for x in rubric['annotations'] if x['state']=='entailed'}
 allowed={tuple(x['edge']):x for x in rubric['annotations']}
 docs={d.evidence_id:d.text for d in card.documents};edges=edge_set(graph);issues=[];records=[]
 if len(edges)!=len(graph['edges']):issues.append('duplicate_edges')
 for e in graph['edges']:
  pair=(e['source'],e['target']);annotation=allowed.get(pair)
  exact=e['evidence_id'] in docs and bool(e['quote']) and e['quote'] in docs[e['evidence_id']]
  supported=bool(annotation and annotation['evidence_id']==e['evidence_id'] and annotation['state']==e['support'])
  temporal=pair[0] in card.variables and pair[1] in card.variables and card.variables[pair[0]]['time']<card.variables[pair[1]]['time']
  records.append(dict(task_id=card.task_id,edge=list(pair),evidence_id=e['evidence_id'],evidence_span_hash=sha(e['quote']),support_state=annotation['state'] if supported else 'unsupported_or_contradicted',exact_quote=exact,temporal_status=temporal,role=e['role'],admission_decision=exact and supported and temporal))
  if not exact:issues.append('invalid_evidence_quote')
  if not supported:issues.append('unsupported_or_misclassified_edge')
  if not temporal:issues.append('chronology_or_unknown_node')
 if not required<=set(edges):issues.append('missing_entailed_edge')
 if not is_acyclic(edges):issues.append('cycle')
 if rubric['flags']['randomized'] and any(b=='A' for a,b in edges):issues.append('randomization_contradiction')
 if rubric['flags']['structural_zero'] and has_path(edges):issues.append('global_exclusion_contradiction')
 return dict(graph_hash=sha(graph),admitted=not issues,reasons=sorted(set(issues)),edges=records)

def identify(card,edges):
 variables=[NS(name=v,time=x['time'],observed=x['observed'],role='baseline' if v not in ('A','Y','U') and x['time']<card.variables['A']['time'] else 'other') for v,x in card.variables.items()]
 task=NS(card=NS(variables=variables,selection='complete_outcomes'))
 return identify_graph(task,edges)

def analyze(card,rubric,frame,answers,critic=None):
 """All support/graph choices fixed before any observable numeric sample is fitted."""
 base=baseline(card);sources=[];audits=[];response_status={}
 for model_answer in answers:
  graphs=(model_answer.get('answer') or {}).get('graphs',[]) if model_answer['status']=='valid' else []
  sources.append(graphs);response_status[model_answer['model']]=model_answer['status']
 raw=[g for gs in sources for g in gs]
 screened=[]
 for source_idx,graphs in enumerate(sources):
  for g in graphs:
   a=compile_graph(card,rubric,g);a['source_index']=source_idx;audits.append(a)
   if a['admitted']:screened.append(g)
 reviews={r['graph_hash']:r for r in ((critic or {}).get('answer') or {}).get('reviews',[])}
 docs={d.evidence_id:d.text for d in card.documents}
 # Critic can only reject with a valid evidence quote AND a compiler-demonstrable
 # violation. It cannot delete credible alternatives on unsupported authority.
 retained=[];critic_audit=[]
 for g in screened:
  r=reviews.get(sha(g));valid=bool(r and r['evidence_id'] in docs and r['quote'] and r['quote'] in docs[r['evidence_id']])
  rejection=bool(valid and r['verdict']=='reject' and not compile_graph(card,rubric,g)['admitted'])
  if not rejection:retained.append(g)
  critic_audit.append(dict(graph_hash=sha(g),review_present=bool(r),valid_quote=valid,rejected=rejection))
 votes={}
 for gs in sources:
  if len(gs)==1:
   for edge in edge_set(gs[0]):votes[edge]=votes.get(edge,0)+1
 voted={edge for edge,count in votes.items() if count>len(sources)/2}
 # Construct an uncited majority graph, as a deliberately unadjudicated baseline.
 majority=[dict(edges=[dict(source=a,target=b) for a,b in sorted(voted)],explanation='strict majority planned denominator four')] if sum(len(gs)==1 for gs in sources)>len(sources)/2 else []
 fits={}
 def fit(z):
  key=tuple(z)
  if key in fits:return fits[key]
  observed=frame.loc[frame.observed==1]
  try:
   result=aipw(observed.Y,observed.A,observed[list(z)].to_numpy(float),folds=3,seed=881113,trim=.01).as_dict() if z else unadjusted(observed.Y,observed.A)
   if any(not math.isfinite(float(result[k])) for k in ('effect','ci_lower','ci_upper')):raise ValueError('nonfinite')
   if result.get('overlap_fraction_below_005_or_above_095',0)>.2:raise ValueError('overlap')
   output=dict(ok=True,**result,adjustment=list(z))
  except (ValueError,TypeError,ArithmeticError):output=dict(ok=False,reason='estimator_or_overlap_failure',adjustment=list(z))
  fits[key]=output;return output
 def combine(graphs,strict=False):
  out=dict(decision='abstain',reason='no_valid_graph',estimate=None,lower=None,upper=None,margin=0.,graph_count=len(graphs),identification=[])
  if not graphs:return out
  if strict and not all(rubric['flags'][k] for k in ('complete','target_valid','positivity')):
   out['reason']='selection_target_or_positivity_unresolved';return out
  if strict and not rubric['flags']['exhaustive']:
   out['reason']='open_world_identification_not_established';return out
  rows=[]
  for g in graphs:
   edges=edge_set(g)
   try:idn=identify(card,edges)
   except ValueError:out['reason']='invalid_graph';return out
   out['identification'].append(idn)
   if idn['status']=='graph_conditional_structural_zero':out['reason']='graph_structural_zero';return out
   if idn['status']!='identified_by_baseline_backdoor':out['reason']='restricted_backdoor_not_identified';return out
   f=fit(idn['adjustment']);rows.append(f)
  if any(not r['ok'] for r in rows):out['reason']='estimator_or_overlap_failure';return out
  radius=.1 if strict else 0.
  lo=min(r['ci_lower'] for r in rows)-radius;hi=max(r['ci_upper'] for r in rows)+radius
  decision='positive' if lo>0 else 'negative' if hi<0 else 'abstain'
  out.update(decision=decision,reason='envelope_excludes_zero' if decision!='abstain' else 'envelope_crosses_zero',lower=lo,upper=hi,margin=max(lo,-hi,0.),conditional_lower=min(r['ci_lower'] for r in rows),conditional_upper=max(r['ci_upper'] for r in rows),estimate=rows[0]['effect'] if len({tuple(r['adjustment']) for r in rows})==1 else None)
  return out
 association=fit(())
 methods={'descriptive_association':dict(decision='abstain',reason='association_only',fit=association,margin=0.)}
 methods['no_llm_evidence']=combine(base,True)
 allbaseline=[(v,'A') for v,x in card.variables.items() if x['observed'] and x['time']<1]+[(v,'Y') for v,x in card.variables.items() if x['observed'] and x['time']<1]+[('A','Y')]
 methods['no_llm_all_baseline']=combine([dict(edges=[dict(source=a,target=b) for a,b in allbaseline])])
 for name,graphs in zip(('google','openai','llama','deepseek'),sources):
  accepted=[g for g in graphs if compile_graph(card,rubric,g)['admitted']]
  methods[name]=combine(accepted,True)
 methods['strict_majority']=combine(majority)
 methods['raw_pool']=combine(raw)
 methods['screened_pool']=combine(screened)
 methods['screened_critic']=combine(retained)
 methods['full_policy']=combine(retained,True)
 methods['always_abstain']=dict(decision='abstain',reason='always_abstain_reference',margin=0.)
 assert set(methods)==set(METHODS)
 return dict(task_id=card.task_id,methods=methods,edge_audits=audits,critic_audit=critic_audit,source_status=response_status,n=len(frame),n_recorded=int(frame.observed.sum()),estimator_fits=list(fits.values()),truth_used=False)
