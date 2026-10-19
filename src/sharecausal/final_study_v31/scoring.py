"""Independent post-seal known-truth scorer. Never imported before sealed scoring."""
import csv,json,math
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.certificates import BinaryDAG,law,effect
from sharecausal.preflight.graph_utils import is_acyclic
from sharecausal.preflight_v23.sealing import write_once
from .engine import METHODS,baseline,edge_set,identify
from .transport import load_records
from .public import MODELS


def csv_write(path,rows):
 if not rows:return
 with Path(path).open('x',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def certificate(card,rubric,worlds):
 flags=rubric['flags'];scms=[BinaryDAG.model_validate(dict(nodes=w['nodes'])) for w in worlds]
 if any(w['pair'] for w in worlds):
  observed=sorted(v for v,x in card.variables.items() if x['observed'])
  if len(scms)!=2 or law(scms[0],observed)!=law(scms[1],observed) or effect(scms[0])==effect(scms[1]):raise ValueError('invalid_observational_equivalence_certificate')
  return 'provably_nonidentified','exact_same_full_observed_law_distinct_effects'
 if flags['structural_zero']:
  if any(effect(s)!=0 for s in scms):raise ValueError('false_structural_zero')
  return 'identified','public_global_exclusion'
 if not all(flags[k] for k in ('exhaustive','complete','target_valid','positivity')):return 'uncertified','no_general_identification_proof_implemented'
 graphs=baseline(card)
 statuses=[]
 for g in graphs:
  edges=edge_set(g)
  if not is_acyclic(edges):continue
  try:statuses.append(identify(card,edges)['status'])
  except ValueError:continue
 if statuses and all(s in ('identified_by_baseline_backdoor','graph_conditional_structural_zero') for s in statuses):return 'identified','all_public_exhaustive_graphs_baseline_backdoor_or_zero'
 return 'uncertified','restricted_backdoor_failure_is_not_general_nonidentification'


def score(out,cards,truth):
 out=Path(out);predictions=json.loads((out/'prediction_batch.json').read_text());assignments=json.loads((out/'families.json').read_text());groups={r['task_id']:r for r in assignments};rubrics=json.loads((out/'public_rubric.json').read_text());calibration=json.loads((out/'calibration.json').read_text());worlds=defaultdict(list)
 for w in truth:
  scm=BinaryDAG.model_validate(dict(nodes=w['nodes']))
  if str(effect(scm))!=w['effect_exact']:raise ValueError('exact_truth_effect_mismatch')
  worlds[w['task_id']].append(w)
 cards={c.task_id:c for c in cards};certs={h:certificate(c,rubrics[h],worlds[h]) for h,c in cards.items()}
 cert_rows=[dict(**groups[h],status=s,reason=r,worlds=len(worlds[h])) for h,(s,r) in certs.items()]
 # Remove duplicate task_id construction before serialization.
 csv_write(out/'identification_certificates.csv',cert_rows)
 rows=[];repeat_rows=[];curves=[]
 def evaluate(p,method,margin,practical):
  h=p['task_id'];result=p['methods'][method];decision=result['decision'];asserted=decision in ('positive','negative') and result['margin']>=margin
  status=certs[h][0];effects=[float(Fraction(w['effect_exact'])) for w in worlds[h]]
  wrong=asserted and any(abs(t)<=practical or (t<0 and decision=='positive') or (t>0 and decision=='negative') for t in effects)
  nonid=asserted and status=='provably_nonidentified';zero=asserted and rubrics[h]['flags']['structural_zero'];target=asserted and not rubrics[h]['flags']['target_valid']
  identified_wrong=wrong and status=='identified';unjustified=bool(nonid or zero or target or identified_wrong)
  return dict(task_id=h,family_id=groups[h]['family_id'],split=groups[h]['split'],sample_repetition=p['sample_repetition'],method=method,assertion=int(asserted),wrong_sign_identified=int(identified_wrong),wrong_sign_all_known_worlds=int(wrong),nonidentification_error=int(nonid),structural_zero_error=int(zero),invalid_target_assertion=int(target),unjustified=int(unjustified),uncertified_assertion=int(asserted and status=='uncertified' and not target),certificate=status,estimate=result.get('estimate'),reason=result['reason'])
 for p in predictions:
  for method in METHODS:
   row=evaluate(p,method,calibration[method]['margin_threshold'],.1)
   (rows if p['sample_repetition']==0 else repeat_rows).append(row)
 evaluation=[r for r in rows if r['split']=='evaluation'];evaluation_n=len({r['task_id'] for r in evaluation});development_n=len({r['task_id'] for r in rows if r['split']=='development'});families=sorted({r['family_id'] for r in evaluation})
 def summarize(values,method,**extra):
  n=len(values);a=sum(r['assertion'] for r in values);e=sum(r['unjustified'] for r in values);unknown=sum(r['uncertified_assertion'] for r in values)
  return dict(method=method,tasks=n,assertions=a,assertion_rate=a/n,abstention_rate=1-a/n,unjustified_assertions=e,selective_risk=e/a if a else None,selective_risk_unresolved_upper=min(1,(e+unknown)/a) if a else None,wrong_sign_identified=sum(r['wrong_sign_identified'] for r in values),wrong_sign_all_known_worlds=sum(r['wrong_sign_all_known_worlds'] for r in values),nonidentification_errors=sum(r['nonidentification_error'] for r in values),structural_zero_errors=sum(r['structural_zero_error'] for r in values),invalid_target_assertions=sum(r['invalid_target_assertion'] for r in values),uncertified_assertions=unknown,utility_penalty2=(a-3*e)/n,utility_penalty5=(a-6*e)/n,utility_penalty10=(a-11*e)/n,**extra)
 summaries=[summarize([r for r in evaluation if r['method']==m],m,calibration_margin=calibration[m]['margin_threshold'],independent_family_units=len(families),matched_target_30_to_50_percent=.3<=sum(r['assertion'] for r in evaluation if r['method']==m)/evaluation_n<=.5) for m in METHODS]
 csv_write(out/'method_comparison.csv',summaries);csv_write(out/'task_scores.csv',rows);csv_write(out/'secondary_sample_scores.csv',repeat_rows)
 for margin in (0.,.025,.05,.1,.2,.3,.4,.5):
  for practical in (.05,.1,.2):
   for method in METHODS:
    values=[evaluate(p,method,margin,practical) for p in predictions if p['sample_repetition']==0 and groups[p['task_id']]['split']=='evaluation']
    curves.append(summarize(values,method,margin_threshold=margin,practical_null=practical))
 csv_write(out/'risk_coverage.csv',curves)
 # Descriptive paired cluster bootstrap: never claim sparse-event calibrated
 # precision. All six tasks move together within each of 14 heldout families.
 rng=np.random.default_rng(881917);draws=rng.integers(0,len(families),(5000,len(families)));contrasts=[]
 family_arrays={}
 for method in METHODS:
  family_arrays[method]=np.array([[sum(r['assertion'] for r in evaluation if r['method']==method and r['family_id']==f),sum(r['unjustified'] for r in evaluation if r['method']==method and r['family_id']==f)] for f in families],float)
 base=family_arrays['no_llm_evidence'];b=base[draws].sum(1)
 for method in METHODS:
  matrix=family_arrays[method];v=matrix[draws].sum(1);valid=(b[:,0]>0)&(v[:,0]>0)
  risk_difference=v[valid,1]/v[valid,0]-b[valid,1]/b[valid,0]
  coverage_difference=(v[:,0]-b[:,0])/evaluation_n
  contrasts.append(dict(method=method,reference='no_llm_evidence',family_units=len(families),bootstrap_repetitions=5000,defined_risk_replicates=int(valid.sum()),risk_difference_lower=float(np.quantile(risk_difference,.025)) if len(risk_difference) else None,risk_difference_upper=float(np.quantile(risk_difference,.975)) if len(risk_difference) else None,coverage_difference_lower=float(np.quantile(coverage_difference,.025)),coverage_difference_upper=float(np.quantile(coverage_difference,.975)),inference='descriptive_cluster_bootstrap_not_validated_for_sparse_events_or_family_superpopulation'))
 csv_write(out/'family_paired_contrasts.csv',contrasts)
 # Four intended coverage targets use development-only thresholds; no test tuning.
 targetrows=[]
 dev=[p for p in predictions if p['sample_repetition']==0 and groups[p['task_id']]['split']=='development']
 for target in (.2,.3,.4,.5):
  for method in METHODS:
   grid=(0.,.025,.05,.1,.2,.3,.4,.5)
   rates={t:sum(evaluate(p,method,t,.1)['assertion'] for p in dev)/len(dev) for t in grid}
   threshold=min(grid,key=lambda t:(abs(rates[t]-target),t))
   targetrows.append(summarize([evaluate(p,method,threshold,.1) for p in predictions if p['sample_repetition']==0 and groups[p['task_id']]['split']=='evaluation'],method,target=target,development_rate=rates[threshold],development_threshold=threshold))
 csv_write(out/'development_target_coverage.csv',targetrows)
 estimation=[]
 for method in METHODS:
  eligible=[]
  for p in predictions:
   h=p['task_id'];r=p['methods'][method]
   if groups[h]['split']=='evaluation' and certs[h][0]=='identified' and r.get('estimate') is not None and r.get('conditional_lower') is not None:
    target=float(Fraction(worlds[h][0]['effect_exact']));eligible.append((r['estimate']-target,r['conditional_lower']<=target<=r['conditional_upper']))
  estimation.append(dict(method=method,identified_sample_intervals=len(eligible),bias=float(np.mean([x[0] for x in eligible])) if eligible else None,rmse=float(np.sqrt(np.mean([x[0]**2 for x in eligible]))) if eligible else None,conditional_sampling_coverage=float(np.mean([x[1] for x in eligible])) if eligible else None,meaning='two nested draws per card, normal intervals conditional on supported identification; not structural confidence coverage'))
 csv_write(out/'causal_estimation_diagnostics.csv',estimation)
 records=load_records(out/'receipts');provider=[]
 for m in MODELS:
  rs=[r for r in records if r['model']==m];counts=Counter(r['status'] for r in rs)
  provider.append(dict(model=m,planned_primary=480,planned_critic=120 if m==MODELS[0] else 0,attempted=len(rs),valid_graph_or_critic=counts['valid'],explicit_abstention=counts['explicit_abstention'],technical_failures=sum(v for k,v in counts.items() if k not in ('valid','explicit_abstention')),null=counts['null_content'],schema_failure=counts['schema_failure'],refusal=counts['refusal'],http_error=counts['http_error'],retried=0,cached_initial=0,cost_usd=sum(r['cost_usd'] or 0 for r in rs),unknown_costs=sum(r['cost_usd'] is None for r in rs),prompt_tokens=sum((r['usage'].get('prompt_tokens') or 0) for r in rs),completion_tokens=sum((r['usage'].get('completion_tokens') or 0) for r in rs),providers=';'.join(sorted({r['provider'] for r in rs if r['provider']}))))
 csv_write(out/'provider_comparison.csv',provider)
 csv_write(out/'provider_failures.csv',[dict(task_id=r['task_id'],model=r['model'],variant=r['variant'],repetition=r['repetition'],status=r['status'],cost_usd=r['cost_usd'],provider_response_id=r['provider_response_id']) for r in records if r['status'] not in ('valid','explicit_abstention')])
 edge_accuracy=[];graphcoverage=[]
 for model in MODELS:
  ps=[p for p in predictions if p['sample_repetition']==0 and groups[p['task_id']]['split']=='evaluation'];idx=MODELS.index(model)
  edge_rows=[e for p in ps for a in p['edge_audits'] if a['source_index']==idx for e in a['edges']]
  graphs=[a for p in ps for a in p['edge_audits'] if a['source_index']==idx]
  required_total=0;recovered=0
  for p in ps:
   required={tuple(a['edge']) for a in rubrics[p['task_id']]['annotations'] if a['state']=='entailed'};required_total+=len(required)
   present={tuple(e['edge']) for a in p['edge_audits'] if a['source_index']==idx for e in a['edges'] if e['admission_decision']};recovered+=len(required&present)
  edge_accuracy.append(dict(model=model,intention_to_evaluate_tasks=evaluation_n,proposed_edges=len(edge_rows),valid_evidence_edges=sum(e['admission_decision'] for e in edge_rows),unsupported_or_misclassified_edges=sum(e['support_state']=='unsupported_or_contradicted' for e in edge_rows),invalid_quote_edges=sum(not e['exact_quote'] for e in edge_rows),edge_support_precision=sum(e['admission_decision'] for e in edge_rows)/len(edge_rows) if edge_rows else None,entailed_edge_recall=recovered/required_total if required_total else None,admitted_graphs=sum(g['admitted'] for g in graphs),total_graphs=len(graphs),independent_human_review=False))
 csv_write(out/'evidence_edge_accuracy.csv',edge_accuracy)
 # Privileged true-graph estimator is produced only here, after seal. It cannot
 # be used for threshold selection or as an ordinary no-LLM comparator.
 import pandas as pd
 from sharecausal.causal.estimators import aipw,unadjusted
 oracle=[];manifest=json.loads((out/'sample_manifest.json').read_text())
 for h,card in cards.items():
  if groups[h]['split']!='evaluation':continue
  w=worlds[h][0];edges=[(a,n['name']) for n in w['nodes'] for a in n['parents']]
  result=dict(task_id=h,certificate=certs[h][0],privileged=True,effect=float(Fraction(w['effect_exact'])),estimate=None,ci_lower=None,ci_upper=None,status='restricted_oracle_unavailable')
  if rubric_target:=rubrics[h]['flags']['target_valid']:
   try:
    identity=identify(card,edges)
    if identity['status']=='graph_conditional_structural_zero':result.update(status='oracle_structural_zero',estimate=0.)
    elif identity['status']=='identified_by_baseline_backdoor' and rubrics[h]['flags']['complete'] and rubrics[h]['flags']['positivity']:
     sample=next(r for r in manifest if r['task_id']==h and r['sample_repetition']==0);frame=pd.DataFrame(json.loads((out/sample['file']).read_text()));z=identity['adjustment']
     fit=aipw(frame.Y,frame.A,frame[z].to_numpy(float),folds=3,seed=881113,trim=.01).as_dict() if z else unadjusted(frame.Y,frame.A)
     result.update(status='oracle_graph_conditional_fit',estimate=fit['effect'],ci_lower=fit['ci_lower'],ci_upper=fit['ci_upper'])
   except (ValueError,ArithmeticError):result['status']='oracle_estimator_failure'
  oracle.append(result)
 csv_write(out/'oracle_upper_bound.csv',oracle)
 dispositions=[]
 for m in METHODS:
  for reason,n in Counter(r['reason'] for r in evaluation if r['method']==m).items():dispositions.append(dict(method=m,reason=reason,tasks=n))
 csv_write(out/'decision_dispositions.csv',dispositions)
 summary=dict(evaluation_cards=evaluation_n,development_cards=development_n,evaluation_family_units=len(families),development_family_units=len({r["family_id"] for r in rows if r["split"]=="development"}),within_family_cards=6,primary_sample_per_card=1,secondary_samples_per_card=1,total_requests=len(records),total_cost_usd=sum(r['cost_usd'] or 0 for r in records),provider=provider,methods=summaries,certificates=dict(Counter(certs[h][0] for h in cards if groups[h]['split']=='evaluation')),minimum_substantive_risk_reduction=.05,superiority_claim=False,limitations=['Single-team evidence/generator authorship, no independent expert adjudication','20 declared mechanisms rather than40; constituent causal motifs have developmental history','14 fixed evaluation family units; sparse bootstrap intervals are descriptive, not calibrated','Unknown identification certificates yield risk bounds, not invented nonidentification proofs','Critic cannot override exact evidence compiler or discard credible alternatives on authority','Oracle uses restricted backdoor/structural-zero criteria, not general causal identification','Finite sample nuisance misspecification and normal conditional intervals; no structural coverage guarantee'])
 write_once(out/'summary.json',summary)
 return summary
