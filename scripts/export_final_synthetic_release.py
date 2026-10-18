"""Explicit post-score synthetic aggregate allowlist; never copies live truth/raw."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
from collections import Counter
from sharecausal.preflight_v23.sealing import verify,digest
from sharecausal.synthetic_final_v1.public import MODELS,SLOTS

def load(p):return json.loads(p.read_text())
def csvrows(p):return list(csv.DictReader(p.open()))
def copy_artifact(source,target):
 if target.exists():
  if digest(source)!=digest(target):raise ValueError('Reviewed artifact differs; preserve and review')
  return
 shutil.copy2(source,target)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--old-pilot',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out=a.output;out.mkdir(parents=True,exist_ok=True);r=a.results
 seal=(r/'prediction_seal_sha256.txt').read_text().strip();verify(r,load(r/'prediction_seal.json'),'prediction_seal.json',seal)
 scorer=load(r/'scorer_receipt.json');assert scorer['seal_verified_before_truth_open']
 summary=load(r/'collection_summary.json');primary=load(r/'prediction_batch.json');repeat=load(r/'stability_prediction_batch.json')
 receipts=[load(p) for p in (r/'provider_responses_private').glob('*.json') if not p.name.endswith(('.raw.json','.inflight.json'))]
 old=load(a.old_pilot/'collection_summary.json');modelrows=[]
 dispositions=[]
 for method in primary[0]['methods']:
  counts=Counter()
  for prediction in primary:
   value=prediction['methods'][method];decision=value['decision'];reason=value.get('reason','')
   slot=None
   if method.startswith('source_'):slot=SLOTS['abcd'.index(method[-1])]
   elif method=='single_preselected_source':slot=SLOTS[0]
   status=prediction['source_audit'][slot]['response_status'] if slot else None
   if decision in ('positive','negative'):kind='directional_assertion'
   elif value.get('structural_effect')==0:kind='public_structural_zero'
   elif slot and status not in ('graph_response','multiple_graphs','source_abstained'):kind='technical_response_failure'
   elif 'estimator_failed' in reason or 'estimator_or_positivity_failure' in reason:kind='estimator_failure'
   elif method=='association_unadjusted':kind='association_only_control'
   else:kind='framework_withheld'
   counts[kind]+=1
  assert sum(counts.values())==len(primary)
  row=dict(method=method,eligible_cards=len(primary),**{k:counts[k] for k in ('directional_assertion','public_structural_zero','technical_response_failure','estimator_failure','association_only_control','framework_withheld')})
  row.update(directional_assertion_rate=counts['directional_assertion']/len(primary),framework_withheld_rate=counts['framework_withheld']/len(primary),no_direction_rate=1-counts['directional_assertion']/len(primary))
  dispositions.append(row)
 for model,slot in zip(MODELS,SLOTS):
  rec=[p for p in receipts if p['model_requested']==model]
  parsed=Counter(p['source_audit'][slot]['response_status'] for p in primary+repeat)
  good=sum(parsed[s] for s in ('graph_response','multiple_graphs','source_abstained'))
  modelrows.append(dict(model=model,planned_primary=24,planned_counterargument=8,attempted=len(rec),
   attempted_primary=sum(x['variant']=='base' for x in rec),attempted_counterargument=sum(x['variant']=='counterargument' for x in rec),
   outer_schema_responses=sum(x['status']=='response' for x in rec),parsed_valid_captures_including_explicit_abstention=good,
   valid_nonabstaining_graph_captures=parsed['graph_response']+parsed['multiple_graphs'],technical_failures=len(rec)-good,
   explicit_epistemic_abstentions=parsed['source_abstained'],parsed_status=dict(parsed),transport_status=dict(Counter(x['status'] for x in rec)),
   providers=dict(Counter(x['provider'] or 'not_reported' for x in rec)),
   reported_cost_usd=sum(x['cost_usd'] or 0 for x in rec),unknown_cost_records=sum(x['cost_usd'] is None for x in rec),
   prompt_tokens=sum(x['usage'].get('prompt_tokens',0) or 0 for x in rec),completion_tokens=sum(x['usage'].get('completion_tokens',0) or 0 for x in rec)))
 report=dict(study='final exploratory synthetic expansion',planned_primary=96,planned_stability=32,planned_posts=128,
  actual_posts=len(receipts),initial_cached_requests_all_models=summary['cached'],scored_primary_cards=len(primary),stability_cards=len(repeat),sample_n_per_card=800,
  model_results=modelrows,reported_cost_usd=summary['cost_usd'],unresolved_costs=summary['unresolved_costs'],
  original_pilot_reported_cost_usd=old['cost_usd'],both_synthetic_batches_reported_cost_usd=old['cost_usd']+summary['cost_usd'],
  batch_upper_reservation_usd=load(r/'request_plan.json')['upper_reservation_usd'],software_cap_usd=25,
  prediction_batch_sha256=digest(r/'prediction_batch.json'),stability_prediction_batch_sha256=digest(r/'stability_prediction_batch.json'),prediction_seal_sha256=seal,
  seal_verified_before_truth_open=True,truth_open_authorized_at=scorer['truth_open_authorized_at'],
  decision_dispositions=dispositions,method_results=csvrows(r/'method_results.csv'),conditional_coverage=load(r/'conditional_coverage.json'),prompt_stability=load(r/'prompt_stability.json'),
  certification_counts=dict(Counter(x['status'] for x in load(r/'scored_results.json')['certifications'].values())),
  limitations=['single-team authorship and development reuse','restricted backdoor and conservative randomization gate','one sample/card and correlated prompt variants','no confirmatory superiority or structural coverage','reported charges not independently invoice-verified','technical failures retained; no-assertion is not all scientific abstention'])
 with (out/'decision_dispositions.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(dispositions[0]));writer.writeheader();writer.writerows(dispositions)
 (out/'completed_run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 allowed=['method_results.csv','risk_abstention.csv','paired_family_contrasts.csv','prompt_stability.csv','conditional_coverage.csv','eligible_source_diagnostics.json']
 for name in allowed:copy_artifact(r/name,out/name)
 with (out/'model_results.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(modelrows[0]));writer.writeheader()
  for row in modelrows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
 for name in ('method_results.csv','risk_abstention.csv'):copy_artifact(a.old_pilot/name,out/('initial_pilot_'+name))
 # Public task descriptions are intentionally released. Hidden CPTs/live truth,
 # observations, transcripts, response IDs and private receipts are not copied.
 copy_artifact(r/'public_tasks.json',out/'public_tasks.json')
 for p in a.figures.glob('synthetic_*'):
  if p.suffix in ('.png','.pdf','.svg'):copy_artifact(p,out/p.name)
 (out/'README.md').write_text('''# Reviewed synthetic aggregate release

This directory deliberately contains public synthetic packets, computed method/model aggregate tables and figures only. It contains no SHARE data or empirical findings, no observed samples, no protected live scoring store, no raw provider responses or response IDs, and no credential. Authoring source is available separately for reproducibility. Initial and expanded pilots remain separate.

Primary denominator24 cards; counterargument repeats8 cards/model, nested within primary tasks. Twelve scenario definitions and26 nested worlds are not proven independent observations. No confirmatory superiority is demonstrated. decision_dispositions.csv separates directional assertions, public zero, technical response failures, dominant estimator failures, the association-only control and framework-withheld conclusions. The latter includes insufficient majorities and identification/envelope withholding; it is not solely an explicit LLM epistemic abstention. Failed responses remain in denominators. Parsed-valid capture is a format/timing status, not correctness or evidence admission. Private raw-fragment audits are not exported as if they were eligible whole-source hypotheses. Scientific no-direction includes structural zero and justified abstention; technical failures are separately reported. Zero-assertion risk is undefined. Family-t intervals are unvalidated; prior undercoverage .097 remains unresolved. Costs are OpenRouter-reported usage charges, not independently reconciled invoices.

The prediction and seal hashes refer to the protected complete batch. No hidden truth was opened by the scorer until seal verification. Local hashes and single-team custody are not external preregistration. Exact internal source slots map to the four real model IDs in completed_run.json; mock-family labels in reused engine code are internal aliases, not fabricated responses.
''')
 manifest={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='release_manifest.json'}
 (out/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(exported_files=len(manifest),actual_posts=len(receipts),reported_cost_usd=summary['cost_usd'],output=str(out.resolve()))))
if __name__=='__main__':main()
