"""Post-score derived accounting and diagnostics. Never changes predictions."""
import argparse,json,csv,shutil,os,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.sealing import digest,verify
from sharecausal.final_study_v31.public import MODELS,Card
from sharecausal.final_study_v31.transport import load_records
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--package',required=True,type=Path);a=p.parse_args();run=a.run.resolve();root=a.package.resolve();tables=root/'tables';tables.mkdir(parents=True,exist_ok=True);os.umask(0o077)
if not (run/'scoring_integrity.json').exists():raise ValueError('post_seal_scoring_required')
seal=json.loads((run/'prediction_seal.json').read_text());verify(run,seal,'prediction_seal.json',(run/'prediction_seal.sha256').read_text().strip())
provenance=json.loads((run/'routing_repair_provenance.json').read_text());original=Path(provenance['original_run']);unique={}
for directory in (original/'receipts',run/'receipts'):
 for r in load_records(directory):
  h=r['request_key']
  if h in unique and sha(unique[h])!=sha(r):raise ValueError('duplicate_request_identity_has_different_receipt')
  unique[h]=r
if any(r['cost_usd'] is None for r in unique.values()):raise ValueError('unknown_cost_not_reportable_as_zero')
def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 path.chmod(0o600)
provider=[]
for m in MODELS:
 rs=[r for r in unique.values() if r['model']==m];counts=Counter(r['status'] for r in rs)
 wrong_slug=sum(r['model']==MODELS[3] and r['settings']['provider']['only']==['openinference'] for r in rs)
 provider.append(dict(model=m,planned_primary_slots=480,attempted=len(rs),valid_graph_or_critic=counts['valid'],explicit_abstention=counts['explicit_abstention'],technical_failures=sum(v for k,v in counts.items() if k not in ('valid','explicit_abstention')),initial_investigator_routing_failures=wrong_slug,null_content=counts['null_content'],schema_failure=counts['schema_failure'],refusal=counts['refusal'],http_error=counts['http_error'],unknown_costs=0,automatic_retries=0,cost_usd=sum(r['cost_usd'] for r in rs),prompt_tokens=sum((r['usage'].get('prompt_tokens') or 0) for r in rs),completion_tokens=sum((r['usage'].get('completion_tokens') or 0) for r in rs),providers=';'.join(sorted({r['provider'] for r in rs if r['provider']}))))
 for stage in ('primary','critic'):
  stage_records=[r for r in rs if (r['variant']=='critic')==(stage=='critic')]
  stage_counts=Counter(r['status'] for r in stage_records)
  provider[-1].update({stage+'_attempted':len(stage_records),stage+'_typed_valid':stage_counts['valid'],stage+'_explicit_abstention':stage_counts['explicit_abstention'],stage+'_technical_failures':sum(v for k,v in stage_counts.items() if k not in ('valid','explicit_abstention'))})
 provider[-1]['primary_cached_in_repaired_schedule']=480 if m!=MODELS[3] else 0
writecsv(tables/'actual_unique_provider_requests.csv',provider)
writecsv(tables/'corrected_schedule_provider_requests.csv',list(csv.DictReader((run/'provider_comparison.csv').open())))
writecsv(tables/'all_unique_request_accounting_PRIVATE.csv',[dict(request_key=r['request_key'],task_id=r['task_id'],model=r['model'],variant=r['variant'],repetition=r['repetition'],status=r['status'],cost_usd=r['cost_usd'],provider=r['provider'],provider_response_id=r['provider_response_id']) for r in unique.values()])
# Materialize the promised per-edge evidence/model/prompt tuple by deterministic
# joins of sealed predictions and sealed base-response receipts.
records=load_records(run/'receipts');index={(r['task_id'],r['model']):r for r in records if r['variant']=='base' and r['repetition']==0};pred=json.loads((run/'prediction_batch.json').read_text());edges=[];dispositions=[];rubric=json.loads((run/'public_rubric.json').read_text())
for p0 in pred:
 if p0['sample_repetition']!=0:continue
 h=p0['task_id']
 for g in p0['edge_audits']:
  model=MODELS[g['source_index']];receipt=index[h,model]
  for edge in g['edges']:edges.append(dict(task_id=h,model_id=model,prompt_hash=receipt['prompt_sha256'],graph_hash=g['graph_hash'],edge='->'.join(edge['edge']),polarity='present',evidence_id=edge['evidence_id'],evidence_span_hash=edge['evidence_span_hash'],support_state=edge['support_state'],temporal_status=edge['temporal_status'],role=edge['role'],uncertainty_reason='author_rubric_credible_alternative' if edge['support_state']=='credible_ambiguous' else ';'.join(g['reasons']),edge_admission=edge['admission_decision'],whole_graph_admission=g['admitted']))
 for method,r in p0['methods'].items():
  if r['decision'] in ('positive','negative'):category='directional_assertion'
  elif method=='descriptive_association':category='association_only_control'
  elif method in ('google','openai','llama','deepseek'):
   model=MODELS[('google','openai','llama','deepseek').index(method)];response=index[h,model]
   if response['status']=='explicit_abstention':category='explicit_source_epistemic_abstention'
   elif response['status']!='valid':category='technical_response_failure_not_scientific_abstention'
   elif not any(g['admitted'] for g in p0['edge_audits'] if g['source_index']==MODELS.index(model)):category='invalid_graph_no_scientific_abstention'
   elif r['reason']=='graph_structural_zero' and rubric[h]['flags']['structural_zero']:category='public_structural_zero'
   else:category='framework_withheld_direction'
  elif r['reason']=='graph_structural_zero' and rubric[h]['flags']['structural_zero']:category='public_structural_zero'
  elif r['reason']=='estimator_or_overlap_failure':category='estimator_or_overlap_failure'
  else:category='framework_withheld_direction'
  dispositions.append(dict(task_id=h,method=method,category=category,reason=r['reason']))
writecsv(tables/'evidence_edge_provenance_PRIVATE.csv',edges)
assignment={r['task_id']:r for r in json.loads((run/'families.json').read_text())};counts=Counter((r['method'],r['category']) for r in dispositions if assignment[r['task_id']]['split']=='evaluation')
writecsv(tables/'disjoint_decision_dispositions.csv',[dict(method=m,category=c,tasks=n) for (m,c),n in sorted(counts.items())])
# Paired prompt/repetition equality: only typed nonabstaining graph sets form a
# graph denominator; equal technical failures never count as agreement successes.
stability=[]
for model in MODELS:
 for variant,repetition in (('base',1),('counterargument',0),('counterargument',1)):
  planned=eligible=equal=0
  for h,entry in assignment.items():
   if entry['split']!='evaluation':continue
   planned+=1;first=index[h,model];second=next(r for r in records if r['task_id']==h and r['model']==model and r['variant']==variant and r['repetition']==repetition)
   if first['status']==second['status']=='valid':
    eligible+=1
    def normalized(r):return sorted(sorted((e['source'],e['target']) for e in g['edges']) for g in r['answer']['graphs'])
    equal+=normalized(first)==normalized(second)
  stability.append(dict(model=model,comparison=f'base0_vs_{variant}{repetition}',planned_pairs=planned,typed_graph_pairs=eligible,equal_graph_sets=equal,equality_fraction=equal/eligible if eligible else None,independent_units='14families_not84independent_pairs'))
writecsv(tables/'prompt_repetition_stability.csv',stability)
# Compare causal decisions already frozen for every alternate condition. This
# descriptive analysis creates no new estimates, thresholds or scientific rules.
member='repeat_condition_predictions.json'
if member not in seal:raise ValueError('repeat_decisions_not_sealed_before_truth')
alternates=json.loads((run/member).read_text());primary={p['task_id']:p for p in pred if p['sample_repetition']==0};calibration=json.loads((run/'calibration.json').read_text());decision_stability=[]
def frozen_direction(result,method):
 return result['decision'] if result['decision'] in ('positive','negative') and result['margin']>=calibration[method]['margin_threshold'] else 'withheld'
for variant,repetition in (('base',1),('counterargument',0),('counterargument',1)):
 selected=[p for p in alternates if p['sample_repetition']==0 and p['prompt_variant']==variant and p['prompt_repetition']==repetition and assignment[p['task_id']]['split']=='evaluation']
 if len(selected)!=84:raise ValueError('incomplete_sealed_repeat_decisions')
 for method in primary[selected[0]['task_id']]['methods']:
  pairs=[(frozen_direction(primary[p['task_id']]['methods'][method],method),frozen_direction(p['methods'][method],method)) for p in selected]
  both=sum(a in ('positive','negative') and b in ('positive','negative') for a,b in pairs)
  flips=sum(a in ('positive','negative') and b in ('positive','negative') and a!=b for a,b in pairs)
  decision_stability.append(dict(method=method,comparison=f'base0_vs_{variant}{repetition}',task_pairs=len(pairs),same_decision=sum(a==b for a,b in pairs),both_directional=both,opposite_directions=flips,base_assertions=sum(a!='withheld' for a,b in pairs),alternate_assertions=sum(b!='withheld' for a,b in pairs),family_units=14,scope='descriptive_sealed_predictions_no_postscore_rule_change'))
writecsv(tables/'causal_decision_repetition_stability.csv',decision_stability)
# Source-blinded critic fidelity against the precommitted author rubric. The
# downstream veto is redundant with the compiler; measure that limitation openly.
from sharecausal.final_study_v31.engine import compile_graph
card_map={c.task_id:c for c in [Card.model_validate(x) for x in json.loads((run/'public_cards.json').read_text())]}
critic_counts=Counter()
for r in records:
 if r['variant']!='critic':continue
 h=r['task_id'];critic_counts['planned_tasks']+=1
 if r['status']!='valid':critic_counts['technical_failures']+=1;continue
 proposals={sha(g):g for rr in records if rr['task_id']==h and rr['variant']=='base' and rr['repetition']==0 and rr['status']=='valid' for g in rr['answer']['graphs']}
 docs={d.evidence_id:d.text for d in card_map[h].documents}
 seen=set()
 for review in r['answer']['reviews']:
  identity=review['graph_hash']
  if identity not in proposals or identity in seen:critic_counts['invalid_or_duplicate_review_keys']+=1;continue
  seen.add(identity);good=compile_graph(card_map[h],rubric[h],proposals[identity])['admitted'];valid_quote=review['evidence_id'] in docs and bool(review['quote']) and review['quote'] in docs[review['evidence_id']]
  critic_counts['reviewed_graphs']+=1;critic_counts['exact_citation_reviews']+=valid_quote
  critic_counts['compiler_admitted_graphs']+=good;critic_counts['compiler_rejected_graphs']+=not good
  critic_counts['reject_admitted_graph']+=good and review['verdict']=='reject'
  critic_counts['reject_rejected_graph']+=(not good) and review['verdict']=='reject'
  critic_counts['retain_admitted_graph']+=good and review['verdict']=='retain'
  critic_counts['uncertain_verdict']+=review['verdict']=='uncertain'
 critic_counts['unreviewed_graphs']+=len(proposals)-len(seen)
writecsv(tables/'critic_fidelity.csv',[dict(**critic_counts,additional_downstream_vetoes_possible_under_rule=0,external_independent_human_review=False)])

# Score original already-sealed decisions using the byte-identical scientific
# scorer. The derived serialization below changes no graph/effect/decision.
legacy=root/'original_pipeline_diagnostic';legacy.mkdir(exist_ok=True)
if not (legacy/'summary.json').exists():
 for name in ('public_cards.json','public_rubric.json','families.json','calibration.json','sample_manifest.json'):
  shutil.copyfile(original/name,legacy/name)
 for f in original.glob('observed-*.json'):shutil.copyfile(f,legacy/f.name)
 (legacy/'receipts').mkdir(exist_ok=True)
 for f in (original/'receipts').glob('*.json'):shutil.copyfile(f,legacy/'receipts'/f.name)
 legacy_predictions=[]
 for split in ('development','evaluation'):
  member='original_attempt_predictions/predictions_'+split+'.json'
  if member not in seal:raise ValueError('original_predictions_not_in_pretruth_seal')
  legacy_predictions+=json.loads((run/member).read_text())
 (legacy/'prediction_batch.json').write_bytes(canonical(legacy_predictions))
 from sharecausal.final_study_v3.scoring import score
 cards=[Card.model_validate(c) for c in json.loads((run/'public_cards.json').read_text())]
 # The original scorer uses its local Card class only for baseline regex/attributes;
 # no type-sensitive payload export occurs in these post-seal certificates.
 score(legacy,cards,json.loads((run/'truth.json').read_text()))
 (legacy/'derived_seal_provenance.json').write_bytes(canonical(dict(parent_prediction_seal_sha256=(run/'prediction_seal.sha256').read_text().strip(),original_prediction_members={n:v for n,v in seal.items() if n.startswith('original_attempt_predictions/')},new_predictions_created=False,serialization_of_presealed_decisions_only=True,original_truth_scored_after_complete_repaired_batch=True)))
summary=dict(actual_unique_requests=len(unique),reported_generation_cost_usd=sum(r['cost_usd'] for r in unique.values()),original_attempts=provenance['original_requests'],canonical_repaired_schedule=len(records),reused_paid_primary_receipts=1440,initial_wrong_provider_slug_rejections=480,corrected_deepseek_attempts=480,source_scientific_modules_identical=all(digest(Path('src/sharecausal/final_study_v3')/n)==digest(Path('src/sharecausal/final_study_v31')/n) for n in ('authoring.py','engine.py','scoring.py')),unknown_costs=0,original_failures_preserved=True,predictions_unchanged=True,full_seal_verified=True,provider_invoice_independently_verified=False)
(root/'actual_execution_accounting.json').write_text(json.dumps(summary,indent=2))
verify(run,seal,'prediction_seal.json',(run/'prediction_seal.sha256').read_text().strip());print(json.dumps(summary))
