"""Pre-truth integration of the already specified repeat decision estimates.

No network, credentials, generator, truth or new decision rules. Checkpoints let
this run alongside collection; the orchestrator remains held until it completes.
"""
import argparse,json,os,time
from pathlib import Path
from datetime import datetime,timezone
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.sealing import digest,write_once
from sharecausal.final_study_v31.workers import Guard


def run(out,max_wait=3600):
 out=Path(out).resolve();root=Path(__file__).resolve().parents[1];os.umask(0o077);os.environ.pop('OPENROUTER_API_KEY',None)
 if (out/'prediction_seal.json').exists() or (out/'scoring_integrity.json').exists():raise ValueError('repeat_predictions_must_precede_seal_and_truth')
 design=json.loads((out/'commitment.json').read_text());inputs=[p for p in out.rglob('*') if p.is_file() and p.name not in ('truth.json','generation.json')]
 inputs += [root/name for name in design['source_sha256'] if Path(name).name not in ('authoring.py','certificates.py','scoring.py')]
 guard=Guard('predict',root,inputs=inputs,output=out,truth=out/'truth.json',generation=out/'generation.json').install()
 from sharecausal.final_study_v31.public import Card,MODELS,messages
 from sharecausal.final_study_v31.engine import analyze
 import pandas as pd
 for name,expected in design['source_sha256'].items():
  if Path(name).name in ('authoring.py','certificates.py','scoring.py'):continue
  if digest(root/name)!=expected:raise ValueError('frozen_engine_source_changed')
 cards={c.task_id:c for c in [Card.model_validate(v) for v in json.loads((out/'public_cards.json').read_text())]};rubric=json.loads((out/'public_rubric.json').read_text());samples=json.loads((out/'sample_manifest.json').read_text());conditions=[('base',1),('counterargument',0),('counterargument',1)]
 guard.inputs.add(Path(__file__).resolve())
 commitment=dict(kind='prespecified_repeat_decisions_same_frozen_engine',source_sha256=digest(Path(__file__)),engine_sha256=digest(root/'src/sharecausal/final_study_v31/engine.py'),public_tasks_sha256=digest(out/'public_cards.json'),sample_manifest_sha256=digest(out/'sample_manifest.json'),conditions=conditions,calibration_sha256=digest(out/'calibration.json'),new_provider_calls=0,rule_changes=False,created_utc=datetime.now(timezone.utc).isoformat())
 if not (out/'repeat_decision_commitment.json').exists():write_once(out/'repeat_decision_commitment.json',commitment)
 checkpoints=out/'repeat_decision_checkpoints';checkpoints.mkdir(exist_ok=True);done={p.stem for p in checkpoints.glob('*.json')};records={};seen=set();start=time.monotonic()
 while len(done)<len(cards):
  if time.monotonic()-start>max_wait:raise ValueError('repeat_collection_wait_limit_reached_no_truth_access')
  for file in (out/'receipts').glob('*.json'):
   if file in seen or file.name.endswith(('.raw.json','.inflight.json')):continue
   record=json.loads(file.read_text());seen.add(file)
   if record['variant']=='critic':continue
   h=record['task_id']
   if h not in cards or record['model'] not in MODELS:raise ValueError('unexpected_repeat_identity')
   if record['cost_usd'] is None:raise ValueError('unreconciled_receipt_not_complete')
   if record['prompt_sha256']!=sha(messages(cards[h],record['variant'],record['repetition'])):raise ValueError('repeat_prompt_binding_failure')
   raw=out/'receipts'/(record['request_key']+'.raw.json')
   if sha(json.loads(raw.read_text()))!=record['response_sha256']:raise ValueError('repeat_raw_binding_failure')
   records[h,record['model'],record['variant'],record['repetition']]=record
  progressed=False
  for h,card in cards.items():
   if h in done:continue
   if not all((h,m,v,r) in records for m in MODELS for v,r in conditions):continue
   batch=[]
   for variant,repetition in conditions:
    selected=[records[h,m,variant,repetition] for m in MODELS]
    for item in samples:
     if item['task_id']!=h:continue
     if digest(out/item['file'])!=item['sha256']:raise ValueError('repeat_observation_binding_failure')
     frame=pd.DataFrame(json.loads((out/item['file']).read_text()));prediction=analyze(card,rubric[h],frame,selected,None)
     prediction.update(sample_repetition=item['sample_repetition'],prompt_variant=variant,prompt_repetition=repetition,critic_scope='base_graph_only_compiler_veto_redundant',new_estimator_or_threshold=False);batch.append(prediction)
   write_once(checkpoints/(h+'.json'),batch);(checkpoints/(h+'.json')).chmod(0o400);done.add(h);progressed=True
   if len(done)%12==0 or len(done)==len(cards):print(json.dumps(dict(repeat_cards_completed=len(done),planned_cards=len(cards),truth_opened=False,new_provider_calls=0)),flush=True)
  if not progressed:time.sleep(5)
 batch=[]
 for h in sorted(cards):batch+=json.loads((checkpoints/(h+'.json')).read_text())
 if not (out/'repeat_condition_predictions.json').exists():write_once(out/'repeat_condition_predictions.json',batch)
 if len(batch)!=len(cards)*3*2:raise ValueError('incomplete_repeat_prediction_batch')
 write_once(out/'repeat_prediction_guard_receipt.json',dict(predictions=len(batch),cards=len(cards),frozen_engine_sha256=commitment['engine_sha256'],prediction_batch_sha256=digest(out/'repeat_condition_predictions.json'),completed_utc=datetime.now(timezone.utc).isoformat(),truth_readable=False,generator_readable=False,network_requests=0,credentials_read=False))
 print(json.dumps(dict(repeat_predictions=len(batch),status='complete_before_truth',new_provider_calls=0)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results-dir',required=True,type=Path);p.add_argument('--max-wait-seconds',type=int,default=3600);a=p.parse_args();run(a.results_dir,a.max_wait_seconds)
