"""Role-separated local workers. No truth input to providers or predictions."""
import json,os,sys
from pathlib import Path
from datetime import datetime,timezone
from sharecausal.pilot_v24.guards import ProcessGuard
from sharecausal.preflight_v23.sealing import digest,verify,write_once

class Guard(ProcessGuard):
 def install(self):
  if self.role=='collect':sys.addaudithook(self.audit);return self
  return super().install()
 def role_output_file(self,path):
  return bool(self.output and path.is_relative_to(self.output) and path not in (self.truth,self.generation) and path.name not in ('public_cards.json','public_rubric.json','families.json','commitment.json','request_plan.json','catalog.json'))
 def audit(self,event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and self.role in ('collect','predict'):
   name=Path(os.fsdecode(args[0])).name
   if any(name.startswith(x) for x in ('authoring.','certificates.','scoring.')):raise PermissionError('truth_source_forbidden')
  if event=='import' and self.role in ('collect','predict') and any(x in str(args[0]) for x in ('.authoring','.certificates','.scoring')):raise PermissionError('truth_import_forbidden')
  if event.startswith('socket.') and self.role=='collect':return
  return super().audit(event,args)

def main():
 role,out,split=sys.argv[1:];out=Path(out).resolve();root=Path(__file__).resolve().parents[3]
 inputs=[out/p for p in ('public_cards.json','public_rubric.json','families.json','commitment.json','request_plan.json','catalog.json','calibration.json','prediction_seal.json','prediction_seal.sha256') if (out/p).exists()]
 # Existing output reads are allowed, excluding truth and generation specifically.
 inputs += [p for p in out.rglob('*') if p.is_file() and p.name not in ('truth.json','generation.json')]
 if role=='generate':inputs.append(out/'generation.json')
 guard=Guard(role,root,inputs=inputs,output=out,truth=out/'truth.json',generation=out/'generation.json').install()
 from sharecausal.preflight.contracts import sha,canonical
 from .public import Card,MODELS
 cards=[Card.model_validate(c) for c in json.loads((out/'public_cards.json').read_text())]
 assignments=json.loads((out/'families.json').read_text());ids={r['task_id'] for r in assignments if split=='all' or r['split']==split.removesuffix('_critic')}
 cards=[c for c in cards if c.task_id in ids]
 committed=json.loads((out/'commitment.json').read_text())
 guard.inputs.update(root/name for name in committed['source_sha256'])
 for name,expected in committed['source_sha256'].items():
  if role in ('collect','predict') and Path(name).name in ('authoring.py','certificates.py','scoring.py'):continue
  if digest(root/name)!=expected:raise ValueError('committed_source_changed')
 for name in ('public_cards.json','public_rubric.json','families.json'):
  if digest(out/name)!=committed['input_sha256'][name]:raise ValueError('committed_public_inputs_changed')
 if role=='generate':
  import numpy as np
  from sharecausal.preflight_v23.certificates import BinaryDAG,law
  plans=json.loads((out/'generation.json').read_text());samples=[]
  for card in cards:
   p=plans[card.task_id];scm=BinaryDAG.model_validate(dict(nodes=p['nodes']));names=sorted(v for v,x in card.variables.items() if x['observed'])
   joint=law(scm,names);states=sorted(joint);cum=np.cumsum([float(joint[x]) for x in states]);cum[-1]=1
   for repetition in range(2):
    rng=np.random.default_rng(p['seed']+repetition*1000003);draws=np.searchsorted(cum,rng.random(p['n']),side='right')
    values={v:np.array([states[i][j] for i in draws],int) for j,v in enumerate(names)}
    if p['recording']=='complete':observed=np.ones(p['n'],int)
    elif p['recording']=='mar':observed=(rng.random(p['n'])<np.where(values['X']==1,.5,.9)).astype(int)
    elif p['recording']=='survival':observed=values['C'].copy()
    else:observed=(rng.random(p['n'])<np.where(values['C']==1,.4,.95)).astype(int)
    payload={v:[int(x) if (v!='Y' or observed[j]) else None for j,x in enumerate(xs)] for v,xs in values.items()};payload['observed']=observed.tolist()
    filename='observed-'+sha([card.task_id,repetition])+'.json'
    if not (out/filename).exists():write_once(out/filename,payload)
    samples.append(dict(task_id=card.task_id,sample_repetition=repetition,file=filename,sha256=digest(out/filename),n=p['n']))
  write_once(out/'sample_manifest.json',samples)
  write_once(out/'generator_receipt.json',dict(truth_opened=False,hidden_nodes_exported=False,samples=len(samples),utc=datetime.now(timezone.utc).isoformat()))
 elif role=='collect':
  from .transport import collect
  collect(cards,out,json.loads((out/'request_plan.json').read_text()),os.environ.get('OPENROUTER_API_KEY'), 'critic' if split.endswith('_critic') else 'primary')
 elif role=='predict':
  import pandas as pd
  from .transport import load_records
  from .engine import analyze,METHODS
  records=load_records(out/'receipts')
  for record in records:
   raw=out/'receipts'/(record['request_key']+'.raw.json')
   if sha(json.loads(raw.read_text()))!=record['response_sha256']:raise ValueError('raw_provider_binding_failure')
  rubrics=json.loads((out/'public_rubric.json').read_text());samples=json.loads((out/'sample_manifest.json').read_text());batch=[];stability=[]
  for card in cards:
   source=[next((r for r in records if r['task_id']==card.task_id and r['model']==m and r['variant']=='base' and r['repetition']==0),dict(model=m,status='not_attempted',answer=None)) for m in MODELS]
   critic=next((r for r in records if r['task_id']==card.task_id and r['variant']=='critic'),None)
   for sample in samples:
    if sample['task_id']!=card.task_id:continue
    if digest(out/sample['file'])!=sample['sha256']:raise ValueError('sample_changed')
    frame=pd.DataFrame(json.loads((out/sample['file']).read_text()))
    pred=analyze(card,rubrics[card.task_id],frame,source,critic);pred['sample_repetition']=sample['sample_repetition'];batch.append(pred)
    if sample['sample_repetition']==0:
     # Repeats' graph and evidence stability needs no further estimator fits.
     from .engine import edge_set
     for m in MODELS:
      rs=[r for r in records if r['task_id']==card.task_id and r['model']==m and r['variant']!='critic']
      stability.append(dict(task_id=card.task_id,model=m,responses=[dict(variant=r['variant'],repetition=r['repetition'],status=r['status'],graph_sets=sorted([edge_set(g) for g in (r['answer'] or {}).get('graphs',[])]) if r['status']=='valid' else None) for r in rs]))
  write_once(out/('predictions_'+split+'.json'),batch);write_once(out/('stability_'+split+'.json'),stability)
  write_once(out/('prediction_guard_'+split+'.json'),dict(truth_access=False,generator_access=False,public_only_graph_decisions=True,source_code_verified=True,utc=datetime.now(timezone.utc).isoformat()))
 elif role=='score':
  seal=json.loads((out/'prediction_seal.json').read_text());expected=(out/'prediction_seal.sha256').read_text().strip();verify(out,seal,'prediction_seal.json',expected)
  guard.truth_ready=True
  stamp=datetime.now(timezone.utc).isoformat()
  authored=json.loads((out/'authoring_commitment.json').read_text())
  if digest(out/'truth.json')!=authored['truth.json']:raise ValueError('authoring_truth_hash_changed')
  from .scoring import score
  result=score(out,cards,json.loads((out/'truth.json').read_text()))
  write_once(out/'scoring_integrity.json',dict(seal_verified_before_truth_open=True,truth_open_authorized_at=stamp,prediction_seal_sha256=expected,truth_sha256=digest(out/'truth.json'),prediction_batch_sha256=digest(out/'prediction_batch.json'),prediction_files_modified=False,single_team_not_external_registration=True))
  verify(out,seal,'prediction_seal.json',expected)
 else:raise ValueError('unknown_worker')

if __name__=='__main__':main()
