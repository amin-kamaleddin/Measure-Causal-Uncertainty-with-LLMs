"""Focused final integration tests; fabricated transport is test-only."""
import json
from pathlib import Path
import httpx
import pytest
from sharecausal.preflight.contracts import canonical
from sharecausal.preflight_v23.packets import PublicTask,task_hash
from sharecausal.preflight_v23.admission import no_llm
from sharecausal.synthetic_final_v1 import runner
from sharecausal.synthetic_final_v1.authoring import build
from sharecausal.synthetic_final_v1.public import MODELS,messages,settings
from sharecausal.synthetic_final_v1.transport import collect
from tests.test_synthetic_live_v1 import catalog

@pytest.fixture
def authored(tmp_path):
 out=tmp_path/'final'; counts=build(out)
 tasks={task_hash(t):t for t in map(PublicTask.model_validate,json.loads((out/'public_tasks.json').read_text()))}
 (out/'design_commitment.json').write_text(json.dumps(dict(counterargument_task_hashes=sorted(tasks)[:8])))
 return out,tasks,counts

def client_for(tasks,unknown=False,evidence_veto=False):
 calls=[]
 lookup={canonical(messages(t,v)):t for t in tasks.values() for v in ('base','counterargument')}
 def handle(request):
  value=json.loads(request.content);calls.append(value);task=lookup[canonical(value['messages'])]
  usage=dict(cost=.0001,prompt_tokens=100,completion_tokens=200)
  if unknown:usage.pop('cost')
  proposals=[p.model_dump() for p in no_llm(task)]
  if evidence_veto:
   for proposal in proposals:
    if proposal['graph']['edges']:proposal['graph']['edges'][0]['evidence_id']='ev-'+'f'*64
  return httpx.Response(200,json=dict(id='gen-test-'+str(len(calls)),model=value['model'],provider='invented-test-provider',usage=usage,
   choices=[dict(finish_reason='stop',message=dict(content=canonical(dict(proposals=[p.model_dump() for p in no_llm(task)])).decode()))]))
 return httpx.Client(transport=httpx.MockTransport(handle)),calls

def test_cards_certificates_and_public_boundary(authored):
 out,tasks,counts=authored
 assert counts==dict(public_cards=24,world_probes=26,declared_families=12)
 from sharecausal.preflight_v23.certificates import Truth,certify
 worlds=list(map(Truth.model_validate,json.loads((out/'scoring_truth.json').read_text())))
 cs=[certify(w,tasks[w.public_task_hash],worlds) for w in worlds]
 assert sum(c['status']=='provably_nonidentified' for c in cs)==4
 assert sum(c['supported_structural_zero'] for c in cs)==2
 for t in tasks.values():
  assert any(e.relation=='ambiguous' for e in t.card.evidence)
  for v in ('base','counterargument'):
   b=canonical(messages(t,v))
   for marker in (b'mergeid',b'population_effect',b'family_id',b'world_id',b'probability_one',b'/Users/',b'OPENROUTER_API_KEY'):
    assert marker not in b
 for m in catalog():
  s=settings(m);assert s['max_tokens']==(8192 if m['id']==MODELS[3] else 4096)
  assert s['provider']['allow_fallbacks'] is False
  assert ('temperature' in s)==(m['id']!=MODELS[1])

def test_128_durable_receipts_resume_and_budget(authored):
 out,tasks,_=authored;client,calls=client_for(tasks)
 with pytest.raises(ValueError,match='batch_reservation'):collect(tasks,out,.001,'DUMMY',client=client,catalog=catalog())
 assert not calls
 # Cap is a committed field: remove test-only plan after rejected pre-network reservation.
 (out/'request_plan.json').unlink()
 records,plan=collect(tasks,out,25.,'DUMMY',client=client,catalog=catalog(),progress=lambda *a,**k:None)
 assert len(records)==len(calls)==len(plan)==128
 assert sum(r['variant']=='counterargument' for r in records)==32
 collect(tasks,out,25.,'DUMMY',client=client,progress=lambda *a,**k:None)
 assert len(calls)==128
 assert json.loads((out/'collection_summary.json').read_text())['cached']==128
 path=next(p for p in (out/'provider_responses_private').glob('*.json') if not p.name.endswith(('.raw.json','.inflight.json')))
 path.unlink()
 collect(tasks,out,25.,'DUMMY',client=client,progress=lambda *a,**k:None)
 assert len(calls)==128 and json.loads((out/'collection_summary.json').read_text())['stopped']

def test_unknown_cost_stops_new_spending_bounded_inflight(authored):
 out,tasks,_=authored;client,calls=client_for(tasks,True)
 records,_=collect(tasks,out,25.,'DUMMY',client=client,catalog=catalog(),progress=lambda *a,**k:None)
 assert 1<=len(calls)<=4 and all(r['status']=='cost_unknown' for r in records)

@pytest.mark.parametrize('veto',[False,True])
def test_guarded_full_pipeline(tmp_path,monkeypatch,veto):
 out=tmp_path/'integration';runner.prepare(out,25.)
 tasks={task_hash(t):t for t in map(PublicTask.model_validate,json.loads((out/'public_tasks.json').read_text()))}
 client,calls=client_for(tasks,evidence_veto=veto);real=runner.worker
 def worker(role,output,cap,digest_value=None):
  if role=='collect':collect(tasks,output,cap,'DUMMY',client=client,catalog=catalog(),progress=lambda *a,**k:None)
  else:real(role,output,cap,digest_value)
 monkeypatch.setattr(runner,'worker',worker);monkeypatch.setenv('OPENROUTER_API_KEY','DUMMY')
 assert runner.run(out,25.,True)==0
 assert len(calls)==128 and len(json.loads((out/'prediction_batch.json').read_text()))==24
 assert len(json.loads((out/'stability_prediction_batch.json').read_text()))==8
 assert json.loads((out/'scorer_receipt.json').read_text())['seal_verified_before_truth_open']
 assert runner.run(out,25.,True)==0 and len(calls)==128
