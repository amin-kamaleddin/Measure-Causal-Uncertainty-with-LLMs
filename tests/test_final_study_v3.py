import json
import subprocess
import sys
from pathlib import Path
import pytest
import httpx
from sharecausal.final_study_v3.authoring import build
from sharecausal.final_study_v3.public import Card,MODELS,messages,decode
from sharecausal.final_study_v3.engine import baseline,compile_graph
from sharecausal.final_study_v3.runner import leakage_audit
from sharecausal.preflight_v23.certificates import BinaryDAG,effect,law

@pytest.fixture(scope='module')
def authored(tmp_path_factory):
 out=tmp_path_factory.mktemp('new-study')/'run';build(out)
 return out

def test_new_family_split_public_boundary_and_truth_consistency(authored):
 cards=[Card.model_validate(c) for c in json.loads((authored/'public_cards.json').read_text())]
 assignments=json.loads((authored/'families.json').read_text());rubric=json.loads((authored/'public_rubric.json').read_text())
 assert len(cards)==120
 assert len({r['family_id'] for r in assignments if r['split']=='evaluation'})==14
 audit=leakage_audit(cards,assignments);assert not audit['cross_split_near_duplicates']
 for card in cards:
  payload=json.dumps(messages(card,'base',0)).lower()
  assert all(s not in payload for s in ('family_id','generator_seed','effect_exact','world_id','mergeid'))
  graphs=baseline(card)
  assert graphs
  if rubric[card.task_id]['flags']['target_valid']:
   assert any(compile_graph(card,rubric[card.task_id],g)['admitted'] for g in graphs)
 truths=json.loads((authored/'truth.json').read_text())
 for w in truths:
  scm=BinaryDAG.model_validate(dict(nodes=w['nodes']));assert str(effect(scm))==w['effect_exact']
 for h in {w['task_id'] for w in truths if w['pair']}:
  pair=[BinaryDAG.model_validate(dict(nodes=w['nodes'])) for w in truths if w['task_id']==h]
  card=next(c for c in cards if c.task_id==h);observed=sorted(v for v,x in card.variables.items() if x['observed'])
  assert law(pair[0],observed)==law(pair[1],observed)
  assert effect(pair[0])!=effect(pair[1])

def test_compiler_exact_quote_and_whole_graph_negative_controls(authored):
 card=Card.model_validate(json.loads((authored/'public_cards.json').read_text())[0]);rubric=json.loads((authored/'public_rubric.json').read_text())[card.task_id]
 graph=baseline(card)[0];assert compile_graph(card,rubric,graph)['admitted']
 graph['edges'][0]['quote']='Invented quote'
 assert not compile_graph(card,rubric,graph)['admitted']
 graph=baseline(card)[0];graph['edges'].pop()
 assert 'missing_entailed_edge' in compile_graph(card,rubric,graph)['reasons']

def test_operational_failures_are_not_abstentions():
 assert decode(dict(choices=[dict(message=dict(content=None))]))[0]=='null_content'
 assert decode(dict(choices=[dict(message=dict(content='not json'))]))[0]=='schema_failure'
 assert decode(dict(choices=[dict(message=dict(refusal='cannot'))]))[0]=='refusal'
 status,value=decode(dict(choices=[dict(message=dict(content=json.dumps(dict(status='abstain',graphs=[],reason='insufficient evidence'))))]))
 assert status=='explicit_abstention'

def test_role_guard_denies_truth_generator_key_and_share(tmp_path):
 code='''import sys,os
from pathlib import Path
from sharecausal.final_study_v3.workers import Guard
out=Path(sys.argv[1]); root=Path.cwd()
g=Guard('predict',root,output=out,truth=out/'truth.json',generation=out/'generation.json').install()
for action in [lambda:(out/'truth.json').read_text(),lambda:(out/'generation.json').read_text(),lambda:os.environ.get('OPENROUTER_API_KEY'),lambda:Path('/tmp/respondents.dta').read_bytes()]:
 try:action()
 except PermissionError:continue
 raise AssertionError('guard bypass')
'''
 result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True)
 assert result.returncode==0,result.stderr

def catalog_fixture():
 return [dict(id=m,supported_parameters=['response_format','temperature'],pricing=dict(prompt='0.0000001',completion='0.0000002')) for m in MODELS]

def test_budget_reservation_and_no_duplicate_post(authored,tmp_path):
 from sharecausal.final_study_v3.transport import plan,collect
 cards=[Card.model_validate(c) for c in json.loads((authored/'public_cards.json').read_text())[:1]]
 catalog=catalog_fixture();planned=plan(cards,catalog,100)
 with pytest.raises(ValueError,match='reservation'):plan(cards,catalog,.000001)
 (tmp_path/'catalog.json').write_text(json.dumps(catalog))
 attempts=[]
 def respond(request):
  body=json.loads(request.content);attempts.append(body)
  return httpx.Response(200,json=dict(id='test-generation-'+str(len(attempts)),model=body['model'],provider=dict(zip(MODELS,('Google AI Studio','OpenAI','DigitalOcean','OpenInference')))[body['model']],usage=dict(cost=.00001,prompt_tokens=1,completion_tokens=1),choices=[dict(message=dict(content=json.dumps(dict(status='abstain',graphs=[],reason='unit-test only'))))]))
 with httpx.Client(transport=httpx.MockTransport(respond)) as client:
  first=collect(cards,tmp_path,planned,'test-not-a-real-key','primary',client=client)
  second=collect(cards,tmp_path,planned,'test-not-a-real-key','primary',client=client)
 assert first['new_posts']==16 and second['new_posts']==0 and second['cached']==16
 assert len(attempts)==16
 receipt=next(p for p in (tmp_path/'receipts').glob('*.raw.json'));receipt.write_text('{}')
 with httpx.Client(transport=httpx.MockTransport(respond)) as client:
  with pytest.raises(ValueError,match='binding'):collect(cards,tmp_path,planned,'test-not-a-real-key','primary',client=client)
 assert len(attempts)==16

def test_unknown_inflight_is_never_resent(authored,tmp_path):
 from sharecausal.final_study_v3.transport import plan,collect
 cards=[Card.model_validate(c) for c in json.loads((authored/'public_cards.json').read_text())[:1]];catalog=catalog_fixture();planned=plan(cards,catalog,100)
 (tmp_path/'catalog.json').write_text(json.dumps(catalog));(tmp_path/'receipts').mkdir();(tmp_path/'receipts'/'uncertain.inflight.json').write_text('{}')
 with httpx.Client(transport=httpx.MockTransport(lambda _:pytest.fail('duplicate paid call'))) as client:
  with pytest.raises(ValueError,match='unresolved_paid'):collect(cards,tmp_path,planned,'test-not-a-real-key','primary',client=client)

def test_guarded_prediction_and_sealed_scoring_independent_toy(tmp_path):
 """Not final-study cards: an independent tiny toy for process/serialization QA."""
 from sharecausal.preflight.contracts import sha,canonical
 from sharecausal.preflight_v23.sealing import digest,seal
 from sharecausal.final_study_v3.engine import METHODS
 from sharecausal.final_study_v3.workers import Guard
 import numpy as np
 docs=[dict(evidence_id='e-'+sha('A cause Y'),text='The design manual records: A directly changes Y.')]
 base=dict(title='Independent unit-test lottery',variables={'A':dict(time=1,observed=True,description='toy setting'),'Y':dict(time=3,observed=True,description='toy response')},query='Unit-test total contrast',documents=docs)
 card=dict(task_id=sha(base),**base);h=card['task_id'];Card.model_validate(card)
 flags=dict(exhaustive=True,randomized=True,complete=True,structural_zero=False,target_valid=True,positivity=True)
 rubric={h:dict(flags=flags,annotations=[dict(edge=['A','Y'],state='entailed',evidence_id=docs[0]['evidence_id'],quote=docs[0]['text'])])}
 nodes=[dict(name='A',parents=[],probability_one=['1/2']),dict(name='Y',parents=['A'],probability_one=['1/8','7/8'])]
 values={'public_cards.json':[card],'public_rubric.json':rubric,'families.json':[dict(task_id=h,family_id='independent-toy-evaluation',split='evaluation')],'truth.json':[dict(task_id=h,world='toy',nodes=nodes,effect_exact='3/4',pair=False,target_valid=True)],'generation.json':{h:dict(nodes=nodes,recording='complete',n=300,seed=1117)},'catalog.json':catalog_fixture(),'request_plan.json':{},'calibration.json':{m:dict(margin_threshold=0.) for m in METHODS}}
 base2=dict(base,title='Independent development unit-test lottery');h2=sha(base2)
 values['public_cards.json'].append(dict(task_id=h2,**base2))
 values['public_rubric.json'][h2]=rubric[h]
 values['families.json'].append(dict(task_id=h2,family_id='independent-toy-development',split='development'))
 values['truth.json'].append(dict(values['truth.json'][0],task_id=h2,world='toy-dev'))
 values['generation.json'][h2]=dict(nodes=nodes,recording='complete',n=300,seed=2229)
 for name,value in values.items():(tmp_path/name).write_bytes(canonical(value))
 (tmp_path/'authoring_commitment.json').write_bytes(canonical({'truth.json':digest(tmp_path/'truth.json')}))
 root=Path(__file__).resolve().parents[1]
 commitment=dict(source_sha256={'scripts/final_study_v3.py':digest(root/'scripts/final_study_v3.py'),'docs/final_study_v3_locked_plan.md':digest(root/'docs/final_study_v3_locked_plan.md')},input_sha256={n:digest(tmp_path/n) for n in ('public_cards.json','public_rubric.json','families.json')})
 (tmp_path/'commitment.json').write_bytes(canonical(commitment));(tmp_path/'receipts').mkdir()
 def worker(role,split):
  r=subprocess.run([sys.executable,'-m','sharecausal.final_study_v3.workers',role,str(tmp_path),split],capture_output=True,text=True)
  assert r.returncode==0,r.stderr
 worker('generate','all');worker('predict','all')
 pred=json.loads((tmp_path/'predictions_all.json').read_text());assert len(pred)==4
 (tmp_path/'prediction_batch.json').write_bytes(canonical(pred))
 # No development observations are scored; calibrate using an independent copy
 # of the toy predictions solely to exercise scorer serialization.
 pred2=[dict(p,task_id=h) for p in pred]
 names=[p.name for p in tmp_path.iterdir() if p.is_file() and p.name not in ('truth.json','generation.json')]
 _,digest_value=seal(tmp_path,names,'prediction_seal.json');(tmp_path/'prediction_seal.sha256').write_text(digest_value)
 worker('score','all')
 integrity=json.loads((tmp_path/'scoring_integrity.json').read_text());assert integrity['seal_verified_before_truth_open']
 assert (tmp_path/'method_comparison.csv').exists()
