"""Step 7 tests use invented provider responses, never real calls/credentials."""
import json
import os
from pathlib import Path
import subprocess
import sys
import httpx
import pytest
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.packets import PublicTask,task_hash
from sharecausal.preflight_v23.admission import no_llm
from sharecausal.synthetic_live_v1.authoring import build
from sharecausal.synthetic_live_v1.public import MODELS,messages,settings,request_key
from sharecausal.synthetic_live_v1.transport import collect,validated,proposals
from sharecausal.synthetic_live_v1 import runner


def catalog():
    return [dict(id=m,pricing=dict(prompt='0.0000001',completion='0.0000005'),
        supported_parameters=['response_format','reasoning']+([] if m==MODELS[1] else ['temperature'])) for m in MODELS]


@pytest.fixture
def authored(tmp_path):
    out=tmp_path/'pilot'; counts=build(out)
    tasks={task_hash(t):t for t in [PublicTask.model_validate(x) for x in json.loads((out/'public_tasks.json').read_text())]}
    return out,tasks,counts


def client_for(tasks,mode='valid'):
    calls=[]
    def handle(request):
        assert request.method=='POST'; value=json.loads(request.content);calls.append(value)
        task=next(t for t in tasks.values() if messages(t)==value['messages'])
        body=dict(id='gen-test-'+str(len(calls)),model=value['model'],provider='invented-test-provider',
            usage=dict(cost=.0001,prompt_tokens=100,completion_tokens=200,total_tokens=300),
            choices=[dict(finish_reason='stop',message=dict(content=canonical(dict(proposals=[p.model_dump() for p in no_llm(task)])).decode()))])
        if mode=='null':body['choices'][0]['message']['content']=None
        if mode=='bad_model':body['model']='unapproved/substitute'
        if mode=='unknown_cost':body['usage'].pop('cost')
        if mode=='overrun':body['usage']['cost']=24.99
        return httpx.Response(200,json=body)
    return httpx.Client(transport=httpx.MockTransport(handle)),calls


def test_new_cards_and_equivalent_pair(authored):
    out,tasks,counts=authored
    assert counts==dict(public_cards=12,world_probes=13,declared_families=9)
    from sharecausal.preflight_v23.certificates import Truth,certify,law
    worlds=[Truth.model_validate(x) for x in json.loads((out/'scoring_truth.json').read_text())]
    certificates=[certify(w,tasks[w.public_task_hash],worlds) for w in worlds]
    assert sum(c['status']=='provably_nonidentified' for c in certificates)==2
    assert sum(c['supported_structural_zero'] for c in certificates)==1
    pair=[w for w in worlds if w.proof.kind=='observational_pair']
    assert pair[0].public_task_hash==pair[1].public_task_hash
    assert law(pair[0].scm,['A','Y'])==law(pair[1].scm,['A','Y'])
    assert pair[0].population_effect_exact!=pair[1].population_effect_exact


def test_payload_identity_privacy_and_luna_routing(authored):
    _,tasks,_=authored
    for task in tasks.values():
        payloads=[canonical(messages(task)) for _ in MODELS]
        assert len(set(payloads))==1
        for marker in (b'mergeid',b'population_effect',b'world_id',b'family_id',b'probability_one',b'/Users/',b'OPENROUTER_API_KEY'):
            assert marker not in payloads[0]
    for m in catalog():
        effective=settings(m)
        assert effective['max_tokens']==4096 and effective['provider']['allow_fallbacks'] is False
        assert ('temperature' in effective)==(m['id']!=MODELS[1])


def test_live_receipts_resume_no_duplicate_billing(authored):
    out,tasks,_=authored; client,calls=client_for(tasks)
    records,schedule=collect(tasks,out,25.,'DUMMY',client=client,catalog=catalog(),progress=lambda *a,**k:None)
    assert len(calls)==len(records)==48
    original={p.name:p.read_bytes() for p in (out/'provider_responses_private').glob('*.json')}
    for r in records: validated(tasks[r['task_sha256']],r['model_requested'],r)
    collect(tasks,out,25.,'DUMMY',client=client,progress=lambda *a,**k:None)
    assert len(calls)==48
    assert original=={p.name:p.read_bytes() for p in (out/'provider_responses_private').glob('*.json')}
    assert json.loads((out/'collection_summary.json').read_text())['cached']==48
    wrong=dict(records[0],prompt_sha256='a'*64)
    with pytest.raises(ValueError): validated(tasks[wrong['task_sha256']],wrong['model_requested'],wrong)


@pytest.mark.parametrize('mode,expected,number',[
    ('null','null_content',48),('bad_model','model_mismatch',48),
    ('unknown_cost','cost_unknown',1),('overrun','response',1)])
def test_failed_calls_keep_denominator_and_billed_cost(authored,mode,expected,number):
    out,tasks,_=authored; client,calls=client_for(tasks,mode)
    records,schedule=collect(tasks,out,25.,'DUMMY',client=client,catalog=catalog(),progress=lambda *a,**k:None)
    assert len(schedule)==48 and len(calls)==number
    assert all(r['status']==expected for r in records)
    assert all(r['cost_usd']==.0001 for r in records) if mode in ('null','bad_model') else True
    if mode=='unknown_cost':assert records[0]['cost_usd'] is None


def test_budget_reserves_entire_batch_before_first_post(authored):
    out,tasks,_=authored;client,calls=client_for(tasks)
    with pytest.raises(ValueError,match='batch_reservation'): collect(tasks,out,.001,'DUMMY',client=client,catalog=catalog())
    assert not calls


def test_concurrent_collector_lock_prevents_posts(authored):
    import fcntl
    out,tasks,_=authored;client,calls=client_for(tasks)
    with (out/'collection.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='concurrent_collection'):
            collect(tasks,out,25.,'DUMMY',client=client,catalog=catalog())
    assert not calls


def test_inflight_uncertainty_prevents_duplicate(authored):
    out,tasks,_=authored;client,calls=client_for(tasks)
    collect(tasks,out,25.,'DUMMY',client=client,catalog=catalog(),progress=lambda *a,**k:None)
    path=next(p for p in (out/'provider_responses_private').glob('*.json') if not p.name.endswith(('.raw.json','.inflight.json')))
    path.unlink()  # Simulate crash after durable raw response, before final receipt.
    collect(tasks,out,25.,'DUMMY',client=client,progress=lambda *a,**k:None)
    assert len(calls)==48
    assert json.loads((out/'collection_summary.json').read_text())['stopped']


def test_refusal_schema_failure_and_duplicate_fields():
    assert proposals(dict(choices=[dict(message=dict(refusal='declined'))]))[0]=='refusal'
    assert proposals(dict(choices=[dict(message=dict(content='{"proposals":[],"proposals":[]}'))]))[0]=='schema_failure'
    assert proposals(dict(choices=[dict(message=dict(content='not JSON'))]))[0]=='schema_failure'
    assert proposals(dict(choices={'unexpected':'shape'}))[0]=='null_content'


def test_helper_subprocess_does_not_receive_key(tmp_path,monkeypatch):
    from sharecausal.synthetic_live_v1.execution import execute
    monkeypatch.setenv('OPENROUTER_API_KEY','TEST_SECRET_NOT_TO_PASS')
    command=[sys.executable,'-c','import os; assert "OPENROUTER_API_KEY" not in os.environ']
    assert execute(command,Path.cwd(),tmp_path).returncode==0
    assert 'TEST_SECRET_NOT_TO_PASS' not in (tmp_path/'commands_and_exit_codes.json').read_text()


def test_full_guarded_integration_and_post_scoring_resume(tmp_path,monkeypatch):
    out=tmp_path/'integration'; runner.prepare(out,25.)
    tasks={task_hash(t):t for t in [PublicTask.model_validate(x) for x in json.loads((out/'public_tasks.json').read_text())]}
    client,calls=client_for(tasks); real_worker=runner.worker
    def worker(role,output,cap,digest_value=None):
        if role=='collect': collect(tasks,output,cap,'DUMMY',client=client,catalog=catalog(),progress=lambda *a,**k:None)
        else:real_worker(role,output,cap,digest_value)
    monkeypatch.setattr(runner,'worker',worker);monkeypatch.setenv('OPENROUTER_API_KEY','DUMMY')
    assert runner.run(out,25.,True)==0
    assert len(calls)==48
    sealed=(out/'prediction_batch.json').read_bytes()
    assert json.loads((out/'scorer_receipt.json').read_text())['seal_verified_before_truth_open']
    assert runner.run(out,25.,True)==0
    assert len(calls)==48 and (out/'prediction_batch.json').read_bytes()==sealed
    for name in ('method_results.csv','graph_admissions.csv','model_results.csv','edge_claim_audit.csv',
                 'usage_costs.csv','risk_abstention.csv','pilot_report.md','scoring_manifest.json'):
        assert (out/name).stat().st_size>0
    manifest=json.loads((out/'prediction_seal.json').read_text())
    assert 'scoring_truth.json' not in manifest and 'generation_plan.json' not in manifest
    # Deliberate post-score tampering must fail verification before a second truth open.
    (out/'prediction_batch.json').chmod(0o600);(out/'prediction_batch.json').write_text('[]')
    with pytest.raises(ValueError,match='sealed_member'): runner.run(out,25.,True)


@pytest.mark.parametrize('role', ['predict','collect'])
def test_worker_guard_denies_truth_and_generator(authored,role):
    out,_,_=authored
    root=Path(__file__).resolve().parents[1]
    code='''
import pathlib,sys
from sharecausal.synthetic_live_v1.workers import Guard
out=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2]);role=sys.argv[3]
g=Guard(role,root,inputs=[],output=out,truth=out/'scoring_truth.json',generation=out/'generation_plan.json').install()
for name in ('scoring_truth.json','generation_plan.json'):
    try:(out/name).read_bytes()
    except PermissionError:pass
    else:raise AssertionError('truth_read_allowed')
try:__import__('sharecausal.preflight_v23.certificates')
except PermissionError:pass
else:raise AssertionError('truth_import_allowed')
'''
    result=subprocess.run([sys.executable,'-c',code,str(out),str(root),role],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
