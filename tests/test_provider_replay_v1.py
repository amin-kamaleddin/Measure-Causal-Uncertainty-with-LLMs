"""Recorded-provider contract tests use invented responses, never real model outputs."""
import copy
import json
from pathlib import Path
import pytest
from sharecausal.preflight.contracts import sha
from sharecausal.preflight_v23.packets import PublicTask,task_hash,prompt
from sharecausal.preflight_v23.admission import Proposal,no_llm
from sharecausal.provider_replay_v1.prediction import adapt,seal
from sharecausal.provider_replay_v1.scoring import score
from test_real_share_followup_v3 import command

ROOT=Path(__file__).resolve().parents[1]
TASKS=ROOT/'tests/fixtures/v2_preflight_v23/public_tasks.json'
TRUTH=ROOT/'tests/fixtures/v2_preflight_v23/private_truth.json'


def pair():
    task=PublicTask.model_validate(json.loads(TASKS.read_text())[1])
    r=dict(schema_version='provider-replay-1',source_type='synthetic_test_record',
           model_requested='synthetic/test-model',model_returned='synthetic/test-model',
           provider_response_id='INVENTED_TEST_ONLY',task_sha256=task_hash(task),
           prompt_sha256=sha(prompt(task)),answer_schema_sha256=sha(Proposal.model_json_schema()),
           variant='base',response_text=json.dumps([no_llm(task)[0].model_dump()]))
    return task,r


def test_real_adapter_contract_with_invented_response():
    t,r=pair();out=adapt(t,r)
    assert out['status']=='graph_response' and out['audits'][0]['admitted']
    assert out['source_type']=='synthetic_test_record'


@pytest.mark.parametrize('field',['task_sha256','prompt_sha256','answer_schema_sha256'])
def test_incompatible_prompt_task_schema_ineligible(field):
    t,r=pair();r[field]='0'*64
    assert adapt(t,r)['status']=='INELIGIBLE_CONTEXT_OR_ENVELOPE'


def test_share_domain_record_not_synthetic_accuracy():
    t,_=pair()
    legacy=dict(model_requested='vendor/model',graph={'edges':[],'abstain':True})
    a=adapt(t,legacy)
    assert a['status']=='INELIGIBLE_CONTEXT_OR_ENVELOPE' and a['proposals']==[]


@pytest.mark.parametrize('wire',['not JSON','[NaN]','{"x":1,"x":2}', '[{"sk-SECRET_SENTINEL":"PRIVATE_SENTINEL"}]'])
def test_malformed_redaction(wire):
    t,r=pair();r['response_text']=wire;a=adapt(t,r)
    assert a['status']=='invalid_response' and a['proposals']==[]
    assert 'SECRET_SENTINEL' not in json.dumps(a) and 'PRIVATE_SENTINEL' not in json.dumps(a)


def test_postseal_known_truth_score_and_no_overwrite(tmp_path):
    t,r=pair();p=tmp_path/'pred.json';seal([t],[r],p)
    with pytest.raises(FileExistsError):seal([t],[r],p)
    scores=score(p,TRUTH)
    assert scores['independent_public_tasks']==1 and scores['scored_world_rows']==1
    assert scores['rows'][0]['certificate']['status']=='identified'
    assert scores['rows'][0]['edge_counts']==[{'tp':1,'fp':0,'fn':0}]


def test_tampered_prediction_rejected_before_truth_read(tmp_path,monkeypatch):
    t,r=pair();p=tmp_path/'pred.json';sealed=seal([t],[r],p)
    sealed['payload']['real_llm_calls']=1;p.write_text(json.dumps(sealed))
    original=Path.read_text
    def guarded(path,*args,**kw):
        assert path!=TRUTH,'Truth opened before seal verification'
        return original(path,*args,**kw)
    monkeypatch.setattr(Path,'read_text',guarded)
    with pytest.raises(ValueError,match='seal'):score(p,TRUTH)


def test_prediction_module_has_no_truth_dependency():
    import sharecausal.provider_replay_v1.prediction as module
    source=Path(module.__file__).read_text()
    assert 'import Truth' not in source and 'private_truth' not in source and 'certificates import' not in source


def test_separate_subprocess_prediction_scoring(tmp_path):
    t,r=pair();tasks=tmp_path/'tasks.json';records=tmp_path/'records.json'
    tasks.write_text(json.dumps([t.model_dump()]));records.write_text(json.dumps([r]))
    pred=tmp_path/'predictions.json';out=tmp_path/'scores.json'
    p=command('provider_replay_v1.py','predict','--tasks',tasks,'--records',records,'--out',pred)
    assert p.returncode==0,(p.stdout,p.stderr)
    s=command('provider_replay_v1.py','score','--predictions',pred,'--truth',TRUTH,'--out',out)
    assert s.returncode==0,(s.stdout,s.stderr)
    assert json.loads(out.read_text())['rows'][0]['edge_counts'][0]['fp']==0
