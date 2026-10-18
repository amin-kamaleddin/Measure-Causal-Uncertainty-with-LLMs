"""Actual v2.4 subprocess experiments and estimator/firewall regressions."""
import ast
from copy import deepcopy
import csv
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import yaml

from sharecausal.preflight.contracts import sha,canonical,make_evidence
from sharecausal.preflight.comparison import Edge
from sharecausal.preflight_v23.packets import PublicTask,task_hash,POLICIES
from sharecausal.preflight_v23.admission import no_llm,proposal,audit
from sharecausal.preflight_v23.certificates import Truth,law,certify
from sharecausal.pilot_v24.capture import capture,parse_wire,vote
from sharecausal.pilot_v24.graphs import identify_graph,d_separated
from sharecausal.pilot_v24.generation import observed_sample
from sharecausal.pilot_v24.observations import read_sample,clean_frame,frame_hash
from sharecausal.pilot_v24.engine import analyze
from sharecausal.pilot_v24.constants import METHODS
from sharecausal.pilot_v24.evaluation import paired_families
from sharecausal.pilot_v24.config import Config

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/v24_pilot_candidate.yaml'
SCRIPT=ROOT/'scripts/v24_pilot_offline.py'
FIX=ROOT/'tests/fixtures/v2_preflight_v23'
MARKER='DUMMY_V24_PRIVATE_SENTINEL'


def load(path):return json.loads(Path(path).read_text())
def tasks():return [PublicTask.model_validate(x) for x in load(FIX/'public_tasks.json')]
def task(title):return next(t for t in tasks() if title in t.card.title)
def cfg():return yaml.safe_load(CONFIG.read_text())
def rules():return load(ROOT/'tests/fixtures/v24_pilot/mock_rules.json')
def models():return cfg()['models']
def selected(t):return {m:capture(t,m,'base',0,rules())['parsed'] for m in models()}

def frame(t,n=800,seed=123):
    data=load(ROOT/'tests/fixtures/v24_pilot/generation_plan.json')['plans'][task_hash(t)]
    sample=observed_sample(t,data,n,seed)
    return pd.DataFrame(sample['rows'],columns=sample['columns'])


def predict(t,f=None,sources=None):return analyze(t,frame(t) if f is None else f,selected(t) if sources is None else sources,models(),cfg()['estimation'])

def rehash(d):
    d=deepcopy(d);d['card']['card_id']='card-'+sha({k:v for k,v in d['card'].items() if k!='card_id'})
    d['task_id']='task-'+sha({k:v for k,v in d.items() if k!='task_id'})
    return PublicTask.model_validate(d)


def cli(tmp_path,config=CONFIG,extra=(),environment=None,directory=None):
    env={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'src')}
    if environment:env.update(environment)
    return subprocess.run([sys.executable,str(SCRIPT),'--config',str(config),'--results-dir',
        str(directory or tmp_path/'runs'),*extra],cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp('v24-completed');r=cli(root,extra=['--new-run'])
    assert r.returncode==0,r.stderr
    out=Path(json.loads(r.stdout)['results_dir'])
    return load(out/'pilot_report.json'),out,json.loads(r.stdout)


@pytest.mark.parametrize('title,adjustment,status',[
    ('Thermostatic',[],'identified_by_baseline_backdoor'),
    ('Two-stage',[],'identified_by_baseline_backdoor'),
    ('hydraulic',[],'identified_by_baseline_backdoor'),
    ('Feedstock',['X'],'identified_by_baseline_backdoor'),
    ('isolated',[],'graph_conditional_structural_zero')])
def test_actual_public_graph_specific_identification(title,adjustment,status):
    t=task(title);p=no_llm(t)[0]
    r=identify_graph(t,[(e.source,e.target) for e in p.graph.edges])
    assert r['status']==status and r['adjustment']==adjustment


def test_latent_backdoor_failure_is_not_a_full_nonidentification_label():
    t=task('Acoustic');r=identify_graph(t,[('U','A'),('U','Y'),('A','Y')])
    assert r['status']=='not_identified_by_baseline_backdoor'
    assert 'not_general_nonidentification' in r['scope']


def test_actual_estimators_total_mediated_effect_and_baseline_AIPW():
    t=task('Two-stage');p=predict(t)
    arm=p['methods']['source_a']
    assert arm['decision']=='positive' and abs(arm['estimate']-.56)<.12
    assert arm['graphs'][0]['fit']['adjustment']==[]
    t=task('Feedstock');p=predict(t);arm=p['methods']['source_a']
    assert arm['graphs'][0]['fit']['estimator']=='cross_fitted_AIPW'
    assert arm['graphs'][0]['fit']['adjustment']==['X']
    assert abs(arm['estimate']-.3)<.12
    assert arm['decision']=='abstain' and arm['conditional_decision']=='positive'
    assert 'public_identification_not_established' in arm['reason']


def test_negative_eligible_effect_is_actually_asserted():
    p=predict(task('Optical'));assert p['methods']['source_a']['decision']=='negative'
    assert p['methods']['source_a']['ci_upper']<0


def test_structural_zero_cannot_be_replaced_by_confounded_observed_difference():
    t=task('isolated');f=frame(t);f['Y']=f.A  # extreme observational discrepancy never overrides exclusion
    p=predict(t,f)
    assert p['methods']['association_unadjusted']['association_direction']=='positive'
    for m,r in p['methods'].items():
        assert r['decision']=='abstain'
        if m in ('source_a','no_llm_evidence_set','robust_credible_alternatives'):
            assert r['structural_effect']==0. and r['estimate'] is None and r['ci_lower'] is None


def test_unknown_recording_blocks_population_effect_assertions():
    p=predict(task('Post-run'))
    assert p['missing_outcome'] and p['n_recorded']<p['n_eligible_sample_units']
    assert all(r['decision']=='abstain' for r in p['methods'].values())


def test_strong_unobserved_Y_C_changes_cannot_change_ANY_prediction_or_sample_digest():
    t=task('Post-run');f=frame(t);assert (f.observed==0).any()
    original=predict(t,f)
    for y,c in [(1.e100,-1.e100),(float('nan'),float('inf')),('hidden value',{MARKER:True})]:
        changed=f.astype(object).copy();changed.loc[changed.observed==0,'Y']=y
        for i in changed.index[changed.observed==0]:changed.at[i,'C']=c
        assert frame_hash(t,changed)==frame_hash(t,f)
        assert canonical(predict(t,changed))==canonical(original)


def test_recorded_outcome_change_changes_the_real_estimate():
    t=task('Thermostatic');f=frame(t);a=predict(t,f)
    f['Y']=1-f['Y'];b=predict(t,f)
    assert a['methods']['source_a']['estimate']==pytest.approx(-b['methods']['source_a']['estimate'])
    assert a['methods']['source_a']['decision']=='positive' and b['methods']['source_a']['decision']=='negative'


@pytest.mark.parametrize('extra',['U','population_effect','mechanism_family'])
def test_hidden_columns_or_frame_metadata_rejected(extra):
    t=tasks()[0];f=frame(t);f[extra]=1
    with pytest.raises(ValueError):predict(t,f)
    f=frame(t);f.attrs[extra]=MARKER
    with pytest.raises(ValueError):predict(t,f)


def test_bad_recorded_numeric_input_is_not_silently_imputed():
    t=task('Thermostatic');f=frame(t);f.loc[0,'Y']=float('nan')
    with pytest.raises(ValueError):predict(t,f)


def test_estimator_failures_remain_abstentions_without_fallback_or_dropped_cards():
    t=task('Thermostatic');f=frame(t);f['A']=0
    p=predict(t,f)
    assert p['estimator_failure_count']>=1
    assert all(r['decision']=='abstain' for r in p['methods'].values())
    assert p['methods']['source_a']['reason']=='admitted_graph_estimator_failed'


@pytest.mark.parametrize('wire',['{broken','null','[] trailing','NaN','[Infinity]',
                                '[{"x":NaN}]','[{"x":1,"x":2}]'])
def test_malformed_nonfinite_duplicate_JSON_response_is_rejected_and_redacted(wire):
    r=parse_wire(tasks()[0],wire)
    assert r['status']=='invalid_response' and r['proposals']==[]
    assert wire not in json.dumps(r)


def test_unknown_response_field_names_or_values_never_enter_audits():
    r=parse_wire(tasks()[0],json.dumps([{MARKER:MARKER}]))
    assert r['status']=='invalid_response' and MARKER not in json.dumps(r)


@pytest.mark.parametrize('kind',['backwards','missing_arrow','missing_citation','unsupported_zero','cycle'])
def test_graph_response_constraints_are_real_pipeline_vetoes(kind):
    t=task('Thermostatic');note=t.card.evidence[0].evidence_id
    if kind=='backwards':raw=proposal([Edge(source='Y',target='A',evidence_id=note)]).model_dump()
    elif kind=='missing_arrow':raw=proposal([Edge(source='U',target='Y',evidence_id=note)]).model_dump()
    elif kind=='missing_citation':raw=proposal([Edge(source='A',target='Y',evidence_id='missing')]).model_dump()
    elif kind=='unsupported_zero':raw=proposal([],zero=True).model_dump()
    else:
        raw=no_llm(t)[0].model_dump();raw['graph']['edges'].append(dict(source='Y',target='A',evidence_id=note))
    a,_=audit(t,raw)
    assert not a['admitted']
    if kind=='missing_arrow':assert ['A','Y'] in a['omitted_supported_edges']
    parsed=parse_wire(t,json.dumps([raw]))
    sources={models()[0]:parsed}
    p=predict(t,sources=sources)
    assert p['methods']['source_a']['decision']=='abstain'
    assert p['planned_source_denominator']==4


def test_public_only_admissions_and_estimator_selection_precede_statistical_fits(monkeypatch):
    import sharecausal.pilot_v24.engine as engine
    real_audit=engine.audit;real_fit=engine.unadjusted;seen=[]
    def checked_audit(*args,**kwargs):seen.append('audit');return real_audit(*args,**kwargs)
    def checked_fit(*args,**kwargs):assert 'audit' in seen;seen.append('fit');return real_fit(*args,**kwargs)
    monkeypatch.setattr(engine,'audit',checked_audit);monkeypatch.setattr(engine,'unadjusted',checked_fit)
    a=predict(task('Thermostatic'))
    assert seen.index('audit')<seen.index('fit') and a['oracle_information_used'] is False


def test_model_missingness_and_graph_sets_cannot_create_extra_ballots():
    t=task('pneumatic');p,q=[r.model_dump() for r in no_llm(t)]
    sources={models()[0]:{'status':'graph_response','proposals':[p]},models()[1]:{'status':'graph_response','proposals':[p]},
        models()[2]:{'status':'multiple_graphs','proposals':[p,q]}}
    graph,v=vote(t,sources,models())
    assert graph is None and v['planned_denominator']==4 and v['sources'][models()[2]]=='multiple_graphs'
    assert v['sources'][models()[3]]=='absent'


def test_cyclic_majority_of_acyclic_complete_graphs_is_not_repaired():
    t=task('Feedstock');d=t.model_dump();d['card']['variables'].append(dict(name='C',role='baseline',time=0,observed=True));t=rehash(d)
    sure=no_llm(t)[0].graph.edges;note=t.card.evidence[0].evidence_id;sources={}
    for model,pairs in zip(models(),[[('U','X'),('X','C')],[('X','C'),('C','U')],[('C','U'),('U','X')]]):
        raw=proposal([*sure,*[Edge(source=a,target=b,evidence_id=note) for a,b in pairs]]).model_dump()
        sources[model]={'status':'graph_response','proposals':[raw]}
    graph,a=vote(t,sources,models()[:3])
    assert graph is None and a['reason']=='cyclic_majority_no_repair'


def test_response_and_prediction_schema_provenance_snapshot():
    t=tasks()[0];r=capture(t,models()[0],'base',0,rules());receipt=r['receipt']
    assert set(receipt)=={'source_type','transport','model_id','variant','repetition','prompt_sha256',
        'task_sha256','response_sha256','schema_sha256','record_sha256'}
    assert receipt['source_type']=='offline_fixture' and receipt['transport']=='local_only_no_sender'
    assert receipt['record_sha256']==sha({k:v for k,v in receipt.items() if k!='record_sha256'})
    p=predict(t)
    assert p['schema_version']=='pilot-prediction-2.4' and tuple(p['methods'])!=() and set(p['methods'])==set(METHODS)
    assert p['planned_source_denominator']==4


def test_evidence_policy_sensitivity_has_no_outcome_input():
    t=task('Thermostatic');p=predict(t)
    c=p['source_audit'][models()[2]]['policy_audits']
    assert not c['credible_alternatives'][0]['admitted'] and c['permissive_sensitivity'][0]['admitted']
    changed=frame(t);changed['Y']=1-changed.Y
    assert predict(t,changed)['source_audit']==p['source_audit']


def test_public_code_cannot_import_truth_generator_or_scorer():
    for name in ('engine.py','capture.py','graphs.py','observations.py'):
        source=(ROOT/'src/sharecausal/pilot_v24'/name).read_text()
        tree=ast.parse(source);imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(any(x in module for x in ('certificates','evaluation','generation')) for module in imports)
        assert 'population_effect_exact' not in source and 'family_id' not in source


def test_family_precision_uses_cards_and_unequal_family_weights_not_people_or_replies():
    a=paired_families({'a':1.,'b':0.,'c':0.,'d':0.},{'a':'f1','b':'f1','c':'f2','d':'f3'})
    assert a['eligible_cards']==4 and a['declared_families']==3 and a['paired_mean_difference']==.25
    assert a['family_bound_unclipped_half_width']>1


def test_full_subprocess_outputs_have_actual_assertions_and_card_denominators(completed):
    report,out,payload=completed
    assert report['public_cards']==9 and report['private_world_probes']==10
    assert report['prediction_rows']==27 and report['method_card_primary_rows']==108
    assert report['declared_development_families']==8 and report['sample_repetitions_nested']==3
    assert payload['assertions_by_method']['source_a']>0 and payload['assertions_by_method']['source_c']==0
    assert all(s['eligible_cards']==9 for s in report['main_summary'])
    for s in report['main_summary']:
        if s['assertions']==0:assert s['unjustified_risk_among_assertions_lower'] is None
    assert report['fixture_mode'] and report['real_llm_calls']==report['openrouter_cost_usd']==report['share_records_used']==0
    assert report['protocol_status']=='candidate_not_frozen' and not report['ready_to_freeze']
    assert report['decision_required_count']==16 and not report['confirmatory_precision_ready']
    assert report['worker_exits']==[{'role':r,'exit_code':0} for r in ('generate','predict','score')]
    for role in ('prediction_guard','scorer','generator'):
        assert (out/(role+'_receipt.json')).exists()
    assert load(out/'prediction_guard_receipt.json')['scorer_imported'] is False
    assert load(out/'scorer_receipt.json')['seal_verified_before_truth_open'] is True


def test_executed_capture_refusals_invalid_and_absent_stay_visible(completed):
    report,out,_=completed;counts=report['response_status_counts_all_variants']
    assert report['mock_capture_records']==144
    assert counts['invalid_response']==18 and counts['absent']==4 and counts['source_abstained']>0 and counts['multiple_graphs']>0
    batch=load(out/'prediction_batch.json')
    assert all(p['planned_source_denominator']==4 for p in batch)
    manifest=load(out/'prediction_seal.json')
    assert {'prediction_batch.json','mock_records.json','design_commitment.json','design_seal.json','sample_manifest.json'}<=manifest.keys()


def test_all_analysis_samples_are_masked_and_identical_across_method_arms(completed):
    _,out,_=completed;samples=load(out/'sample_manifest.json');predictions=load(out/'prediction_batch.json')
    assert len(samples)==27
    for row in samples:
        data=load(out/row['file']);assert 'U' not in data['columns']
        cols=data['columns'];r=cols.index('observed');y=cols.index('Y')
        for values in data['rows']:
            if not values[r]:
                assert values[y] is None
                if 'C' in cols:assert values[cols.index('C')] is None
        p=next(p for p in predictions if p['sample_id']==row['sample_id'])
        t=next(t for t in tasks() if task_hash(t)==row['task_hash'])
        assert p['analysis_sample_sha256']==frame_hash(t,read_sample(t,data))


def test_fully_parsed_JSON_CSV_manifests_and_fixed_curves(completed):
    _,out,_=completed;m=load(out/'content_manifest.json')
    for section,items in m.items():
        for name,value in items.items():
            path=(out if section=='outputs' else ROOT)/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==value
    for path in out.rglob('*.json'):json.loads(path.read_text())
    assert len(list(csv.DictReader((out/'main_summary.csv').open())))==12
    assert len(list(csv.DictReader((out/'per_card.csv').open())))==108
    curves=list(csv.DictReader((out/'risk_assertion_curves.csv').open()))
    assert len(curves)==144 and all(r['eligible_cards']=='9' for r in curves)
    assert {r['practical_zero'] for r in curves}=={'0.0','0.05','0.1'}
    assert any(r['coverage_matched']=='False' for r in curves)
    assert len(list(csv.DictReader((out/'paired_family_differences.csv').open())))==24


def test_previous_failed_statistical_gates_are_rerun_without_being_weakened(completed):
    report,out,_=completed;r=report['precision_validation']
    assert r['family_t_comparator']['coverage_gate_counts']['FAIL_UNDERCOVERAGE']==92
    assert r['family_t_comparator']['minimum_simulated_coverage']==.097
    assert r['bounded_family_hoeffding']['minimum_mean_half_width']>.49
    assert len(list(csv.DictReader((out/'precision_validation.csv').open())))==594


def test_same_public_evidence_and_observed_law_private_world_swap_leaves_FULL_prediction_seal_unchanged(tmp_path):
    data=cfg();truth=tmp_path/'truth.json';truth.write_text((FIX/'private_truth.json').read_text());data['scoring_truth']=str(truth)
    data['sample_repetitions']=1;configuration=tmp_path/'config.yaml';configuration.write_text(yaml.safe_dump(data))
    results=[]
    for swap in (False,True):
        if swap:
            w=load(truth)
            assert w[0]['public_task_hash']==w[1]['public_task_hash']
            assert law(Truth.model_validate(w[0]).scm,['A','Y'])==law(Truth.model_validate(w[1]).scm,['A','Y'])
            for k in ('scm','population_effect_exact'):w[0][k],w[1][k]=w[1][k],w[0][k]
            truth.write_text(json.dumps(w))
        r=cli(tmp_path,configuration,extra=['--new-run']);assert r.returncode==0,r.stderr
        results.append(Path(json.loads(r.stdout)['results_dir']))
    first,second=results
    for name in ('prediction_seal.json','design_seal.json','prediction_batch.json','mock_records.json','sample_manifest.json'):
        assert (first/name).read_bytes()==(second/name).read_bytes()
    assert load(first/'scorer_receipt.json')['scoring_truth_sha256']!=load(second/'scorer_receipt.json')['scoring_truth_sha256']
    certs=load(second/'scored_results.json')['certifications']
    assert certs['world-pump-0']['status']==certs['world-pump-1']['status']=='provably_nonidentified'
    p=next(p for p in load(second/'prediction_batch.json') if p['public_task_hash']==load(truth)[0]['public_task_hash'])
    assert all(m['decision']=='abstain' for m in p['methods'].values())


def test_private_observational_equivalence_gives_identical_generated_sample_and_prediction_bytes():
    t=task('pneumatic');worlds=[Truth.model_validate(x) for x in load(FIX/'private_truth.json')][:2]
    payloads=[observed_sample(t,{'nodes':w.scm.model_dump()['nodes'],'recording':'complete'},800,999) for w in worlds]
    assert canonical(payloads[0])==canonical(payloads[1])
    predictions=[predict(t,pd.DataFrame(p['rows'],columns=p['columns'])) for p in payloads]
    assert canonical(predictions[0])==canonical(predictions[1])


@pytest.mark.parametrize('role',['generate','predict','score'])
def test_process_guard_blocks_truth_keys_sockets_and_role_writes_before_seal(tmp_path,role):
    code='''import json,os,socket,sys
from pathlib import Path
from sharecausal.pilot_v24.guards import ProcessGuard
root=Path(sys.argv[1]);out=Path(sys.argv[2]);role=sys.argv[3]
truth=out/'truth.json';generation=out/'generation.json'
g=ProcessGuard(role,root,output=out,truth=truth,generation=generation).install()
blocked=[]
for name,operation in [
 ('truth',lambda:truth.read_text()),
 ('key',lambda:os.environ['OPENROUTER_API_KEY']),
 ('approval',lambda:os.environ['SHARECAUSAL_LIVE_LLM_APPROVED']),
 ('socket',lambda:socket.socket()),
 ('SHARE',lambda:Path('/tmp/not_share.dta').read_bytes()),
 ('credential',lambda:Path('/tmp/.env').read_text()),
 ('prediction_rewrite',lambda:(out/'prediction_seal.json').write_text('{}'))]:
 try:operation()
 except PermissionError:blocked.append(name)
 else:raise AssertionError('guard did not reject operation')
if role=='predict':
 try:generation.read_text()
 except PermissionError:blocked.append('generation')
 else:raise AssertionError('generator plan readable')
import secrets
print(json.dumps({'blocked':blocked,'runtime_source_imported':True}))
'''
    r=subprocess.run([sys.executable,'-c',code,str(ROOT),str(tmp_path),role],cwd=ROOT,
        env={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'src'),'OPENROUTER_API_KEY':MARKER,
             'SHARECAUSAL_LIVE_LLM_APPROVED':'1'},capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert len(json.loads(r.stdout)['blocked'])==(8 if role=='predict' else 7)
    assert MARKER not in r.stdout+r.stderr


def test_score_worker_cannot_read_truth_or_rewrite_predictions_with_bad_seal(completed,tmp_path):
    _,out,_=completed;job=load(out/'job_score.json');job['prediction_digest']='0'*64
    # A distinct internal job file uses the same sealed outputs and invalid expected digest.
    path=tmp_path/'job_score.json';path.write_text(json.dumps(job))
    before=hashlib.sha256((out/'prediction_batch.json').read_bytes()).hexdigest()
    r=subprocess.run([sys.executable,'-m','sharecausal.pilot_v24.workers','score',str(path)],cwd=ROOT,
        env={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
    assert r.returncode==2 and json.loads(r.stderr)['stage']=='worker_score'
    assert hashlib.sha256((out/'prediction_batch.json').read_bytes()).hexdigest()==before


def test_actual_guarded_cli_blocks_all_network_credential_SHARE_IO_with_dummy_environment(tmp_path):
    hook=tmp_path/'hook';hook.mkdir()
    (hook/'sitecustomize.py').write_text('''import os,sys
_original=os._Environ.__getitem__
def no_keys(self,key):
 if key in ('OPENROUTER_API_KEY','SHARECAUSAL_LIVE_LLM_APPROVED'):raise RuntimeError('credential read attempted')
 return _original(self,key)
os._Environ.__getitem__=no_keys
def audit(event,args):
 if event.startswith('socket.'):raise RuntimeError('socket attempted')
 if event=='open' and isinstance(args[0],str):
  p=args[0].lower()
  if any(s in p for s in ('share-research/private','/.env','/.aws/','/.ssh/')) or p.endswith(('.zip','.dta','.sav','.parquet','.feather')):
   raise RuntimeError('private IO attempted')
sys.addaudithook(audit)
''')
    env={'PYTHONPATH':str(hook)+os.pathsep+str(ROOT/'src'),'OPENROUTER_API_KEY':MARKER,'SHARECAUSAL_LIVE_LLM_APPROVED':'1'}
    runs=[cli(tmp_path,extra=['--new-run'],environment=env) for _ in range(2)]
    for r in runs:assert r.returncode==0,r.stderr
    data=[json.loads(r.stdout) for r in runs]
    assert data[0]['content_manifest_sha256']==data[1]['content_manifest_sha256']
    for row in data:
        for p in Path(row['results_dir']).rglob('*'):
            if p.is_file():assert MARKER not in p.read_text()


def test_overwrite_refusal_preserves_existing_outputs_and_new_run_succeeds(tmp_path):
    path=tmp_path/'outputs';path.mkdir();marker=path/'keep.txt';marker.write_text('preserve')
    r=cli(tmp_path,directory=path)
    assert r.returncode==2 and json.loads(r.stderr)['stage']=='output' and marker.read_text()=='preserve'
    assert sorted(p.name for p in path.iterdir())==['keep.txt']


@pytest.mark.parametrize('mutation,stage',[
 ('private_path','configuration'),('credential_path','configuration'),('unknown_key','configuration'),
 ('live_spend','configuration'),('frozen_claim','configuration'),('missing_packet','public_inputs'),
 ('corrupted_packet','public_inputs'),('truth_as_public','public_inputs')])
def test_cli_bad_paths_or_inputs_fail_closed_with_redaction(tmp_path,mutation,stage):
    d=cfg()
    if mutation=='private_path':d['public_tasks']='~/share-research/private/'+MARKER+'.dta'
    elif mutation=='credential_path':d['public_tasks']='~/.aws/credentials'
    elif mutation=='unknown_key':d[MARKER]=MARKER
    elif mutation=='live_spend':d['authorized_openrouter_usd']=1
    elif mutation=='frozen_claim':d['protocol_status']='frozen'
    elif mutation=='missing_packet':d['public_tasks']=str(tmp_path/'missing.json')
    elif mutation=='truth_as_public':d['public_tasks']=d['scoring_truth']
    else:
        values=load(FIX/'public_tasks.json');values[0][MARKER]=MARKER
        p=tmp_path/'packets.json';p.write_text(json.dumps(values));d['public_tasks']=str(p)
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(d));r=cli(tmp_path,config=p,extra=['--new-run'])
    assert r.returncode==2 and not r.stdout,r.stderr
    assert json.loads(r.stderr)['stage']==stage
    assert MARKER not in r.stderr and json.loads(r.stderr)['live_fallback'] is False


def test_worker_entry_rejects_credential_or_truth_file_before_open(tmp_path):
    for name in (str(FIX/'private_truth.json'),'/tmp/.env'):
        r=subprocess.run([sys.executable,'-m','sharecausal.pilot_v24.workers','predict',name],cwd=ROOT,
            env={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
        assert r.returncode==2 and json.loads(r.stderr)['stage']=='worker_predict'


def test_no_live_CLI_option_exists(tmp_path):
    r=cli(tmp_path,extra=['--live'])
    assert r.returncode==2 and json.loads(r.stderr)['stage']=='arguments'


@pytest.mark.parametrize('field,value',[('prompt_sha256','0'*64),('model_id','mock-family-b/fixture'),('record_sha256','0'*64)])
def test_tampered_mock_provenance_fails_binding(field,value):
    from sharecausal.pilot_v24.capture import validate_record
    t=tasks()[0];r=capture(t,models()[0],'base',0,rules())
    assert validate_record(t,r,models()[0],'base',0)
    r['receipt'][field]=value
    assert not validate_record(t,r,models()[0],'base',0)


def test_parsed_response_cannot_be_modified_after_receipt_capture():
    from sharecausal.pilot_v24.capture import validate_record
    t=tasks()[0];r=capture(t,models()[0],'base',0,rules())
    r['parsed']['proposals']=[]
    assert not validate_record(t,r,models()[0],'base',0)


def test_new_backdoor_d_separation_matches_existing_independent_graph_library_exhaustively():
    import itertools
    import networkx as nx
    names=['U','X','A','Y'];possible=list(itertools.combinations(names,2))
    for mask in range(2**len(possible)):
        edges=[e for i,e in enumerate(possible) if mask&(1<<i)]
        g=nx.DiGraph();g.add_nodes_from(names);g.add_edges_from(e for e in edges if e[0]!='A')
        for conditioned in ([],['X']):
            assert d_separated(edges,conditioned)==nx.is_d_separator(g,{'A'},{'Y'},set(conditioned))


def test_association_intervals_are_not_scored_as_causal_ATE_interval_coverage(completed):
    _,out,_=completed;rows=load(out/'scored_results.json')['world_estimation_diagnostics']
    association=[r for r in rows if r['method']=='association_unadjusted']
    assert all(r['estimate_target']=='observed_case_association' and r['interval_covers_identified_truth_ONLY'] is None for r in association)
