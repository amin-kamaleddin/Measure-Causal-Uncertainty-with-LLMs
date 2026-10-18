"""Step 6C.2 regressions and actual subprocess execution; fixtures are fictional."""
import ast
from copy import deepcopy
import csv
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
import pytest
import yaml

from sharecausal.preflight.contracts import PublicWorldCardV2,sha,make_evidence
from sharecausal.preflight.comparison import Edge
from sharecausal.preflight.budget import FakePrice,SimulatedBudget
from sharecausal.preflight_v23.packets import PublicTask,public_task,task_hash,exports,task_content,POLICIES
from sharecausal.preflight_v23.admission import proposal,audit,no_llm,majority
from sharecausal.preflight_v23.certificates import Truth,BinaryDAG,BinaryNode,law,effect,certify
from sharecausal.preflight_v23.precision import (PrecisionPlan,diagnostics,simulate_case,family_intervals,
                                               bounded_half_width,coverage_gate,wilson)
from sharecausal.preflight_v23.scoring import score,fixed_curves
from sharecausal.preflight_v23.families import FamilyDefinition,CardAssignment,audit_families
from sharecausal.preflight_v23.sealing import seal,verify,write_once,digest
from sharecausal.preflight_v23.transport import MockTransport,validate_record
from sharecausal.preflight_v23 import runner

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests/fixtures/v2_preflight_v23'
CONFIG=ROOT/'configs/v23_preflight_candidate.yaml'
SCRIPT=ROOT/'scripts/v23_preflight_offline.py'
MARKER='DUMMY_PRIVATE_TEST_SENTINEL_23'


def load(name):return json.loads((FIX/name).read_text())
def tasks():return [PublicTask.model_validate(x) for x in load('public_tasks.json')]
def worlds():return [Truth.model_validate(x) for x in load('private_truth.json')]
def config():return yaml.safe_load(CONFIG.read_text())
def by_title(text):return next(t for t in tasks() if text in t.card.title)

def rehash(data):
    d=deepcopy(data);d['card']['card_id']='card-'+sha({k:v for k,v in d['card'].items() if k!='card_id'})
    d['task_id']='task-'+sha({k:v for k,v in d.items() if k!='task_id'})
    return PublicTask.model_validate(d)

def edge(task,a,b):
    e=next(e for e in task.card.evidence if (e.source,e.target)==(a,b))
    return Edge(source=a,target=b,evidence_id=e.evidence_id)

def fast_plan():
    p=config()['precision'];p.update(card_counts=[60],family_counts=[10],conditional_error_a=[.2],
        latent_family_correlations=[.6],source_b_dropout=[0.,.35],repetitions=200)
    return PrecisionPlan.model_validate(p)


def cli(tmp_path,config_file=CONFIG,extra=(),env=None):
    # Do not enumerate inherited environment values (a real key may exist).
    environ={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'src')}
    if env:environ.update(env)
    return subprocess.run([sys.executable,str(SCRIPT),'--config',str(config_file),
        '--results-dir',str(tmp_path/'runs'),'--new-run',*extra],cwd=ROOT,
        env=environ,capture_output=True,text=True,timeout=120)


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    out=tmp_path_factory.mktemp('v23-completed')
    return runner.run_preflight(CONFIG,out)


def test_exact_joint_witness_and_different_population_interventions():
    ws=worlds()[:2];t=by_title('pneumatic')
    expected={(0,0):F(2,5),(0,1):F(1,10),(1,0):F(1,10),(1,1):F(2,5)}
    assert law(ws[0].scm,['A','Y'])==law(ws[1].scm,['A','Y'])==expected
    assert [effect(w.scm) for w in ws]==[F(3,5),F(0)]
    assert ws[0].public_task_hash==ws[1].public_task_hash==task_hash(t)
    assert all(certify(w,t,ws)['status']=='provably_nonidentified' for w in ws)
    assert len(exports([t]))==1


def test_public_exports_allowlist_order_dedup_and_no_truth_metadata():
    ts=tasks();a=exports(ts);b=exports(list(reversed(ts)))
    assert a==b and list(a)==sorted(a)
    for name,raw in a.items():
        assert name=='packet_'+hashlib.sha256(raw).hexdigest()+'.json'
        for w in worlds():assert w.world_id.encode() not in raw and w.family_id.encode() not in raw
        assert b'probability_one' not in raw and b'population_effect_exact' not in raw
    with pytest.raises(ValueError):exports(ts+[ts[0]])
    with pytest.raises(ValueError):public_task(worlds()[0])


@pytest.mark.parametrize('key',['population_effect_exact','probability_one','generator_seed',MARKER])
def test_extra_public_fields_fail_closed(key):
    data=tasks()[0].model_dump();data[key]=MARKER
    with pytest.raises(ValueError):PublicTask.model_validate(data)


def test_mutating_nested_frozen_card_or_credibility_invalidates_export():
    t=tasks()[0];t.credibility.pop()
    with pytest.raises(ValueError):public_task(t)
    t=tasks()[0];t.card.evidence.pop()
    with pytest.raises(ValueError):public_task(t)


def test_task_id_does_not_trigger_secret_scan_but_dummy_key_text_does():
    assert public_task(tasks()[0])['task_id'].startswith('task-')
    d=tasks()[0].model_dump();d['card']['title']='sk-dummysecret12345678'
    with pytest.raises(ValueError):public_task(rehash(d))


@pytest.mark.parametrize('policy',POLICIES)
def test_supported_edge_admitted_every_policy(policy):
    t=by_title('Thermostatic');a,p=audit(t,proposal([edge(t,'A','Y')]).model_dump(),policy)
    assert a['admitted'] and a['edge_tiers']==['supported']


def test_credible_latent_alternatives_are_hypotheses_not_proved_zeros():
    t=by_title('pneumatic');p=proposal([edge(t,'U','A'),edge(t,'U','Y')])
    a,_=audit(t,p.model_dump(),'credible_alternatives')
    assert a['admitted'] and a['conditional_graph_zero'] and not a['supported_exclusion']
    assert not a['causal_zero_claim_justified'] and a['edge_tiers']==['credible_ambiguous']*2
    assert not audit(t,p.model_dump(),'supported_only')[0]['admitted']
    p=proposal(p.graph.edges,zero=True)
    assert 'unsupported_exclusion_claim' in audit(t,p.model_dump())[0]['edge_tiers']


def test_unrelated_valid_citation_is_not_credible_primary_evidence():
    t=by_title('pneumatic');note=next(e for e in t.card.evidence if e.kind=='general')
    p=proposal([Edge(source='A',target='Y',evidence_id=note.evidence_id)]).model_dump()
    for policy in ('supported_only','credible_alternatives'):
        a,_=audit(t,p,policy);assert not a['admitted'] and a['edge_tiers']==['unrelated']
    assert audit(t,p,'permissive_sensitivity')[0]['admitted']


def test_ambiguous_edge_without_predeclared_credibility_rule_is_ungrounded():
    t=by_title('pneumatic');d=t.model_dump();d['credibility']=[];t=rehash(d)
    a,_=audit(t,proposal([edge(t,'A','Y')]).model_dump())
    assert not a['admitted'] and 'unrelated' in a['edge_tiers']


@pytest.mark.parametrize('policy',POLICIES)
def test_uncited_or_invalid_citation_rejected_even_permissive(policy):
    t=tasks()[0]
    for citation in ('','not-an-evidence-id'):
        a,_=audit(t,proposal([Edge(source='A',target='Y',evidence_id=citation)]).model_dump(),policy)
        assert not a['admitted'] and 'invalid_citation' in a['edge_tiers']


@pytest.mark.parametrize('policy',POLICIES)
def test_randomized_incoming_assignment_and_impossible_time_veto(policy):
    t=by_title('Thermostatic');note=t.card.evidence[0]
    a,_=audit(t,proposal([Edge(source='U',target='A',evidence_id=note.evidence_id)]).model_dump(),policy)
    assert a['format_valid'] and not a['admitted'] and 'contradicted' in a['edge_tiers']
    a,_=audit(t,proposal([Edge(source='Y',target='A',evidence_id=note.evidence_id)]).model_dump(),policy)
    assert not a['format_valid'] and not a['admitted']


def test_unsupported_empty_exclusion_is_not_primary_graph_evidence():
    t=tasks()[0]
    assert not audit(t,proposal().model_dump())[0]['admitted']
    z=by_title('isolated');a,_=audit(z,no_llm(z)[0].model_dump())
    assert a['admitted'] and a['supported_exclusion'] and a['conditional_graph_zero']
    p=proposal([Edge(source='A',target='Y',evidence_id=z.card.evidence[0].evidence_id)])
    assert not audit(z,p.model_dump(),'permissive_sensitivity')[0]['admitted']


def test_format_cycle_malformed_response_and_hostile_keys_redacted():
    t=tasks()[0];note=t.card.evidence[0]
    cyc={'schema_version':'graph-proposal-2.3','structural_zero_claim':False,
         'graph':{'schema_version':'preflight-graph-2.2','edges':[dict(source='A',target='Y',evidence_id=note.evidence_id),
         dict(source='Y',target='A',evidence_id=note.evidence_id)],'adjustment':[],'abstain':False,'abstention_reason':''}}
    for p in (None,[],{},cyc,{MARKER:MARKER}):
        a,_=audit(t,p);assert not a['admitted'] and MARKER not in json.dumps(a)


def test_majority_missing_model_multigraph_tie_stay_in_planned_denominator():
    t=tasks()[0];p=no_llm(t)[0].model_dump();q=no_llm(t)[1].model_dump();models=list('abcd')
    for sources in ({'a':[p],'b':[p]}, {'a':[p],'b':[p],'c':[q],'d':[q]},
                    {'a':[p,q],'b':[p],'c':[p]}):
        r=majority(t,sources,models,'credible_alternatives')
        assert r['denominator_all_planned_models']==4 and r['reason']=='no_strict_majority'
        assert r['graph'] is None
    r=majority(t,{'a':[p],'b':[p],'c':[p]},models,'credible_alternatives')
    assert r['graph']==[('A','Y')] and r['decision']=='abstain'


def test_majority_cycle_not_repaired():
    # Same-time latent/baseline nodes permit three individually acyclic orders.
    t=by_title('Feedstock');d=t.model_dump()
    d['card']['variables'].append(dict(name='C',role='baseline',time=0,observed=True))
    note=d['card']['evidence'][0]['evidence_id'];t=rehash(d)
    models=list('abc');sources={}
    for m,pairs in zip(models,[[('U','X'),('X','C')],[('X','C'),('C','U')],[('C','U'),('U','X')]]):
        required=[edge(t,'X','A'),edge(t,'X','Y')]
        sources[m]=[proposal(required+[Edge(source=a,target=b,evidence_id=note) for a,b in pairs]).model_dump()]
    r=majority(t,sources,models,'permissive_sensitivity')
    assert r['reason']=='cycle_no_repair' and r['decision']=='abstain' and r['graph'] is None


def test_observed_proxy_or_selection_warning_does_not_prove_nonidentification():
    ws=worlds();ts={task_hash(t):t for t in tasks()}
    for w in ws:
        if w.family_id in ('proxy_common_drive','measured_common_cause','outcome_selection'):
            assert certify(w,ts[w.public_task_hash],ws)['status']=='uncertified'
    w=ws[-1];d=w.model_dump();d['proof']={'kind':'observational_pair','witness_ids':[ws[0].world_id,ws[1].world_id]}
    altered=Truth.model_validate(d)
    assert certify(altered,ts[w.public_task_hash],ws)['status']=='uncertified'


def test_randomization_with_observed_mediator_identifies_total_effect():
    ts={task_hash(t):t for t in tasks()};w=next(w for w in worlds() if w.family_id=='randomized_mediation')
    c=certify(w,ts[w.public_task_hash],worlds())
    assert c['status']=='identified' and effect(w.scm)==F(14,25)


def test_missing_witness_or_unequal_law_does_not_certify():
    ws=worlds()[:2];t=by_title('pneumatic')
    assert certify(ws[0],t,[ws[0]])['status']=='uncertified'
    d=ws[0].model_dump();d['scm']['nodes'][-1]['probability_one']=['1/10','9/10'];d['population_effect_exact']='4/5'
    w=Truth.model_validate(d)
    assert certify(w,t,[w,ws[1]])['status']=='uncertified'


def test_public_randomization_invalidates_confounded_matching_witness():
    t=by_title('pneumatic');d=t.model_dump();d['card']['assignment']='randomized_documented'
    d['card']['evidence'].append(make_evidence(kind='assignment',source='A',target='Y',relation='supported',statement='Independent switch lottery.'))
    t=rehash(d);pair=[]
    for w in worlds()[:2]:
        d=w.model_dump();d['public_task_hash']=task_hash(t);pair.append(Truth.model_validate(d))
    assert certify(pair[0],t,pair)['status']=='uncertified'
    with pytest.raises(ValueError):certify(pair[1],t,pair)


@pytest.mark.parametrize('change',['effect','probability','cycle','binding'])
def test_corrupted_private_proof_contract(change):
    d=worlds()[0].model_dump()
    if change=='effect':d['population_effect_exact']='0'
    elif change=='probability':d['scm']['nodes'][0]['probability_one']=['2']
    elif change=='cycle':d['scm']['nodes'][0]['parents']=['Y'];d['scm']['nodes'][0]['probability_one']=['1/2','1/2']
    else:
        d['public_task_hash']='0'*64;w=Truth.model_validate(d)
        with pytest.raises(ValueError):certify(w,tasks()[0],worlds())
        return
    with pytest.raises(ValueError):Truth.model_validate(d)


def test_primary_components_bounds_card_dedup_and_no_assertion_risk():
    ts={task_hash(t):t for t in tasks()};ws=worlds();cs={w.world_id:certify(w,ts[w.public_task_hash],ws) for w in ws}
    s=score({h:'positive' for h in ts},ws,cs,ts,.1)
    assert s['eligible_cards']==9 and s['assertions']==9
    assert s['component_counts']==dict(wrong_sign_identified_nonnull=1,assertion_on_proven_nonidentification=1,supported_structural_zero_violation=1)
    assert s['known_unjustified_count']==3 and s['unresolved_additional_count']==3
    assert s['unjustified_rate_lower']==3/9 and s['unjustified_rate_upper']==6/9
    s=score({h:'abstain' for h in ts},ws,cs,ts,.1)
    assert s['assertion_rate']==0 and s['risk_denominator_asserted_cards']==0
    assert s['unjustified_risk_among_assertions_lower'] is None
    assert s['wrong_sign_identified_nonnull_per_all_assertions'] is None
    with pytest.raises(ValueError):score({h:'positive' for h in ts},ws+[ws[0]],cs,ts,.1)


def test_curves_thresholds_fixed_full_denominator_and_unmatched_detection(completed):
    _,out,_=completed;rows=list(csv.DictReader((out/'risk_assertion_curves.csv').open()))
    assert len(rows)==45 and all(int(r['eligible_cards'])==9 for r in rows)
    assert {r['practical_zero'] for r in rows}=={'0.0','0.05','0.1'}
    zero=[r for r in rows if r['probe']=='abstain_contract_probe']
    assert all(r['coverage_matched']=='False' and r['unjustified_risk_among_assertions_lower']=='' for r in zero)
    assert any(r['coverage_matched']=='True' for r in rows if r['threshold_fixed_before_truth']=='0.5')
    assert completed[0]['actual_model_coverage_matching'].startswith('UNMATCHED')


def test_distribution_free_bound_cannot_collapse_at_zero_events():
    means,intervals=family_intervals(np.zeros((4,10)))
    assert np.all(means==0) and np.all(intervals['family_t_comparator']==0)
    assert np.all(intervals['bounded_family_hoeffding']>.8)
    assert bounded_half_width([.5,.5])>bounded_half_width([.1]*10)
    assert bounded_half_width([.9,.1])>bounded_half_width([.5,.5])


@pytest.mark.parametrize('successes,expected',[(100,'FAIL_UNDERCOVERAGE'),(190,'FAIL_MC_UNSTABLE'),(200,'PASS_SCENARIO_COVERAGE')])
def test_coverage_gate_uses_MC_uncertainty(successes,expected):
    r=coverage_gate(successes,200,.93,.02)
    assert r['coverage_gate']==expected
    assert 0<=r['coverage_MC_lower']<=r['coverage_MC_upper']<=1


def test_simulation_targets_both_absolute_per_card_contrasts_not_conditional_difference():
    plan=fast_plan()
    for d in (.025,.05):
        rows=simulate_case(120,20,d,.2,.6,0.,'shared_uniform',plan)
        assert all(abs(r['mean_estimate']-d)<.02 for r in rows)
        assert all(r['difference_per_eligible_card']==d for r in rows)
        assert all(r['MC_repetitions']==200 for r in rows)


def test_rare_event_negative_control_fails_t_coverage_bounded_interval_stays_wide():
    rows=simulate_case(60,10,.002,.005,.6,0.,'shared_uniform',fast_plan(),'rare_event_negative_control')
    t,b=rows
    assert t['zero_event_fraction']>.5 and t['coverage_gate']=='FAIL_UNDERCOVERAGE'
    assert b['observed_coverage']==1 and b['coverage_gate']=='PASS_SCENARIO_COVERAGE'
    assert not b['precision_gate'] and b['mean_half_width']>.8


def test_source_dropout_not_new_evaluation_units_and_detects_unmatched_arms():
    rows=simulate_case(120,20,.025,.2,.6,.35,'conditionally_independent',fast_plan())
    assert all(r['cards']==120 and r['families']==20 and not r['coverage_matched_population'] for r in rows)
    assert all(r['assertion_rate_b']<.4 for r in rows)


def test_precision_full_grid_probes_10_20_30_families_and_fails_confirmatory_gate(completed):
    report,out,_=completed;rows=list(csv.DictReader((out/'precision_diagnostics.csv').open()))
    assert {int(r['families']) for r in rows}=={10,20,30}
    assert {int(r['cards']) for r in rows}=={60,120,180}
    planning=[r for r in rows if r['scenario_kind']=='planning']
    assert {float(r['difference_per_eligible_card']) for r in planning}=={.025,.05}
    assert any(r['coverage_gate']=='FAIL_UNDERCOVERAGE' for r in rows)
    gate=report['precision_gate']
    assert not gate['confirmatory_precision_ready'] and not gate['precision_passed_in_simulated_grid']
    assert gate['coverage_validation_passed_in_simulated_grid'] and gate['failed_scenarios']


def test_family_definitions_are_more_than_renamed_IDs_and_no_heldout_scoring():
    ts={task_hash(t):t for t in tasks()};d=load('families.json')
    defs=[FamilyDefinition.model_validate(r) for r in d['definitions']]
    assignments=[CardAssignment.model_validate(r) for r in d['assignments']]
    a=audit_families(ts,defs,assignments)
    assert a['development_families']==8 and a['heldout_cards_scored']==0 and a['candidate_heldout_family_definitions']==3
    fake=defs[-1].model_dump();fake['family_id']='renamed';fake['mechanism_features']=defs[0].mechanism_features
    with pytest.raises(ValueError):audit_families(ts,defs+[FamilyDefinition.model_validate(fake)],assignments)
    direct=[x for x in assignments if x.family_id=='randomized_direct'];d=direct[0].model_dump();d['family_id']='randomized_mediation'
    wrong=[CardAssignment.model_validate(d) if x is direct[0] else x for x in assignments]
    with pytest.raises(ValueError):audit_families(ts,defs,wrong)


def test_meaningful_public_mechanism_variation_and_exposed_status(completed):
    ts=tasks()
    assert len({tuple(v.name for v in t.card.variables) for t in ts})>=5
    assert len({t.card.selection for t in ts})==2
    assert len({t.card.assignment for t in ts})==2
    assert completed[0]['family_audit']['independent_family_validation']=='DECISION_REQUIRED'


def test_no_truth_imports_or_parameters_in_public_admission_or_transport():
    for name in ('packets.py','admission.py','transport.py','families.py'):
        source=(ROOT/'src/sharecausal/preflight_v23'/name).read_text();tree=ast.parse(source)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any('certificates' in n or 'scoring' in n for n in imports)
        assert 'population_effect' not in source if name!='packets.py' else True


def test_isolated_mock_transport_and_receipt_payload_bindings():
    t=tasks()[0];price=FakePrice.model_validate(json.loads((ROOT/config()['fake_pricing']).read_text())[0])
    transport=MockTransport();answers,r=transport.record(t,price.model_id,'base',0,price,SimulatedBudget(1.,1.),8192,2048)
    assert transport.calls==1 and validate_record(t,answers,r.model_dump(),'base',0,price.model_id)
    for key,val in [('response_sha256','0'*64),('model_id','mock-family-b/fixture'),('repetition',1)]:
        d=r.model_dump();d[key]=val
        assert not validate_record(t,answers,d,'base',0,price.model_id)
    assert not validate_record(t,[],r.model_dump(),'base',0,price.model_id)


def test_immutable_seal_rejects_member_and_seal_replacement(tmp_path):
    write_once(tmp_path/'p.json',{'decision':'abstain'});m,h=seal(tmp_path,['p.json'],'seal.json')
    verify(tmp_path,m,'seal.json',h)
    with pytest.raises(FileExistsError):write_once(tmp_path/'seal.json',{})
    (tmp_path/'p.json').write_text('{}')
    with pytest.raises(ValueError):verify(tmp_path,m,'seal.json',h)
    (tmp_path/'seal.json').write_text('{}')
    with pytest.raises(ValueError):verify(tmp_path,m,'seal.json',h)


def test_entire_batch_and_definitions_exist_before_first_truth_open(tmp_path,monkeypatch):
    original=runner.load_json;seen=[]
    def guarded(path):
        if path==config()['private_truth']:
            m=json.loads((tmp_path/'prediction_seal.json').read_text())
            assert {'design_commitment.json','design_seal.json','comparison_batch.json','mock_records.json','fixed_contract_probes.json'}<=m.keys()
            for n,h in m.items():assert digest(tmp_path/n)==h
            assert json.loads((tmp_path/'design_commitment.json').read_text())['candidate_config']['fixed_margin_thresholds']==[0.,.25,.5,.75,1.]
            seen.append(True)
        return original(path)
    monkeypatch.setattr(runner,'load_json',guarded)
    monkeypatch.setattr(runner,'diagnostics',lambda p:[])
    runner.run_preflight(CONFIG,tmp_path)
    assert seen==[True]


def test_truth_swap_does_not_change_predictions_or_design_commitment(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'diagnostics',lambda p:[])
    _,out,_=runner.run_preflight(CONFIG,tmp_path/'a');original=runner.load_json
    def swapped(path):
        d=original(path)
        if path==config()['private_truth']:
            for k in ('scm','population_effect_exact'):d[0][k],d[1][k]=d[1][k],d[0][k]
        return d
    monkeypatch.setattr(runner,'load_json',swapped)
    _,other,_=runner.run_preflight(CONFIG,tmp_path/'b')
    for name in ('prediction_seal.json','design_seal.json','comparison_batch.json','mock_records.json','fixed_contract_probes.json'):
        assert (out/name).read_bytes()==(other/name).read_bytes()
    assert (out/'identification_audit.json').read_bytes()!=(other/'identification_audit.json').read_bytes()


def test_modified_prediction_during_truth_read_is_fatal(tmp_path,monkeypatch):
    original=runner.load_json
    def tamper(path):
        d=original(path)
        if path==config()['private_truth']:(tmp_path/'comparison_batch.json').write_text('{}')
        return d
    monkeypatch.setattr(runner,'load_json',tamper)
    with pytest.raises(runner.PreflightFailure) as exc:runner.run_preflight(CONFIG,tmp_path)
    assert exc.value.stage=='private_scoring' and not (tmp_path/'preflight_report.json').exists()


def test_fake_cost_16N_retry_and_zero_authorization(completed):
    report,_,_=completed;c=report['cost_envelopes']
    assert [r['planned_calls'] for r in c]==[320,960,1920]
    assert [r['maximum_attempts'] for r in c]==[640,1920,3840]
    assert all(r['pricing_label']=='FICTIONAL' and not r['live_feasible_under_current_authorization'] for r in c)
    assert not c[-1]['within_hypothetical_caps']


def test_full_output_JSON_CSV_and_manifest_source_integrity(completed):
    report,out,digest_value=completed
    assert len(list((out/'public').glob('*.json')))==9
    manifest=json.loads((out/'content_manifest.json').read_text())
    assert digest(out/'content_manifest.json')==digest_value
    for section in ('inputs','sources','outputs'):
        for path,value in manifest[section].items():assert digest((out if section=='outputs' else ROOT)/path)==value
    for p in out.rglob('*.json'):json.loads(p.read_text())
    for p in out.glob('*.csv'):assert list(csv.DictReader(p.open()))
    assert report['fixture_mode'] and report['real_llm_calls']==report['share_records_used']==report['openrouter_cost_usd']==0
    assert report['decision_required_count']==16 and not report['ready_to_freeze']
    assert report['mock_transport_calls']==144


def test_cli_in_real_subprocess_blocks_network_keys_SHARE_and_credentials_and_repeats(tmp_path):
    hook=tmp_path/'hook';hook.mkdir()
    (hook/'sitecustomize.py').write_text('''import sys,os
_original=os._Environ.__getitem__
def no_keys(self,key):
    if key in ('OPENROUTER_API_KEY','SHARECAUSAL_LIVE_LLM_APPROVED'):raise RuntimeError('key access forbidden')
    return _original(self,key)
os._Environ.__getitem__=no_keys
def audit(event,args):
    if event.startswith('socket.'):raise RuntimeError('socket attempt forbidden')
    if event=='open' and isinstance(args[0],str):
        p=args[0].lower()
        if any(s in p for s in ('share-research/private','/.env','/.aws/','/.ssh/')) or p.endswith(('.dta','.zip','.sav','.parquet','.feather')):
            raise RuntimeError('credential or data IO forbidden')
sys.addaudithook(audit)
''')
    env={'PYTHONPATH':str(hook)+os.pathsep+str(ROOT/'src'),'OPENROUTER_API_KEY':MARKER,'SHARECAUSAL_LIVE_LLM_APPROVED':'1'}
    results=[cli(tmp_path,env=env) for _ in range(2)]
    for r in results:assert r.returncode==0,r.stderr
    outputs=[json.loads(r.stdout) for r in results]
    assert outputs[0]['content_manifest_sha256']==outputs[1]['content_manifest_sha256']
    assert outputs[0]['results_dir']!=outputs[1]['results_dir']
    for d in outputs:
        for p in Path(d['results_dir']).rglob('*'):
            if p.is_file():assert MARKER not in p.read_text()
    assert cli(tmp_path,extra=['--require-ready']).returncode==3
    assert cli(tmp_path,extra=['--require-precision']).returncode==4


@pytest.mark.parametrize('change,expected_stage',[
    ('missing_packet','public_packets'),('corrupted_packet','public_packets'),('duplicate_card','public_packets'),
    ('hostile_packet_key','public_packets'),('hostile_config_key','configuration'),
    ('authorized_spending','configuration'),('approved_decision','decisions'),
    ('missing_price','pricing'),('bad_truth','private_scoring'),('credential_path','public_packets')])
def test_cli_invalid_inputs_actionable_redacted_nonzero(tmp_path,change,expected_stage):
    cfg=config()
    if change=='missing_packet':cfg['public_tasks']=str(tmp_path/'missing.json')
    elif change=='hostile_config_key':cfg[MARKER]=MARKER
    elif change=='authorized_spending':cfg['authorized_openrouter_usd']=1
    elif change=='credential_path':cfg['public_tasks']='~/share-research/private/'+MARKER+'.dta'
    elif change=='approved_decision':
        d=load('decisions.json');d[0]['status']='APPROVED';p=tmp_path/'decisions.json';p.write_text(json.dumps(d));cfg['decisions']=str(p)
    elif change=='missing_price':
        d=json.loads((ROOT/cfg['fake_pricing']).read_text());d[0]['input_usd_per_million']=None
        p=tmp_path/'prices.json';p.write_text(json.dumps(d));cfg['fake_pricing']=str(p)
    elif change=='bad_truth':
        d=load('private_truth.json');d[0]['population_effect_exact']='0'
        p=tmp_path/'truth.json';p.write_text(json.dumps(d));cfg['private_truth']=str(p)
    else:
        d=load('public_tasks.json')
        if change=='corrupted_packet':d[0]['card']['title']=MARKER
        elif change=='hostile_packet_key':d[0][MARKER]=MARKER
        else:d.append(d[0])
        p=tmp_path/'packets.json';p.write_text(json.dumps(d));cfg['public_tasks']=str(p)
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg))
    r=cli(tmp_path,p)
    assert r.returncode==2 and not r.stdout,r.stderr
    assert json.loads(r.stderr)['stage']==expected_stage and MARKER not in r.stderr
    assert json.loads(r.stderr)['live_fallback'] is False


def test_original_frozen_protocol_check_unchanged():
    r=subprocess.run([sys.executable,'scripts/check_protocol_freeze.py'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0 and '14 checked' in r.stdout


def test_same_underlying_card_cannot_be_counted_twice_by_changing_credibility_wrapper():
    t=tasks()[0];d=t.model_dump();d['credibility']=[];other=rehash(d)
    assert task_hash(t)!=task_hash(other)
    with pytest.raises(ValueError):exports([t,other])


@pytest.mark.parametrize('kind',['assignment','exclusion'])
def test_documented_public_facts_require_bound_support(kind):
    t=by_title('Thermostatic') if kind=='assignment' else by_title('isolated')
    d=t.model_dump();d['card']['evidence']=[e for e in d['card']['evidence'] if e['kind']!=kind]
    with pytest.raises(ValueError):rehash(d)


def test_declared_parent_must_have_actual_functional_influence():
    with pytest.raises(ValueError):BinaryNode(name='Y',parents=['A'],probability_one=['1/2','1/2'])


def test_design_input_mutation_after_prediction_seal_is_fatal(tmp_path,monkeypatch):
    cfg_file=tmp_path/'config.yaml';cfg_file.write_text(CONFIG.read_text());original=runner.load_json
    def tamper(path):
        d=original(path)
        if path==config()['private_truth']:cfg_file.write_text(cfg_file.read_text()+'\n# changed after seal\n')
        return d
    monkeypatch.setattr(runner,'load_json',tamper);monkeypatch.setattr(runner,'diagnostics',lambda p:[])
    with pytest.raises(runner.PreflightFailure) as exc:runner.run_preflight(cfg_file,tmp_path/'out')
    assert exc.value.stage=='artifacts' and not (tmp_path/'out/preflight_report.json').exists()


def test_directional_claim_on_private_zero_alone_is_uncertified_not_supported_zero_error():
    ws=worlds();ts={task_hash(t):t for t in tasks()};certs={w.world_id:certify(w,ts[w.public_task_hash],ws) for w in ws}
    h=next(w.public_task_hash for w in ws if w.family_id=='proxy_common_drive')
    result=score({k:'positive' if k==h else 'abstain' for k in ts},ws,certs,ts,.1)
    assert result['known_unjustified_count']==0 and result['unresolved_additional_count']==1
    assert result['component_counts']['supported_structural_zero_violation']==0


@pytest.mark.parametrize('mutation',['unapproved_policy','missing_precision','infeasible_contrast'])
def test_missing_or_posthoc_design_options_are_rejected(mutation):
    d=config()
    if mutation=='unapproved_policy':d['candidate_primary_policy']='choose_by_scores'
    elif mutation=='missing_precision':del d['precision']['minimum_coverage']
    else:d['precision']['differences_per_card']=[.5]
    with pytest.raises(ValueError):runner.Config.model_validate(d)


def test_network_veto_blocks_attempt_without_reading_credentials(monkeypatch):
    from sharecausal.preflight.runner import network_veto
    monkeypatch.setenv('OPENROUTER_API_KEY',MARKER)
    with network_veto():
        with pytest.raises(RuntimeError):socket.getaddrinfo('example.invalid',443)
        with pytest.raises(RuntimeError):socket.create_connection(('example.invalid',443))


@pytest.mark.parametrize('policy',POLICIES)
def test_supported_arrow_cannot_be_silently_omitted_from_complete_hypothesis(policy):
    t=by_title('Thermostatic');note=t.card.evidence[0]
    p=proposal([Edge(source='U',target='Y',evidence_id=note.evidence_id)])
    a,_=audit(t,p.model_dump(),policy)
    assert not a['admitted']
    assert 'contradicted' in a['edge_tiers']
    assert ['A','Y'] in a['omitted_supported_edges']


def test_public_exclusion_vetoes_indirect_causal_path():
    t=by_title('isolated');d=t.model_dump()
    d['card']['variables'].insert(-1,dict(name='M',role='mediator',time=2,observed=True))
    t=rehash(d);note=t.card.evidence[0]
    p=proposal([edge(t,'X','Y'),Edge(source='A',target='M',evidence_id=note.evidence_id),
                Edge(source='M',target='Y',evidence_id=note.evidence_id)])
    for policy in POLICIES:
        a,_=audit(t,p.model_dump(),policy)
        assert a['format_valid'] and not a['admitted'] and 'contradicted' in a['edge_tiers']


@pytest.mark.parametrize('conflict',['assignment','exclusion','timing','contradictory_statements'])
def test_incoherent_stipulated_public_facts_rejected_before_mock_capture(conflict):
    t=by_title('Thermostatic');d=t.model_dump()
    if conflict=='assignment':
        d['card']['evidence'].append(make_evidence(kind='edge',source='U',target='A',relation='supported',statement='A hidden source sets the switch.'))
    elif conflict=='exclusion':
        d['card']['structural_exclusion']=True
        d['card']['evidence'].append(make_evidence(kind='exclusion',source='A',target='Y',relation='supported',statement='All causal paths are excluded.'))
    elif conflict=='timing':
        d['card']['evidence'].append(make_evidence(kind='edge',source='Y',target='A',relation='supported',statement='The later detector causes an earlier switch.'))
    else:
        d['card']['evidence'].append(make_evidence(kind='edge',source='A',target='Y',relation='contradicted',statement='The supported lead is also declared impossible.'))
    with pytest.raises(ValueError):rehash(d)
