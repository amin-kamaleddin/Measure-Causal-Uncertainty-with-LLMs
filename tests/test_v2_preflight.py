"""Scientific and privacy acceptance tests for the v2.2 OFFLINE preflight."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import yaml

from sharecausal.preflight.contracts import (PublicWorldCardV2, public_payload, public_bytes,
    packet_hash, public_exports, sha, make_evidence, canonical)
from sharecausal.preflight.certificates import PrivateWorldTruthV2, certify, observational_joint, intervention_effect
from sharecausal.preflight.comparison import (Edge, GraphAnswer, edge_support, graph_audit, no_llm_baseline,
    majority, calibrate_public_margins)
from sharecausal.preflight.budget import FakePrice, cost_envelope, SimulatedBudget
from sharecausal.preflight.provenance import MockTransport, ProviderResponseProvenance, validate_record
from sharecausal.preflight.design import PrecisionConfig, precision_diagnostics, SplitUnit, lint_split
from sharecausal.preflight.runner import Config, run_preflight, network_veto, safe_path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests/fixtures/v2_preflight'
CONFIG = ROOT / 'configs/v2_preflight_candidate.yaml'
MARKER = 'UNTRUSTED_SECRET_FIELD_DO_NOT_EXPORT_722'


def load(name): return json.loads((FIX/name).read_text())
def cards(): return [PublicWorldCardV2.model_validate(x) for x in load('public_packets.json')]
def worlds(): return [PrivateWorldTruthV2.model_validate(x) for x in load('private_truth_fixture.json')]
def config(): return yaml.safe_load(CONFIG.read_text())
def price(): return FakePrice.model_validate(load('fake_pricing.json')[0])

def rehash_card(data):
    data = deepcopy(data); data.pop('card_id', None)
    data['card_id'] = 'card-'+sha(data)
    return PublicWorldCardV2.model_validate(data)


def run_cli(tmp_path, *, config_file=CONFIG, extra=(), env_extra=None):
    env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT/'src')
    if env_extra: env.update(env_extra)
    return subprocess.run([sys.executable, 'scripts/v2_preflight_offline.py', '--config', str(config_file),
            '--results-dir', str(tmp_path/'output'), '--new-run', *extra], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=60)


def test_exact_witness_observational_equivalence_and_intervention_difference():
    a,b,*_ = worlds(); c=cards()[0]
    expected = {(0,0):F(2,5),(0,1):F(1,10),(1,0):F(1,10),(1,1):F(2,5)}
    assert observational_joint(a.true_scm) == observational_joint(b.true_scm) == expected
    assert intervention_effect(a.true_scm) == F(3,5)
    assert intervention_effect(b.true_scm) == 0
    assert sum(expected.values()) == 1
    assert a.public_card_hash == b.public_card_hash == packet_hash(c)
    assert certify(a,c,worlds())['status'] == certify(b,c,worlds())['status'] == 'provably_nonidentified'
    assert public_bytes(c) == public_bytes(cards()[0])


def test_public_allowlist_bytes_filenames_and_order_independent_of_private_truth():
    cs=cards(); ws=worlds()
    exports=public_exports(cs)
    assert exports == public_exports(list(reversed(cs)))
    for name,raw in exports.items():
        assert name == 'packet_'+hashlib.sha256(raw).hexdigest()+'.json'
        for w in ws:
            for secret in (w.world_id,w.mechanism_family,w.narrative_style,str(w.generator_seed)):
                assert secret.encode() not in raw and secret not in name
        for forbidden in (b'true_scm',b'population_effect',b'potential_outcomes',b'world-a',b'p_a_u0'):
            assert forbidden not in raw
    for w in ws:
        with pytest.raises(ValueError): public_payload(w)
    with pytest.raises(ValueError): public_payload({'card':cs[0].model_dump(),'true_ate':42})
    assert len(exports)==4 and len(ws)==5  # ambiguous worlds share one model input, not two task IDs


@pytest.mark.parametrize('change', ['unknown_private','bad_role','U_observed','backwards','modified_evidence'])
def test_public_schema_rejects_bad_metadata_and_mutated_evidence(change):
    data=cards()[0].model_dump()
    if change=='unknown_private': data['true_ate']=MARKER
    elif change=='bad_role': data['variables'][0]['role']='outcome'
    elif change=='U_observed': data['variables'][2]['observed']=True
    elif change=='backwards': data['variables'][0]['time']=5
    else: data['evidence'][0]['statement']=MARKER
    with pytest.raises(ValueError): rehash_card(data)


def test_semantic_marker_supplement_does_not_replace_field_allowlist():
    data=cards()[0].model_dump();data['title']='generator_seed private marker'
    c=rehash_card(data)
    with pytest.raises(ValueError):public_bytes(c)


def test_missing_or_invalid_certificate_stays_uncertified_despite_latent_labels():
    w=worlds()[-1]; c=cards()[-1]
    assert w.backdoor_failure and w.structural_zero_graph
    result=certify(w,c,worlds())
    assert result['status']=='uncertified' and not result['supported_structural_zero']
    bad=worlds()[0].model_dump();bad['certificate']['witness_world_ids']=['world-a','world-missing']
    assert certify(PrivateWorldTruthV2.model_validate(bad),cards()[0],worlds())['status']=='uncertified'


def test_witness_changed_joint_or_additional_information_cannot_certify():
    ws=worlds();bad=ws[1].model_dump();bad['true_scm']['p_a_u0']=.3
    altered=PrivateWorldTruthV2.model_validate(bad)
    assert certify(ws[0],cards()[0],[ws[0],altered,*ws[2:]])['status']=='uncertified'
    # A private randomized mechanism is insufficient if that assumption is not public.
    data=ws[0].model_dump();data['certificate']['kind']='randomized_complete';data['certificate']['witness_world_ids']=[]
    assert certify(PrivateWorldTruthV2.model_validate(data),cards()[0],ws)['status']=='uncertified'


def test_extra_observed_variable_exceeds_binary_certificate_scope():
    data=cards()[0].model_dump();data['variables'].append(dict(name='X',role='baseline',time=0,observed=True))
    c=rehash_card(data);ws=worlds();w=ws[0].model_dump()
    w['public_card_hash']=packet_hash(c);w['certificate']['public_card_hash']=packet_hash(c)
    assert certify(PrivateWorldTruthV2.model_validate(w),c,ws)['status']=='uncertified'


def test_identified_and_supported_zero_certificates():
    ws=worlds();cs=cards()
    assert certify(ws[2],cs[1],ws)['status']=='identified'
    zero=certify(ws[3],cs[2],ws)
    assert zero['status']=='identified' and zero['supported_structural_zero']
    assert zero['functional']=='0'


def test_population_and_finite_truth_cannot_be_fabricated():
    data=worlds()[0].model_dump();data['population_effect']=.9
    with pytest.raises(ValueError):PrivateWorldTruthV2.model_validate(data)
    data=worlds()[0].model_dump();data['finite_sample_effect']=.9
    with pytest.raises(ValueError):PrivateWorldTruthV2.model_validate(data)


def test_public_baseline_has_identical_input_and_no_private_dependencies():
    from sharecausal.preflight import comparison, contracts, provenance
    cs=cards()
    assert len(no_llm_baseline(cs[0]))==2
    assert no_llm_baseline(cs[2])[0].edges==[]
    for mod in (comparison,contracts,provenance):
        source=inspect.getsource(mod)
        for excluded in ('from .certificates','PrivateWorldTruthV2','generator_seed','from .scoring'):
            # The public text marker lint is a supplemental explicit check, not a truth import.
            if mod is contracts and excluded=='generator_seed':continue
            assert excluded not in source


def test_all_five_evidence_rubric_categories_and_format_separation():
    unknown,randomized,zero,_=cards()
    e=lambda c,i:Edge(source=c.evidence[i].source,target=c.evidence[i].target,evidence_id=c.evidence[i].evidence_id)
    assert edge_support(randomized,e(randomized,0))=='supported'
    assert edge_support(zero,e(zero,0))=='contradicted'
    assert edge_support(unknown,e(unknown,0))=='ambiguous'
    assert edge_support(unknown,Edge(source='A',target='Y',evidence_id=unknown.evidence[-1].evidence_id))=='unrelated'
    assert edge_support(unknown,Edge(source='A',target='Y',evidence_id='unverified hypothesis'))=='invalid_citation'
    answer=GraphAnswer(schema_version='preflight-graph-2.2',edges=[e(unknown,0)],adjustment=[],abstain=False,abstention_reason='').model_dump()
    strict,_=graph_audit(unknown,answer,'supported_only')
    permissive,_=graph_audit(unknown,answer,'retain_alternatives')
    assert strict['format_valid'] and not strict['admitted'] and permissive['admitted']
    answer['edges'][0]['evidence_id']='missing'
    assert graph_audit(unknown,answer,'retain_alternatives')[0]['reason']=='evidence_policy_veto'
    answer['edges'][0]['target']='U'
    assert not graph_audit(unknown,answer,'retain_alternatives')[0]['format_valid']


def test_majority_tie_absence_abstention_multiple_graphs_and_empty_exclusion():
    c=cards()[1];g=no_llm_baseline(c)[0].model_dump();z=cards()[2]
    assert majority(c,{'a':[g]},['a','b'],'supported_only')['reason']=='no_strict_majority'
    assert majority(c,{'a':[g,g]},['a'],'supported_only')['source_audit']['a']=='absent_or_multiple_graphs'
    abst=GraphAnswer(schema_version='preflight-graph-2.2',edges=[],adjustment=[],abstain=True,abstention_reason='unclear').model_dump()
    assert majority(c,{'a':[abst]},['a'],'supported_only')['reason']=='no_strict_majority'
    zero=majority(z,{'a':[no_llm_baseline(z)[0].model_dump()]},['a'],'supported_only')
    assert zero['structural_zero'] and zero['decision']=='abstain'


def test_directed_majority_cycle_is_not_silently_repaired():
    data=cards()[0].model_dump()
    for name in ['X','M','C']:data['variables'].append(dict(name=name,role='baseline',time=0,observed=True))
    for a,b in [('X','M'),('M','C'),('C','X')]:
        data['evidence'].append(make_evidence(kind='edge',source=a,target=b,relation='ambiguous',statement='Fictional unordered baseline possibility.'))
    card=rehash_card(data)
    edges={ (e.source,e.target):Edge(source=e.source,target=e.target,evidence_id=e.evidence_id) for e in card.evidence if e.kind=='edge'}
    ballots={}
    for name,pairs in zip('abc',[[('X','M'),('M','C')],[('M','C'),('C','X')],[('C','X'),('X','M')]]):
        ballots[name]=[GraphAnswer(schema_version='preflight-graph-2.2',edges=[edges[p] for p in pairs],adjustment=[],abstain=False,abstention_reason='').model_dump()]
    result=majority(card,ballots,list('abc'),'retain_alternatives')
    assert result['decision']=='abstain' and result['reason']=='cyclic_majority_not_repaired' and result['graph'] is None


def test_scoring_denominators_nonidentification_and_structural_zero_are_separate():
    from sharecausal.preflight.scoring import score_contract
    cs={packet_hash(c):c for c in cards()};ws=worlds()
    positive=score_contract(canonical({h:'positive' for h in cs}),ws,cs,.1)
    abstain=score_contract(canonical({h:'abstain' for h in cs}),ws,cs,.1)
    assert positive['eligible_cards']==4 and positive['world_probes']==5
    assert positive['assertions']==4 and positive['known_unjustified_rate_lower']==.5
    assert positive['unjustified_rate_upper_with_uncertified']==.75
    comp=positive['components_per_all_eligible_cards']
    assert comp['assertion_on_proven_nonidentification']==.25
    assert comp['supported_structural_zero_violation']==.25
    assert positive['uncertified_assertions']==1
    assert abstain['selective_wrong_sign_risk_identified_nonnull'] is None
    assert abstain['abstention_rate']==1 and abstain['abstention_penalized_candidate_loss']['0.25']==.25
    assert next(t for t in positive['tasks'] if t['card_hash']==ws[0].public_card_hash)['compatible_with_all_prespecified_worlds'] is False
    with pytest.raises(ValueError):score_contract(canonical({}),ws,cs,.1)


@pytest.mark.parametrize('field', ['mechanism_family','narrative_style','seeds','card_hash'])
def test_split_overlap_lint(field):
    data=load('split_fixture.json')
    data[2][field]=data[0][field]
    with pytest.raises(ValueError):lint_split([SplitUnit.model_validate(d) for d in data])


def test_precision_does_not_count_repeated_seeds_as_independent_units():
    data=config()['precision'];data.update(card_counts=[20],family_counts=[5],assertion_rates=[.5],repetitions=100,probes_per_card=1)
    first=precision_diagnostics(PrecisionConfig.model_validate(data))
    data['probes_per_card']=100
    repeated=precision_diagnostics(PrecisionConfig.model_validate(data))
    for a,b in zip(first,repeated):
        assert a['mean_family_cluster_se']==b['mean_family_cluster_se']
        assert b['mean_misleading_seed_independent_se']==pytest.approx(a['mean_misleading_seed_independent_se']/10)
        assert a['mean_approximate_95_half_width']>0


def test_unknown_fake_prices_capabilities_and_token_context_fail_closed():
    data=load('fake_pricing.json')[0]
    for key in ['input_usd_per_million','output_usd_per_million','supports_json_schema']:
        bad={**data,key:None}
        with pytest.raises(ValueError):FakePrice.model_validate(bad).validate_known()
    with pytest.raises(ValueError):price().cost(20000,100)
    row=cost_envelope([price()]*1,20,2,2,1,8192,1024,8,25)
    assert row['planned_calls']==80 and row['maximum_attempts']==160
    assert row['hypothetical_total_usd']==1.6384 and not row['live_feasible_under_current_authorization']


def test_budget_reservation_reconciliation_and_global_and_per_model_stops():
    b=SimulatedBudget(1,.6);r=b.reserve('a',.5)
    with pytest.raises(ValueError):b.reserve('a',.2)
    b.reconcile(r,.4)
    assert float(b.spent['a'])==.4
    assert b.stopped
    b=SimulatedBudget(1,1);r=b.reserve('a',.6)
    with pytest.raises(ValueError):b.reserve('b',.5)
    b=SimulatedBudget(1,1);r=b.reserve('a',.1)
    with pytest.raises(ValueError):b.reconcile(r,.2)
    assert b.stopped and float(b.spent['a'])==.2


def receipt():
    card=cards()[1];p=price();b=SimulatedBudget(25,8)
    a,record=MockTransport().record(card,p.model_id,'base',0,p,b,8192,1024,1)
    return card,a,record


def test_mock_provider_and_fixture_use_same_source_neutral_graph_validation():
    c,a,p=receipt()
    status,_=validate_record(c,a,p.model_dump(),'supported_only')
    fixture={'source_type':'offline_fixture','fixture_id':'dummy_record','prompt_sha256':p.prompt_sha256,'response_sha256':p.response_sha256}
    assert validate_record(c,a,fixture,'supported_only')[0]==status
    assert status['format_valid'] and status['admitted']
    assert p.source_type=='provider_response' and p.record_origin=='mock_transport'


@pytest.mark.parametrize('field,value', [
    ('prompt_sha256','0'*64),('response_sha256','0'*64),('card_sha256','0'*64),
    ('schema_sha256','0'*64),('evidence_sha256','0'*64),('provider_fallback_allowed',True),
    ('retry_count',5),('output_tokens',20000),('response_timestamp','2026-10-16'),
    ('record_origin','recorded_provider'),('failure','timeout'),('returned_model_id','other/model'),
])
def test_malformed_future_provenance_never_votes(field,value):
    c,a,p=receipt();data=p.model_dump();data[field]=value
    result,_=validate_record(c,a,data,'supported_only')
    assert not result['admitted'] and result['reason']=='provenance_invalid'
    assert MARKER not in json.dumps(result)


def test_calibration_is_public_only_and_cannot_create_assertions():
    with pytest.raises(ValueError):calibrate_public_margins([dict(method='a',decision='abstain',margin=0.,true_ate=1)],.5)
    result=calibrate_public_margins([dict(method='a',decision='abstain',margin=100.)],.5)
    assert result['a']['achieved_assertion_rate']==0 and not result['a']['target_feasible']


def test_private_and_credentials_paths_rejected_before_resolving(monkeypatch):
    def forbidden(*args):raise AssertionError('forbidden path inspected')
    monkeypatch.setattr(Path,'resolve',forbidden)
    for p in ['~/share-research/private','.env','/tmp/private/data.json','/tmp/secret.dta',
              '~/.config/openrouter/credentials.json','/tmp/api_key.json']:
        with pytest.raises(ValueError):safe_path(p)


def test_network_guard_rejects_outbound_even_with_environment_key(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY',MARKER)
    with network_veto():
        with pytest.raises(RuntimeError):socket.create_connection(('example.invalid',443))
        with pytest.raises(RuntimeError):socket.getaddrinfo('example.invalid',443)


def test_truth_file_not_opened_until_complete_public_batch_sealed(tmp_path,monkeypatch):
    from sharecausal.preflight import runner
    original=runner.load_json
    seen=[]
    def guarded(path):
        if path==config()['private_truth']:
            assert (tmp_path/'prediction_seal.json').is_file()
            seal=json.loads((tmp_path/'prediction_seal.json').read_text())
            assert 'comparison_batch.json' in seal and 'decision_contract_probes.json' in seal
            for name,digest in seal.items():assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==digest
            seen.append('truth_opened')
        return original(path)
    monkeypatch.setattr(runner,'load_json',guarded)
    report,_,_=run_preflight(CONFIG,tmp_path)
    assert seen==['truth_opened'] and report['ready_to_freeze'] is False


def test_real_cli_outputs_reproducible_manifests_no_network_and_no_key_reads(tmp_path):
    # A subprocess startup hook blocks actual socket attempts and secret-file/key access.
    hook=tmp_path/'hook';hook.mkdir()
    (hook/'sitecustomize.py').write_text('''import sys,os
_original=os._Environ.__getitem__
def no_keys(self,key):
    if key in ('OPENROUTER_API_KEY','SHARECAUSAL_LIVE_LLM_APPROVED'):raise RuntimeError('key access forbidden')
    return _original(self,key)
os._Environ.__getitem__=no_keys
def audit(event,args):
    if event.startswith('socket.'):raise RuntimeError('outbound attempt forbidden')
    if event=='open' and isinstance(args[0],str):
        p=args[0]
        if 'share-research/private' in p or '/.env' in p or '/.aws/' in p:raise RuntimeError('private IO forbidden')
sys.addaudithook(audit)
''')
    env={'PYTHONPATH':str(hook)+os.pathsep+str(ROOT/'src'),'OPENROUTER_API_KEY':MARKER,'SHARECAUSAL_LIVE_LLM_APPROVED':'1'}
    results=[run_cli(tmp_path,env_extra=env) for _ in range(2)]
    for r in results:assert r.returncode==0,r.stderr
    payloads=[json.loads(r.stdout) for r in results]
    assert payloads[0]['results_dir']!=payloads[1]['results_dir']
    assert payloads[0]['content_manifest_sha256']==payloads[1]['content_manifest_sha256']
    out=Path(payloads[0]['results_dir']);report=json.loads((out/'preflight_report.json').read_text())
    assert report['fixture_mode'] is True and report['protocol_status']=='candidate_not_frozen'
    assert report['real_llm_calls']==report['openrouter_cost_usd']==report['share_records_used']==0
    assert report['decision_required_count']==16 and report['mock_transport_calls']==64
    assert not report['ready_to_freeze'] and report['diagnostics_only']
    manifest=json.loads((out/'content_manifest.json').read_text())
    for rel,digest in manifest['outputs'].items():assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==digest
    for file in out.rglob('*'):
        if file.is_file():assert MARKER not in file.read_text()
    assert len(list((out/'public').glob('*.json')))==4
    assert len(report['precision'])==36
    assert [c['planned_calls'] for c in report['cost_envelopes']]==[320,960,1920]
    records=json.loads((out/'mock_provider_records.json').read_text())
    for card_hash in {r['card_hash'] for r in records}:
        for variant in ['base','challenge']:
            subset=[r for r in records if r['card_hash']==card_hash and r['variant']==variant]
            assert len({r['provenance']['prompt_sha256'] for r in subset})==1
    assert run_cli(tmp_path,extra=['--require-ready']).returncode==3


@pytest.mark.parametrize('change', ['missing_quantity','unknown_scientific_option','authorized_spend','unknown_price','frozen_claim'])
def test_cli_invalid_inputs_fail_closed_with_redacted_nonzero_diagnostics(tmp_path,change):
    cfg=config()
    if change=='missing_quantity':del cfg['input_token_reservation']
    elif change=='unknown_scientific_option':cfg['support_policies']=['choose_after_scores']
    elif change=='authorized_spend':cfg['authorized_openrouter_usd']=1
    elif change=='frozen_claim':cfg['protocol_status']='frozen'
    else:
        data=load('fake_pricing.json');data[0]['input_usd_per_million']=None
        path=tmp_path/'prices.json';path.write_text(json.dumps(data));cfg['fake_pricing']=str(path)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    result=run_cli(tmp_path,config_file=path)
    assert result.returncode==2 and MARKER not in result.stderr
    assert not result.stdout


def test_cli_redacts_hostile_configuration_field_names(tmp_path):
    cfg=config();cfg[MARKER]=MARKER
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    result=run_cli(tmp_path,config_file=path)
    assert result.returncode==2 and MARKER not in result.stderr and not result.stdout


def test_content_manifest_records_all_sources_and_preserves_v1(tmp_path):
    report,out,_=run_preflight(CONFIG,tmp_path)
    manifest=json.loads((out/'content_manifest.json').read_text())
    for rel,digest in manifest['sources'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
    result=subprocess.run([sys.executable,'scripts/check_protocol_freeze.py'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0 and '14 checked' in result.stdout


def test_dependency_free_graph_checks_against_exhaustive_three_node_topologies():
    from itertools import permutations,combinations
    from sharecausal.preflight.graph_utils import is_acyclic,has_path
    nodes=('A','X','Y');possible=[(a,b) for a in nodes for b in nodes if a!=b]
    for size in range(7):
        for edges in combinations(possible,size):
            topological=any(all(order.index(a)<order.index(b) for a,b in edges) for order in permutations(nodes))
            assert is_acyclic(edges)==topological
            assert has_path(edges)==(('A','Y') in edges or (('A','X') in edges and ('X','Y') in edges))
    assert not is_acyclic([('A','A')])


def test_randomization_fact_vetoes_contradictory_incoming_assignment_arrow():
    c=cards()[1]
    graph=GraphAnswer(schema_version='preflight-graph-2.2',edges=[Edge(source='U',target='A',evidence_id=c.evidence[0].evidence_id)],adjustment=[],abstain=False,abstention_reason='')
    for policy in ('supported_only','retain_alternatives'):
        audit,_=graph_audit(c,graph.model_dump(),policy)
        assert audit['format_valid'] and not audit['admitted'] and 'contradicted' in audit['support']


def test_supported_public_information_can_invalidate_an_otherwise_matching_witness():
    cs=cards();data=cs[0].model_dump();e=data['evidence'][0]
    replacement={k:v for k,v in e.items() if k!='evidence_id'};replacement['relation']='supported'
    data['evidence'][0]={'evidence_id':'ev-'+sha(replacement),**replacement}
    c=rehash_card(data);ws=worlds();pair=[]
    for w in ws[:2]:
        d=w.model_dump();d['public_card_hash']=packet_hash(c);d['certificate']['public_card_hash']=packet_hash(c)
        pair.append(PrivateWorldTruthV2.model_validate(d))
    assert certify(pair[0],c,pair)['status']=='uncertified'
    with pytest.raises(ValueError):certify(pair[1],c,pair)


def test_identified_wrong_sign_component_and_unknown_bounds_use_card_denominators():
    from sharecausal.preflight.scoring import score_contract
    ws=worlds();data=ws[2].model_dump()
    data['true_scm'].update(p_y0=.8,p_y1=.2);data['population_effect']=-.6;data['finite_sample_effect']=-.6
    data['potential_outcomes']=[list(reversed(p)) for p in data['potential_outcomes']]
    ws[2]=PrivateWorldTruthV2.model_validate(data)
    cs={packet_hash(c):c for c in cards()}
    result=score_contract(canonical({h:'positive' for h in cs}),ws,cs,.1)
    assert result['components_per_all_eligible_cards']['wrong_sign_identified_nonnull']==.25
    assert result['known_unjustified_rate_lower']==.75 and result['unjustified_rate_upper_with_uncertified']==1


def test_changing_compatible_private_SCMs_cannot_change_public_predictions_or_seal(tmp_path,monkeypatch):
    from sharecausal.preflight import runner
    _,first,_=run_preflight(CONFIG,tmp_path/'first')
    original=runner.load_json
    def swap_hidden(path):
        data=original(path)
        if path==config()['private_truth']:
            for key in ('true_scm','population_effect','potential_outcomes','finite_sample_effect','backdoor_failure','structural_zero_graph'):
                data[0][key],data[1][key]=data[1][key],data[0][key]
        return data
    monkeypatch.setattr(runner,'load_json',swap_hidden)
    _,second,_=run_preflight(CONFIG,tmp_path/'second')
    for name in ['comparison_batch.json','mock_provider_records.json','decision_contract_probes.json','prediction_seal.json','public_manifest.json']:
        assert (first/name).read_bytes()==(second/name).read_bytes()
    assert (first/'identification_audit.json').read_bytes()!=(second/'identification_audit.json').read_bytes()


def test_effective_output_limit_cannot_exceed_reserved_tokens():
    c,a,p=receipt();data=p.model_dump()
    for key in ('requested_settings','effective_settings'):data[key]['max_output_tokens']=20000
    assert validate_record(c,a,data,'supported_only')[0]['reason']=='provenance_invalid'
