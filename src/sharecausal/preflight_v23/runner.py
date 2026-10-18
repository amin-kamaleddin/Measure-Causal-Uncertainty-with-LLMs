"""Offline-only v2.3 orchestration, pre-capture definitions and full-batch sealing."""
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import importlib.metadata
import json
import platform
import tempfile
from typing import Literal
import yaml
from pydantic import Field,model_validator
from sharecausal.preflight.contracts import Strict,canonical,sha
from sharecausal.preflight.runner import network_veto,safe_path,write_csv
from sharecausal.preflight.budget import FakePrice,SimulatedBudget,cost_envelope
from . import VERSION
from .packets import PublicTask,exports,task_hash,RUBRIC_VERSION,POLICIES
from .admission import comparisons,Proposal
from .families import FamilyDefinition,CardAssignment,audit_families
from .transport import MockTransport,validate_record
from .precision import PrecisionPlan,diagnostics
from .sealing import write_once,seal,verify,digest,definition_commitment

ROOT=Path(__file__).resolve().parents[3]
STAGES=('configuration','public_packets','design_families','pricing','decisions','output',
        'mock_prediction','prediction_seal','private_scoring','precision','artifacts')


class PreflightFailure(Exception):
    def __init__(self,stage,invalid=True):
        self.stage=stage;self.invalid=invalid
        super().__init__('offline_preflight_failed')


@contextmanager
def stage(name):
    try:yield
    except PreflightFailure:raise
    except Exception as exc:raise PreflightFailure(name,isinstance(exc,(ValueError,OSError,TypeError,KeyError))) from exc


class Decision(Strict):
    id:str
    status:Literal['DECISION_REQUIRED']
    review_recommendation_not_approval:str
    required_gate:str


class Config(Strict):
    schema_version:Literal['preflight-config-2.3']
    protocol_status:Literal['candidate_not_frozen']
    authorized_openrouter_usd:int=Field(ge=0,le=0)
    public_tasks:str
    private_truth:str
    families:str
    decisions:str
    fake_pricing:str
    candidate_primary_policy:Literal['credible_alternatives']
    policy_approval:Literal['DECISION_REQUIRED']
    models:list[str]=Field(min_length=4,max_length=4)
    variants:list[Literal['base','challenge']]
    repetitions:int=Field(ge=1,le=10)
    max_retries:int=Field(ge=0,le=3)
    input_token_reservation:int=Field(gt=0)
    output_token_reservation:int=Field(gt=0)
    hypothetical_global_cap_usd:float=Field(ge=0)
    hypothetical_per_model_cap_usd:float=Field(ge=0)
    planning_card_counts:list[int]
    assertion_target:float=Field(gt=0,lt=1)
    matching_tolerance:float=Field(gt=0,lt=1)
    practical_zero_sensitivity:list[float]
    fixed_margin_thresholds:list[float]
    precision:PrecisionPlan

    @model_validator(mode='after')
    def prespecified_candidate(self):
        if set(self.models)!={f'mock-family-{x}/fixture' for x in 'abcd'} or len(set(self.models))!=4:
            raise ValueError('four_distinct_mock_families_required')
        if self.variants!=['base','challenge'] or self.planning_card_counts!=[20,60,120]:
            raise ValueError('candidate_schedule_required')
        if (self.practical_zero_sensitivity!=[0.,.05,.1] or self.fixed_margin_thresholds!=[0.,.25,.5,.75,1.]):
            raise ValueError('predeclared_diagnostic_grid_required')
        return self


def input_path(path):return safe_path(path if Path(path).is_absolute() else ROOT/path)

def load_json(path):return json.loads(input_path(path).read_text())


def _run(config_path,results_dir,new_run=False):
    with stage('configuration'):
        cfg_file=input_path(config_path);cfg=Config.model_validate(yaml.safe_load(cfg_file.read_text()))
    with stage('public_packets'):
        tasks=[PublicTask.model_validate(r) for r in load_json(cfg.public_tasks)]
        exported=exports(tasks);tasks=sorted(tasks,key=task_hash);by_hash={task_hash(t):t for t in tasks}
    with stage('design_families'):
        family_data=load_json(cfg.families)
        if set(family_data)!={'definitions','assignments'}:raise ValueError('family_fields_invalid')
        families=[FamilyDefinition.model_validate(r) for r in family_data['definitions']]
        assignments=[CardAssignment.model_validate(r) for r in family_data['assignments']]
        family_audit=audit_families(by_hash,families,assignments)
    with stage('pricing'):
        prices=[FakePrice.model_validate(r) for r in load_json(cfg.fake_pricing)]
        if len(prices)!=len(cfg.models) or {p.model_id for p in prices}!=set(cfg.models):raise ValueError('price_catalog_mismatch')
        for price in prices:
            price.validate_known();price.cost(cfg.input_token_reservation,cfg.output_token_reservation)
    with stage('decisions'):
        decisions=[Decision.model_validate(r) for r in load_json(cfg.decisions)]
        if len(decisions)!=16 or len({d.id for d in decisions})!=16:raise ValueError('sixteen_pending_decisions_required')
        # Public/design input commitments are captured before any mock response.
        initial_inputs=[cfg_file]+[input_path(getattr(cfg,k)) for k in
                                  ('public_tasks','families','decisions','fake_pricing')]
        initial_digests={str(p):digest(p) for p in initial_inputs}
    with stage('output'):
        output=safe_path(results_dir)
        if output.exists() and (not output.is_dir() or (not new_run and any(output.iterdir()))):
            raise ValueError('new_empty_output_required')
        output.mkdir(parents=True,exist_ok=True)
        if new_run:output=Path(tempfile.mkdtemp(prefix='run-',dir=output))
        (output/'public').mkdir()
        for name,raw in exported.items():
            with (output/'public'/name).open('xb') as f:f.write(raw)
        write_once(output/'public_manifest.json',{name:sha(json.loads(raw)) for name,raw in exported.items()})
        design=definition_commitment(ROOT,cfg.model_dump(),{'version':RUBRIC_VERSION,'policies':list(POLICIES),
            'schema':Proposal.model_json_schema(),'primary':'credible_alternatives','approval':'DECISION_REQUIRED',
            'credibility':'explicit_public_candidate_rule_not_automated_semantic_confidence'},
            family_data,[d.model_dump() for d in decisions])
        design['pre_capture_input_sha256']={str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT)
            else str(Path(p)):h for p,h in initial_digests.items()}
        write_once(output/'design_commitment.json',design)
        design_names=['design_commitment.json','public_manifest.json']+['public/'+n for n in exported]
        design_seal,design_hash=seal(output,design_names,'design_seal.json')
    with stage('mock_prediction'):
        transport=MockTransport();budget=SimulatedBudget(cfg.hypothetical_global_cap_usd,cfg.hypothetical_per_model_cap_usd)
        records=[];batch={}
        for task in tasks:
            selected={}
            for model in cfg.models:
                price=next(p for p in prices if p.model_id==model)
                for variant in cfg.variants:
                    for repetition in range(cfg.repetitions):
                        answers,receipt=transport.record(task,model,variant,repetition,price,budget,
                            cfg.input_token_reservation,cfg.output_token_reservation)
                        if not validate_record(task,answers,receipt.model_dump(),variant,repetition,model):
                            raise ValueError('mock_record_contract_failed')
                        records.append({'answers':answers,'receipt':receipt.model_dump()})
                        if variant=='base' and repetition==0:selected[model]=answers
            batch[task_hash(task)]=comparisons(task,selected,cfg.models)
        # No outcome estimation or scoring labels. Fixed probes are NOT model predictions.
        probes={name:{h:{'decision':decision,'margin':(i+1)/len(tasks) if decision!='abstain' else 0.}
                      for i,h in enumerate(sorted(by_hash))} for name,decision in (
                       ('positive_contract_probe','positive'),('negative_contract_probe','negative'),
                       ('abstain_contract_probe','abstain'))}
        write_once(output/'comparison_batch.json',batch);write_once(output/'mock_records.json',records)
        write_once(output/'fixed_contract_probes.json',probes)
    with stage('prediction_seal'):
        verify(output,design_seal,'design_seal.json',design_hash)
        names=design_names+['design_seal.json','comparison_batch.json','mock_records.json','fixed_contract_probes.json']
        prediction_seal,prediction_hash=seal(output,names,'prediction_seal.json')
        verify(output,prediction_seal,'prediction_seal.json',prediction_hash)
    with stage('private_scoring'):
        # FIRST private truth-file open; every prompt, graph, rule, probe, split and scoring definition fixed.
        from .certificates import Truth,certify
        from .scoring import score,fixed_curves
        worlds=[Truth.model_validate(r) for r in load_json(cfg.private_truth)]
        if len({w.world_id for w in worlds})!=len(worlds) or {w.public_task_hash for w in worlds}!=set(by_hash):
            raise ValueError('truth_card_coverage_invalid')
        mapped={a.task_hash:a.family_id for a in assignments}
        if any(mapped[w.public_task_hash]!=w.family_id for w in worlds):raise ValueError('truth_family_mismatch')
        certs={w.world_id:certify(w,by_hash[w.public_task_hash],worlds) for w in worlds}
        probes=json.loads((output/'fixed_contract_probes.json').read_bytes())
        scores={name:score({h:r['decision'] for h,r in probe.items()},worlds,certs,by_hash,.1)
                for name,probe in probes.items()}
        curve=fixed_curves(probes,cfg.fixed_margin_thresholds,cfg.practical_zero_sensitivity,
            worlds,certs,by_hash,cfg.assertion_target,cfg.matching_tolerance)
        verify(output,prediction_seal,'prediction_seal.json',prediction_hash)
    with stage('precision'):
        precision=diagnostics(cfg.precision)
        bounded=[r for r in precision if r['procedure']=='bounded_family_hoeffding']
        gate={'candidate_procedure':'bounded_family_hoeffding','PI_approval':'DECISION_REQUIRED',
            'coverage_validation_passed_in_simulated_grid':all(r['coverage_gate']=='PASS_SCENARIO_COVERAGE' for r in bounded),
            'precision_passed_in_simulated_grid':all(r['precision_gate'] for r in bounded),
            'confirmatory_precision_ready':False,
            'independent_family_assumption_verified':False,
            'coverage_gate_counts':dict(Counter(r['coverage_gate'] for r in precision)),
            'failed_scenarios':[{'scenario_id':r['scenario_id'],'procedure':r['procedure'],
                'coverage_gate':r['coverage_gate'],'precision_gate':r['precision_gate']}
                for r in precision if r['coverage_gate']!='PASS_SCENARIO_COVERAGE' or not r['precision_gate']],
            'limitation':'Only these assumed scenarios were simulated; bounded intervals are conservative, family independence unverified, width target unapproved.'}
        gate['candidate_statistical_gate_passed']=bool(bounded) and (
            gate['coverage_validation_passed_in_simulated_grid'] and gate['precision_passed_in_simulated_grid'])
        gate['confirmatory_precision_ready']=(gate['candidate_statistical_gate_passed'] and
            gate['independent_family_assumption_verified'] and gate['PI_approval']!='DECISION_REQUIRED')
        gate['blockers']=[name for name,ok in (
            ('coverage_or_MC_stability',gate['coverage_validation_passed_in_simulated_grid']),
            ('interval_width',gate['precision_passed_in_simulated_grid']),
            ('independent_family_validation',gate['independent_family_assumption_verified']),
            ('PI_approval',gate['PI_approval']!='DECISION_REQUIRED')) if not ok]
        costs=[{**cost_envelope(prices,n,len(cfg.variants),cfg.repetitions,cfg.max_retries,
                    cfg.input_token_reservation,cfg.output_token_reservation,cfg.hypothetical_per_model_cap_usd,
                    cfg.hypothetical_global_cap_usd),'pricing_label':'FICTIONAL'} for n in cfg.planning_card_counts]
    with stage('artifacts'):
        verify(output,prediction_seal,'prediction_seal.json',prediction_hash)
        if any(digest(Path(p))!=h for p,h in initial_digests.items()):
            raise ValueError('pre_capture_input_changed')
        if any(digest(ROOT/p)!=h for p,h in design['source_sha256'].items()):
            raise ValueError('committed_runtime_source_changed')
        firewall={'passed_engineering_checks':True,'truth_opened_after_full_prediction_seal':True,
            'definitions_sealed_before_capture':True,'public_export_policy':'explicit_allowlist_content_hash_sorted',
            'private_values_used_for_admission':False,'independent_custody':'DECISION_REQUIRED',
            'ambiguous_pairs_same_public_payload':all(len({w.public_task_hash for w in worlds if w.world_id in
                x.proof.witness_ids})==1 for x in worlds if x.proof.kind=='observational_pair')}
        for name,data in [('identification_audit',certs),('family_audit',family_audit),('firewall_audit',firewall),
                          ('scoring_contract_diagnostics',scores),('precision_gate',gate),('decision_register',[d.model_dump() for d in decisions])]:
            write_once(output/(name+'.json'),data)
        write_csv(output/'precision_diagnostics.csv',precision);write_csv(output/'risk_assertion_curves.csv',curve)
        write_csv(output/'cost_envelope.csv',[{k:v for k,v in c.items() if k!='per_model_hypothetical_usd'} for c in costs])
        admission_counts={policy:dict(Counter(a['reason'] for item in batch.values()
                for rows in item['policies'][policy]['sources'].values() for a in rows)) for policy in POLICIES}
        # BLAS configuration is obtained locally without enumerating credentials/environment.
        import numpy as np
        blas=np.__config__.show(mode='dicts')
        report={'version':VERSION,'protocol_status':'candidate_not_frozen','ready_to_freeze':False,
            'fixture_mode':True,'real_llm_calls':0,'openrouter_cost_usd':0,'authorized_openrouter_usd':0,
            'share_records_used':0,'decision_required_count':len(decisions),'public_cards':len(tasks),
            'world_probes':len(worlds),'mock_transport_calls':transport.calls,'admission_counts':admission_counts,
            'identification_status_counts':dict(Counter(c['status'] for c in certs.values())),
            'effect_decisions':{'abstain':len(tasks),'reason':'graph_only_no_empirical_outcome_estimation',
                'candidate_primary_majority_reasons':dict(Counter(item['policies']['credible_alternatives']['majority']['reason']
                    for item in batch.values()))},
            'actual_model_coverage_matching':'UNMATCHED_zero_assertions',
            'precision_rows':len(precision),'precision_gate':gate,'family_audit':family_audit,
            'cost_envelopes':costs,'pricing_label':'FICTIONAL_not_actual_provider_prices',
            'simulated_usage_cost_NOT_spending':float(sum(budget.spent.values())),
            'prediction_seal_sha256':prediction_hash,'design_seal_sha256':design_hash,
            'environment':{'python':platform.python_version(),'architecture':platform.machine(),
              'system':platform.system(),'numpy_build':blas,
              'packages':{p:importlib.metadata.version(p) for p in ('pydantic','numpy','scipy','PyYAML')}},
            'warning':'Exposed engineering fixtures and assumed precision scenarios only. No LLM performance, validated confirmatory coverage, independent holdout or superiority.'}
        write_once(output/'preflight_report.json',report)
        md=['# v2.3 offline candidate diagnostics','',
            'candidate_not_frozen; ready_to_freeze=false; authorized spending USD 0.',
            f'{len(tasks)} public development cards; {len(worlds)} private world probes; {len(decisions)} DECISION_REQUIRED.',
            'No real LLM calls, provider requests, credentials or SHARE records.',
            'Candidate primary credibility requires explicit public stipulations; human PI approval remains pending.',
            'All majority effects abstain; risk curves are fixed scoring-contract probes, not model performance.',
            '','## Precision validation','',
            f"Coverage gate counts: {gate['coverage_gate_counts']}.",
            f"Bounded-family coverage passed in grid: {gate['coverage_validation_passed_in_simulated_grid']}.",
            f"Bounded-family precision passed: {gate['precision_passed_in_simulated_grid']}.",
            'Conservative finite-family bounds can be scientifically valid under family independence and still unusably wide.',
            'Family-t undercoverage/Monte Carlo instability are failures, not approved operating characteristics.',
            '','## FICTIONAL cost envelopes','',
            '| Cards | Planned | Max attempts | Fictional USD | Within hypothetical caps |',
            '| --- | --- | --- | --- | --- |']
        md += [f"| {c['independent_cards']} | {c['planned_calls']} | {c['maximum_attempts']} | {c['hypothetical_total_usd']:.4f} | {c['within_hypothetical_caps']} |" for c in costs]
        md += ['','Candidate heldout definitions have no generated/scored cards and no independent custody.']
        (output/'diagnostics.md').write_text('\n'.join(md)+'\n')
        inputs=[cfg_file]+[input_path(getattr(cfg,k)) for k in ('public_tasks','private_truth','families','decisions','fake_pricing')]
        source=design['source_sha256']
        manifest={'inputs':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):digest(p) for p in inputs},
                  'sources':source,'outputs':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
        write_once(output/'content_manifest.json',manifest)
        return report,output,digest(output/'content_manifest.json')


def run_preflight(config_path,results_dir,*,new_run=False):
    with network_veto():return _run(config_path,results_dir,new_run)
