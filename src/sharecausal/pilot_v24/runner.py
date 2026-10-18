"""Separate-process offline pilot; orchestrator never opens generator/scoring truth."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import yaml
from .config import Config
from .guards import ProcessGuard,forbidden

ROOT=Path(__file__).resolve().parents[3]


class PilotFailure(Exception):
    def __init__(self,stage,diagnostic=None):
        self.stage=stage;self.diagnostic=diagnostic;super().__init__('offline_pilot_failed')


def resolve(path):
    p=Path(path).expanduser()
    if not p.is_absolute():p=ROOT/p
    if forbidden(p):raise ValueError('forbidden_synthetic_path')
    p=p.resolve()
    if forbidden(p):raise ValueError('forbidden_resolved_path')
    return p


def run(config_path,results_dir,new_run=False):
    current='configuration'
    try:
        cfgpath=resolve(config_path);output=resolve(results_dir)
        guard=ProcessGuard('orchestrator',ROOT,inputs=[cfgpath],output=output).install()
        cfg=Config.model_validate(yaml.safe_load(cfgpath.read_text()));data=cfg.model_dump()
        path_fields=('public_tasks','generation_plan','scoring_truth','families','decisions','mock_rules','precision_reference_config')
        for k in path_fields:data[k]=str(resolve(data[k]))
        guard.truth=Path(data['scoring_truth']);guard.generation=Path(data['generation_plan'])
        public_inputs=[cfgpath]+[Path(data[k]) for k in path_fields if k not in ('generation_plan','scoring_truth')]
        guard.inputs.update(public_inputs)
        from sharecausal.preflight.contracts import sha,canonical
        from sharecausal.preflight_v23.packets import PublicTask,exports,task_hash
        from sharecausal.preflight_v23.sealing import write_once,seal,verify,digest
        from sharecausal.preflight.runner import write_csv
        from sharecausal.preflight_v23.families import FamilyDefinition,CardAssignment,audit_families
        from . import VERSION
        current='public_inputs'
        tasks=[PublicTask.model_validate(r) for r in json.loads(Path(data['public_tasks']).read_text())]
        exported=exports(tasks);by_hash={task_hash(t):t for t in tasks}
        families=json.loads(Path(data['families']).read_text())
        family_audit=audit_families(by_hash,[FamilyDefinition.model_validate(x) for x in families['definitions']],
            [CardAssignment.model_validate(x) for x in families['assignments']])
        decisions=json.loads(Path(data['decisions']).read_text())
        if len(decisions)!=16 or len({d['id'] for d in decisions})!=16 or any(d['status']!='DECISION_REQUIRED' for d in decisions):
            raise ValueError('decisions_must_remain_pending')
        rules=json.loads(Path(data['mock_rules']).read_text())
        if set(rules)!={'schema_version','absent_d_on_unknown_recording','malformed_c_challenge'} or rules['schema_version']!='mock-rules-2.4':
            raise ValueError('mock_rules_invalid')
        if any(type(rules[k]) is not bool for k in rules if k!='schema_version'):raise ValueError('mock_flags_must_be_boolean')
        current='output'
        if output.exists() and (not output.is_dir() or (not new_run and any(output.iterdir()))):raise ValueError('new_empty_output_required')
        output.mkdir(parents=True,exist_ok=True)
        if new_run:output=Path(tempfile.mkdtemp(prefix='run-',dir=output))
        guard.output=output
        (output/'public').mkdir()
        for name,raw in exported.items():
            with (output/'public'/name).open('xb') as f:f.write(raw)
        write_once(output/'public_manifest.json',{n:sha(json.loads(raw)) for n,raw in exported.items()})
        write_once(output/'mock_rules_public.json',rules)
        sources=sorted((ROOT/'src/sharecausal/pilot_v24').glob('*.py'))
        sources+=sorted((ROOT/'src/sharecausal/preflight_v23').glob('*.py'))
        sources+=sorted((ROOT/'src/sharecausal/preflight').glob('*.py'))
        sources += [ROOT/'src/sharecausal/causal/estimators.py',ROOT/'src/sharecausal/causal/robust_conclusions.py',
            ROOT/'scripts/v24_pilot_offline.py',ROOT/'scripts/build_v24_generation_plan.py',ROOT/'docs/freeze_manifest_v1.json']
        guard.inputs.update(sources)
        input_hash={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):digest(p) for p in public_inputs}
        code_hash={str(p.relative_to(ROOT)):digest(p) for p in sources}
        # Config paths are normalized relative to root in the scientific commitment, not random run paths.
        design={'schema_version':'pilot-design-2.4','candidate_config':cfg.model_dump(),
                'public_input_sha256':input_hash,'source_sha256':code_hash,'decision_status':'DECISION_REQUIRED',
                'family_audit':family_audit,'comparison_unit':'public_card_primary_sample_zero_nested_repeats',
                'custody':'separate_local_processes_not_independent_people'}
        write_once(output/'design_commitment.json',design)
        design_names=['public/'+n for n in exported]+['public_manifest.json','design_commitment.json','mock_rules_public.json']
        design_seal,design_digest=seal(output,design_names,'design_seal.json')
        worker_status=[]
        def worker(role,extra=None):
            job={'config':data,'output':str(output),**(extra or {})};path=output/('job_'+role+'.json')
            write_once(path,job)
            child=subprocess.run([sys.executable,'-m','sharecausal.pilot_v24.workers',role,str(path)],cwd=ROOT,
                capture_output=True,text=True,timeout=240)
            worker_status.append({'role':role,'exit_code':child.returncode})
            if child.returncode:
                diagnostic={'role':role,'exit_code':child.returncode}
                try:
                    decoded=json.loads(child.stderr)
                    if decoded.get('exception_type') in ('ValueError','PermissionError','TypeError','ImportError','ModuleNotFoundError','OSError','Exception'):
                        diagnostic['exception_type']=decoded['exception_type']
                    names={p.stem for p in (ROOT/'src/sharecausal/pilot_v24').glob('*.py')}
                    positions=decoded.get('code_positions',[])
                    diagnostic['code_positions']=[{'module':p['module'],'line':p['line']} for p in positions
                        if type(p) is dict and p.get('module') in names and type(p.get('line')) is int and p['line']>0]
                except (ValueError,TypeError,KeyError):pass
                raise PilotFailure('worker_'+role,diagnostic)
            # Raw child diagnostics are never forwarded, even when a dependency fails.
        current='worker_generate';worker('generate')
        current='worker_predict';worker('predict',{'design_seal':design_seal,'design_digest':design_digest})
        verify(output,design_seal,'design_seal.json',design_digest)
        current='prediction_seal'
        samples=json.loads((output/'sample_manifest.json').read_text())
        names=design_names+['design_seal.json','sample_manifest.json','generator_receipt.json',
            'mock_records.json','prediction_batch.json','prediction_guard_receipt.json']+[r['file'] for r in samples]
        prediction_seal,prediction_digest=seal(output,names,'prediction_seal.json')
        verify(output,prediction_seal,'prediction_seal.json',prediction_digest)
        current='worker_score';worker('score',{'prediction_digest':prediction_digest})
        verify(output,prediction_seal,'prediction_seal.json',prediction_digest)
        current='precision_diagnostics'
        from sharecausal.preflight_v23.precision import PrecisionPlan,diagnostics
        reference=yaml.safe_load(Path(data['precision_reference_config']).read_text())
        precision=diagnostics(PrecisionPlan.model_validate(reference['precision']))
        current='reporting'
        scored=json.loads((output/'scored_results.json').read_text())
        batch=json.loads((output/'prediction_batch.json').read_text());records=json.loads((output/'mock_records.json').read_text())
        from .capture import parse_wire
        response_counts=dict(Counter(r['parsed']['status'] for r in records))
        tables=('main_summary','per_card','nested_repetition_summary','risk_assertion_curves',
                'paired_family_differences','world_estimation_diagnostics')
        for name in tables:write_csv(output/(name+'.csv'),scored[name])
        write_csv(output/'precision_validation.csv',precision)
        precision_summary={method:{'scenarios':len(rows),
            'coverage_gate_counts':dict(Counter(r['coverage_gate'] for r in rows)),
            'minimum_simulated_coverage':min(r['observed_coverage'] for r in rows),
            'maximum_simulated_coverage':max(r['observed_coverage'] for r in rows),
            'minimum_mean_half_width':min(r['mean_half_width'] for r in rows),
            'maximum_mean_half_width':max(r['mean_half_width'] for r in rows)} for method in
                ('family_t_comparator','bounded_family_hoeffding')
            for rows in [[r for r in precision if r['procedure']==method]]}
        costs={'fixture_mode':True,'real_llm_calls':0,'openrouter_cost_usd':0,'authorized_openrouter_usd':0,
               'mock_capture_records':len(records),'actual_provider_prices':'not_queried',
               'invoice_status':'no_provider_requests_no_invoice','no_live_path':True}
        write_once(output/'offline_cost_manifest.json',costs)
        report={'version':VERSION,'protocol_status':'candidate_not_frozen','ready_to_freeze':False,
            'confirmatory_precision_ready':False,'fixture_mode':True,'real_llm_calls':0,
            'openrouter_cost_usd':0,'share_records_used':0,'authorized_openrouter_usd':0,
            'decision_required_count':len(decisions),'public_cards':len(tasks),
            'declared_development_families':scored['declared_families'],'private_world_probes':scored['world_probes'],
            'sample_repetitions_nested':cfg.sample_repetitions,'sample_size':cfg.sample_size,
            'primary_sample_repetition':cfg.primary_sample_repetition,'prediction_rows':len(batch),
            'method_card_primary_rows':len(scored['per_card']),'mock_capture_records':len(records),
            'response_status_counts_all_variants':response_counts,'worker_exits':worker_status,
            'main_summary':scored['main_summary'],'precision_validation':precision_summary,
            'estimator_fit_failures':sum(p['estimator_failure_count'] for p in batch),
            'prediction_seal_sha256':prediction_digest,'design_seal_sha256':design_digest,
            'family_audit':family_audit,'main_experiment_status':'executed_exposed_development_exploratory_only',
            'environment':{'python':platform.python_version(),'machine':platform.machine(),
                'packages':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','pydantic','PyYAML')}},
            'limitations':['No independently authored or held-out validation tasks.',
                'Credible alternatives are candidate author stipulations, not expert-adjudicated causal knowledge.',
                'Model-conditional normal/AIPW intervals do not cover graph or selection uncertainty.',
                'Paired family-t intervals are exploratory and prior coverage failures persist.',
                'Conservative family bounds remain too wide; no .025/.05 powered comparison.',
                'Scoring is a separate guarded process, not independent human custody.',
                'Conservative public randomization gate withholds directions on uncertified observational cards.']}
        write_once(output/'pilot_report.json',report)
        lines=['# v2.4 executed offline exploratory pilot','','candidate_not_frozen; ready_to_freeze=false; all live costs USD 0.',
            '',f'{len(tasks)} exposed cards / {scored["declared_families"]} declared families; repeats are nested.',
            '','| Method | Assertions/cards | Unjustified bounds | Matched |',
            '| --- | --- | --- | --- |']
        lines += [f"| {r['method']} | {r['assertions']}/{r['eligible_cards']} | {r['unjustified_rate_lower']:.3f}–{r['unjustified_rate_upper']:.3f} | {r['coverage_matched']} |" for r in scored['main_summary']]
        lines += ['','No real LLM performance, superiority, confirmatory precision or protocol freeze is established.',
            'Family-t undercoverage and conservative-width failures remain in precision_validation.csv.',
            'Zero observed unjustified claims on these exposed cards is not evidence of calibration or accuracy.']
        (output/'pilot_report.md').write_text('\n'.join(lines)+'\n')
        if any(digest(ROOT/p if not Path(p).is_absolute() else p)!=h for p,h in input_hash.items()):raise ValueError('public_input_changed')
        if any(digest(ROOT/p)!=h for p,h in code_hash.items()):raise ValueError('source_changed')
        gen_receipt=json.loads((output/'generator_receipt.json').read_text())
        score_receipt=json.loads((output/'scorer_receipt.json').read_text())
        all_inputs={**input_hash,cfg.generation_plan:gen_receipt['generation_plan_sha256'],cfg.scoring_truth:score_receipt['scoring_truth_sha256']}
        write_once(output/'content_manifest.json',{'inputs':all_inputs,'sources':code_hash,
            'outputs':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*'))
                if p.is_file() and not p.name.startswith('job_')}})
        return report,output,digest(output/'content_manifest.json')
    except PilotFailure:raise
    except Exception as exc:raise PilotFailure(current) from exc
