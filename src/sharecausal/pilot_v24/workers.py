"""Role entry point. Install file/key/socket guards BEFORE scientific imports."""
import json
from pathlib import Path
import sys


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if len(args)!=2 or args[0] not in ('generate','predict','score'):
        print('{"status":"error","stage":"worker_arguments"}',file=sys.stderr);return 2
    role,jobname=args
    try:
        # Job contains only paths/public configuration, no truth parameters.
        from .guards import ProcessGuard,forbidden
        lexical=Path(jobname).expanduser().absolute()
        if forbidden(lexical) or lexical.name!='job_'+role+'.json' or lexical.is_symlink():
            raise ValueError('internal_role_job_path_required')
        jobpath=lexical.resolve()
        if forbidden(jobpath):raise ValueError('forbidden_role_job')
        job=json.loads(jobpath.read_text())
        root=Path(__file__).resolve().parents[3];output=Path(job['output']).resolve()
        cfg=job['config'];truth=Path(cfg['scoring_truth']);generation=Path(cfg['generation_plan'])
        paths=[jobpath,Path(cfg['public_tasks'])]
        if role=='generate':paths.append(generation)
        if role=='score':paths.append(Path(cfg['families']))
        if role=='predict':
            paths.extend(output/n for n in job['design_seal'])
            paths.extend(output/n for n in ('design_seal.json','mock_rules_public.json','sample_manifest.json'))
        if role=='score':paths.append(output/'prediction_seal.json')
        if any(forbidden(p) for p in [output,truth,generation,*paths]):raise ValueError('forbidden_role_path')
        guard=ProcessGuard(role,root,inputs=paths,output=output,truth=truth,generation=generation).install()
        from .config import Config
        Config.model_validate(cfg)
        from sharecausal.preflight.contracts import sha,canonical
        from sharecausal.preflight_v23.packets import PublicTask,task_hash
        from sharecausal.preflight_v23.sealing import write_once,seal,verify,digest
        tasks={task_hash(t):t for t in [PublicTask.model_validate(r) for r in json.loads(Path(cfg['public_tasks']).read_text())]}
        if role=='generate':
            from .generation import observed_sample
            plan=json.loads(generation.read_text())
            if set(plan)!={'schema_version','plans'} or plan['schema_version']!='generation-plan-2.4' or set(plan['plans'])!=set(tasks):
                raise ValueError('generation_plan_inventory_invalid')
            samples=[];sample_dir=output/'observed';sample_dir.mkdir()
            for h,task in sorted(tasks.items()):
                if set(plan['plans'][h])!={'nodes','recording'}:raise ValueError('generation_entry_invalid')
                for rep in range(cfg['sample_repetitions']):
                    seed=cfg['sampling_seed']+int(sha([h,rep])[:12],16)
                    values=observed_sample(task,plan['plans'][h],cfg['sample_size'],seed)
                    sample_id='sample-'+sha([h,rep]);name='observed/'+sample_id+'.json'
                    data={'schema_version':'observed-sample-2.4','task_hash':h,'sample_id':sample_id,**values}
                    write_once(output/name,data)
                    samples.append({'task_hash':h,'sample_id':sample_id,'sample_repetition':rep,'file':name,'sha256':digest(output/name)})
            write_once(output/'sample_manifest.json',samples)
            write_once(output/'generator_receipt.json',{'source_type':'independent_synthetic_exposed_CPT_fixture',
                'generation_plan_sha256':digest(generation),'sample_files':len(samples),
                'hidden_variables_exported':False,'unrecorded_outcome_values_exported':False,
                'scoring_truth_opened':False,'guard_role':'generate'})
        elif role=='predict':
            from .capture import capture,validate_record
            from .engine import analyze
            from .observations import read_sample
            from sharecausal.preflight_v23.sealing import verify
            verify(output,job['design_seal'],'design_seal.json',job['design_digest'])
            records=[];selected={}
            rules=json.loads((output/'mock_rules_public.json').read_text())
            for h,task in sorted(tasks.items()):
                chosen={}
                for model in cfg['models']:
                    for variant in cfg['variants']:
                        for rep in range(cfg['source_repetitions']):
                            record=capture(task,model,variant,rep,rules)
                            if not validate_record(task,record,model,variant,rep):raise ValueError('mock_receipt_invalid')
                            records.append(record)
                            if variant=='base' and rep==0:chosen[model]=record['parsed']
                selected[h]=chosen
            # Graph generation above never receives sample statistics.
            samples=json.loads((output/'sample_manifest.json').read_text());batch=[]
            guard.inputs.update(output/r['file'] for r in samples)
            for item in samples:
                if digest(output/item['file'])!=item['sha256']:raise ValueError('observed_input_modified')
                task=tasks[item['task_hash']];data=json.loads((output/item['file']).read_text())
                if data['task_hash']!=item['task_hash'] or data['sample_id']!=item['sample_id']:raise ValueError('sample_binding_invalid')
                frame=read_sample(task,data)
                prediction=analyze(task,frame,selected[item['task_hash']],cfg['models'],cfg['estimation'])
                prediction.update(sample_id=item['sample_id'],sample_repetition=item['sample_repetition'])
                batch.append(prediction)
            write_once(output/'mock_records.json',records)
            write_once(output/'prediction_batch.json',batch)
            write_once(output/'prediction_guard_receipt.json',{'role':'predict','scoring_truth_readable':False,
                'generator_plan_readable':False,'scorer_imported':any(x.startswith('sharecausal.pilot_v24.evaluation') for x in sys.modules),
                'public_prompt_only_generation':True,'samples':len(batch),'mock_capture_records':len(records)})
            verify(output,job['design_seal'],'design_seal.json',job['design_digest'])
        else:
            # Verify all samples, graph responses, predictions, code/design commitments BEFORE opening truth.
            manifest=json.loads((output/'prediction_seal.json').read_text())
            guard.inputs.update(output/name for name in manifest)
            verify(output,manifest,'prediction_seal.json',job['prediction_digest'])
            guard.truth_ready=True
            from .evaluation import evaluate
            raw=json.loads(truth.read_text())
            family=json.loads(Path(cfg['families']).read_text())
            result=evaluate(json.loads((output/'prediction_batch.json').read_text()),raw,tasks,family['assignments'],cfg)
            write_once(output/'scored_results.json',result)
            write_once(output/'scorer_receipt.json',{'role':'score','seal_verified_before_truth_open':True,
                'prediction_seal_sha256':job['prediction_digest'],'scoring_truth_sha256':digest(truth),
                'prediction_files_updated':False,'guard_role':'score',
                'custody':'separate_process_single_team_exposed_not_independent_personnel'})
            verify(output,manifest,'prediction_seal.json',job['prediction_digest'])
        print(json.dumps({'status':'ok','role':role}));return 0
    except Exception as exc:
        # No exception message, field names, raw response, path or private local in worker stderr.
        frames=[];tb=exc.__traceback__
        while tb:
            file=Path(tb.tb_frame.f_code.co_filename)
            # Source/dependency code positions only; no exception values, locals or data filenames.
            if file.is_relative_to(Path(__file__).resolve().parent):
                frames.append({'module':file.stem,'line':tb.tb_lineno})
            tb=tb.tb_next
        print(json.dumps({'status':'error','stage':'worker_'+role,
                         'exception_type':type(exc).__name__ if type(exc).__name__ in (
                             'ValueError','PermissionError','TypeError','ImportError','ModuleNotFoundError','OSError') else 'Exception',
                         'code_positions':frames,
                         'hint':'Check declared synthetic inputs, role guards and immutable seals; no live fallback.'}),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
