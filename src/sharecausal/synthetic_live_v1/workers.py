"""Separate guarded generation, provider collection, prediction and scoring."""
import json
import os
from pathlib import Path
import sys
from sharecausal.pilot_v24.guards import ProcessGuard


class Guard(ProcessGuard):
    def install(self):
        if self.role!='collect': return super().install()
        # Only the sender may read the one explicitly authorized environment value.
        sys.addaudithook(self.audit); return self

    def role_output_file(self,path):
        if super().role_output_file(path): return True
        if self.role=='predict' and path.parent==self.output and path.name in {
            'graph_audit_batch.json','graph_admissions.csv','edge_claim_audit.csv'}: return True
        if self.role=='collect' and path.is_relative_to(self.output):
            return (path.parent==self.output/'provider_responses_private' or
                    path.parent==self.output and path.name in {
                        'catalog_snapshot.json','request_plan.json','collection_status.json','collection_summary.json',
                        'collection.lock',
                        'catalog_snapshot.json.pending','request_plan.json.pending',
                        'collection_status.json.pending','collection_summary.json.pending'})
        return False

    def audit(self,event,args):
        if event=='open' and self.role in ('collect','predict') and isinstance(args[0],(str,bytes,os.PathLike)):
            candidate=Path(os.fsdecode(args[0]))
            if candidate.stem=='authoring' or candidate.name.startswith('authoring.'):
                raise PermissionError('authoring_truth_source_forbidden')
        if event=='import' and self.role in ('collect','predict') and any(
            str(args[0]).startswith(x) for x in (
                'sharecausal.synthetic_live_v1.authoring','sharecausal.synthetic_live_v1.reporting',
                'sharecausal.preflight_v23.certificates','sharecausal.pilot_v24.generation',
                'sharecausal.pilot_v24.evaluation')):
            raise PermissionError('truth_module_forbidden')
        if event.startswith('socket.') and self.role=='collect': return
        return super().audit(event,args)


def main():
    role,path=sys.argv[1:]; job=json.loads(Path(path).read_text())
    out=Path(job['output']).resolve(); root=Path(__file__).resolve().parents[3]
    paths=[Path(path),out/'public_tasks.json',out/'design_commitment.json']
    if role=='generate': paths.append(out/'generation_plan.json')
    if role=='predict':
        paths += [out/'sample_manifest.json',out/'request_manifest.jsonl']
        paths += list((out/'provider_responses_private').glob('*.json'))
    if role=='score': paths += [out/'prediction_seal.json',out/'families.json']
    guard=Guard(role,root,inputs=paths,output=out,truth=out/'scoring_truth.json',generation=out/'generation_plan.json').install()
    from sharecausal.preflight.contracts import sha
    from sharecausal.preflight_v23.packets import PublicTask, task_hash
    from sharecausal.preflight_v23.sealing import write_once,digest,verify
    from .public import MODELS,SLOTS,ESTIMATION,SCORING
    tasks={task_hash(t):t for t in [PublicTask.model_validate(x) for x in json.loads((out/'public_tasks.json').read_text())]}
    design=json.loads((out/'design_commitment.json').read_text())
    guard.inputs.update(root/name for name in design['source_sha256'])
    if any(digest(root/name)!=expected for name,expected in design['source_sha256'].items()
           if not (role in ('collect','predict') and Path(name).name=='authoring.py')):
        raise ValueError('committed_source_changed')
    if digest(out/'public_tasks.json')!=design['public_tasks_sha256']: raise ValueError('committed_tasks_changed')
    if role=='generate':
        from sharecausal.pilot_v24.generation import observed_sample
        plan=json.loads((out/'generation_plan.json').read_text()); manifest=[]
        (out/'observed').mkdir()
        for h,task in sorted(tasks.items()):
            data=observed_sample(task,plan['plans'][h],800,71807+int(sha([h,0])[:12],16))
            name='observed/sample-'+sha([h,0])+'.json'; sid='sample-'+sha([h,0])
            write_once(out/name,dict(schema_version='observed-sample-2.4',task_hash=h,sample_id=sid,**data))
            manifest.append(dict(task_hash=h,sample_id=sid,sample_repetition=0,file=name,sha256=digest(out/name)))
        write_once(out/'sample_manifest.json',manifest)
        write_once(out/'generator_receipt.json',dict(role=role,generation_plan_sha256=digest(out/'generation_plan.json'),
            scoring_truth_opened=False,hidden_variables_exported=False,sample_files=len(manifest)))
    elif role=='collect':
        from .transport import collect
        key=os.environ.get('OPENROUTER_API_KEY')
        if not key: raise ValueError('authorized_key_not_in_worker_environment')
        collect(tasks,out,job['cap'],key)
    elif role=='predict':
        from .records import validated
        from sharecausal.provider_replay_v1.prediction import adapt
        from sharecausal.preflight_v23.packets import prompt
        from sharecausal.preflight_v23.admission import Proposal
        from sharecausal.pilot_v24.engine import analyze
        from sharecausal.pilot_v24.observations import read_sample
        selected={h:{} for h in tasks}
        manifest=[json.loads(line) for line in (out/'request_manifest.jsonl').read_text().splitlines()]
        for item in manifest:
            h=item['task_sha256']; model=item['model']; slot=SLOTS[MODELS.index(model)]
            parsed=dict(status=item['status'],proposals=[],rejected_proposals=0)
            if item.get('record_file'):
                record=validated(tasks[h],model,json.loads((out/item['record_file']).read_text()))
                if record.design_commitment_sha256!=sha(design): raise ValueError('provider_design_binding_failed')
                if (digest(out/record.raw_file)!=item['raw_file_sha256']
                    or sha(json.loads((out/record.raw_file).read_text()))!=record.response_sha256
                    or digest(out/item['record_file'])!=item['record_sha256']):
                    raise ValueError('raw_response_or_receipt_changed')
                if record.status=='response':
                    envelope=dict(schema_version='provider-replay-1',source_type='recorded_provider_response',
                        model_requested=model,model_returned=record.model_returned,provider_response_id=record.provider_response_id,
                        task_sha256=h,prompt_sha256=sha(prompt(tasks[h])),
                        answer_schema_sha256=sha(Proposal.model_json_schema()),variant='base',response_text=record.proposals_wire)
                    value=adapt(tasks[h],envelope)
                    parsed=dict(status=value['status'],proposals=value['proposals'],rejected_proposals=0)
            selected[h][slot]=parsed
        from .audits import graph_audits,write_csv
        graph_records=graph_audits(tasks,manifest,out)
        write_once(out/'graph_audit_batch.json',graph_records)
        write_csv(out/'graph_admissions.csv',graph_records['graphs'])
        write_csv(out/'edge_claim_audit.csv',graph_records['edges'])
        samples=json.loads((out/'sample_manifest.json').read_text()); batch=[]
        guard.inputs.update(out/x['file'] for x in samples)
        for item in samples:
            if digest(out/item['file'])!=item['sha256']: raise ValueError('sample_digest_mismatch')
            h=item['task_hash']; frame=read_sample(tasks[h],json.loads((out/item['file']).read_text()))
            prediction=analyze(tasks[h],frame,selected[h],SLOTS,ESTIMATION)
            prediction.update(sample_id=item['sample_id'],sample_repetition=0,
                source_slot_mapping=dict(zip(SLOTS,MODELS)),source_type='recorded_provider_response')
            batch.append(prediction)
        write_once(out/'prediction_batch.json',batch)
        write_once(out/'prediction_guard_receipt.json',dict(role=role,scoring_truth_readable=False,
            generator_plan_readable=False,truth_module_imported=False,source_slots_are_internal_aliases=True,
            share_records_in_prompts=0,provider_prompts_receive_no_observed_samples=True))
    elif role=='score':
        seal=json.loads((out/'prediction_seal.json').read_text())
        guard.inputs.update(out/n for n in seal)
        verify(out,seal,'prediction_seal.json',job['prediction_digest'])
        guard.truth_ready=True
        from sharecausal.pilot_v24.evaluation import evaluate
        result=evaluate(json.loads((out/'prediction_batch.json').read_text()),
            json.loads((out/'scoring_truth.json').read_text()),tasks,
            json.loads((out/'families.json').read_text())['assignments'],SCORING)
        write_once(out/'scored_results.json',result)
        write_once(out/'scorer_receipt.json',dict(role=role,seal_verified_before_truth_open=True,
            prediction_seal_sha256=job['prediction_digest'],scoring_truth_sha256=digest(out/'scoring_truth.json'),
            prediction_files_updated=False,custody='single_team_separate_local_processes'))
        verify(out,seal,'prediction_seal.json',job['prediction_digest'])
    else: raise ValueError('unknown_worker')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        # Fixed diagnostic types and code positions; no values, secrets or records.
        positions=[]; tb=exc.__traceback__
        while tb:
            p=Path(tb.tb_frame.f_code.co_filename)
            if 'sharecausal' in p.parts: positions.append(dict(module=p.stem,line=tb.tb_lineno))
            tb=tb.tb_next
        print(json.dumps(dict(status='worker_failed',type=type(exc).__name__,positions=positions)),file=sys.stderr)
        sys.exit(2)
