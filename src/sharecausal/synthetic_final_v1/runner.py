"""Local orchestrator: no pre-seal truth access; post-score artifact hashes include truth."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime,timezone
from sharecausal.preflight.contracts import sha, canonical
from sharecausal.preflight_v23.packets import PublicTask,task_hash
from sharecausal.preflight_v23.sealing import write_once,seal,verify,digest
from .public import MODELS,SLOTS,ESTIMATION,SCORING,messages,schema
from .execution import execute,record

ROOT=Path(__file__).resolve().parents[3]


def worker(role,out,cap,digest_value=None):
    job=out/('job_'+role+'.json')
    data=dict(output=str(out),cap=cap,prediction_digest=digest_value)
    if not job.exists(): write_once(job,data)
    elif json.loads(job.read_text())!=data: raise ValueError('worker_job_changed')
    command=[sys.executable,'-m','sharecausal.synthetic_final_v1.workers',role,str(job)]
    child_env=dict(os.environ)
    if role!='collect': child_env.pop('OPENROUTER_API_KEY',None)
    completed=execute(command,ROOT,out,env=child_env)
    if completed.returncode: raise RuntimeError('worker_'+role+'_failed')


def prepare(out,cap):
    if out.is_relative_to(ROOT) or out==out.parent: raise ValueError('private_output_must_be_outside_repository')
    if not out.exists():
        command=[sys.executable,'-c',
            'import sys,json; from sharecausal.synthetic_final_v1.authoring import build; print(json.dumps(build(sys.argv[1])))',str(out)]
        result=execute(command,ROOT,out)
        if result.returncode: raise RuntimeError('task_authoring_failed')
    tasks=[PublicTask.model_validate(x) for x in json.loads((out/'public_tasks.json').read_text())]
    if len(tasks)!=24 or len({task_hash(t) for t in tasks})!=24: raise ValueError('twenty_four_unique_cards_required')
    source_paths=sorted((ROOT/'src/sharecausal').rglob('*.py'))+[ROOT/'scripts/synthetic_final_v1.py']
    commitment=dict(version='synthetic-live-1',protocol_status='candidate_not_frozen',study_kind='exploratory',
        software_cap_usd=cap,planned_primary_attempts=96,planned_stability_attempts=32,max_post_attempts=128,retries=0,
        counterargument_task_hashes=sorted(task_hash(t) for t in tasks)[:8],
        bounded_concurrency=4,max_tokens_deepseek=8192,max_tokens_other=4096,
        analysis_plan_sha256=digest(ROOT/'docs/final_analysis_plan_exploratory.md'),
        models=list(MODELS),slots=dict(zip(SLOTS,MODELS)),estimation=ESTIMATION,scoring=SCORING,
        sample_size=800,sample_repetitions=1,subgroups='none_prespecified',
        assignment_of_cards='content_hash_order_independent_of_truth',
        public_tasks_sha256=digest(out/'public_tasks.json'),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        environment=dict(python=platform.python_version(),platform=platform.platform(),
            packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','pydantic','httpx','matplotlib')}),
        custody='single_team_authored_separate_guarded_processes_not_independent_registration')
    target=out/'design_commitment.json'
    if not target.exists(): write_once(target,commitment)
    elif json.loads(target.read_text())!=commitment: raise ValueError('committed_code_or_design_changed_new_run_required')
    manifest=[dict(task_id=t.task_id,task_sha256=task_hash(t),card_sha256=sha(t.card.model_dump()),
        prompt_sha256=sha(messages(t)),schema_sha256=sha(schema()),variant='base',repetition=0) for t in tasks]
    if not (out/'task_manifest.json').exists(): write_once(out/'task_manifest.json',manifest)
    return tasks


def run(out,cap,live):
    os.umask(0o077)
    tasks=prepare(out,cap)
    print(json.dumps(dict(proposed_primary_calls=96,stability_calls=32,maximum_posts=128,retries=0,
        software_cap_usd=cap,live=live,results_dir=str(out))),flush=True)
    if not live: return 0
    if not os.environ.get('OPENROUTER_API_KEY'): raise ValueError('OPENROUTER_API_KEY_missing')
    # Frozen-protocol integrity is a precondition to all POSTs.
    checked=execute([sys.executable,'scripts/check_protocol_freeze.py'],ROOT,out)
    if checked.returncode: raise ValueError('frozen_protocol_check_failed')
    if not (out/'sample_manifest.json').exists(): worker('generate',out,cap)
    if not (out/'prediction_batch.json').exists():
        worker('collect',out,cap)
        from .transport import Receipt
        records={}
        for p in (out/'provider_responses_private').glob('*.json'):
            if p.name.endswith(('.raw.json','.inflight.json')): continue
            r=Receipt.model_validate(json.loads(p.read_text()))
            records[(r.task_sha256,r.model_requested,r.variant)]=(p,r)
        request_manifest=[]
        stability=set(json.loads((out/'design_commitment.json').read_text())['counterargument_task_hashes'])
        for task in tasks:
            h=task_hash(task)
            for variant in (['base','counterargument'] if h in stability else ['base']):
              for m in MODELS:
                item=dict(task_sha256=h,model=m,variant=variant,status='not_attempted_endpoint_budget_or_unknown_prior_outcome')
                if (h,m,variant) in records:
                    p,r=records[h,m,variant]
                    item.update(status=r.status,record_file=str(p.relative_to(out)),record_sha256=digest(p),
                                raw_file_sha256=digest(out/r.raw_file),cost_usd=r.cost_usd,
                                prompt_sha256=r.prompt_sha256,schema_sha256=r.schema_sha256)
                request_manifest.append(item)
        request_path=out/'request_manifest.jsonl'
        if not request_path.exists(): request_path.write_text(''.join(canonical(r).decode()+'\n' for r in request_manifest))
        elif request_path.read_text()!=''.join(canonical(r).decode()+'\n' for r in request_manifest):
            raise ValueError('sealed_request_inventory_changed')
        worker('predict',out,cap)
    if not (out/'prediction_seal.json').exists():
        names=['public_tasks.json','task_manifest.json','request_manifest.jsonl','design_commitment.json',
            'sample_manifest.json','generator_receipt.json','prediction_batch.json','prediction_guard_receipt.json',
            'request_plan.json','catalog_snapshot.json','graph_audit_batch.json','graph_admissions.csv','edge_claim_audit.csv','stability_prediction_batch.json']
        names += [x['file'] for x in json.loads((out/'sample_manifest.json').read_text())]
        names += [str(p.relative_to(out)) for p in sorted((out/'provider_responses_private').glob('*.json'))]
        _, sealed=seal(out,names,'prediction_seal.json')
        (out/'prediction_batch_sha256.txt').write_text(digest(out/'prediction_batch.json')+'\n')
        (out/'prediction_seal_sha256.txt').write_text(sealed+'\n')
        for n in names+['prediction_seal.json','prediction_batch_sha256.txt','prediction_seal_sha256.txt']:
            (out/n).chmod(0o400)
    sealed=(out/'prediction_seal_sha256.txt').read_text().strip()
    verify(out,json.loads((out/'prediction_seal.json').read_text()),'prediction_seal.json',sealed)
    if not (out/'scored_results.json').exists(): worker('score',out,cap,sealed)
    # Reporting only consumes post-seal results, never performs requests or alters decisions.
    from .reporting import report,final_diagnostics
    report(out)
    final_diagnostics(out)
    return 0 if json.loads((out/'collection_summary.json').read_text())['actual_posts_total']==128 else 3


def main():
    parser=argparse.ArgumentParser(description='Final expansion: 128 fresh synthetic graph attempts, exploratory, private')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--live',action='store_true')
    parser.add_argument('--max-cost-usd',type=float,default=25.)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--key-env-file',type=Path,
        help='Explicit existing credential file; load only OPENROUTER_API_KEY without printing it')
    parser.add_argument('--key-file',type=Path,
        help='Explicit existing file containing only the raw authorized key; never copied or logged')
    args=parser.parse_args()
    if not 0<args.max_cost_usd<=25: parser.error('cap must be >0 and <=25 USD')
    out=args.results_dir.expanduser().resolve(); start=datetime.now(timezone.utc).isoformat()
    try:
        if args.live and args.key_file and not os.environ.get('OPENROUTER_API_KEY'):
            credential=args.key_file.expanduser().read_text().strip()
            if not credential or any(c.isspace() for c in credential): raise ValueError('invalid_raw_key_file')
            os.environ['OPENROUTER_API_KEY']=credential
            del credential
        if args.live and args.key_env_file and not os.environ.get('OPENROUTER_API_KEY'):
            import shlex
            matches=[]
            for line in args.key_env_file.expanduser().read_text().splitlines():
                name,separator,value=line.removeprefix('export ').partition('=')
                if separator and name.strip()=='OPENROUTER_API_KEY':
                    pieces=shlex.split(value,comments=True)
                    if len(pieces)!=1: raise ValueError('invalid_credential_assignment')
                    matches.append(pieces[0])
            if len(matches)!=1 or not matches[0]: raise ValueError('one_key_assignment_required')
            os.environ['OPENROUTER_API_KEY']=matches[0]
        code=run(out,args.max_cost_usd,args.live)
    except Exception as exc:
        print(json.dumps(dict(status='failed',exception=type(exc).__name__,
            hint='Inspect guarded worker diagnostics; preserve outputs and resume with identical code/config. No automatic paid retries.')),
            file=sys.stderr)
        code=2
    if out.is_dir():record(out,[sys.executable,*sys.argv],code,start,datetime.now(timezone.utc).isoformat(),cwd=str(ROOT))
    return code
