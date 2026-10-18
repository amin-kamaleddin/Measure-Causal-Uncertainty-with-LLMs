"""Durable, bounded, synthetic-only OpenRouter transport.

Deliberately does not call the SHARE-specific complete()/GraphAnswer parser.
Reuse catalog, base endpoint and conservative estimate from the existing client.
No automatic POST retry: uncertain network outcomes must not cause duplicate bills.
"""
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
from pydantic import Field
from typing import Literal
from sharecausal.llm.openrouter import BASE, available_models, upper_cost_estimate
from sharecausal.preflight.contracts import Strict, canonical, sha
from sharecausal.preflight_v23.packets import task_hash
from .public import MODELS, messages, schema, settings, request_key, Answer


def now(): return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    temporary.chmod(0o600); os.replace(temporary,path)


from .records import Receipt, validated


def proposals(body):
    choices=body.get('choices')
    choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {}
    msg=choice.get('message')
    msg=msg if isinstance(msg,dict) else {}
    if msg.get('refusal'): return 'refusal',None,choice.get('finish_reason')
    raw=msg.get('content')
    if not isinstance(raw,str) or not raw.strip(): return 'null_content',None,choice.get('finish_reason')
    try:
        # Preserve duplicate-key rejection through the existing parser in prediction;
        # reject duplicates at the wrapper too, before Pydantic can collapse them.
        from sharecausal.pilot_v24.capture import _unique_object, _reject_constant
        decoded=json.loads(raw,object_pairs_hook=_unique_object,parse_constant=_reject_constant)
        answer=Answer.model_validate(decoded)
        return 'response',canonical([p.model_dump() for p in answer.proposals]).decode(),choice.get('finish_reason')
    except (ValueError,TypeError,RecursionError): return 'schema_failure',None,choice.get('finish_reason')


def collect(tasks, out, cap, key, *, client=None, catalog=None, progress=print):
    # POSIX advisory lock serializes resume invocations, including the check/create
    # gap around inflight records. Never unlink the lock inode while another process waits.
    import fcntl
    with (out/'collection.lock').open('a') as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('concurrent_collection_forbidden') from None
        try: return _collect(tasks,out,cap,key,client=client,catalog=catalog,progress=progress)
        finally: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def _collect(tasks, out, cap, key, *, client=None, catalog=None, progress=print):
    if not 0 < cap <= 25 or not key: raise ValueError('authorized_key_and_cap_required')
    owned=client is None; client=client or httpx.Client(timeout=150., follow_redirects=False)
    try:
        catalog_path=out/'catalog_snapshot.json'
        if catalog is None:
            catalog=(json.loads(catalog_path.read_text()) if catalog_path.exists()
                     else available_models(transport=client))
        by_id={m['id']:m for m in catalog}
        selected={m:by_id[m] for m in MODELS if m in by_id}
        if not catalog_path.exists(): atomic(catalog_path,[selected[m] for m in selected])
        else:
            # Resume with the exact previously committed capabilities/prices.
            selected={m['id']:m for m in json.loads(catalog_path.read_text())}
        request_dir=out/'provider_responses_private'; request_dir.mkdir(exist_ok=True,mode=0o700)
        schedule=[]
        stability=set(json.loads((out/'design_commitment.json').read_text())['counterargument_task_hashes'])
        for h,task in sorted(tasks.items()):
          for variant in (['base','counterargument'] if h in stability else ['base']):
            for m in MODELS:
                if m not in selected:
                    schedule.append(dict(task_sha256=h,model=m,variant=variant,status='endpoint_unavailable')); continue
                effective=settings(selected[m]); chat=messages(task,variant)
                reserve=upper_cost_estimate(selected[m],len(canonical(chat)),effective['max_tokens'])
                if not math.isfinite(reserve) or reserve<=0: raise ValueError('unknown_or_invalid_pricing')
                schedule.append(dict(task_sha256=h,model=m,variant=variant,settings=effective,
                    prompt_sha256=sha(chat),schema_sha256=sha(schema()),
                    idempotency_key=request_key(h,m,chat,effective,variant),reserve_usd=reserve))
        plan=out/'request_plan.json'
        if not plan.exists(): atomic(plan,dict(planned_primary_attempts=96,planned_stability_attempts=32,max_post_attempts=128,
            retries=0,software_cap_usd=cap,upper_reservation_usd=sum(x.get('reserve_usd',0) for x in schedule),schedule=schedule))
        else:
            old=json.loads(plan.read_text())
            if old['schedule']!=schedule or old['software_cap_usd']!=cap: raise ValueError('request_plan_changed')
        reserve_total=sum(x.get('reserve_usd',0) for x in schedule)
        if reserve_total>cap: raise ValueError('batch_reservation_exceeds_cap')
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event, Lock
        stop=Event(); status_lock=Lock()
        records=[]; cached=0; new=0
        # Whole-batch reservations fit cap; all submitted and pending requests are
        # covered simultaneously, including four bounded concurrent POSTs.
        # Unknown prior outcomes are never resent and block all new spending.
        for item in schedule:
            if item.get('status')=='endpoint_unavailable': continue
            identity=item['idempotency_key']
            rp=request_dir/(identity+'.json')
            if (request_dir/(identity+'.inflight.json')).exists() and not rp.exists(): stop.set()
            if rp.exists():
                r=validated(tasks[item['task_sha256']],item['model'],json.loads(rp.read_text()))
                if r.cost_usd is None or r.cost_usd>r.reserved_usd: stop.set()
        def perform(item):
            nonlocal cached,new
            if item.get('status')=='endpoint_unavailable': return
            h=item['task_sha256']; m=item['model']; variant=item['variant']; identity=item['idempotency_key']
            record_path=request_dir/(identity+'.json'); raw_path=request_dir/(identity+'.raw.json')
            inflight=request_dir/(identity+'.inflight.json')
            if record_path.exists():
                record=validated(tasks[h],m,json.loads(record_path.read_text())).model_dump()
                if sha(json.loads((out/record['raw_file']).read_text()))!=record['response_sha256']:
                    raise ValueError('provider_raw_digest_mismatch')
                with status_lock:
                    records.append(record); cached+=1
                return
            if stop.is_set() or inflight.exists(): return
            payload=dict(model=m,messages=messages(tasks[h],variant),**item['settings'],
                response_format=dict(type='json_schema',json_schema=dict(name='synthetic_graphs',strict=True,schema=schema())),
                usage=dict(include=True))
            # Persist reservation before touching network. No secrets or headers in receipts.
            atomic(inflight,dict(**item,request_timestamp=now(),state='reserved_before_network'))
            started=time.monotonic(); stamp=now(); body={}; code=None; finish=None; wire=None; status='transport_error'
            try:
                response=client.post(BASE+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload)
                code=response.status_code
                try: body=response.json()
                except ValueError: body=dict(non_json_body_sha256=sha(response.text))
                if not isinstance(body,dict): body=dict(unexpected_body=body)
                # Save immediately, including billed malformed/null/error responses.
                atomic(raw_path,body)
                if response.is_error: status='http_error'
                elif body.get('error'): status='provider_error'
                else: status,wire,finish=proposals(body)
            except httpx.TransportError:
                atomic(raw_path,dict(transport_error=True,outcome='unknown_no_retry'))
                body=json.loads(raw_path.read_text())
            usage=body.get('usage'); usage=usage if isinstance(usage,dict) else {}; cost=usage.get('cost')
            if cost is not None and (not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0): cost=None
            # Explicit HTTP errors with no generation ID are retained at zero reported charge;
            # transport uncertainty or missing usage on a generated response blocks further spend.
            if cost is None and status=='http_error' and not body.get('id'): cost=0.
            if status=='response' and body.get('model')!=m: status='model_mismatch'; wire=None
            if status=='response' and (not body.get('provider') or not body.get('id')):
                status='provider_error'; wire=None
            transport_outcome=status
            if cost is None: status='cost_unknown'; stop.set()
            record=Receipt(version='synthetic-provider-1',source_type='recorded_provider_response',
                endpoint=BASE+'/chat/completions',variant=variant,repetition=0,model_family=m.split('/')[0],
                catalog_snapshot_sha256=sha(json.loads(catalog_path.read_text())),
                design_commitment_sha256=sha(json.loads((out/'design_commitment.json').read_text())) if (out/'design_commitment.json').exists() else None,
                idempotency_key=identity,task_sha256=h,prompt_sha256=item['prompt_sha256'],
                schema_sha256=item['schema_sha256'],model_requested=m,model_returned=body.get('model'),
                provider=body.get('provider'),provider_response_id=body.get('id'),request_timestamp=stamp,
                response_timestamp=now(),settings=item['settings'],status=status,
                transport_outcome=transport_outcome,http_status=code,
                reserved_usd=item['reserve_usd'],cost_usd=cost,usage=usage,
                latency_seconds=time.monotonic()-started,response_sha256=sha(body),proposals_wire=wire,
                raw_file=str(raw_path.relative_to(out)),finish_reason=finish).model_dump()
            atomic(record_path,record)
            if cost is not None and cost>item['reserve_usd']: stop.set()
            with status_lock:
                records.append(record); new+=1
                atomic(out/'collection_status.json',dict(new_posts_this_invocation=new,cached=cached,
                    accounted_cost_usd=sum(r['cost_usd'] if r['cost_usd'] is not None else r['reserved_usd'] for r in records),
                    stopped=stop.is_set(),complete_records=len(records)))
                progress(json.dumps(dict(completed=len(records),planned=128,model=m,variant=variant,status=status,cost_usd=cost)),flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(perform,schedule))
        stopped=stop.is_set()
        atomic(out/'collection_summary.json',dict(new_posts_this_invocation=new,cached=cached,
            actual_posts_total=len(records),planned=128,cost_usd=sum(r['cost_usd'] or 0 for r in records),
            unresolved_costs=sum(r['cost_usd'] is None for r in records),stopped=stopped,
            unresolved_inflight_records=len(list(request_dir.glob('*.inflight.json')))-len(records),
            endpoint_unavailable=sum(i.get('status')=='endpoint_unavailable' for i in schedule)))
        return records,schedule
    finally:
        if owned: client.close()
