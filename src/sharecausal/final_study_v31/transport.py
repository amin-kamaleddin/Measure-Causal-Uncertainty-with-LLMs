"""Durable real-provider receipts, whole-study reservation, no paid retries."""
import json,math,time,fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Event,Lock
from pathlib import Path
import httpx
from sharecausal.llm.openrouter import BASE,available_models,upper_cost_estimate
from sharecausal.synthetic_final_v1.transport import atomic,now
from sharecausal.preflight.contracts import canonical,sha
from .public import MODELS,messages,schema,settings,decode


def plan(cards,catalog,cap):
 by_id={m['id']:m for m in catalog}
 if set(MODELS)-set(by_id):raise ValueError('exact_model_endpoint_unavailable')
 schedule=[]
 for card in sorted(cards,key=lambda x:x.task_id):
  for variant in ('base','counterargument'):
   for repetition in range(2):
    for model in MODELS:
     chat=messages(card,variant,repetition);effective=settings(by_id[model])
     # Provider price ceilings are twice catalog rates; reserve that ceiling,
     # not an optimistic historical mean. Full UTF-8 bytes conservatively bound
     # prompt tokens. Schemas and request framing included in the bound.
     prompt_bound=len(canonical(chat))+len(canonical(schema()))+4096
     reserve=2*upper_cost_estimate(by_id[model],prompt_bound,effective['max_tokens'])
     schedule.append(dict(task_id=card.task_id,model=model,variant=variant,repetition=repetition,settings=effective,reserve_usd=reserve,prompt_sha256=sha(chat),schema_sha256=sha(schema())))
 critic_model=MODELS[0];cfg=settings(by_id[critic_model])
 critic_reserve=2*upper_cost_estimate(by_id[critic_model],300000,cfg['max_tokens'])*len(cards)
 total=sum(x['reserve_usd'] for x in schedule)+critic_reserve
 if not math.isfinite(total) or total<=0 or total>cap:raise ValueError('whole_study_reservation_exceeds_cap')
 return dict(primary=schedule,critic_model=critic_model,critic_max_calls=len(cards),critic_reserve_usd=critic_reserve,upper_reservation_usd=total,cap_usd=cap,retries=0,concurrency=64)


def collect(cards,out,study_plan,key,stage,progress=print,client=None):
 out=Path(out);directory=out/'receipts';directory.mkdir(mode=0o700,exist_ok=True)
 with (out/'collector.lock').open('a') as lock:
  try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise ValueError('concurrent_collection_forbidden') from None
  return _collect(cards,out,directory,study_plan,key,stage,progress,client)


def _collect(cards,out,directory,study_plan,key,stage,progress,client):
 if not key or not 0<study_plan['cap_usd']<=100:raise ValueError('key_and_approved_cap_required')
 by_id={c.task_id:c for c in cards};schedule=[dict(x) for x in study_plan['primary'] if x['task_id'] in by_id]
 catalog={m['id']:m for m in json.loads((out/'catalog.json').read_text())}
 if stage=='critic':
  schedule=[]
  for card in cards:
   graphs=[]
   for r in load_records(directory):
    if r['task_id']==card.task_id and r['variant']=='base' and r['repetition']==0 and r['status']=='valid':graphs.extend(r['answer']['graphs'])
   unique={sha(g):g for g in graphs};graphs=[unique[h] for h in sorted(unique)]
   model=study_plan['critic_model'];chat=messages(card,'critic',0,graphs)
   effective=settings(catalog[model]);reserve=2*upper_cost_estimate(catalog[model],len(canonical(chat))+len(canonical(schema(True)))+4096,effective['max_tokens'])
   if len(canonical(chat))>290000:raise ValueError('critic_prompt_exceeds_reservation')
   schedule.append(dict(task_id=card.task_id,model=model,variant='critic',repetition=0,settings=effective,reserve_usd=reserve,prompt_sha256=sha(chat),schema_sha256=sha(schema(True)),graphs=graphs))
 for item in schedule:item['request_key']=sha({k:v for k,v in item.items() if k not in ('graphs','request_key')})
 for p in directory.glob('*.inflight.json'):
  if not (directory/(p.name.removesuffix('.inflight.json')+'.json')).exists():raise ValueError('unresolved_paid_request_no_automatic_retry')
 existing=load_records(directory)
 if any(r['cost_usd'] is None or r['cost_usd']>r['reserve_usd'] for r in existing):raise ValueError('unreconciled_cost_blocks_new_calls')
 spent=sum(r['cost_usd'] for r in existing)
 pending=sum(i['reserve_usd'] for i in schedule if not (directory/(i['request_key']+'.json')).exists())
 if spent+pending>study_plan['cap_usd']:raise ValueError('spending_ceiling')
 own=client is None;client=client or httpx.Client(timeout=httpx.Timeout(660,connect=30),follow_redirects=False)
 stop=Event();mutex=Lock();count=0;cached=0
 def perform(item):
  nonlocal count,cached
  identity=item['request_key'];rp=directory/(identity+'.json');raw=directory/(identity+'.raw.json');inflight=directory/(identity+'.inflight.json')
  if not rp.exists() and (out/'legacy_receipts'/(identity+'.json')).exists():
   import shutil
   legacy=out/'legacy_receipts'
   for suffix in ('.json','.raw.json','.inflight.json'):
    old=legacy/(identity+suffix)
    if old.exists():shutil.copyfile(old,directory/(identity+suffix))
  if rp.exists():
   record=json.loads(rp.read_text())
   if record['request_key']!=identity or record['prompt_sha256']!=item['prompt_sha256'] or record['response_sha256']!=sha(json.loads(raw.read_text())):raise ValueError('cached_receipt_binding_failure')
   with mutex:cached+=1
   return
  if stop.is_set():return
  chat=messages(by_id[item['task_id']],item['variant'],item['repetition'],item.get('graphs'))
  if sha(chat)!=item['prompt_sha256']:raise ValueError('prompt_changed')
  payload=dict(model=item['model'],messages=chat,**item['settings'],usage=dict(include=True),response_format=dict(type='json_schema',json_schema=dict(name='evidence_critic' if stage=='critic' else 'evidence_graphs',strict=True,schema=schema(stage=='critic'))))
  atomic(inflight,dict(**{k:v for k,v in item.items() if k!='graphs'},reserved_at=now(),state='reserved_before_network'))
  started=now();clock=time.monotonic();body={};code=None;status='transport_error';answer=None
  try:
   response=client.post(BASE+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload);code=response.status_code
   try:body=response.json()
   except ValueError:body=dict(non_json_body_sha256=sha(response.text))
   if not isinstance(body,dict):body=dict(unexpected_body=body)
   atomic(raw,body)
   if response.is_error:status='http_error'
   elif body.get('error'):status='provider_error'
   elif body.get('model')!=item['model']:status='model_mismatch'
   elif not body.get('id') or not body.get('provider'):status='provenance_failure'
   else:status,answer=decode(body,stage=='critic')
  except httpx.TransportError:
   body=dict(transport_error=True,outcome_unknown=True);atomic(raw,body)
  usage=body.get('usage') or {};cost=usage.get('cost')
  if not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0:cost=None
  if cost is None and status=='http_error' and not body.get('id'):cost=0.
  provider_expected=dict(zip(MODELS,('Google AI Studio','OpenAI','DigitalOcean','OpenInference')))[item['model']]
  if body.get('provider') and body['provider']!=provider_expected:
   status='unapproved_provider';answer=None;stop.set()
  if cost is None or cost>item['reserve_usd']:stop.set()
  record=dict(**{k:v for k,v in item.items() if k!='graphs'},started_utc=started,ended_utc=now(),latency_seconds=time.monotonic()-clock,status=status,answer=answer,http_status=code,cost_usd=cost,usage=usage,model_returned=body.get('model'),provider=body.get('provider'),provider_response_id=body.get('id'),response_sha256=sha(body),source_type='actual_openrouter_response')
  atomic(rp,record)
  with mutex:
   count+=1
   with (out/'request_ledger.jsonl').open('a') as stream:stream.write(canonical(dict(request_key=identity,status=status,cost_usd=cost,ended_utc=record['ended_utc'])).decode()+'\n');stream.flush()
   if count%16==0 or stop.is_set():progress(json.dumps(dict(stage=stage,completed_new=count,cached=cached,planned=len(schedule),model=item['model'],status=status,stopped=stop.is_set())),flush=True)
 try:
  with ThreadPoolExecutor(max_workers=64) as pool:
   futures=[pool.submit(perform,i) for i in schedule]
   for future in as_completed(futures):future.result()
 finally:
  if own:client.close()
 records=load_records(directory)
 summary=dict(stage=stage,new_posts=count,cached=cached,planned=len(schedule),total_receipts=len(records),cost_usd=sum(r['cost_usd'] or 0 for r in records),unknown_costs=sum(r['cost_usd'] is None for r in records),stopped=stop.is_set())
 atomic(out/(stage+'_collection_summary_'+sha(sorted(by_id))+'.json'),summary)
 if stop.is_set():raise ValueError('provider_privacy_or_cost_integrity_stop')
 if count+cached!=len(schedule):raise ValueError('incomplete_request_batch')
 return summary


def load_records(directory):
 return [json.loads(p.read_text()) for p in sorted(directory.glob('*.json')) if not p.name.endswith(('.raw.json','.inflight.json'))]
