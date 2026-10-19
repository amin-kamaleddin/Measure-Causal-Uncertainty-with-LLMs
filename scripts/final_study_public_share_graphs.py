"""Missing public-evidence SHARE prompt repeats. No empirical-data imports/reads."""
import argparse,json,os,time,math,hashlib,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import httpx
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.synthetic_final_v1.transport import atomic,now
from sharecausal.final_study_v31.public import MODELS,settings
from sharecausal.llm.schemas import GraphAnswer
from sharecausal.llm.openrouter import BASE,upper_cost_estimate
from sharecausal.pilot_v24.capture import _unique_object,_reject_constant

SOURCES=[
 dict(id='D1',url='https://share-eric.eu/fileadmin/user_upload/Other_Publications/ScalesManual_rel.9-0-0_updated.pdf',note='The activities module asks about participation during the previous twelve months. A is Wave8 club/sport/social-group participation. This defines a reported behavior,not an assigned intervention.'),
 dict(id='D2',url='https://share-eric.eu/fileadmin/user_upload/Other_Publications/ScalesManual_rel.9-0-0_updated.pdf',note='EURO-D aggregates twelve depressive symptom items; caseness is scale>=4. Y is Wave9 EURO-D. D8 is Wave8 depression and may overlap the prior-year exposure reference period. Do not assume D8 precedes all participation.'),
 dict(id='D3',url='https://share-eric.eu/data/faqs-support',note='Wave7 used retrospective SHARELIFE and regular interview universes. Childhood recall H and regular earlier depression are not automatically jointly available as a common baseline.'),
 dict(id='D4',url='https://share-eric.eu/fileadmin/user_upload/SHARE_Working_Paper/SHARE_WP_104-2025.pdf',note='Participation,attrition and sample development must be considered in longitudinal SHARE. S is later observation/survival; depressive symptoms are undefined after death. Unknown survival is not known survival.'),
 dict(id='L1',url='https://doi.org/10.1093/aje/kwv015',note='Croezen and colleagues studied social participation and depression using observational fixed-effects analyses. An observational analysis does not establish every proposed direct causal arrow.'),
 dict(id='L2',url='https://link.springer.com/article/10.1186/s12916-026-04846-4',note='Wang and colleagues used repeated observational social engagement histories and targeted estimation with explicit time-varying assumptions. Their target differs from one Wave8 participation report followed by Wave9 symptoms; no estimates are supplied here.'),
 dict(id='L3',url='https://proceedings.mlr.press/v337/li26e.html',note='Li and Russo treat language models as imperfect sources of causal constraints. Agreement and fluent explanations do not prove a causal graph or intervention effect.')]

def schema():
 s=GraphAnswer.model_json_schema()
 def walk(x):
  if isinstance(x,dict):
   for k in ('minItems','maxItems','default'):x.pop(k,None)
   if x.get('type')=='object':x['required']=list(x.get('properties',{}));x['additionalProperties']=False
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(s);return s

def messages(variant,rep):
 return [dict(role='system',content='You are auditing causal assumptions from public documentation and literature. No respondent data or empirical estimates are provided. Propose a complete hypothesis DAG or explicitly abstain. Every evidence_reference must cite an existing ID and exact quoted substring,or say unverified hypothesis. Documentation of measurement is not proof of a causal arrow. Do not invent studies or infer effects from model agreement.'),dict(role='user',content=canonical(dict(question='Assess the identifiability and graph assumptions for setting Wave8 prior-year club participation A and later Wave9 EURO-D Y. B=pre-exposure demographics/country,H=retrospective childhood circumstances,U=unmeasured health/social background,D8=temporally ambiguous Wave8 depression,S=Wave9 survival/recording. Retain ambiguity about timing,confounding and selection; no unconditional effect sign is requested.',public_sources=SOURCES,condition=variant,repetition=rep,instruction='Challenge each proposed edge and consider counterarguments.' if variant=='counterargument' else 'Audit all public evidence before proposing a graph.',answer_schema=schema())).decode())]

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--catalog',required=True,type=Path);p.add_argument('--routing-run',required=True,type=Path);p.add_argument('--key-file',required=True,type=Path);a=p.parse_args();os.umask(0o077);out=a.output.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=True)
 catalog={m['id']:m for m in json.loads(a.catalog.read_text())};planned=[]
 for model in MODELS:
  for variant in ('base','counterargument'):
   for repetition in range(2):
    chat=messages(variant,repetition);cfg=settings(catalog[model]);reserve=2*upper_cost_estimate(catalog[model],len(canonical(chat))+len(canonical(schema()))+4096,cfg['max_tokens']);item=dict(model=model,variant=variant,repetition=repetition,settings=cfg,prompt_sha256=sha(chat),schema_sha256=sha(schema()),reserve_usd=reserve);item['request_key']=sha(item);planned.append(item)
 routing=json.loads((a.routing_run/'routing_repair_provenance.json').read_text());primary_plan=json.loads((a.routing_run/'request_plan.json').read_text());union=primary_plan['upper_reservation_usd']+routing['prior_actual_charges_and_historical_known_synthetic_usd']+sum(x['reserve_usd'] for x in planned)
 if union>100:raise ValueError('joint_software_reservation_exceeds100')
 commitment=dict(kind='public_documentation_only_no_empirical_results',sources=SOURCES,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),schema=schema(),schedule=planned,joint_upper_reservation_usd=union)
 target=out/'public_prompt_commitment.json'
 if not target.exists():atomic(target,commitment);target.chmod(0o400)
 elif json.loads(target.read_text())!=commitment:raise ValueError('public_prompt_commitment_changed')
 key=a.key_file.expanduser().read_text().strip()
 if not key or any(c.isspace() for c in key):raise ValueError('invalid_runtime_key')
 for i in planned:
  if (out/(i['request_key']+'.inflight.json')).exists() and not (out/(i['request_key']+'.json')).exists():raise ValueError('unresolved_call_no_retry')
 with httpx.Client(timeout=httpx.Timeout(660,connect=30),follow_redirects=False) as client:
  def call(item):
   h=item['request_key'];receipt=out/(h+'.json');raw=out/(h+'.raw.json')
   if receipt.exists():
    r=json.loads(receipt.read_text())
    if r['response_sha256']!=sha(json.loads(raw.read_text())):raise ValueError('cached_public_response_changed')
    return
   atomic(out/(h+'.inflight.json'),dict(**item,reserved_at=now()));start=now();body={};code=None;status='transport_error';answer=None
   try:
    response=client.post(BASE+'/chat/completions',headers={'Authorization':'Bearer '+key},json=dict(model=item['model'],messages=messages(item['variant'],item['repetition']),**item['settings'],usage=dict(include=True),response_format=dict(type='json_schema',json_schema=dict(name='public_share_graph',strict=True,schema=schema()))));code=response.status_code
    try:body=response.json()
    except ValueError:body=dict(non_json_body_sha256=sha(response.text))
    if not isinstance(body,dict):body=dict(unexpected_body=body)
    atomic(raw,body)
    if response.is_error:status='http_error'
    elif body.get('error'):status='provider_error'
    elif body.get('model')!=item['model']:status='model_mismatch'
    elif body.get('provider')!=dict(zip(MODELS,('Google AI Studio','OpenAI','DigitalOcean','OpenInference')))[item['model']]:status='unapproved_provider'
    elif not body.get('id'):status='provenance_failure'
    else:
     choices=body.get('choices');choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {};msg=choice.get('message');msg=msg if isinstance(msg,dict) else {};content=msg.get('content')
     if msg.get('refusal'):status='refusal'
     elif not isinstance(content,str) or not content.strip():status='null_content'
     else:
      try:answer=GraphAnswer.model_validate(json.loads(content,object_pairs_hook=_unique_object,parse_constant=_reject_constant)).model_dump();status='explicit_abstention' if answer['abstain'] else 'valid'
      except (ValueError,TypeError,RecursionError):status='schema_or_graph_failure'
   except httpx.TransportError:atomic(raw,dict(transport_outcome_unknown=True));body=dict(transport_outcome_unknown=True)
   usage=body.get('usage') or {};cost=usage.get('cost') if isinstance(usage,dict) else None
   if cost is None and status=='http_error' and not body.get('id'):cost=0.
   record=dict(**item,status=status,answer=answer,cost_usd=cost,usage=usage,http_status=code,provider=body.get('provider'),provider_response_id=body.get('id'),model_returned=body.get('model'),started_utc=start,ended_utc=now(),response_sha256=sha(body),respondent_data_sent=False,empirical_estimates_sent=False)
   atomic(receipt,record)
   if cost is None or cost>item['reserve_usd']:raise ValueError('public_call_billing_unresolved')
   if status=='unapproved_provider':raise ValueError('unapproved_public_graph_provider')
  with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(call,planned))
 key=None
 records=[json.loads((out/(i['request_key']+'.json')).read_text()) for i in planned]
 edges=[]
 for r in records:
  for edge in (r['answer'] or {}).get('edges',[]):
   reference=edge['evidence_reference'];traceable=any(s['id'] in reference for s in SOURCES)
   edges.append(dict(model=r['model'],variant=r['variant'],repetition=r['repetition'],**edge,traceable_source_id=traceable,causal_support='hypothesis_not_validated_by_observational_or_measurement_source'))
 atomic(out/'graph_decisions.json',dict(records=records,edges=edges,causal_conclusion='ABSTAIN',independent_causal_adjudication=False))
 from sharecausal.preflight_v23.sealing import seal
 names=[f.name for f in out.iterdir() if f.is_file() and f.name!='public_graph_seal.json']
 if not (out/'public_graph_seal.json').exists():seal(out,names,'public_graph_seal.json')
 atomic(out/'summary.json',dict(requests=len(records),cost_usd=sum(r['cost_usd'] for r in records),valid=sum(r['status']=='valid' for r in records),explicit_abstention=sum(r['status']=='explicit_abstention' for r in records),technical_failures=sum(r['status'] not in ('valid','explicit_abstention') for r in records),proposed_edges=len(edges),causal_conclusion='ABSTAIN',public_sources_only=True,provider_receipts_private=True))
 print('Public-only SHARE graph repetitions complete; no participant or empirical values sent.')
if __name__=='__main__':main()
