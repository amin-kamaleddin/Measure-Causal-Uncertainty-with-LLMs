"""Reconstruct exact hash-bound public payloads from sealed inputs, without keys."""
import argparse,json,os
from pathlib import Path
from sharecausal.final_study_v31.public import Card,messages,schema
from sharecausal.final_study_v31.transport import load_records
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.sealing import verify,digest
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--package',required=True,type=Path);a=p.parse_args();run=a.run.resolve();root=a.package.resolve();out=root/'exact_public_request_payloads_PRIVATE';out.mkdir(mode=0o700,exist_ok=True);os.umask(0o077)
seal=json.loads((run/'prediction_seal.json').read_text());verify(run,seal,'prediction_seal.json',(run/'prediction_seal.sha256').read_text().strip());assert (run/'scoring_integrity.json').exists()
old=Path(json.loads((run/'routing_repair_provenance.json').read_text())['original_run']);cards={c.task_id:c for c in [Card.model_validate(x) for x in json.loads((run/'public_cards.json').read_text())]};seen={}
for directory in (old,run):
 records=load_records(directory/'receipts');graphs={}
 for r in records:
  if r['variant']=='base' and r['repetition']==0 and r['status']=='valid':
   g=graphs.setdefault(r['task_id'],{})
   for item in r['answer']['graphs']:g[sha(item)]=item
 for r in records:
  critic=r['variant']=='critic';proposals=[g for h,g in sorted(graphs.get(r['task_id'],{}).items())] if critic else None
  chat=messages(cards[r['task_id']],r['variant'],r['repetition'],proposals)
  if sha(chat)!=r['prompt_sha256'] or sha(schema(critic))!=r['schema_sha256']:raise ValueError('reconstructed_prompt_or_schema_hash_mismatch')
  payload=dict(model=r['model'],messages=chat,**r['settings'],usage=dict(include=True),response_format=dict(type='json_schema',json_schema=dict(name='evidence_critic' if critic else 'evidence_graphs',strict=True,schema=schema(critic))))
  h=r['request_key'];body=canonical(payload)
  if h in seen and seen[h]!=sha(payload):raise ValueError('same_request_different_payload')
  (out/(h+'.json')).write_bytes(body);seen[h]=sha(payload)
report=dict(unique_exact_payloads=len(seen),prompt_and_schema_hashes_match_every_receipt=True,reconstruction_after_score=True,source='sealed public cards,typed receipts,executed message/schema functions; not a verbatim network capture',credentials_included=False,headers_included=False,share_data_included=False,predictions_modified=False,payload_sha256=seen)
(root/'exact_public_payload_verification.json').write_bytes(canonical(report));print(json.dumps({k:v for k,v in report.items() if k!='payload_sha256'}))
