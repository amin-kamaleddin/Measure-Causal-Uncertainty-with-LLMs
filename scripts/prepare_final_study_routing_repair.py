"""No network: archive/reuse completed original calls for prospective routing fix."""
import argparse,json,shutil,os,hashlib
from pathlib import Path
from sharecausal.preflight.contracts import canonical
from sharecausal.preflight_v23.sealing import digest
from sharecausal.final_study_v31.authoring import build
from sharecausal.final_study_v31.public import MODELS
p=argparse.ArgumentParser();p.add_argument('--original',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();original=a.original.resolve();out=a.output.resolve();os.umask(0o077)
if (original/'scoring_integrity.json').exists() or (original/'scored_results.json').exists():raise ValueError('truth_already_opened_amendment_forbidden')
files=list((original/'receipts').glob('*.json'));records=[(f,json.loads(f.read_text())) for f in files if not f.name.endswith(('.raw.json','.inflight.json'))]
if len(records)!=1956 or len(list((original/'receipts').glob('*.inflight.json')))!=len(records):raise ValueError('original_calls_must_drain_completely_1956')
if any(r['cost_usd'] is None for _,r in records):raise ValueError('unresolved_original_billing')
if not out.exists():build(out)
for n in ('public_cards.json','public_rubric.json','families.json','authoring_commitment.json'):
 # Hidden truth/generation equality is established by the authoring receipts;
 # this setup never opens either hidden file.
 if digest(out/n)!=digest(original/n):raise ValueError('original_public_or_hidden_authoring_changed')
for folder in ('receipts','legacy_receipts'):(out/folder).mkdir(exist_ok=True,mode=0o700)
for f,r in records:
 for suffix in ('.json','.raw.json','.inflight.json'):
  source=original/'receipts'/(r['request_key']+suffix);target=out/'legacy_receipts'/source.name
  if not target.exists():shutil.copyfile(source,target)
  if r['variant']!='critic' and r['model']!=MODELS[3]:
   target=out/'receipts'/source.name
   if not target.exists():shutil.copyfile(source,target)
for n in ('catalog.json','sample_manifest.json','generator_receipt.json','calibration.json'):
 if not (out/n).exists():shutil.copyfile(original/n,out/n)
for f in original.glob('observed-*.json'):
 if not (out/f.name).exists():shutil.copyfile(f,out/f.name)
prior=sum(r['cost_usd'] for _,r in records)
provenance=dict(original_run=str(original),original_commitment_sha256=digest(original/'commitment.json'),original_calibration_sha256=digest(original/'calibration.json'),original_requests=len(records),original_reported_charges_usd=prior,historical_known_synthetic_cost_usd=.1636867377,prior_actual_charges_and_historical_known_synthetic_usd=prior+.1636867377,expected_new_primary_requests=480,expected_primary_paid_receipt_reuse=1440,critic_exact_key_reuse_allowed=True,initial_deepseek_pipeline_rejections=sum(r['model']==MODELS[3] for _,r in records),all_original_receipts_preserved=True,original_truth_not_scored=True,scientific_rules_unchanged=True,independent_replication=False)
target=out/'routing_repair_provenance.json'
if not target.exists():target.write_bytes(canonical(provenance))
elif json.loads(target.read_text())!=provenance:raise ValueError('routing_provenance_changed')
print(json.dumps(provenance))
