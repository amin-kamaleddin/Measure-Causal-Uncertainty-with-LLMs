"""Resumable final-study orchestrator. No truth file is opened here."""
import argparse,json,os,sys,subprocess,platform
from pathlib import Path
from datetime import datetime,timezone
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.sealing import digest,write_once,seal,verify
from sharecausal.synthetic_final_v1.execution import execute,record
from sharecausal.synthetic_final_v1.transport import atomic
from .public import Card,MODELS

ROOT=Path(__file__).resolve().parents[3]

def worker(role,out,split,key=None):
 env=dict(os.environ);env.pop('OPENROUTER_API_KEY',None)
 if role=='collect':env['OPENROUTER_API_KEY']=key
 result=execute([sys.executable,'-m','sharecausal.final_study_v3.workers',role,str(out),split],ROOT,out,env=env)
 if result.returncode:raise ValueError('worker_'+role+'_failed')

def leakage_audit(cards,assignments):
 """Full evidence prose comparison; common boilerplate excluded transparently."""
 from sklearn.feature_extraction.text import TfidfVectorizer
 import numpy as np
 split={r['task_id']:r['split'] for r in assignments};docs=[]
 for c in cards:
  # Compare causal statements, preserving directions and uncertainty. Domain
  # titles and UUIDs must not make duplicate causal evidence appear different.
  docs.append(' '.join(d.text for d in c.documents if any(t in d.text for t in ('direct','causal','mechanism','hypothetical','act on','input to'))))
 matrix=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit_transform(docs)
 similarity=(matrix@matrix.T).toarray();pairs=[]
 for i,c in enumerate(cards):
  for j in range(i):
   if split[c.task_id]!=split[cards[j].task_id] and similarity[i,j]>=.96:pairs.append([c.task_id,cards[j].task_id,float(similarity[i,j])])
 if pairs:raise ValueError('cross_split_near_duplicate_causal_evidence')
 return dict(cross_split_threshold=.96,cross_split_near_duplicates=pairs,comparison='TFIDF character3-5 on causal evidence, no UUID/title assistance',max_cross_split_similarity=max(float(similarity[i,j]) for i,c in enumerate(cards) for j in range(i) if split[c.task_id]!=split[cards[j].task_id]),independence='20 declared mechanisms; not 40 independent narrative families; no external author review')

def prepare(out,cap,catalog=None):
 if out.is_relative_to(ROOT) or out==out.parent:raise ValueError('private_results_must_be_outside_git')
 if not out.exists():
  child=execute([sys.executable,'-c','import sys,json;from sharecausal.final_study_v3.authoring import build;print(json.dumps(build(sys.argv[1])))',str(out)],ROOT,out)
  if child.returncode:raise ValueError('authoring_failed')
 cards=[Card.model_validate(c) for c in json.loads((out/'public_cards.json').read_text())]
 assignments=json.loads((out/'families.json').read_text())
 if len(cards)!=120 or len({c.task_id for c in cards})!=120:raise ValueError('120_unique_cards_required')
 if not (out/'leakage_audit.json').exists():write_once(out/'leakage_audit.json',leakage_audit(cards,assignments))
 if not (out/'catalog.json').exists():
  from sharecausal.llm.openrouter import available_models
  catalog=catalog or available_models()
  selected=[m for m in catalog if m['id'] in MODELS]
  if len(selected)!=4:raise ValueError('exact_four_models_unavailable')
  write_once(out/'catalog.json',selected)
 from .transport import plan
 planned=plan(cards,json.loads((out/'catalog.json').read_text()),cap)
 if not (out/'request_plan.json').exists():write_once(out/'request_plan.json',planned)
 elif json.loads((out/'request_plan.json').read_text())!=planned:raise ValueError('request_plan_changed')
 paths=sorted((ROOT/'src/sharecausal').rglob('*.py'))+[ROOT/'scripts/final_study_v3.py',ROOT/'docs/final_study_v3_locked_plan.md']
 commitment=dict(version='evidence-grounded-final-study-3',status='prospectively_locked_exploratory',cap_usd=cap,models=list(MODELS),
  planned_primary_calls=1920,critic_ceiling=120,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
  input_sha256={p:digest(out/p) for p in ('public_cards.json','public_rubric.json','families.json','authoring_commitment.json','catalog.json','request_plan.json','leakage_audit.json')},
  python=platform.python_version(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
 if not (out/'commitment.json').exists():write_once(out/'commitment.json',commitment)
 elif json.loads((out/'commitment.json').read_text())!=commitment:raise ValueError('source_or_design_changed_after_freeze')
 for name in (*commitment['input_sha256'],'commitment.json'): (out/name).chmod(0o400)
 return cards,assignments

def calibrate(out):
 from .engine import METHODS
 predictions=json.loads((out/'predictions_development.json').read_text());primary=[p for p in predictions if p['sample_repetition']==0];grid=[0.,.025,.05,.1,.2,.3,.4,.5]
 result={}
 for method in METHODS:
  rates={t:sum(p['methods'][method]['decision'] in ('positive','negative') and p['methods'][method]['margin']>=t for p in primary)/len(primary) for t in grid}
  threshold=min(grid,key=lambda t:(abs(rates[t]-.4),t))
  result[method]=dict(margin_threshold=threshold,development_assertion_rate=rates[threshold],target=.4,tolerance=.1,selection_uses_truth=False)
 write_once(out/'calibration.json',result)
 return result

def run(out,cap,key,live):
 os.umask(0o077);cards,assignments=prepare(out,cap)
 if not live:return 0
 checked=execute([sys.executable,'scripts/check_protocol_freeze.py'],ROOT,out)
 if checked.returncode:raise ValueError('frozen_protocol_failure')
 if not (out/'sample_manifest.json').exists():worker('generate',out,'all')
 for split in ('development','evaluation'):
  if not (out/('predictions_'+split+'.json')).exists():
   if split=='evaluation' and not (out/'calibration.json').exists():raise ValueError('calibration_not_frozen')
   worker('collect',out,split,key)
   worker('collect',out,split+'_critic',key)
   worker('predict',out,split)
  if split=='development' and not (out/'calibration.json').exists():calibrate(out);(out/'calibration.json').chmod(0o400)
 if not (out/'prediction_batch.json').exists():write_once(out/'prediction_batch.json',sum([json.loads((out/('predictions_'+s+'.json')).read_text()) for s in ('development','evaluation')],[]))
 if not (out/'prediction_seal.json').exists():
  from .transport import load_records
  if len(load_records(out/'receipts'))!=2040:raise ValueError('incomplete_entire_provider_batch')
  names=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ('truth.json','generation.json','commands_and_exit_codes.json','collector.lock') and not p.name.endswith('.pending')]
  _,sealed=seal(out,names,'prediction_seal.json');(out/'prediction_seal.sha256').write_text(sealed+'\n')
  for name in names+['prediction_seal.json','prediction_seal.sha256']:(out/name).chmod(0o400)
 if not (out/'scoring_integrity.json').exists():worker('score',out,'all')
 return 0

def main():
 parser=argparse.ArgumentParser(description='Final study:120 synthetic cards,1920+120 bounded real calls, local/private/resumable')
 parser.add_argument('--live',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--results-dir',required=True,type=Path);parser.add_argument('--max-cost-usd',type=float,default=100);parser.add_argument('--key-file',type=Path)
 args=parser.parse_args()
 if not (args.live ^ args.prepare) or not 0<args.max_cost_usd<=100:parser.error('choose --live or --prepare; authorized cap >0 and <=100')
 out=args.results_dir.expanduser().resolve();started=datetime.now(timezone.utc).isoformat();key=None
 try:
  if args.live:
   key=os.environ.get('OPENROUTER_API_KEY')
   if not key and args.key_file:key=args.key_file.expanduser().read_text().strip()
   if not key or any(c.isspace() for c in key):raise ValueError('authorized_raw_key_unavailable')
  code=run(out,args.max_cost_usd,key,args.live)
 except Exception as exc:
  print(json.dumps(dict(status='failed',exception=type(exc).__name__,reason=str(exc) if isinstance(exc,ValueError) else 'inspect_private_worker_diagnostics')),file=sys.stderr);code=2
 finally:key=None
 if out.exists():record(out,[sys.executable,*sys.argv],code,started,datetime.now(timezone.utc).isoformat(),cwd=str(ROOT))
 return code
