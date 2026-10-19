"""Independent toy fixture; never score or tune the actual heldout study."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
from sharecausal.preflight.contracts import canonical,sha
from sharecausal.preflight_v23.sealing import digest
from sharecausal.final_study_v31.public import Card,MODELS,messages


def test_repeat_decisions_are_guarded_and_all_conditions_complete(tmp_path):
 root=Path(__file__).resolve().parents[1]
 document=dict(evidence_id='e-independent-repeat-unit',text='The design manual records: A directly changes Y.')
 content=dict(title='Independent repeat estimator unit fixture',variables={'A':dict(time=1,observed=True,description='toy assignment'),'Y':dict(time=3,observed=True,description='toy response')},query='Independent toy total contrast',documents=[document]);card=Card.model_validate(dict(task_id=sha(content),**content));h=card.task_id
 values={'public_cards.json':[card.model_dump()],'public_rubric.json':{h:dict(annotations=[dict(edge=['A','Y'],state='entailed',evidence_id=document['evidence_id'],quote=document['text'])],flags=dict(exhaustive=True,randomized=True,complete=True,structural_zero=False,target_valid=True,positivity=True))},'calibration.json':{},'commitment.json':dict(source_sha256={'src/sharecausal/final_study_v31/engine.py':digest(root/'src/sharecausal/final_study_v31/engine.py')})}
 for name,value in values.items():(tmp_path/name).write_bytes(canonical(value))
 samples=[]
 for r in range(2):
  rng=np.random.default_rng(9871+r);A=rng.binomial(1,.5,300);Y=rng.binomial(1,.1+.8*A);name='independent-sample-'+str(r)+'.json';(tmp_path/name).write_bytes(canonical(dict(A=A.tolist(),Y=Y.tolist(),observed=[1]*300)));samples.append(dict(task_id=h,sample_repetition=r,file=name,sha256=digest(tmp_path/name)))
 (tmp_path/'sample_manifest.json').write_bytes(canonical(samples));(tmp_path/'receipts').mkdir()
 for variant,repetition in [('base',1),('counterargument',0),('counterargument',1)]:
  for model in MODELS:
   key=sha([model,variant,repetition]);raw=dict(unit_test_fixture_only=True);(tmp_path/'receipts'/(key+'.raw.json')).write_bytes(canonical(raw))
   record=dict(task_id=h,model=model,variant=variant,repetition=repetition,request_key=key,prompt_sha256=sha(messages(card,variant,repetition)),response_sha256=sha(raw),status='explicit_abstention',answer=dict(status='abstain',graphs=[],reason='unit fixture only'),cost_usd=0.)
   (tmp_path/'receipts'/(key+'.json')).write_bytes(canonical(record))
 result=subprocess.run([sys.executable,str(root/'scripts/final_study_repeat_decisions.py'),'--results-dir',str(tmp_path),'--max-wait-seconds','5'],capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 predictions=json.loads((tmp_path/'repeat_condition_predictions.json').read_text());receipt=json.loads((tmp_path/'repeat_prediction_guard_receipt.json').read_text())
 assert len(predictions)==6 and receipt['network_requests']==0 and receipt['truth_readable'] is False
 assert {(r['prompt_variant'],r['prompt_repetition']) for r in predictions}=={('base',1),('counterargument',0),('counterargument',1)}
 assert all(r['truth_used'] is False for r in predictions)
 assert not (tmp_path/'truth.json').exists()
