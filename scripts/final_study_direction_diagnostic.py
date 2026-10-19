"""Post-score decomposition of the frozen direction/practical-null endpoint."""
import argparse,json,csv
from pathlib import Path
from fractions import Fraction
from collections import defaultdict,Counter
from sharecausal.preflight_v23.sealing import verify
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--package',required=True,type=Path);a=p.parse_args();r=a.run.resolve();root=a.package.resolve()
integrity=json.loads((r/'scoring_integrity.json').read_text());assert integrity['seal_verified_before_truth_open'];seal=json.loads((r/'prediction_seal.json').read_text());verify(r,seal,'prediction_seal.json',(r/'prediction_seal.sha256').read_text().strip())
scores=list(csv.DictReader((r/'task_scores.csv').open()));pred={p['task_id']:p for p in json.loads((r/'prediction_batch.json').read_text()) if p['sample_repetition']==0};truth=defaultdict(list)
for w in json.loads((r/'truth.json').read_text()):truth[w['task_id']].append(float(Fraction(w['effect_exact'])))
counts={m:Counter() for m in next(iter(pred.values()))['methods']}
for row in scores:
 if row['split']!='evaluation' or row['assertion']!='1':continue
 h=row['task_id'];m=row['method'];d=pred[h]['methods'][m]['decision'];values=truth[h];wrong=any((t<0 and d=='positive') or (t>0 and d=='negative') for t in values);near=any(abs(t)<=.1 for t in values);identified=row['certificate']=='identified'
 assert int(row['wrong_sign_all_known_worlds'])==int(wrong or near)
 assert int(row['wrong_sign_identified'])==int((wrong or near) and identified)
 c=counts[m];c['assertions']+=1;c['literal_wrong_direction_all_worlds']+=wrong;c['practical_null_assertions_all_worlds']+=near;c['literal_wrong_direction_identified']+=wrong and identified;c['practical_null_assertions_identified']+=near and identified
rows=[dict(method=m,**{k:c[k] for k in ('assertions','literal_wrong_direction_all_worlds','practical_null_assertions_all_worlds','literal_wrong_direction_identified','practical_null_assertions_identified')},frozen_composite_endpoint_unchanged=True) for m,c in counts.items()]
with (root/'tables/wrong_direction_vs_practical_null.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
verify(r,seal,'prediction_seal.json',(r/'prediction_seal.sha256').read_text().strip());print(json.dumps(rows))
