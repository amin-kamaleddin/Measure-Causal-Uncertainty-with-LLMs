"""Descriptive graph diagnostics for sealed public-document-only SHARE calls."""
import argparse,json,csv
from pathlib import Path
from collections import Counter
from sharecausal.empirical_v3.graphs import backdoor
from sharecausal.preflight_v23.sealing import digest
p=argparse.ArgumentParser();p.add_argument('--package',required=True,type=Path);a=p.parse_args();root=a.package.resolve();directory=root/'public_share_graph_repeats';data=json.loads((directory/'graph_decisions.json').read_text());records=data['records'];rows=[]
for r in records:
 answer=r.get('answer') or {};edges=[(e['source'],e['target']) for e in answer.get('edges',[])];status='NO_GRAPH';sets=None
 if edges:
  try:
   sets=backdoor(edges);status='CONDITIONAL_BACKDOOR' if sets else 'NO_OBSERVED_BACKDOOR'
  except ValueError:status='INVALID_CYCLIC_GRAPH'
 rows.append(dict(model=r['model'],variant=r['variant'],repetition=r['repetition'],provider_status=r['status'],source_abstention_flag=bool(answer.get('abstain')),nonempty_graph=bool(edges),edge_count=len(edges),graph_status=status,measured_backdoor_sets_ignoring_selection=json.dumps(sets),unmeasured_fork=('U','A') in edges and ('U','Y') in edges,D8_to_A=('D8','A') in edges,A_to_D8=('A','D8') in edges,source_id_traceable_edges=sum(e['traceable_source_id'] for e in data['edges'] if e['model']==r['model'] and e['variant']==r['variant'] and e['repetition']==r['repetition']),validated_causal_edges=0,causal_verdict='ABSTAIN',cost_usd=r['cost_usd']))
tables=root/'tables';tables.mkdir(exist_ok=True)
with (tables/'public_share_graph_diagnostics_PRIVATE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(requests=len(rows),nonempty_graphs=sum(r['nonempty_graph'] for r in rows),graph_status_counts=dict(Counter(r['graph_status'] for r in rows)),source_abstention_flags=sum(r['source_abstention_flag'] for r in rows),abstention_flag_with_nonempty_graph=sum(r['source_abstention_flag'] and r['nonempty_graph'] for r in rows),technical_failures=sum(r['provider_status'] not in ('valid','explicit_abstention') for r in rows),unmeasured_fork_graphs=sum(r['unmeasured_fork'] for r in rows),D8_to_A_graphs=sum(r['D8_to_A'] for r in rows),A_to_D8_graphs=sum(r['A_to_D8'] for r in rows),proposed_edges=len(data['edges']),traceable_source_id_edges=sum(e['traceable_source_id'] for e in data['edges']),validated_causal_edges=0,reported_cost_usd=sum(r['cost_usd'] for r in rows),causal_verdict='ABSTAIN',empirical_estimates_sent=False,respondent_data_sent=False,source_decision_sha256=digest(directory/'graph_decisions.json'),restricted_backdoor_not_general_identification=True,conceptual_blocks_not_all_jointly_observed=True)
(root/'public_share_graph_diagnostics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
