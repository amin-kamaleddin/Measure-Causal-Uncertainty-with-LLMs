"""Known-truth scoring is deliberately a separate, post-seal entry point."""
import json
from pathlib import Path
from sharecausal.preflight.contracts import sha


def score(predictions_path, truth_path):
    sealed=json.loads(Path(predictions_path).read_text())
    if sealed['seal_sha256']!=sha(sealed['payload']):raise ValueError('Prediction seal mismatch')
    # Truth is not opened until the entire prediction seal has been verified.
    from sharecausal.preflight_v23.certificates import Truth, certify, edges
    from sharecausal.preflight_v23.packets import PublicTask, task_hash
    from .prediction import adapt
    worlds=[Truth.model_validate(w) for w in json.loads(Path(truth_path).read_text())]
    rows=[]
    for unit in sealed['payload']['predictions']:
        task=PublicTask.model_validate(unit['task']);p=unit['prediction']
        matching=[w for w in worlds if w.public_task_hash==task_hash(task)]
        if not matching:raise ValueError('Missing matching truth world')
        for w in matching:
            cert=certify(w,task,worlds)
            raw=p['proposals'] if p['status'] in ['graph_response','multiple_graphs'] else []
            proposed=[{(e['source'],e['target']) for e in r['graph']['edges']} for r in raw]
            # Set-valued response alternatives are separate hypotheses, never extra independent units.
            truth_edges=set(edges(w.scm))
            rows.append(dict(task_sha256=task_hash(task),world_id=w.world_id,family_id=w.family_id,
                status=p['status'],certificate=cert,source_type=p.get('source_type','ineligible'),
                n_proposals=len(proposed),edge_counts=[dict(tp=len(s&truth_edges),fp=len(s-truth_edges),
                    fn=len(truth_edges-s)) for s in proposed],
                interpretation='graph_hypothesis_audit_not_causal_accuracy_or_independent_replication'))
    return dict(version='provider-replay-1',prediction_seal=sealed['seal_sha256'],
                scored_world_rows=len(rows),independent_public_tasks=len({r['task_sha256'] for r in rows}),
                family_ids=sorted({r['family_id'] for r in rows}),rows=rows,
                limitation='Repeated responses/worlds are nested. No synthetic real-provider responses collected by this task.')
