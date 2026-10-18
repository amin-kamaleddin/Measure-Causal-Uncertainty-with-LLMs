"""Author exposed pilot generation CPT fixture, separate from scoring truth access."""
from pathlib import Path
import json
from sharecausal.preflight.contracts import canonical
from sharecausal.preflight_v23.certificates import Truth
from sharecausal.preflight_v23.packets import PublicTask,task_hash

ROOT=Path(__file__).resolve().parents[1]


def build():
    tasks={task_hash(t):t for t in [PublicTask.model_validate(r) for r in json.loads(
        (ROOT/'tests/fixtures/v2_preflight_v23/public_tasks.json').read_text())]}
    # Authoring sees exposed synthetic truth; this script is NOT run by the predictor/scorer.
    worlds=[Truth.model_validate(r) for r in json.loads((ROOT/'tests/fixtures/v2_preflight_v23/private_truth.json').read_text())]
    plans={}
    for w in worlds:
        plans.setdefault(w.public_task_hash,{'nodes':w.scm.model_dump()['nodes'],
            'recording':'complete' if tasks[w.public_task_hash].card.selection=='complete_outcomes' else 'post_inspection_dependent'})
    (ROOT/'tests/fixtures/v24_pilot/generation_plan.json').write_bytes(canonical({
        'schema_version':'generation-plan-2.4','plans':dict(sorted(plans.items()))}))
    print(json.dumps({'exposed_cards':len(plans),'scoring_truth_file_written':False}))


if __name__=='__main__':build()
