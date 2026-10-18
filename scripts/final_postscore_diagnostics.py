"""Post-score-only diagnostic export repair. Never changes scientific predictions."""
import json
from collections import Counter
from sharecausal.synthetic_final_v1.reporting import json_file,csv_file,method_name
from sharecausal.synthetic_final_v1.public import MODELS
def final_diagnostics(out):
    """Post-seal stability and eligible-source diagnostics; no new decisions."""
    from sharecausal.synthetic_final_v1.public import SLOTS
    from sharecausal.pilot_v24.capture import parse_wire
    primary={p['public_task_hash']:p for p in json.loads((out/'prediction_batch.json').read_text())}
    repeat={p['public_task_hash']:p for p in json.loads((out/'stability_prediction_batch.json').read_text())}
    result=json.loads((out/'scored_results.json').read_text())
    records={}
    for p in (out/'provider_responses_private').glob('*.json'):
        if p.name.endswith(('.raw.json','.inflight.json')): continue
        r=json.loads(p.read_text()); records[r['task_sha256'],r['model_requested'],r['variant']]=r
    stability=[]; eligible=[]
    for slot,model in zip(SLOTS,MODELS):
        valid_pairs=0; equal=0; decision_equal=0; status_equal=0
        for h,alt in repeat.items():
            base=primary[h]
            status_equal+=base['source_audit'][slot]['response_status']==alt['source_audit'][slot]['response_status']
            decision_equal+=base['methods']['source_'+'abcd'[MODELS.index(model)]]['decision']==alt['methods']['source_'+'abcd'[MODELS.index(model)]]['decision']
            r1=records.get((h,model,'base')); r2=records.get((h,model,'counterargument'))
            valid=lambda p:p['source_audit'][slot]['response_status'] in ('graph_response','multiple_graphs')
            if valid(base) and valid(alt) and r1 and r2:
                valid_pairs+=1
                def signature(r):
                    return sorted((sorted((e['source'],e['target']) for e in p['graph']['edges']), sorted(p['graph']['adjustment']),p['structural_zero_claim']) for p in json.loads(r['proposals_wire']))
                equal+=signature(r1)==signature(r2)
        stability.append(dict(model=model,planned_pairs=8,eligible_graph_pairs=valid_pairs,
            identical_graph_sets=equal,identical_graph_set_rate=equal/valid_pairs if valid_pairs else None,
            equal_primary_causal_decisions=decision_equal,identical_response_status=status_equal,
            warning='Matching abstentions can conceal technical failures; graph stability uses valid nonabstaining pairs only.'))
        counts=Counter(); failed=0; explicit=0
        for h,p in primary.items():
            status=p['source_audit'][slot]['response_status']
            failed+=status not in ('graph_response','multiple_graphs','source_abstained')
            explicit+=status=='source_abstained'
            graphs=p['methods']['source_'+'abcd'[MODELS.index(model)]].get('graphs',[])
            for graph in graphs: counts[(graph.get('identification') or {}).get('status','not_admitted')]+=1
        eligible.append(dict(model=model,primary_tasks=24,valid_parsed_primary=24-failed,
            failed_primary=failed,explicit_epistemic_abstentions=explicit,eligible_method_graph_identification=dict(counts)))
    coverage=[]
    for m in result['main_summary']:
        probes=[p for p in result['world_estimation_diagnostics'] if p['method']==m['method']]
        defined=[p['interval_covers_identified_truth_ONLY'] for p in probes if p['interval_covers_identified_truth_ONLY'] is not None]
        coverage.append(dict(method=method_name(m['method']),identified_world_intervals=len(defined),
            intervals_covering_identified_truth=sum(defined),conditional_coverage=sum(defined)/len(defined) if defined else None,
            warning='Nested probes, one sample/card; not a calibrated structural or repeated-sampling coverage estimate.'))
    json_file(out/'prompt_stability.json',stability)
    json_file(out/'eligible_source_diagnostics.json',eligible)
    json_file(out/'conditional_coverage.json',coverage)
    csv_file(out/'prompt_stability.csv',stability)
    csv_file(out/'conditional_coverage.csv',coverage)

def main():
    import argparse
    from pathlib import Path
    from sharecausal.preflight_v23.sealing import verify,digest
    p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,required=True);a=p.parse_args();out=a.results_dir.resolve()
    receipt=json.loads((out/'scorer_receipt.json').read_text());assert receipt['seal_verified_before_truth_open']
    seal=(out/'prediction_seal_sha256.txt').read_text().strip()
    verify(out,json.loads((out/'prediction_seal.json').read_text()),'prediction_seal.json',seal)
    before={name:digest(out/name) for name in ('prediction_batch.json','stability_prediction_batch.json','scored_results.json')}
    final_diagnostics(out)
    assert before=={name:digest(out/name) for name in before}
    verify(out,json.loads((out/'prediction_seal.json').read_text()),'prediction_seal.json',seal)
    fixed=dict(correction='Post-score export handles rejected graph identification=None; no scientific code or prediction changed.',
        script_sha256=digest(Path(__file__)),prediction_and_score_sha256=before,seal_reverified=True,paid_requests=0,
        new_truth_reads=0,exit_status=0)
    (out/'postscore_diagnostics_repair_receipt.json').write_text(json.dumps(fixed,indent=2)+'\n')
    print(json.dumps(fixed))
if __name__=='__main__':main()
