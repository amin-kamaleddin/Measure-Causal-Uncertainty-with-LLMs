"""Offline-only Step 6C.2 candidate diagnostics. No live mode exists."""
import argparse,json,sys


class Parser(argparse.ArgumentParser):
    def error(self,message):
        self.exit(2,'{"status":"error","stage":"arguments","hint":"Use --help; no live mode exists."}\n')


HINTS={
 'configuration':'Check v2.3 schema, zero authorization, pending policy approval and numeric candidate grids.',
 'public_packets':'Check packet presence, exact schema, immutable content/evidence IDs and duplicate cards.',
 'design_families':'Check complete development assignments, distinct mechanism definitions and pending heldout custody.',
 'pricing':'Check FICTIONAL price/capability catalog, four mock models and token/context reservations.',
 'decisions':'Provide exactly 16 unique DECISION_REQUIRED entries; recommendations are not approvals.',
 'output':'Use a new/empty synthetic output directory or --new-run; private/credential paths are forbidden.',
 'mock_prediction':'Check local mock receipt/schema bindings and byte-token reservations; no fallback exists.',
 'prediction_seal':'A sealed prediction/design member was changed or missing; preserve it and start a new run.',
 'private_scoring':'Check synthetic SCM CPTs, public bindings, witness proofs and seal integrity; no fallback exists.',
 'precision':'Check feasible marginal error contrasts, dropout and independent-card/family grid.',
 'artifacts':'Check synthetic output IO and seal integrity.'}


def main(argv=None):
    parser=Parser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--results-dir',required=True)
    parser.add_argument('--new-run',action='store_true');parser.add_argument('--require-ready',action='store_true')
    parser.add_argument('--require-precision',action='store_true',help='Exit 4 when confirmatory precision is not ready.')
    args=parser.parse_args(argv)
    try:
        from sharecausal.preflight_v23.runner import run_preflight,PreflightFailure
        report,out,digest=run_preflight(args.config,args.results_dir,new_run=args.new_run)
    except Exception as exc:
        # Only fixed enums/strings; never echo Pydantic field locations, paths, values or exception messages.
        name=getattr(exc,'stage',None)
        safe_stage=name if name in HINTS else 'internal'
        print(json.dumps({'status':'error','stage':safe_stage,'live_fallback':False,
              'hint':HINTS.get(safe_stage,'Offline implementation failure; inspect locally with synthetic inputs.')}),file=sys.stderr)
        return 2 if getattr(exc,'invalid',False) else 1
    print(json.dumps({'results_dir':str(out),'content_manifest_sha256':digest,
        **{k:report[k] for k in ('version','protocol_status','ready_to_freeze','fixture_mode','real_llm_calls',
            'openrouter_cost_usd','share_records_used','decision_required_count','public_cards','world_probes','mock_transport_calls')},
        'confirmatory_precision_ready':report['precision_gate']['confirmatory_precision_ready']},indent=2))
    if args.require_precision and not report['precision_gate']['confirmatory_precision_ready']:return 4
    if args.require_ready and not report['ready_to_freeze']:return 3
    return 0


if __name__=='__main__':raise SystemExit(main())
