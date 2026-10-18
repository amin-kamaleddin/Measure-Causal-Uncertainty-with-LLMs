"""Execute the offline-only v2.4 exploratory causal estimation pilot."""
import argparse,json,sys


class Parser(argparse.ArgumentParser):
    def error(self,message):self.exit(2,'{"status":"error","stage":"arguments","hint":"Use --help; no live mode exists."}\n')


def main(argv=None):
    parser=Parser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--results-dir',required=True)
    parser.add_argument('--new-run',action='store_true')
    args=parser.parse_args(argv)
    try:
        from sharecausal.pilot_v24.runner import run
        report,out,digest=run(args.config,args.results_dir,args.new_run)
    except Exception as exc:
        allowed={'configuration','public_inputs','output','worker_generate','worker_predict','worker_score',
                 'prediction_seal','precision_diagnostics','reporting'}
        stage=getattr(exc,'stage','configuration')
        print(json.dumps({'status':'error','stage':stage if stage in allowed else 'configuration',
            'hint':'Check candidate schemas, declared synthetic paths, empty output directory, worker guards and sealed inputs. No live fallback exists.',
            'worker_diagnostic':getattr(exc,'diagnostic',None),
            'live_fallback':False}),file=sys.stderr);return 2
    print(json.dumps({'results_dir':str(out),'content_manifest_sha256':digest,
        **{k:report[k] for k in ('version','protocol_status','ready_to_freeze','confirmatory_precision_ready',
            'fixture_mode','real_llm_calls','openrouter_cost_usd','share_records_used','public_cards',
            'method_card_primary_rows','prediction_rows','decision_required_count')},
        'assertions_by_method':{r['method']:r['assertions'] for r in report['main_summary']}},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
