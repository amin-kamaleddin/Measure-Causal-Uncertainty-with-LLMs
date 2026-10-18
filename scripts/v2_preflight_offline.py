"""Execute candidate-study diagnostics only. No live mode or key handling."""
import argparse
import json
import sys


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, '{"status":"error","error_code":"invalid_offline_arguments","hint":"Use --help; no live mode exists."}\n')


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--new-run', action='store_true')
    parser.add_argument('--require-ready', action='store_true', help='Return exit 3 when PI decisions remain pending.')
    args = parser.parse_args(argv)
    try:
        from sharecausal.preflight.runner import run_preflight
        report, output, digest = run_preflight(args.config, args.results_dir, new_run=args.new_run)
    except Exception as exc:
        # Do not emit Pydantic locations, raw exception strings, input paths, keys or locals.
        code = 'offline_preflight_invalid_input' if isinstance(exc, (ValueError, OSError)) else 'offline_preflight_failed'
        print(json.dumps({'status':'error','error_code':code,'live_fallback':False,
              'hint':'Check required candidate config fields, fixture schema, certificates, split and fake price inputs; use a new/empty results directory or --new-run.'}), file=sys.stderr)
        return 2 if code.endswith('invalid_input') else 1
    print(json.dumps({'results_dir':str(output),'content_manifest_sha256':digest,
          **{k:report[k] for k in ('version','fixture_mode','real_llm_calls','openrouter_cost_usd','share_records_used',
             'protocol_status','ready_to_freeze','decision_required_count','public_cards','world_probes','mock_transport_calls')}},indent=2))
    return 3 if args.require_ready and report['decision_required_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
