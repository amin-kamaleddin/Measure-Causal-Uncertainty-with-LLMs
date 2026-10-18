"""Execute Step 6B offline synthetic bridge; no live mode exists."""
import argparse
import json
import sys

from sharecausal.benchmark.bridge_v2_errors import safe_failure


class OfflineArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default error echoes raw argument values, which may be secrets.
        self.exit(2, json.dumps({'status': 'error', 'error_code': 'invalid_arguments',
                               'stage': 'startup', 'live_fallback': False,
                               'message': 'Invalid CLI arguments. Use --help for supported offline options.'}) + '\n')


def main(argv=None):
    parser = OfflineArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--new-run', action='store_true',
                        help='Create a fresh run-* child under --results-dir; preserve all prior files.')
    args = parser.parse_args(argv)
    try:
        from sharecausal.benchmark.bridge_v2_offline import run_offline_smoke
        report = run_offline_smoke(args.fixture_dir, args.results_dir, new_run=args.new_run)
    except Exception as exc:
        diagnostic, status = safe_failure(exc)
        print(json.dumps(diagnostic, sort_keys=True), file=sys.stderr)
        return status
    print(json.dumps({k: report[k] for k in ('kind', 'implementation_version', 'results_dir',
          'prediction_batch_sha256', 'fixture_mode', 'real_llm_calls',
          'openrouter_cost_usd', 'share_records_used', 'n_tasks', 'n_method_rows', 'summary')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
