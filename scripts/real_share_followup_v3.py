#!/usr/bin/env python3
"""Run only saved-response, local SHARE sensitivity analysis. No live transport."""
import argparse
import json
import sys
from pathlib import Path
import os
import tempfile
import traceback


def main():
    # Only plotting caches, never respondent data; valid within the workspace sandbox.
    cache=tempfile.TemporaryDirectory(prefix='sharecausal-plot-cache-')
    os.environ['MPLCONFIGDIR']=cache.name
    from sharecausal.empirical_v3.io import install_offline_guard
    install_offline_guard()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--record',type=Path,action='append',required=True)
    p.add_argument('--comparison-dir',type=Path,required=True)
    p.add_argument('--prior-results',type=Path,required=True)
    p.add_argument('--independent-audit-dir',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/real_share_followup_v3.yaml'))
    p.add_argument('--results-dir',type=Path,required=True)
    args=p.parse_args()
    stage=['startup']
    try:
        from sharecausal.empirical_v3.runner import run
        r=run(args,stage)
    except Exception as exc:
        # No repr/str/traceback: errors can contain identifiers, values or secrets.
        kind=type(exc).__name__
        safe=kind if kind in {'ValueError','TypeError','RuntimeError','FileNotFoundError','PermissionError','KeyError','OSError',
                              'ValidationError','JSONDecodeError','ImportError','ModuleNotFoundError'} else 'AnalysisError'
        hints={'saved_graph_audit':'Check four saved graph records, hashes and comparison exports.',
               'configuration':'Check versioned YAML fields and local_only mode.',
               'archive_metadata':'Check all three ZIPs and original release status labels.',
               'panel_construction':'Check module joins, missing codes and contradictory status records.',
               'matched_estimation':'Check complete-case rows, two exposure arms and country clusters.',
               'selection_diagnostics':'Check survivor cells and death/outcome consistency.',
               'country_childhood_overlap':'Check eligibility, support thresholds and complete covariates.',
               'private_reporting':'Check private output permissions and plotting dependencies.',
               'startup':'Choose an empty private output location; check local dependencies.'}
        print(json.dumps({'status':'FAILED','stage':stage[0],'error_type':safe,
                          'action':hints.get(stage[0],'Inspect local configuration; no live fallback.'),
                          'code_locations':[{'module':Path(f.filename).name,'line':f.lineno}
                              for f in traceback.extract_tb(exc.__traceback__)
                              if '/sharecausal/empirical_v3/' in f.filename],
                          'redaction':'exception content suppressed; no respondent values logged'}),file=sys.stderr)
        cache.cleanup()
        return 2
    print(json.dumps({'status':'SUCCESS','results_dir':str(args.results_dir),
                      'eligible':r['share_records_used'],'main_n':r['estimates']['aipw_demographic_country']['n'],
                      'decision':r['causal_decision'],'real_llm_calls':0,'openrouter_cost_usd':0,'uploads':0}))
    cache.cleanup()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
