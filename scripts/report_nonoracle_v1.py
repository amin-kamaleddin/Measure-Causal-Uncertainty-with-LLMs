"""Transparent *posthoc descriptive* Monte Carlo report for locked v1 outputs.

Does not rerun or retune the frozen estimator, never reads SHARE, and does
not claim independent replication or valid structural confidence intervals.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def stratified_bootstrap(rows: pd.DataFrame, policy: dict, *, n_boot: int = 2000, seed: int = 20261016):
    if n_boot < 200:
        raise ValueError('Require at least 200 bootstrap repetitions')
    rng = np.random.default_rng(seed)
    mechanisms = sorted(rows['scenario'].unique())
    # Shared draw indices give a paired bootstrap across methods. Mechanism
    # class is fixed; within-class independent simulated seeds are resampled.
    replicate_key = []
    for mechanism in mechanisms:
        keys = sorted(rows.loc[rows.scenario == mechanism, 'seed'].unique())
        replicate_key.append([(mechanism, k) for k in keys])
    if len({len(x) for x in replicate_key}) != 1:
        raise ValueError('Unequal seed counts across mechanism classes')
    per_class = len(replicate_key[0])
    draw = rng.integers(0, per_class, size=(n_boot, len(mechanisms), per_class))
    summaries = []
    for method, config in sorted(policy.items()):
        sub = rows.loc[rows.method == method].set_index(['scenario', 'seed'])
        masks, errors, nonid = [], [], []
        for key in replicate_key:
            one = sub.loc[key]
            mask = ((one['decision'] != 'abstain') &
                    (one['assertion_margin'].astype(float) > config['threshold'])).to_numpy()
            masks.append(mask)
            errors.append(one['sign_error'].to_numpy(dtype=int) * mask)
            nonid.append(one['asserted_nonidentifiable_truth'].to_numpy(dtype=int) * mask)
        mask = np.asarray(masks)[np.arange(len(mechanisms))[None, :, None], draw]
        wrong = np.asarray(errors)[np.arange(len(mechanisms))[None, :, None], draw]
        invalid = np.asarray(nonid)[np.arange(len(mechanisms))[None, :, None], draw]
        count = mask.sum(axis=(1, 2))
        coverage = count / mask[0].size
        if (count > 0).any():
            with np.errstate(invalid='ignore', divide='ignore'):
                risks = np.where(count > 0, wrong.sum(axis=(1, 2)) / count, np.nan)
                id_risks = np.where(count > 0, invalid.sum(axis=(1, 2)) / count, np.nan)
        else:
            risks = np.full(n_boot, np.nan)
            id_risks = np.full(n_boot, np.nan)
        def interval(x):
            finite = x[np.isfinite(x)]
            return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))] if len(finite) else None
        summaries.append({'method': method, 'bootstrap_design':'resample seeds within fixed mechanism family',
                          'bootstrap_repetitions': n_boot,
                          'assertion_rate_MC_interval': interval(coverage),
                          'false_sign_risk_MC_interval': interval(risks),
                          'nonidentifiable_assertion_fraction_MC_interval': interval(id_risks),
                          'unidentifiable_risk_replicates': int(np.sum(count == 0))})
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description='Report the locked non-oracle synthetic results')
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()
    path = Path(args.results_dir)
    locked = json.loads((path/'locked_split_summary.json').read_text())
    df = pd.read_csv(path/'heldout/nonoracle_run_level.csv')
    if locked['real_llm_calls'] != 0 or locked['share_records_used'] != 0:
        raise ValueError('Wrong input type; not a synthetic offline run')
    if len(df) != len(locked['generator_families']) * len(locked['heldout_seeds']) * 10:
        raise ValueError('Unexpected heldout method-level row count')
    intervals = stratified_bootstrap(df, locked['calibration'])
    out = {'report_type':'descriptive_MC_bootstrap_post_frozen_run',
           'not_external_preregistration': True,
           'not_structural_confidence_intervals': True,
           'frozen_config_sha256':locked['config_sha256'],
           'heldout_simulations':len(locked['generator_families'])*len(locked['heldout_seeds']),
           'methods':[{**locked['heldout_target_coverage_comparison'][item['method']],**item}
                      for item in intervals]}
    (path/'nonoracle_mc_uncertainty.json').write_text(json.dumps(out,indent=2))
    pd.DataFrame(out['methods']).to_csv(path/'nonoracle_mc_uncertainty.csv',index=False)
    lines = ['# Internal frozen v1 — executed non-oracle synthetic challenge', '',
             f"Protocol configuration SHA-256: `{locked['config_sha256']}`.",
             f"Number of heldout synthetic datasets: {out['heldout_simulations']}; simulated people per dataset: {locked['n_per_replication']}.",
             'All graph proposals were fixed **offline synthetic fixtures**, not API-based LLMs.',
             'Observed outcomes only entered the blinded analysis; truth was used for scoring after predictions.',
             '', '| Method | Assertions | Wrong signs | Nonidentified claims | Matched at 50%? |',
             '|---|---:|---:|---:|---|']
    for item in out['methods']:
        lines.append(f"| {item['method']} | {item['heldout_assertions']}/{item['n_heldout']} | "
                     f"{item['wrong_sign_count']} | {item['not_identified_claims']} | "
                     f"{'yes' if item['comparable_at_target_coverage'] else 'no'} |")
    lines += ['', '**Interpretation:** The screened graph-set method was not coverage-matched at the primary target and made claims on nonidentified ground truth. No performance or novelty advantage has been demonstrated. The strict unmeasured-U arm always abstains, hence its selective risk is undefined. The no-LLM graph and naive voting arms tie on these fixed fixtures.',
              '', 'Bootstrap intervals in `nonoracle_mc_uncertainty.json` resample simulated seeds *within the five fixed mechanism families*. They quantify illustrative Monte Carlo variation conditional on this mechanism mix. They are neither survey-design inference, real-LLM performance intervals nor confidence intervals over unknown graph structures.',
              '', 'This internally frozen evaluation followed earlier exploratory simulations; it is **not external preregistration**. The full result requires independent mechanism families and real-model comparisons before causal-methodology publication claims.']
    (path/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'reported','n_bootstrap':2000, 'methods':len(out['methods']),
                      'heldout_simulations':out['heldout_simulations'],
                      'report':str((path/'REPORT.md').resolve())},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
