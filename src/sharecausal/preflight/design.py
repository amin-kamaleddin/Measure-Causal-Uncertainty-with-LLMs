"""Split lint and offline precision diagnostics: cards/families are clusters."""
import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import t
from .contracts import Strict


class SplitUnit(Strict):
    card_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    split: Literal['development', 'heldout']
    mechanism_family: str
    narrative_style: str
    seeds: list[int] = Field(min_length=1)


def lint_split(units):
    if len({u.card_hash for u in units}) != len(units):
        raise ValueError('card_overlap_or_duplicate')
    groups = {s: [u for u in units if u.split == s] for s in ('development', 'heldout')}
    if any(not g for g in groups.values()):
        raise ValueError('both_splits_required')
    a, b = groups.values()
    overlaps = {
        'family': sorted({u.mechanism_family for u in a} & {u.mechanism_family for u in b}),
        'style': sorted({u.narrative_style for u in a} & {u.narrative_style for u in b}),
        'seed': sorted({s for u in a for s in u.seeds} & {s for u in b for s in u.seeds}),
    }
    if any(overlaps.values()):
        raise ValueError('split_overlap')
    return {'passed': True, 'overlaps': overlaps, 'independent_public_cards': len(units),
            'counts': {s: {'cards': len(g), 'families': len({u.mechanism_family for u in g}),
                            'styles': len({u.narrative_style for u in g}),
                            'seed_probes': sum(len(u.seeds) for u in g)} for s, g in groups.items()},
            'limitation': 'Disjoint labels require PI review of scientific family/style independence.'}


class PrecisionConfig(Strict):
    card_counts: list[int] = Field(min_length=1)
    family_counts: list[int] = Field(min_length=1)
    assertion_rates: list[float] = Field(min_length=1)
    assumed_conditional_error_a: float = Field(gt=0, lt=1)
    assumed_conditional_error_b: float = Field(ge=0, lt=1)
    family_icc: float = Field(gt=0, lt=1)
    probes_per_card: int = Field(ge=1)
    repetitions: int = Field(ge=100, le=10000)
    simulation_seed: int = Field(ge=0)
    candidate_half_width: float = Field(gt=0)

    @model_validator(mode='after')
    def coherent(self):
        if self.assumed_conditional_error_b >= self.assumed_conditional_error_a:
            raise ValueError('candidate_error_difference_must_be_positive')
        if any(not 0 < a <= 1 for a in self.assertion_rates):
            raise ValueError('invalid_candidate_coverage')
        if any(f < 3 or n < f or n % f for n in self.card_counts for f in self.family_counts):
            raise ValueError('balanced_independent_card_family_counts_required')
        return self


def precision_diagnostics(cfg: PrecisionConfig):
    rows = []
    p = cfg.assumed_conditional_error_a
    concentration = 1/cfg.family_icc - 1
    for n in cfg.card_counts:
        for families in cfg.family_counts:
            for coverage in cfg.assertion_rates:
                for coupling in ('nested_optimistic', 'conditionally_independent'):
                    critical = float(t.ppf(.975, families-1))
                    rng = np.random.default_rng(cfg.simulation_seed + n + families)
                    estimates, cluster_ses, naive_ses, covers, excludes = [], [], [], [], []
                    target = coverage * (p - cfg.assumed_conditional_error_b)
                    for _ in range(cfg.repetitions):
                        q = np.repeat(rng.beta(p*concentration, (1-p)*concentration, families), n//families)
                        u = rng.random(n)
                        v = u if coupling == 'nested_optimistic' else rng.random(n)
                        asserted = rng.random(n) < coverage
                        delta = ((u < q).astype(float) - (v < q*cfg.assumed_conditional_error_b/p)) * asserted
                        estimate = float(delta.mean())
                        family_means = delta.reshape(families, -1).mean(axis=1)
                        se = float(family_means.std(ddof=1)/math.sqrt(families))
                        half = critical*se
                        estimates.append(estimate); cluster_ses.append(se)
                        naive_ses.append(float(delta.std(ddof=1)/math.sqrt(n*cfg.probes_per_card)))
                        covers.append(abs(estimate-target) <= half)
                        excludes.append(abs(estimate) > half)
                    half = critical*float(np.mean(cluster_ses))
                    rows.append({'independent_cards': n, 'independent_families': families,
                        'probes_per_card_not_independent': cfg.probes_per_card,
                        'assumed_assertion_rate': coverage, 'error_coupling': coupling,
                        'assumed_difference_per_eligible_card': target,
                        'monte_carlo_mean_difference': float(np.mean(estimates)),
                        'monte_carlo_sd': float(np.std(estimates, ddof=1)),
                        'mean_family_cluster_se': float(np.mean(cluster_ses)),
                        'mean_misleading_seed_independent_se': float(np.mean(naive_ses)),
                        'mean_approximate_95_half_width': half,
                        'scenario_interval_coverage_not_a_guarantee': float(np.mean(covers)),
                        'scenario_excludes_zero_fraction_not_validated_power': float(np.mean(excludes)),
                        'candidate_half_width_met': half <= cfg.candidate_half_width,
                        'status': 'candidate_assumptions_not_LLM_performance'})
    return rows
