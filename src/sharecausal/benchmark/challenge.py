"""Previously unevaluated synthetic mechanisms reserved as a new challenge set.

No SHARE data or OpenRouter output informs these mechanisms.  Truth and
mechanism labels are available to *scoring only*.  Estimation code receives a
plain DataFrame with observable columns, never this return object's metadata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .scm import SCMData

CHALLENGE_MECHANISMS = (
    'threshold_nonlinear', 'proxy_hidden', 'informative_dropout',
    'near_zero_heterogeneous', 'overlap_collapse',
)
# This is ONLY for scoring; it must not be imported by the blinded analyzer.
NONIDENTIFIABLE_GROUND_TRUTH = frozenset({'proxy_hidden', 'informative_dropout'})


def generate_challenge(mechanism: str, n: int, seed: int) -> SCMData:
    if mechanism not in CHALLENGE_MECHANISMS:
        raise ValueError(f'Unknown challenge mechanism: {mechanism}')
    if n < 80:
        raise ValueError('At least 80 observations required')
    rng = np.random.default_rng(seed + 389129)
    u = rng.normal(size=n)
    x1 = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    x2 = (0.65 * u + rng.normal(scale=0.85, size=n)
          if mechanism == 'proxy_hidden' else rng.normal(size=n))
    logit = -0.15 + (3.5 if mechanism == 'overlap_collapse' else 1.05) * x1 - 0.65 * x2
    if mechanism == 'proxy_hidden':
        logit += 1.35 * u
    p = expit(logit)
    a = rng.binomial(1, p)
    noise = rng.standard_t(df=7, size=n) * 0.75
    base = 0.8 * x1 + 0.7 * x2 + noise
    if mechanism == 'threshold_nonlinear':
        base += 0.75 * (x1 > 0.4) + 0.45 * np.sin(2 * x2)
    if mechanism == 'proxy_hidden':
        base += 1.6 * u
    # Seed parity assigns both beneficial and harmful interventions; no ATE
    # direction is supplied to the blinded inference routine.
    direction = 1.0 if seed % 2 else -1.0
    tau = (direction * 0.65 if mechanism != 'near_zero_heterogeneous'
           else 0.65 * np.where(x1 >= 0, 1.0, -1.0))
    m0 = 0.25 * x1 + rng.normal(scale=0.7, size=n)
    m1 = m0 + 0.8
    mediator_effect = 0.65 if mechanism == 'threshold_nonlinear' else 0.0
    y0 = base + mediator_effect * m0
    y1 = base + tau + mediator_effect * m1
    y = np.where(a == 1, y1, y0)
    m = np.where(a == 1, m1, m0)
    c = 0.6 * a + 0.5 * y + rng.normal(scale=0.6, size=n)
    if mechanism == 'informative_dropout':
        # In the analysis, Y for R=0 is removed, exactly as it would be in
        # incomplete follow-up data. Full outcomes remain here for scoring.
        p_observed = expit(1.0 - 0.65 * a - 0.8 * y)
        observed = rng.binomial(1, p_observed)
    else:
        observed = np.ones(n, dtype=int)
    edges = [('X1', 'A'), ('X2', 'A'), ('X1', 'Y'), ('X2', 'Y'),
             ('A', 'Y'), ('X1', 'M'), ('A', 'M'), ('A', 'C'), ('Y', 'C')]
    if mechanism == 'proxy_hidden':
        edges += [('U', 'A'), ('U', 'Y'), ('U', 'X2')]
    if mechanism == 'threshold_nonlinear':
        edges += [('M', 'Y')]
    if mechanism == 'informative_dropout':
        edges += [('A', 'S'), ('Y', 'S')]
    frame = pd.DataFrame({'X1': x1, 'X2': x2, 'A': a, 'Y': y,
                          'M': m, 'C': c, 'observed': observed})
    return SCMData(frame, float(np.mean(y1-y0)), mechanism, tuple(edges))
