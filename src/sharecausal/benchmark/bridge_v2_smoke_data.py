"""Tiny independent synthetic integration examples, not v2 holdout mechanisms.

Only the orchestration/scoring side imports this module. Parameters have no
SHARE source. These three examples do not support a performance comparison.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class SmokeTask:
    frame: pd.DataFrame
    true_ate: float
    identified: bool
    label: str


def smoke_tasks() -> list[SmokeTask]:
    tasks = []
    for index, label in enumerate(('measured_smoke', 'latent_smoke', 'selection_smoke')):
        rng = np.random.default_rng(60601 + index)
        n = 320
        x1, x2, u = rng.normal(size=(3, n))
        latent = index == 1
        a = rng.binomial(1, expit(0.5 * x1 - 0.3 * x2 + (0.8 * u if latent else 0)))
        y = 1.1 * a + 0.7 * x1 + 0.4 * x2 + rng.normal(size=n) + (u if latent else 0)
        m = 0.6 * a + rng.normal(size=n)
        c = 0.3 * y + rng.normal(size=n)
        observed = (rng.binomial(1, expit(1 - 0.5 * y)) if index == 2 else np.ones(n, dtype=int))
        frame = pd.DataFrame(dict(X1=x1, X2=x2, A=a, M=m, Y=y, C=c, observed=observed))
        tasks.append(SmokeTask(frame, 1.1, index == 0, label))
    return tasks
