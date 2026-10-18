"""Wave-specific descriptive weights. No substitute for longitudinal weights."""
from __future__ import annotations
import numpy as np

def weighted_mean(y, weight):
    a, w = np.asarray(y, dtype=float), np.asarray(weight, dtype=float)
    ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
    return float(np.average(a[ok], weights=w[ok])) if ok.any() else float('nan')
