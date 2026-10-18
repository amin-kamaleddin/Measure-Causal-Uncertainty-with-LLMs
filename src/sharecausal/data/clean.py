"""SHARE cleaning: distinguish survey special codes and true not observed."""
import numpy as np
import pandas as pd

def survey_numeric(s: pd.Series, allowed: set[int] | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce').astype(float)
    x = x.mask(x < 0)  # SHARE -1 don't know, -2 refusal, -9 inapplicable
    if allowed is not None:
        x = x.where(x.isin(allowed))
    return x

def eurod(s: pd.Series) -> pd.Series:
    return survey_numeric(s, set(range(13)))

def binary_activity(s: pd.Series) -> pd.Series:
    return survey_numeric(s, {0, 1})

def childhood_poor(s: pd.Series) -> pd.Series:
    value = survey_numeric(s, {1, 2, 3})
    return pd.Series(np.where(value.isna(), np.nan, (value == 3).astype(float)), index=s.index)
