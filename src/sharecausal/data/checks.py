"""Ingestion invariants: missing, uniqueness, cross-country surprises."""
from __future__ import annotations
import pandas as pd

def unique_key(frame: pd.DataFrame, name: str) -> None:
    if 'mergeid' not in frame.columns or frame.mergeid.isna().any():
        raise ValueError(f'{name}: no complete mergeid')
    if frame.mergeid.duplicated().any():
        raise ValueError(f'{name}: duplicate mergeid, unsafe many-to-many merge')

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, name: str, how='left') -> pd.DataFrame:
    unique_key(left, 'left-of-' + name)
    unique_key(right, name)
    return left.merge(right, on='mergeid', how=how, validate='one_to_one', sort=False)
