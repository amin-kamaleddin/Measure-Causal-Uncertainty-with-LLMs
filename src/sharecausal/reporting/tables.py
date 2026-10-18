"""Aggregate summary tables; do not produce respondent-level exports."""
import pandas as pd

def baseline(df):
    rows=[]
    for arm,part in df.groupby('A'):
        rows.append({'exposed':int(arm),'n_eligible':len(part),
                     'n_wave9_eurod':int(part.Y.notna().sum()),
                     'mean_eurod_w9_observed':float(part.Y.mean()),
                     'mean_eurod_w8_observed':float(part.D8.mean()),
                     'mean_age':float(part.age.mean()),
                     'female_fraction':float(part.female.mean()),
                     'poor_childhood_fraction_available':float(part.poor.mean()),
                     'n_childhood_poor_valid':int(part.poor.notna().sum()),
                     'attrition_fraction':float(part.Y.isna().mean()),
                     'known_deceased':int(part.w9_deceased.sum())})
    return pd.DataFrame(rows)
