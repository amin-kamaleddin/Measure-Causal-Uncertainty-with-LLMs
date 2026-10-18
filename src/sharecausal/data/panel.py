"""Wave 8-anchored SHARE linkage with explicit respondent universes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .archives import read_module
from .clean import binary_activity, childhood_poor, eurod, survey_numeric
from .checks import safe_merge
from .variables import MAP

@dataclass
class PanelResult:
    data: pd.DataFrame
    flow: dict
    diagnostics: dict


def _read(paths, wave, module, renames=None):
    d = read_module(paths[wave], wave, module, MAP[(wave,module)])
    return d.rename(columns=renames or {})


def construct(paths) -> PanelResult:
    ac = _read(paths, 8, 'ac', {'country':'country8'})
    step = {'wave8_ac_module': len(ac)}
    cv8 = _read(paths, 8, 'cv_r', {'country':'cv8_country', 'interview':'interview8',
                                  'age_int':'age8','gender':'gender8',
                                  'int_year':'year8', 'int_month':'month8','deceased':'deceased8'})
    d = safe_merge(ac,cv8,'w8_cv')
    step['wave8_main_interview'] = int((d.interview8 == 1).sum())
    d = d.loc[d.interview8 == 1].copy()
    d['country_mismatch8'] = d.country8.ne(d.cv8_country)
    if d.country_mismatch8.any():
        raise ValueError(f'{d.country_mismatch8.sum()} contradictory country codes within Wave 8')
    d['A'] = binary_activity(d.ac035d5)
    step['valid_wave8_activity'] = int(d.A.notna().sum())
    d = d.loc[d.A.notna()].copy()
    step['wave8_exposed'] = int((d.A==1).sum())
    step['wave8_unexposed'] = int((d.A==0).sum())
    d = safe_merge(d,_read(paths,8,'gv_health',{'eurod':'eurod8'}),'w8_health')
    d = safe_merge(d,_read(paths,8,'gv_isced',{'isced1997_r':'education8'}),'w8_isced')
    d = safe_merge(d,_read(paths,8,'gv_weights'),'w8_weights')
    d = safe_merge(d,_read(paths,7,'cc', {'country':'country7','cc733_':'childhood_fin',
                                          'cc008_':'childhood_books','cc729_':'childhood_lonely'}),'w7_cc')
    d = safe_merge(d,_read(paths,7,'gv_weights'),'w7_weights')
    d = safe_merge(d,_read(paths,7,'gv_health',{'eurod':'eurod7'}),'w7_health')
    d = safe_merge(d,_read(paths,7,'cv_r',{'interview':'interview7','int_year':'year7',
                                         'int_month':'month7'}),'w7_cv')
    d = safe_merge(d,_read(paths,9,'cv_r',{'country':'country9','interview':'interview9',
                 'int_year':'year9','int_month':'month9','deceased':'deceased9'}),'w9_cv')
    d = safe_merge(d,_read(paths,9,'gv_health',{'eurod':'eurod9','eurodcat':'eurodcat9'}),'w9_health')
    d['Y'] = eurod(d.eurod9).where(d.interview9 == 1)
    d['D8'] = eurod(d.eurod8)
    d['D7'] = eurod(d.eurod7).where(d.interview7 == 1)
    d['age'] = survey_numeric(d.age8).where(lambda x:(x>=50)&(x<=110))
    d['female'] = survey_numeric(d.gender8,{1,2}).map({1:0,2:1})
    d['education'] = survey_numeric(d.education8, set(range(7)))
    d['country'] = survey_numeric(d.country8)
    d['poor'] = childhood_poor(d.childhood_fin)
    d['observed9'] = d.Y.notna().astype(int)
    d['w9_deceased'] = d.deceased9.eq(1).fillna(False)
    step['w7_record_linked'] = int(d.childhood_fin.notna().sum())
    step['w7_sharelife_flag1'] = int(d.SHARELIFE.eq(1).sum())
    step['w7_valid_childhood_financial'] = int(d.poor.notna().sum())
    step['w7_pre_wave8_eurod'] = int(d.D7.notna().sum())
    step['w7_both_childhood_and_regular_depression'] = int((d.D7.notna()&d.poor.notna()).sum())
    step['w8_eurod_valid'] = int(d.D8.notna().sum())
    step['w9_cv_record_linked'] = int(d.interview9.notna().sum())
    step['w9_known_deceased'] = int(d.w9_deceased.sum())
    step['w9_main_interview'] = int(d.interview9.eq(1).sum())
    step['w9_end_of_life_interview'] = int(d.interview9.eq(2).sum())
    step['w9_valid_eurod'] = int(d.Y.notna().sum())
    # Report calendar anomalies explicitly; do not manufacture interview dates.
    t8 = pd.to_numeric(d.year8,errors='coerce')*12+pd.to_numeric(d.month8,errors='coerce')
    t9 = pd.to_numeric(d.year9,errors='coerce')*12+pd.to_numeric(d.month9,errors='coerce')
    valid_time = d.interview9.eq(1)&t8.gt(0)&t9.gt(0)&d.month8.between(1,12)&d.month9.between(1,12)
    d['invalid_time_order'] = valid_time & t9.le(t8)
    step['w9_nonforward_interview_time'] = int(d.invalid_time_order.sum())
    d.loc[d.invalid_time_order, 'Y'] = np.nan
    d['observed9'] = d.Y.notna().astype(int)
    step['w9_valid_eurod_forward_order'] = int(d.Y.notna().sum())
    country_mismatch9 = (d.country9.notna() & d.country.ne(d.country9))
    diag = {
      'country_changes_between_w8_w9': int(country_mismatch9.sum()),
      'exposure_code_counts': {str(k):int(v) for k,v in ac.ac035d5.value_counts(dropna=False).items()},
      'w8_year_counts': {str(k):int(v) for k,v in d.year8.value_counts(dropna=False).items()},
      'w9_year_counts': {str(k):int(v) for k,v in d.year9.value_counts(dropna=False).items()},
      'w9_obs_by_exposure': {str(int(a)):{'eligible':len(g),'observed':int(g.observed9.sum()),
                   'known_deceased':int(g.w9_deceased.sum())} for a,g in d.groupby('A')},
      'per_country': {str(int(k)):{'w8_eligible':len(g),'w9_outcomes':int(g.Y.notna().sum()),
                                'w9_deceased':int(g.w9_deceased.sum())} for k,g in d.groupby('country')},
      'missing_covariates': {c:int(d[c].isna().sum()) for c in ['age','female','education','D8','poor','D7']},
      'childhood_financial_counts': {str(k):int(v) for k,v in d.childhood_fin.value_counts(dropna=False).items()},
      'eurod9_distribution':{str(k):int(v) for k,v in d.Y.value_counts(dropna=False).items()},
    }
    return PanelResult(data=d,flow=step,diagnostics=diag)
