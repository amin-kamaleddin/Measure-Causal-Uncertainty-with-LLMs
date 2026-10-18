import numpy as np
import pandas as pd
from sharecausal.data.clean import survey_numeric,binary_activity,childhood_poor,eurod

def test_special_codes_not_zero():
    x=binary_activity(pd.Series([0,1,-1,-2,-9,np.nan,3]))
    assert x.iloc[:2].tolist()==[0,1]
    assert x.iloc[2:].isna().all()

def test_childhood_categories():
    p=childhood_poor(pd.Series([1,2,3,4,5,-1,-2,np.nan]))
    assert p.iloc[:3].tolist()==[0,0,1]
    assert p.iloc[3:].isna().all()

def test_eurod_range():
    y=eurod(pd.Series([0,12,13,-9,-1]))
    assert y.iloc[:2].tolist()==[0,12]
    assert y.iloc[2:].isna().all()
