"""Small *invented* Stata ZIPs: these contain no SHARE respondent records."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import pytest
import zipfile
from sharecausal.data.variables import MAP
from sharecausal.data.archives import NAME

@pytest.fixture
def mock_archives(tmp_path):
    n=40
    rng=np.random.default_rng(2)
    ids=[f'MOCK_{i:03d}' for i in range(n)]
    for w in [7,8,9]:
        zpath=tmp_path/NAME.format(wave=w)
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
            for (wave,module),columns in MAP.items():
                if wave!=w:continue
                d=pd.DataFrame(index=range(n))
                for col in columns:
                    if col=='mergeid':d[col]=ids
                    elif col=='country':d[col]=np.where(np.arange(n)%2==0,11,13)
                    else:d[col]=0
                if w==8:
                    if module=='cv_r':
                        d['age_int']=65;d['gender']=np.where(np.arange(n)%3==0,2,1)
                        d['int_year']=2019;d['int_month']=12;d['interview']=1
                    if module=='ac':
                        d['ac035d5']=np.arange(n)%2;d.loc[0,'ac035d5']=-2
                    if module=='gv_health':d['eurod']=2
                    if module=='gv_isced':d['isced1997_r']=3
                    if module=='gv_weights':d['cciw_w8_main']=1;d['dw_w8']=1
                if w==7:
                    if module=='cc':d['cc733_']=np.where(np.arange(n)%3==0,3,2)
                    if module=='gv_weights':d['SHARELIFE']=1
                    if module=='cv_r':d['interview']=1
                    if module=='gv_health':d['eurod']=-9
                if w==9:
                    if module=='cv_r':
                        d['interview']=1;d['int_year']=2022;d['int_month']=1;d['deceased']=0
                    if module=='gv_health':d['eurod']=np.arange(n)%8;d['eurodcat']=0
                buff=io.BytesIO();d.to_stata(buff,write_index=False,version=118)
                z.writestr(f'sharew{w}_rel9-0-0_{module}.dta',buff.getvalue())
    return tmp_path
