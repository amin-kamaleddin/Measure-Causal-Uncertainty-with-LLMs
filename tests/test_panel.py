import pytest
import pandas as pd
from sharecausal.data.panel import construct
from sharecausal.data.archives import locate_archives
from sharecausal.data.checks import safe_merge

def test_real_zip_api_with_synthetic_fixture(mock_archives):
    result=construct(locate_archives(mock_archives))
    f=result.flow
    assert f['wave8_ac_module']==40
    assert f['valid_wave8_activity']==39
    assert f['w9_valid_eurod']==39
    assert f['w9_nonforward_interview_time']==0
    assert f['w7_pre_wave8_eurod']==0
    assert result.data.A.isin([0,1]).all()
    assert result.data.mergeid.is_unique

def test_no_accidental_many_to_many():
    left=pd.DataFrame({'mergeid':['a','b']})
    right=pd.DataFrame({'mergeid':['a','a'],'exposure':[0,1]})
    with pytest.raises(ValueError,match='duplicate'):
        safe_merge(left,right,'unsafe')
