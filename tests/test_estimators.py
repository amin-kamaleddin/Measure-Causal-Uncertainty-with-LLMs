import numpy as np
from sharecausal.causal.estimators import aipw,unadjusted
from sharecausal.benchmark.scm import generate

def test_crossfit_known_truth():
    x=generate('measured',n=4500,seed=901)
    d=x.frame
    fit=aipw(d.Y,d.A,d[['X1','X2']].to_numpy(),seed=19)
    assert abs(fit.effect-x.true_ate)<.2
    assert fit.ci_lower<fit.ci_upper
    assert fit.n==4500

def test_unadjusted_returns_diff():
    a=np.r_[np.zeros(10),np.ones(10)]
    y=np.r_[np.ones(10),np.ones(10)*3]
    assert abs(unadjusted(y,a)['effect']-2)<1e-10

def test_invalid_binary_rejected():
    import pytest
    with pytest.raises(ValueError):aipw(np.ones(35),np.ones(35)*2,np.ones((35,2)))
