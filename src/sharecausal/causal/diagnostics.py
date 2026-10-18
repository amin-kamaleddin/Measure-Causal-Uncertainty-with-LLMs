"""Summary diagnostics only; no participant-level result storage."""
import numpy as np

def standardized_mean_difference(x,a,w=None):
    x=np.asarray(x,dtype=float);a=np.asarray(a,dtype=int)
    w=np.ones(len(x)) if w is None else np.asarray(w,dtype=float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    mu=[np.average(x[ok&(a==k)],weights=w[ok&(a==k)]) for k in (0,1)]
    v=[np.average((x[ok&(a==k)]-mu[k])**2,weights=w[ok&(a==k)]) for k in (0,1)]
    return float((mu[1]-mu[0])/np.sqrt((v[0]+v[1])/2)) if sum(v)>0 else float('nan')
