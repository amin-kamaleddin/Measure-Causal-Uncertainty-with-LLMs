"""Simplified MAR observation probability, not death/principal-stratum correction."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def observation_summary(df):
    out={}
    for arm in [0,1]:
        d=df.loc[df.A==arm]
        out[str(arm)]={'n':len(d),'observed_w9':int(d.Y.notna().sum()),
                       'proportion_observed':float(d.Y.notna().mean()),
                       'known_deceased':int(d.w9_deceased.sum())}
    return out

def illustrative_mar_weights(X, observed, max_weight=10):
    """In-sample diagnostic only; not a cross-fitted selection estimator."""
    a=np.asarray(observed,dtype=int);X=np.asarray(X,dtype=float)
    if a.min()==a.max():raise ValueError('No selection variation')
    fit=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)).fit(X,a)
    p=fit.predict_proba(X)[:,1]
    return np.minimum(1/np.clip(p,.01,1),max_weight)
