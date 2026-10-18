"""Cross-fitted AIPW with auditable overlap and (optional) cluster influence SE."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge

@dataclass
class Estimate:
    effect:float
    se:float
    ci_lower:float
    ci_upper:float
    n:int
    treated:int
    control:int
    overlap_fraction_below_005_or_above_095:float
    propensity_min:float
    propensity_max:float
    method:str
    def as_dict(self):
        return dict(self.__dict__)


def _regression():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),Ridge(alpha=3))

def _propensity():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=1.0,max_iter=700))

def _interval(psi,cluster=None):
    n=len(psi); c=psi-np.mean(psi)
    if cluster is None:
        se=np.std(c,ddof=1)/np.sqrt(n)
    else:
        cl=np.asarray(cluster)
        levels=np.unique(cl)
        if len(levels)<3: raise ValueError('Clustered standard error requires at least three clusters')
        sums=np.array([c[cl==z].sum() for z in levels]);m=len(levels)
        se=np.sqrt(m/(m-1)*np.sum(sums*sums)/n**2)
    estimate=float(np.mean(psi));z=norm.ppf(0.975)
    return estimate,float(se),float(estimate-z*se),float(estimate+z*se)


def aipw(y, a, X, folds=3, seed=11, cluster=None, trim=0.01, return_scores=False) -> Estimate:
    y,a,X=np.asarray(y,dtype=float),np.asarray(a,dtype=int),np.asarray(X,dtype=float)
    if X.ndim!=2 or len(y)!=len(a) or len(y)!=len(X):raise ValueError('Mismatched shapes')
    if not np.isfinite(y).all() or not np.isfinite(X).all() or not np.isin(a,[0,1]).all():
        raise ValueError('Caller must provide complete numeric rows and binary treatment')
    if len(y)<30 or min(np.bincount(a,minlength=2))<folds*3:
        raise ValueError('Insufficient treated/control cases for cross-fitting')
    if not 0<trim<0.5:raise ValueError('Invalid positivity clipping')
    e, m0, m1 = [np.zeros(len(a)) for _ in range(3)]
    split=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    for train,test in split.split(X,a):
        e[test]=_propensity().fit(X[train],a[train]).predict_proba(X[test])[:,1]
        m0[test]=_regression().fit(X[train][a[train]==0],y[train][a[train]==0]).predict(X[test])
        m1[test]=_regression().fit(X[train][a[train]==1],y[train][a[train]==1]).predict(X[test])
    exc=float(np.mean((e<0.05)|(e>0.95)))
    ec=np.clip(e,trim,1-trim)
    psi=m1-m0+a*(y-m1)/ec-(1-a)*(y-m0)/(1-ec)
    eff,se,lo,hi=_interval(psi,cluster=cluster)
    output=Estimate(eff,se,lo,hi,len(a),int(sum(a)),int(sum(1-a)),exc,float(e.min()),float(e.max()),
                    '3-fold cross-fitted AIPW (logit propensity; ridge outcome)')
    return (output,psi) if return_scores else output


def unadjusted(y,a,cluster=None):
    y=np.asarray(y,dtype=float);a=np.asarray(a,dtype=int)
    if len(y)!=len(a) or not np.isin(a,[0,1]).all(): raise ValueError('Invalid treatment/outcome')
    if min(np.bincount(a,minlength=2))<3: raise ValueError('Insufficient cells')
    e=np.mean(a);psi=(a*(y-np.mean(y[a==1]))/e - (1-a)*(y-np.mean(y[a==0]))/(1-e))
    diff=float(np.mean(y[a==1])-np.mean(y[a==0]))
    _,se,_,_=_interval(psi, cluster=cluster)
    z=norm.ppf(.975)
    return {'effect':diff,'se':se,'ci_lower':diff-z*se,'ci_upper':diff+z*se,
            'mean_exposed':float(y[a==1].mean()),'mean_unexposed':float(y[a==0].mean()),'n':len(y)}


def frame_matrix(df:pd.DataFrame, continuous=('age','female','education'), categories=('country',)):
    """Complete-case pre-exposure covariates with prespecified one-hot country."""
    subset=df[list(continuous)+list(categories)].copy()
    if subset.isna().any().any():raise ValueError('Covariates require complete cases')
    values=pd.get_dummies(subset,columns=list(categories),drop_first=True,dtype=float)
    return values.to_numpy(dtype=float)
