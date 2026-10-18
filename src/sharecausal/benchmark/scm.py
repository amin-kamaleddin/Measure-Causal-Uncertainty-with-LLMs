"""Independent synthetic structural causal mechanisms with coupled do(A=0,1) truth.

No SHARE value, distribution or respondent enters these equations. Graph labels are
oracle generator knowledge, NOT evidence that real LLMs found the graphs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import expit

SCENARIOS=('measured','hidden','mediator','collider','weak_positivity','nonlinear','heterogeneity','selection')

@dataclass
class SCMData:
    frame:pd.DataFrame
    true_ate:float
    scenario:str
    true_edges:tuple[tuple[str,str],...]


def generate(scenario='measured',n=700,seed=11) -> SCMData:
    if scenario not in SCENARIOS:raise ValueError('Unknown synthetic SCM')
    if n<50:raise ValueError('n too small')
    r=np.random.default_rng(seed)
    x1=r.normal(size=n); x2=r.normal(size=n); u=r.normal(size=n)
    z=-.25+(.95 if scenario!='weak_positivity' else 4.4)*x1-.6*x2
    if scenario=='hidden': z+=1.0*u
    ps=expit(z)
    a=r.binomial(1,ps)
    eps=r.normal(scale=.85,size=n)
    tau=-.85+(.65*(x1>0) if scenario=='heterogeneity' else 0)
    base=1.15*x1+.65*x2+.12*x1*x2+eps
    if scenario=='nonlinear':base+=.65*x1**2+np.sin(2*x2)
    if scenario=='hidden':base+=1.25*u
    mediator0=.25*x1 + r.normal(scale=.6,size=n)
    mediator1=mediator0+.85
    b=.85 if scenario=='mediator' else 0.
    y0=base+b*mediator0
    y1=base+tau+b*mediator1
    y=np.where(a==1,y1,y0)
    m=np.where(a==1,mediator1,mediator0)
    # C is a descendant/collider only in the dedicated collider stress scenario.
    c=(.65*a+.6*y+r.normal(scale=.5,size=n) if scenario=='collider'
       else r.normal(scale=.5,size=n))
    if scenario=='selection':
        survival_prob=expit(1.2-.4*a-.65*y)
        observed=r.binomial(1,survival_prob)
    else:observed=np.ones(n,dtype=int)
    data=pd.DataFrame({'X1':x1,'X2':x2,'A':a,'Y':y,'M':m,'C':c,'observed':observed})
    edges=(('X1','A'),('X2','A'),('X1','Y'),('X2','Y'),('A','Y'))
    if scenario=='hidden':edges+=(('U','A'),('U','Y'))
    edges+=(('A','M'),('X1','M'))  # M generated from A and X1 in every mechanism
    if scenario=='mediator':edges+=(('M','Y'),)
    if scenario=='collider':edges+=(('A','C'),('Y','C'))
    if scenario=='selection':edges+=(('A','S'),('Y','S'))
    return SCMData(data,float(np.mean(y1-y0)),scenario,edges)
