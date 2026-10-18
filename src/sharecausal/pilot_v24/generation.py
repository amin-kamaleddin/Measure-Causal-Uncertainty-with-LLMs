"""Generator process only: SCM CPTs never cross the analysis boundary."""
from fractions import Fraction as F
import numpy as np
import pandas as pd
from sharecausal.preflight_v23.certificates import BinaryDAG,law
from sharecausal.preflight.contracts import sha
from .observations import frame_payload


def observed_sample(task,plan,n,seed):
    scm=BinaryDAG.model_validate({'nodes':plan['nodes']})
    rng=np.random.default_rng(seed)
    names=sorted(v.name for v in task.card.variables if v.observed)
    joint=law(scm,names);states=sorted(joint)
    # Inverse-CDF coupling gives byte-identical observations for observationally
    # equivalent private SCMs, rather than merely equality in distribution.
    cumulative=np.cumsum([float(joint[x]) for x in states]);cumulative[-1]=1.
    draws=np.searchsorted(cumulative,rng.random(n),side='right')
    values={name:np.array([states[i][j] for i in draws],dtype=int) for j,name in enumerate(names)}
    observed=np.ones(n,dtype=int)
    if plan['recording']=='post_inspection_dependent':
        observed=(rng.random(n)<np.where(values['C']==1,.4,.95)).astype(int)
    elif plan['recording']!='complete':raise ValueError('unknown_synthetic_recording_rule')
    measured={v.name:values[v.name] for v in task.card.variables if v.observed}
    measured['observed']=observed
    # Hidden U and all unrecorded Y/descendants are removed before crossing processes.
    return frame_payload(task,pd.DataFrame(measured))
