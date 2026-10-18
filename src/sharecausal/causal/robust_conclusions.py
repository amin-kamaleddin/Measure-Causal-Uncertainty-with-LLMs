"""Identification-first direction rule; never call intervals structural CIs."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ConditionalInterval:
    graph: str
    estimate: float
    lower: float
    upper: float
    identified: bool


def assess(intervals: list[ConditionalInterval], shifts=(0.0,), expected_graphs=None) -> dict:
    if not intervals:
        return {'decision':'abstain','reason':'No graph results','envelope':None}
    if expected_graphs is not None and set(x.graph for x in intervals)!=set(expected_graphs):
        return {'decision':'abstain','reason':'Supported graph results incomplete','envelope':None}
    if any(not x.identified for x in intervals):
        return {'decision':'abstain','reason':'At least one supported graph not identified',
                'nonidentified_graphs':[x.graph for x in intervals if not x.identified],
                'envelope':None}
    if not shifts: raise ValueError('At least one confounding scenario required')
    bounds=[(x.lower + float(shift),x.upper + float(shift)) for x in intervals for shift in shifts]
    low, high = min(q[0] for q in bounds),max(q[1] for q in bounds)
    decision='negative' if high<0 else 'positive' if low>0 else 'abstain'
    return {'decision':decision,'reason':('All supported scenarios have same sign' if decision!='abstain'
            else 'Robustness envelope includes zero'),
            'envelope':{'lower':float(low),'upper':float(high)},
            'interpretation':'Graph/scenario robustness envelope, not a simultaneous 95% CI'}
