"""Observed-variable backdoor criterion; explicitly DOES NOT solve general ID."""
from __future__ import annotations
import itertools
import networkx as nx
from .graphs import GraphSpec


def backdoor_sets(spec: GraphSpec, treatment='A', outcome='Y', max_covariates=12):
    g = spec.networkx()
    if treatment not in g or outcome not in g: raise ValueError('Treatment/outcome not in graph')
    if spec.selection_warning:
        return []  # no general identification of selected-population effect
    candidates = sorted((spec.observed-set([treatment,outcome]))-nx.descendants(g,treatment))
    if len(candidates)>max_covariates:
        raise ValueError('Too many observed adjustment candidates for exhaustive enumeration')
    bd = g.copy()
    bd.remove_edges_from(list(g.out_edges(treatment)))
    out = []
    for n in range(len(candidates)+1):
        for subset in itertools.combinations(candidates,n):
            if nx.is_d_separator(bd,{treatment},{outcome},set(subset)):
                out.append(tuple(subset))
        if out: break  # return minimal-cardinality candidates, not all supersets
    return out


def identify(spec:GraphSpec):
    sets=backdoor_sets(spec)
    return {'graph':spec.name,'identified_by_backdoor':bool(sets),
            'adjustment_sets':[list(x) for x in sets],
            'selection_warning':spec.selection_warning,
            'rationale':spec.rationale,
            'status':('assumption_conditional' if sets else 'not_identified_by_backdoor')}
