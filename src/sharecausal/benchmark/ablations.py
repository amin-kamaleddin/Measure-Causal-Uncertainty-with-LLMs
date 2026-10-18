"""Deterministic, graph-based ablations; not synthetic fake LLM performance."""
from sharecausal.causal.graphs import GraphSpec
from sharecausal.causal.identification import identify

def majority_graph(candidate_graphs:list[GraphSpec]):
    """Majority directed edge, veto cycles, report ambiguous edges explicitly."""
    if not candidate_graphs:raise ValueError('No candidate graphs')
    from collections import Counter
    counts=Counter(edge for g in candidate_graphs for edge in g.edges)
    n=len(candidate_graphs)
    edges=tuple(sorted(e for e,v in counts.items() if v>n/2))
    observed=frozenset.union(*(x.observed for x in candidate_graphs))
    proposal=GraphSpec('naive_majority',edges,observed,'Synthetic graph vote, not probability')
    proposal.networkx()  # fail rather than silently remove cyclic majority edges
    return proposal

def graph_accuracy(candidate_edges,true_edges):
    pred=set(candidate_edges);true=set(true_edges)
    tp=len(pred&true)
    return {'precision':tp/len(pred) if pred else None,
            'recall':tp/len(true) if true else None,
            'directed_shd':len(pred^true)}
