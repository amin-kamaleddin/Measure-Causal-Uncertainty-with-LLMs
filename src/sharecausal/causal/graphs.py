"""Auditable causal DAG specs, not ML-discovered certainty."""
from __future__ import annotations
from dataclasses import dataclass
import networkx as nx

@dataclass(frozen=True)
class GraphSpec:
    name: str
    edges: tuple[tuple[str,str], ...]
    observed: frozenset[str]
    rationale: str
    supported: bool = True
    selection_warning: bool = False

    def networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph(self.edges)
        graph.add_nodes_from(self.observed)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(f'{self.name}: graph contains cycle')
        return graph

BASE = (('B','A'),('B','Y'),('A','Y'))
SPECS = (
 GraphSpec('pre_depression', BASE+(('D8','A'),('D8','Y')),frozenset({'A','Y','B','D8'}),
           'Requires D8 temporally preceding all prior-year exposure: not demonstrated.'),
 GraphSpec('depression_mediator', BASE+(('A','D8'),('D8','Y')),frozenset({'A','Y','B','D8'}),
           'W8 EURO-D may be affected by participation.'),
 GraphSpec('unmeasured_frailty', BASE+(('U','A'),('U','Y')),frozenset({'A','Y','B'}),
           'Unobserved health/social-selection driver; nonidentifiable.'),
 GraphSpec('observed_survivors', BASE+(('A','S'),('Y','S')),frozenset({'A','Y','B','S'}),
           'Analysis conditioned on W9 observation; standard backdoor insufficient.',selection_warning=True),
)

def edge_frequencies(graphs: list[GraphSpec]) -> dict[str,float]:
    """Votes are frequencies only, NOT calibrated probabilities."""
    if not graphs: return {}
    edges = set.union(*(set(g.edges) for g in graphs))
    return {f'{u}->{v}':sum((u,v) in g.edges for g in graphs)/len(graphs)
            for u,v in sorted(edges)}


def validate_temporal_graph(spec:GraphSpec, node_time:dict[str,int]) -> None:
    spec.networkx()
    for a,b in spec.edges:
        if a in node_time and b in node_time and node_time[a] > node_time[b]:
            raise ValueError(f'Backward-in-time edge {a}->{b} in {spec.name}')
