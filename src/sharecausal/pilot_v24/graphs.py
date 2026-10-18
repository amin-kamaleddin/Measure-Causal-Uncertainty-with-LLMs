"""Public graph-conditional backdoor checks without truth or NetworkX imports."""
from itertools import combinations
from sharecausal.preflight.graph_utils import has_path,is_acyclic


def _ancestors(nodes,edges):
    out=set(nodes)
    while True:
        extended=out|{a for a,b in edges if b in out}
        if extended==out:return out
        out=extended


def d_separated(edges,conditioned):
    """Moralize the ancestral backdoor DAG; A,Y are fixed query endpoints."""
    edges=[(a,b) for a,b in edges if a!='A']
    nodes=_ancestors({'A','Y'}|set(conditioned),edges)
    adjacency={v:set() for v in nodes}
    for a,b in edges:
        if a in nodes and b in nodes:adjacency[a].add(b);adjacency[b].add(a)
    for child in nodes:
        parents=[a for a,b in edges if b==child and a in nodes]
        for a,b in combinations(parents,2):adjacency[a].add(b);adjacency[b].add(a)
    seen=set(conditioned);pending=['A']
    while pending:
        node=pending.pop()
        if node in seen:continue
        if node=='Y':return False
        seen.add(node);pending.extend(adjacency[node]-seen)
    return True


def identify_graph(task,pairs):
    pairs=[tuple(p) for p in pairs];nodes={v.name:v for v in task.card.variables}
    if not is_acyclic(pairs) or any(a not in nodes or b not in nodes or nodes[a].time>nodes[b].time for a,b in pairs):
        raise ValueError('invalid_causal_graph')
    if not has_path(pairs):return {'status':'graph_conditional_structural_zero','adjustment':[],
        'structural_effect':0.,'scope':'conditional_on_complete_causal_graph'}
    if task.card.selection!='complete_outcomes':return {'status':'selection_unresolved','adjustment':None,
        'scope':'no_selection_identification_proof_implemented'}
    candidates=sorted(v.name for v in nodes.values() if v.observed and v.role=='baseline')
    for k in range(len(candidates)+1):
        for z in combinations(candidates,k):
            if d_separated(pairs,z):return {'status':'identified_by_baseline_backdoor','adjustment':list(z),
                'scope':'conditional_on_complete_causal_graph'}
    return {'status':'not_identified_by_baseline_backdoor','adjustment':None,
            'scope':'restricted_criterion_failure_not_general_nonidentification_proof'}


def public_direction_eligible(task):
    return (task.card.assignment=='randomized_documented' and task.card.selection=='complete_outcomes'
        and any(e.kind=='assignment' and e.relation=='supported' and (e.source,e.target)==('A','Y')
                for e in task.card.evidence))
