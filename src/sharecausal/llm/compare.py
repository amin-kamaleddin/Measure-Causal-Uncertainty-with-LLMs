"""Consistent prompts and mock graph fixtures; live calls explicitly opt-in."""
from __future__ import annotations
from .schemas import GraphAnswer,Arrow
from .prompts import messages,hash_messages

def mock_graphs():
    def arrow(src,dst):
        return Arrow(source=src,target=dst,causal_role='hypothesis',
                     temporal_rationale='Synthetic public fixture, not scientific validation',
                     evidence_reference='unverified hypothesis',uncertainty='High')
    return [GraphAnswer(edges=[arrow('B','A'),arrow('B','Y'),arrow('A','Y')],
                  abstain=False,abstention_reason='',contested_assumptions=['unmeasured confounding']),
            GraphAnswer(edges=[arrow('B','A'),arrow('B','Y'),arrow('A','D8'),arrow('D8','Y')],
                  abstain=True,abstention_reason='Exposure timing ambiguous',contested_assumptions=['D8 timing'])]

def dry_run():
    fixtures=mock_graphs()
    return {'mode':'DRY_RUN_OFFLINE','live_requests':0,'actual_llm_results':0,'cost_usd':0.0,
            'prompt_sha256':{v:hash_messages(v) for v in ('base','counterargument')},
            'mock_valid_graphs':len(fixtures),'schema':'GraphAnswer',
            'warning':'Mock graph fixtures are not empirical LLM results; no provider availability/price confirmed'}


def summarize_graphs(records):
    """Descriptive LLM graph disagreement, NOT edge truth probabilities.

    Accepts completed OpenRouter records (if live) and validates them again.
    Naive majority is reported as a comparator and is NOT causally identified.
    """
    from collections import Counter,defaultdict
    from itertools import combinations
    from statistics import mean
    if not records:
        return {'n_graphs':0,'warning':'No live responses; no LLM accuracy established'}
    by_model=defaultdict(list)
    edge_lists=[]
    for record in records:
        v=GraphAnswer.model_validate(record['graph'])
        edges=frozenset((x.source,x.target) for x in v.edges)
        edge_lists.append(edges)
        by_model[record.get('model_requested',record.get('model','unknown'))].append(edges)
    edges=Counter(edge for graph in edge_lists for edge in graph)
    n=len(edge_lists)
    majority=sorted([list(pair) for pair,count in edges.items() if count>n/2])
    import networkx as nx
    majority_dag=nx.is_directed_acyclic_graph(nx.DiGraph(majority))
    overlap=[len(a&b)/len(a|b) if a|b else 1. for a,b in combinations(edge_lists,2)]
    return {
       'n_graphs':n,
       'n_models':len(by_model),
       'distinct_models':sorted(by_model),
       'edge_frequencies_not_probabilities':{f'{a}->{b}':count/n for (a,b),count in sorted(edges.items())},
       'naive_majority_arrows_without_identification_validation':majority,
       'naive_majority_is_acyclic':bool(majority_dag),
       'mean_pairwise_jaccard':float(mean(overlap)) if overlap else None,
       'n_abstained':sum(bool(r['graph'].get('abstain')) for r in records),
       'caveat':'Agreement measures shared responses to prompts, not causal graph correctness. '
                'Prompt/model variants, provider routing and data contamination can induce correlated errors.'}
