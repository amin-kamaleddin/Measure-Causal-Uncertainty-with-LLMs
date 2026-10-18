"""Pre-estimation, graph-specific backdoor audit, never an LLM truth vote."""
from itertools import combinations
import json
from pathlib import Path
import networkx as nx
from sharecausal.llm.schemas import GraphAnswer
from .io import digest, write_csv, write_json


def backdoor(edges, observed=('B', 'H', 'D8')):
    g = nx.DiGraph()
    g.add_nodes_from(['A', 'Y', *observed])
    g.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError('Cyclic candidate graph')
    eligible = sorted(set(observed) - nx.descendants(g, 'A') - {'A', 'Y'})
    cut = g.copy()
    cut.remove_edges_from(list(g.out_edges('A')))
    for size in range(len(eligible) + 1):
        sets = [list(s) for s in combinations(eligible, size)
                if nx.is_d_separator(cut, {'A'}, {'Y'}, set(s))]
        if sets:
            return sets
    return []


def scenarios():
    common = [('B','A'), ('B','Y'), ('H','A'), ('H','Y'), ('A','Y'), ('D8','Y')]
    result = []
    for timing in ['pre', 'post']:
        for latent in [False, True]:
            for selection in [False, True]:
                edges = common + ([('D8','A')] if timing == 'pre' else [('A','D8')])
                if latent:
                    edges += [('U','A'), ('U','Y')]
                if selection:
                    # R is response among survivors; V survival precedes R.
                    # Y here has meaning only in the V=1 domain. No Y->V arrow.
                    edges += [('A','V'), ('H','V'), ('V','R'), ('A','R'), ('Y','R')]
                sets = backdoor(edges)
                result.append(dict(
                    scenario=f'D8_{timing}_U{int(latent)}_selection{int(selection)}',
                    edges=edges, timing='HYPOTHESIS_ONLY_' + timing,
                    provenance='analyst_theoretical_sensitivity_not_provider_response',
                    measured_backdoor_sets_ignoring_truncation=sets,
                    identification=('NO_observed_backdoor' if not sets else
                                    'SELECTION_TRUNCATION_UNRESOLVED_backdoor_not_sufficient' if selection else
                                    'HYPOTHETICAL_complete_outcome_backdoor_only'),
                    causal_verdict='ABSTAIN',
                    limitations=('B and H are conceptual blocks; demographic proxies do not establish full H measurement. '
                                 'D8 timing unresolved. Selection/truncation requires additional assumptions; '
                                 'survivor conditioning is not an always-survivor effect. No general ID proof.')))
    return result


def audit_records(paths):
    if len(paths) != 4:
        raise ValueError('Exactly four saved model records required')
    rows, provenance, sources, pairs, settings = [], [], [], [], []
    prompt_hashes, schema_hashes, ids = set(), set(), set()
    for path in paths:
        item = json.loads(Path(path).read_text())
        mid = item['model_requested']
        if mid in ids:
            raise ValueError('Repeated model')
        ids.add(mid)
        answer = GraphAnswer.model_validate(item['graph'])
        edges = [(e.source, e.target) for e in answer.edges]
        if not edges and not answer.abstain:
            raise ValueError('Missing graph requires explicit source abstention')
        if answer.abstain and not answer.abstention_reason.strip():
            raise ValueError('Source abstention requires reason')
        sets = backdoor(edges) if edges else None
        if edges:
            pairs.append(set(edges))
        prompt_hashes.add(item['input_sha256'])
        schema_hashes.add(item['schema_sha256'])
        rows.append(dict(model=mid, edges=len(edges), edge_pairs=json.dumps(edges), source_abstained=answer.abstain,
                         observed_backdoor_sets=json.dumps(sets),
                         status='NO_GRAPH' if not edges else ('NO_OBSERVED_BACKDOOR' if not sets else 'CONDITIONAL_BACKDOOR'),
                         D8_to_A=('D8','A') in edges, selection=('Y','S') in edges or any(b=='S' for a,b in edges),
                         causal_decision='ABSTAIN'))
        for e in answer.edges:
            provenance.append(dict(model=mid, arrow=f'{e.source}->{e.target}',
                                   evidence=e.evidence_reference,
                                   admissibility='UNVERIFIED_not_causal_evidence',
                                   temporal_issue=e.source=='D8' and e.target=='A'))
        sources.append(dict(file=Path(path).name, sha256=digest(path)))
        settings.append({k:item.get(k) for k in ['model_requested','model','input_sha256','schema_sha256',
                        'seed_sent','temperature_sent','max_tokens_sent','reasoning_effort_sent','usage']})
    if len(prompt_hashes) != 1 or len(schema_hashes) != 1 or None in prompt_hashes | schema_hashes:
        raise ValueError('Saved graph prompt/schema mismatch')
    return dict(n_models=4, nonempty=len(pairs), unique_arrows=len(set().union(*pairs)),
                edge_records=len(provenance), explicitly_unverified=sum(
                    e['evidence'].strip().lower() == 'unverified hypothesis' for e in provenance),
                explicit_links_or_doi=sum('https://' in e['evidence'].lower() or 'doi.org/' in e['evidence'].lower()
                                         for e in provenance),
                models=rows, edge_provenance=provenance, source_hashes=sources, generation_settings=settings,
                frequency_is_validity=False, no_graph_is_absence_vote=False)


def write_audit(root, audit):
    write_json(root/'graph_audit.json', audit)
    write_csv(root/'graph_identification.csv', audit['models'])
    write_csv(root/'edge_provenance_PRIVATE.csv', audit['edge_provenance'])
    theory = scenarios()
    write_json(root/'theoretical_dags.json', theory)
    write_csv(root/'theoretical_identification.csv', [dict(
        scenario=s['scenario'], timing=s['timing'], sets=json.dumps(s['measured_backdoor_sets_ignoring_truncation']),
        identification=s['identification'], decision=s['causal_verdict'], limitations=s['limitations']) for s in theory])
    return theory
