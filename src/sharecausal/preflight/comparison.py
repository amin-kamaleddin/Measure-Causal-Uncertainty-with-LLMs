"""Public-only baseline, support rubric and fair candidate vote rules."""
from collections import Counter
import math
from typing import Literal

from pydantic import Field, model_validator
from .contracts import Strict, PublicWorldCardV2, prompt_payload, sha
from .graph_utils import is_acyclic, has_path


class Edge(Strict):
    source: str = Field(pattern=r'^[A-Z]$')
    target: str = Field(pattern=r'^[A-Z]$')
    evidence_id: str


class GraphAnswer(Strict):
    schema_version: Literal['preflight-graph-2.2']
    edges: list[Edge] = Field(max_length=40)
    adjustment: list[str]
    abstain: bool
    abstention_reason: str

    @model_validator(mode='after')
    def graph_contract(self):
        pairs = [(e.source, e.target) for e in self.edges]
        if len(pairs) != len(set(pairs)) or not is_acyclic(pairs):
            raise ValueError('duplicate_or_cyclic_graph')
        if self.abstain and (self.edges or self.adjustment or not self.abstention_reason.strip()):
            raise ValueError('malformed_abstention')
        if not self.abstain and self.abstention_reason:
            raise ValueError('malformed_nonabstention')
        if len(set(self.adjustment)) != len(self.adjustment):
            raise ValueError('duplicate_adjustment')
        return self


def edge_support(card: PublicWorldCardV2, edge: Edge) -> str:
    evidence = {e.evidence_id: e for e in card.evidence}.get(edge.evidence_id)
    if evidence is None:
        return 'invalid_citation'
    if evidence.kind != 'edge' or (evidence.source, evidence.target) != (edge.source, edge.target):
        return 'unrelated'
    return evidence.relation


def graph_audit(card, answer, policy):
    if policy not in ('supported_only', 'retain_alternatives'):
        raise ValueError('unknown_support_policy')
    try:
        graph = GraphAnswer.model_validate(answer)
        nodes = {v.name: v for v in card.variables}
        for e in graph.edges:
            if e.source not in nodes or e.target not in nodes or nodes[e.source].time > nodes[e.target].time:
                raise ValueError('unknown_or_temporally_invalid_edge')
        if any(x not in nodes or not nodes[x].observed or nodes[x].role != 'baseline' for x in graph.adjustment):
            raise ValueError('ineligible_adjustment')
    except (ValueError, TypeError):
        return {'format_valid': False, 'admitted': False, 'reason': 'graph_contract_invalid', 'support': []}, None
    statuses = [edge_support(card, e) for e in graph.edges]
    if graph.abstain:
        return {'format_valid': True, 'admitted': False, 'reason': 'source_abstained', 'support': statuses}, graph
    causal_path = has_path([(e.source, e.target) for e in graph.edges])
    if not causal_path and not card.structural_exclusion:
        statuses.append('ambiguous')  # Graph exclusion is not proved by absent edge citations.
    if causal_path and card.structural_exclusion:
        statuses.append('contradicted')
    if card.assignment == 'randomized_documented' and any(e.target == 'A' for e in graph.edges):
        statuses.append('contradicted')
    allowed = {'supported'} if policy == 'supported_only' else {'supported', 'ambiguous', 'unrelated'}
    admitted = all(s in allowed for s in statuses)
    return {'format_valid': True, 'admitted': admitted,
            'reason': 'candidate_policy_admission' if admitted else 'evidence_policy_veto',
            'support': statuses, 'support_policy': policy,
            'human_adjudication': 'not_performed'}, graph


def _answer(edges=(), *, abstain=False):
    return GraphAnswer(schema_version='preflight-graph-2.2', edges=list(edges), adjustment=[],
                       abstain=abstain, abstention_reason='Public information unresolved' if abstain else '')


def no_llm_baseline(card: PublicWorldCardV2) -> list[GraphAnswer]:
    # Same card as every mock source. No truth, effects, outcomes or certificate imports.
    if card.structural_exclusion:
        return [_answer()]
    supported = [Edge(source=e.source, target=e.target, evidence_id=e.evidence_id)
                 for e in card.evidence if e.kind == 'edge' and e.relation == 'supported']
    if card.assignment == 'randomized_documented':
        return [_answer(supported)]
    possible = [Edge(source=e.source, target=e.target, evidence_id=e.evidence_id)
                for e in card.evidence if e.kind == 'edge' and e.relation == 'ambiguous']
    # Deliberately retain alternatives, rather than pretend uncertainty is a true DAG.
    return [_answer(supported + [e for e in possible if e.source != 'U']),
            _answer(supported + [e for e in possible if e.source == 'U'])]


def majority(card, by_model: dict, planned_models: list[str], policy):
    """One first-variant/first-repetition ballot per model; all planned models count.

    Multiple graph answers from a model are an explicit abstention (no extra votes).
    Evidence screening precedes voting. No tie repair, direction flip or cycle repair.
    """
    if len(planned_models) != len(set(planned_models)) or not planned_models:
        raise ValueError('invalid_model_denominator')
    votes, valid, audit = Counter(), [], {}
    for model in planned_models:
        records = by_model.get(model, [])
        if len(records) != 1:
            audit[model] = 'absent_or_multiple_graphs'; continue
        status, graph = graph_audit(card, records[0], policy)
        audit[model] = status['reason']
        if not status['admitted']:
            continue
        valid.append(graph)
        votes.update((e.source, e.target) for e in graph.edges)
    n = len(planned_models)
    pairs = sorted(e for e, count in votes.items() if count > n / 2)
    if not pairs and sum(not g.edges for g in valid) <= n / 2:
        return {'decision': 'abstain', 'reason': 'no_strict_majority', 'graph': None,
                'denominator': n, 'source_audit': audit}
    if not is_acyclic(pairs):
        return {'decision': 'abstain', 'reason': 'cyclic_majority_not_repaired', 'graph': None,
                'denominator': n, 'source_audit': audit}
    return {'decision': 'abstain', 'reason': 'graph_only_no_outcome_estimation_in_preflight',
            'graph': pairs, 'structural_zero': not has_path(pairs),
            'denominator': n, 'source_audit': audit}


def public_comparisons(card, model_ids, mock_answers, policies):
    result = {'input_sha256': sha(prompt_payload(card)),
              'baseline_input_sha256': sha(prompt_payload(card)),
              'baseline_graphs': [g.model_dump() for g in no_llm_baseline(card)], 'policies': {}}
    for policy in policies:
        by_model = {m: mock_answers.get(m, []) for m in model_ids}
        result['policies'][policy] = {
            'individual_sources': {m: [graph_audit(card, a, policy)[0] for a in answers]
                                   for m, answers in by_model.items()},
            'majority': majority(card, by_model, model_ids, policy)}
    return result


def calibrate_public_margins(rows, target):
    if not 0 < target < 1 or any(set(r) != {'method', 'decision', 'margin'} for r in rows):
        raise ValueError('public_calibration_fields_only')
    if any(r['decision'] not in ('positive', 'negative', 'abstain')
           or not math.isfinite(r['margin']) or r['margin'] < 0 for r in rows):
        raise ValueError('invalid_public_margin')
    result = {}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == method]
        margins = sorted([r['margin'] for r in group if r['decision'] != 'abstain' and r['margin'] > 0], reverse=True)
        goal = max(1, round(target*len(group)))
        cutoff = ((margins[goal-1]+margins[goal])/2 if len(margins) > goal else 0.0)
        retained = sum(r['decision'] != 'abstain' and r['margin'] > cutoff for r in group)
        result[method] = {'threshold': cutoff, 'available_assertions': len(margins),
                          'achieved_assertion_rate': retained/len(group),
                          'target_feasible': len(margins) >= goal and abs(retained/len(group)-target) <= 1/len(group)}
    return result
