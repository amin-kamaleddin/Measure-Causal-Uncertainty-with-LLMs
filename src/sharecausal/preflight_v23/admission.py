"""Truth-blind candidate admission. Plausibility is a public stipulation, not confidence."""
from collections import Counter
from typing import Literal
from pydantic import model_validator
from sharecausal.preflight.contracts import Strict
from sharecausal.preflight.comparison import GraphAnswer, Edge
from sharecausal.preflight.graph_utils import has_path, is_acyclic
from .packets import POLICIES, RUBRIC_VERSION, prompt
from sharecausal.preflight.contracts import sha


class Proposal(Strict):
    schema_version: Literal['graph-proposal-2.3']
    graph: GraphAnswer
    structural_zero_claim: bool

    @model_validator(mode='after')
    def coherent(self):
        if self.structural_zero_claim and (self.graph.abstain or has_path([(e.source,e.target) for e in self.graph.edges])):
            raise ValueError('incoherent_zero_claim')
        return self


def proposal(edges=(), *, abstain=False, zero=False):
    return Proposal(schema_version='graph-proposal-2.3', structural_zero_claim=zero,
        graph=GraphAnswer(schema_version='preflight-graph-2.2', edges=list(edges), adjustment=[],
            abstain=abstain, abstention_reason='Public evidence unresolved' if abstain else ''))


def support(task, edge):
    e = next((e for e in task.card.evidence if e.evidence_id == edge.evidence_id), None)
    if e is None: return 'invalid_citation'
    if e.kind != 'edge' or (e.source,e.target) != (edge.source,edge.target): return 'unrelated'
    if e.relation == 'ambiguous':
        return 'credible_ambiguous' if any(r.evidence_id==e.evidence_id for r in task.credibility) else 'unrelated'
    return e.relation


def supported_exclusion(task):
    return task.card.structural_exclusion and any(e.kind=='exclusion' and e.relation=='supported'
        and (e.source,e.target)==('A','Y') for e in task.card.evidence)


def audit(task, raw, policy='credible_alternatives'):
    if policy not in POLICIES: raise ValueError('unknown_admissibility_policy')
    base = {'rubric_version': RUBRIC_VERSION, 'policy': policy,
            'credibility_authority': 'candidate_public_stipulation_PI_approval_pending',
            'human_adjudication': 'not_performed'}
    try:
        p=Proposal.model_validate(raw)
        nodes={v.name:v for v in task.card.variables}
        for e in p.graph.edges:
            if e.source not in nodes or e.target not in nodes or nodes[e.source].time > nodes[e.target].time:
                raise ValueError('temporal_or_node_invalid')
        if any(x not in nodes or not nodes[x].observed or nodes[x].role!='baseline' for x in p.graph.adjustment):
            raise ValueError('invalid_adjustment')
    except (ValueError,TypeError):
        return {**base,'format_valid':False,'admitted':False,'reason':'format_or_temporal_invalid','edge_tiers':[]},None
    tiers=[support(task,e) for e in p.graph.edges]
    if p.graph.abstain:
        return {**base,'format_valid':True,'admitted':False,'reason':'source_abstained','edge_tiers':tiers},p
    pairs=[(e.source,e.target) for e in p.graph.edges]
    required={(e.source,e.target) for e in task.card.evidence if e.kind=='edge' and e.relation=='supported'}
    omitted=sorted(required-set(pairs))
    if omitted:tiers.append('contradicted')
    conditional_zero=not has_path(pairs)
    if task.card.assignment=='randomized_documented' and any(b=='A' for a,b in pairs):
        tiers.append('contradicted')
    if supported_exclusion(task) and not conditional_zero: tiers.append('contradicted')
    # Hard vetoes apply even to deliberately permissive sensitivity hypotheses.
    if p.structural_zero_claim and not supported_exclusion(task): tiers.append('unsupported_exclusion_claim')
    if not pairs and not supported_exclusion(task): tiers.append('unrelated')
    allowed={'supported'}
    if policy!='supported_only': allowed.add('credible_ambiguous')
    if policy=='permissive_sensitivity': allowed.add('unrelated')
    admitted=all(s in allowed for s in tiers)
    return {**base,'format_valid':True,'admitted':admitted,
            'reason':'candidate_admission' if admitted else 'evidence_veto','edge_tiers':tiers,
            'conditional_graph_zero':conditional_zero,
            'omitted_supported_edges':[list(e) for e in omitted],
            'supported_exclusion':bool(supported_exclusion(task)),
            'causal_zero_claim_justified':bool(p.structural_zero_claim and supported_exclusion(task))},p


def no_llm(task):
    """Same public card, evidence and credibility commitments as mock instruments."""
    if supported_exclusion(task):
        return [proposal([Edge(source=e.source,target=e.target,evidence_id=e.evidence_id)
                for e in task.card.evidence if e.kind=='edge' and e.relation=='supported'])]
    sure=[Edge(source=e.source,target=e.target,evidence_id=e.evidence_id)
          for e in task.card.evidence if e.kind=='edge' and e.relation=='supported']
    maybe=[Edge(source=e.source,target=e.target,evidence_id=e.evidence_id)
           for e in task.card.evidence if e.kind=='edge' and e.relation=='ambiguous'
           and any(r.evidence_id==e.evidence_id for r in task.credibility)]
    if not maybe: return [proposal(sure)]
    # Prespecified set-valued alternatives; a set never gets multiple ballots.
    return [proposal(sure+[e for e in maybe if e.source!='U']),
            proposal(sure+[e for e in maybe if e.source=='U'])]


def majority(task, sources, models, policy):
    if not models or len(set(models))!=len(models): raise ValueError('invalid_model_denominator')
    votes=Counter(); accepted=[]; source_audit={}
    for model in models:
        rows=sources.get(model,[])
        if len(rows)!=1:
            source_audit[model]='absent_or_multiple_graphs';continue
        a,p=audit(task,rows[0],policy);source_audit[model]=a['reason']
        if a['admitted']:
            accepted.append(p);votes.update((e.source,e.target) for e in p.graph.edges)
    n=len(models);pairs=sorted(e for e,c in votes.items() if c>n/2)
    reason='graph_only_no_effect_estimation'
    if not pairs and sum(not p.graph.edges for p in accepted)<=n/2:reason='no_strict_majority'
    elif not is_acyclic(pairs):reason='cycle_no_repair'
    return {'decision':'abstain','reason':reason,'graph':pairs if reason=='graph_only_no_effect_estimation' else None,
            'denominator_all_planned_models':n,'source_audit':source_audit,
            'conditional_graph_zero':not has_path(pairs) if reason=='graph_only_no_effect_estimation' else None,
            'supported_exclusion':bool(supported_exclusion(task))}


def comparisons(task,sources,models):
    baseline=[p.model_dump() for p in no_llm(task)]
    return {'prompt_hash':sha(prompt(task)),'no_llm_prompt_hash':sha(prompt(task)),
            'baseline_graphs':baseline,'candidate_primary_policy':'credible_alternatives',
            'PI_policy_approval':'DECISION_REQUIRED',
            'policies':{policy:{'sources':{m:[audit(task,r,policy)[0] for r in sources.get(m,[])] for m in models},
                'baseline_audits':[audit(task,r,policy)[0] for r in baseline],
                'majority':majority(task,sources,models,policy)} for policy in POLICIES}}
