"""Public design metadata: engineering lint, never independent authorship proof."""
from typing import Literal
from pydantic import Field
from sharecausal.preflight.contracts import Strict,sha


class FamilyDefinition(Strict):
    family_id:str
    cohort:Literal['development_exposed','candidate_heldout_definition_only']
    mechanism_features:list[str]=Field(min_length=2)
    narrative_domain:str
    custody:Literal['single_team_exposed_not_independent','DECISION_REQUIRED']
    rationale:str=Field(min_length=40)


class CardAssignment(Strict):
    task_hash:str=Field(pattern=r'^[a-f0-9]{64}$')
    family_id:str
    narrative_style:str


def signature(task):
    return sha({'variables':[(v.name,v.role,v.time,v.observed) for v in task.card.variables],
        'assignment':task.card.assignment,'selection':task.card.selection,'exclusion':task.card.structural_exclusion,
        'potential_edges':sorted((e.source,e.target,e.relation) for e in task.card.evidence if e.kind=='edge')})


def audit_families(tasks,definitions,assignments):
    if len({d.family_id for d in definitions})!=len(definitions):raise ValueError('duplicate_family_ID')
    defs={d.family_id:d for d in definitions};mapped={a.task_hash:a for a in assignments}
    if len(mapped)!=len(assignments) or set(mapped)!=set(tasks):raise ValueError('duplicate_or_unassigned_card')
    signatures={}
    for h,task in tasks.items():
        a=mapped[h]
        if a.family_id not in defs or defs[a.family_id].cohort!='development_exposed':
            raise ValueError('no_heldout_packets_may_be_scored_in_this_gate')
        s=signature(task)
        if s in signatures and signatures[s]!=a.family_id:raise ValueError('renamed_mechanism_family')
        signatures[s]=a.family_id
    fingerprints={}
    for d in definitions:
        fp=sha(sorted(set(d.mechanism_features)))
        if fp in fingerprints:raise ValueError('renamed_family_definition')
        fingerprints[fp]=d.family_id
    held=[d for d in definitions if d.cohort=='candidate_heldout_definition_only']
    if not held:raise ValueError('candidate_heldout_definitions_required')
    return {'engineering_lint_passed':True,'development_cards':len(tasks),
            'development_families':len(set(a.family_id for a in assignments)),
            'candidate_heldout_family_definitions':len(held),'heldout_cards_scored':0,
            'independent_family_validation':'DECISION_REQUIRED',
            'truth_custody':'DECISION_REQUIRED_single_team_local_seal_only',
            'historical_cases':'v1/v2.1/v2.2 already inspected exploratory; not unseen holdouts',
            'limitations':'Feature/name disjointness cannot establish semantic novelty or independent authorship.'}
