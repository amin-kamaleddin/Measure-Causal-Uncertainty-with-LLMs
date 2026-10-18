"""Public-only, versioned credibility commitments; no truth/scoring imports."""
from typing import Literal
import re
from pydantic import Field, model_validator
from sharecausal.preflight.contracts import Strict, PublicWorldCardV2, public_payload, sha, canonical
from sharecausal.preflight.graph_utils import is_acyclic, has_path

RUBRIC_VERSION = 'admissibility-2.3.1'
POLICIES = ('supported_only', 'credible_alternatives', 'permissive_sensitivity')


class CredibilityRule(Strict):
    evidence_id: str
    basis_evidence_id: str
    mechanism: Literal['common_cause', 'direct_channel', 'mediated_channel', 'recording_process']
    rationale: str = Field(min_length=30, max_length=600)
    status: Literal['candidate_stipulation_requires_PI_review']


class PublicTask(Strict):
    schema_version: Literal['public-task-2.3']
    task_id: str = Field(pattern=r'^task-[a-f0-9]{64}$')
    card: PublicWorldCardV2
    rubric_version: Literal['admissibility-2.3.1']
    credibility: list[CredibilityRule]

    @model_validator(mode='after')
    def binding(self):
        evidence = {e.evidence_id: e for e in self.card.evidence}
        required={(e.source,e.target) for e in evidence.values() if e.kind=='edge' and e.relation=='supported'}
        contradicted={(e.source,e.target) for e in evidence.values() if e.kind=='edge' and e.relation=='contradicted'}
        timing={v.name:v.time for v in self.card.variables}
        if (required & contradicted or not is_acyclic(required)
                or any(timing[a]>timing[b] for a,b in required)
                or (self.card.assignment=='randomized_documented' and any(b=='A' for a,b in required))
                or (self.card.structural_exclusion and has_path(required))):
            raise ValueError('inconsistent_supported_public_facts')
        for condition, kind in ((self.card.assignment == 'randomized_documented', 'assignment'),
                                (self.card.structural_exclusion, 'exclusion')):
            if condition and not any(e.kind == kind and e.relation == 'supported'
                    and (e.source, e.target) == ('A', 'Y') for e in self.card.evidence):
                raise ValueError('documented_public_fact_requires_bound_evidence')
        if len({r.evidence_id for r in self.credibility}) != len(self.credibility):
            raise ValueError('duplicate_credibility_rule')
        for r in self.credibility:
            e, basis = evidence.get(r.evidence_id), evidence.get(r.basis_evidence_id)
            if (e is None or e.kind != 'edge' or e.relation != 'ambiguous' or basis is None
                    or basis.relation not in ('supported', 'ambiguous') or basis.evidence_id == e.evidence_id):
                raise ValueError('invalid_public_credibility_basis')
        if self.task_id != 'task-' + sha(task_content(self)):
            raise ValueError('public_task_content_hash_mismatch')
        return self


def task_content(task):
    return {'schema_version': task.schema_version, 'card': public_payload(task.card),
            'rubric_version': task.rubric_version,
            'credibility': [{k: getattr(r, k) for k in (
                'evidence_id', 'basis_evidence_id', 'mechanism', 'rationale', 'status')}
                for r in task.credibility]}


def public_task(task):
    if type(task) is not PublicTask:
        raise ValueError('public_task_type_required')
    data = {'task_id': task.task_id, **task_content(task)}
    PublicTask.model_validate(data)  # Frozen models can still contain mutable lists.
    raw = canonical(data).lower()
    if any(x in raw for x in (b'population_effect', b'potential_outcomes', b'generator_seed',
                             b'world_id', b'openrouter', b'mergeid')) or re.search(rb'\bsk-[a-z0-9]{8,}', raw):
        raise ValueError('private_marker_in_public_task')
    return data


def task_hash(task):
    return sha(public_task(task))


def prompt(task, variant='base'):
    if variant not in ('base', 'challenge'):
        raise ValueError('invalid_prompt_variant')
    return {'public_task': public_task(task), 'variant': variant,
            'instruction': 'Return cited complete graph hypotheses or abstain. Graph exclusion is conditional, not a proved zero.',
            'answer_schema': 'graph-proposal-2.3'}


def exports(tasks):
    hashes = [task_hash(t) for t in tasks]
    cards = [sha(public_payload(t.card)) for t in tasks]
    if not tasks or len(set(hashes)) != len(hashes) or len(set(cards)) != len(cards):
        raise ValueError('duplicate_or_missing_public_card')
    return dict(sorted((f'packet_{task_hash(t)}.json', canonical(public_task(t))) for t in tasks))
