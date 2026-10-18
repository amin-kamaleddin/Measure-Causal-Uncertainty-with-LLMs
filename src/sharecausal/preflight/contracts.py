"""Public contracts and one-way serialization; deliberately no truth imports."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


Support = Literal['supported', 'contradicted', 'ambiguous', 'unrelated', 'invalid_citation']


class Variable(Strict):
    name: Literal['A', 'Y', 'U', 'X', 'M', 'C']
    role: Literal['treatment', 'outcome', 'baseline', 'mediator', 'post_outcome', 'latent_hypothesis']
    time: int
    observed: bool


class Evidence(Strict):
    evidence_id: str = Field(pattern=r'^ev-[a-f0-9]{64}$')
    kind: Literal['edge', 'assignment', 'exclusion', 'general']
    source: str
    target: str
    relation: Support
    statement: str = Field(min_length=1, max_length=600)
    reliability: Literal['stipulated_synthetic']
    version: Literal['fictional-evidence-1']

    @model_validator(mode='after')
    def immutable_id(self):
        if self.evidence_id != 'ev-' + sha(evidence_content(self)):
            raise ValueError('evidence_content_hash_mismatch')
        return self


def evidence_content(e) -> dict:
    return {k: getattr(e, k) for k in ('kind', 'source', 'target', 'relation', 'statement', 'reliability', 'version')}


def make_evidence(**kwargs) -> dict:
    content = dict(reliability='stipulated_synthetic', version='fictional-evidence-1', **kwargs)
    return {'evidence_id': 'ev-' + sha(content), **content}


class Estimand(Strict):
    treatment: Literal['A']
    outcome: Literal['Y']
    contrast: Literal['E[Y(do(A=1))-Y(do(A=0))]']
    intervention: str = Field(min_length=1, max_length=500)
    population: str = Field(min_length=1, max_length=500)
    observation_target: Literal['Y_defined_for_all_eligible_units']
    consistency: Literal['stipulated_no_interference_and_single_treatment_version']
    positivity: Literal['both_interventions_possible_in_target']


class PublicWorldCardV2(Strict):
    schema_version: Literal['public-world-2.2']
    card_id: str = Field(pattern=r'^card-[a-f0-9]{64}$')
    title: str = Field(min_length=1, max_length=200)
    variables: list[Variable] = Field(min_length=2, max_length=6)
    estimand: Estimand
    assignment_description: str = Field(min_length=1, max_length=600)
    assignment: Literal['randomized_documented', 'assignment_unknown']
    selection_description: str = Field(min_length=1, max_length=600)
    selection: Literal['complete_outcomes', 'selection_unknown']
    structural_exclusion: bool
    evidence: list[Evidence] = Field(min_length=1, max_length=30)

    @model_validator(mode='after')
    def valid_packet(self):
        variables = {v.name: v for v in self.variables}
        if len(variables) != len(self.variables) or not {'A', 'Y'} <= variables.keys():
            raise ValueError('invalid_variable_roles')
        if (variables['A'].role != 'treatment' or variables['Y'].role != 'outcome'
                or not variables['A'].observed or not variables['Y'].observed
                or variables['A'].time >= variables['Y'].time):
            raise ValueError('invalid_variable_timing')
        for v in self.variables:
            if v.name == 'U' and (v.observed or v.role != 'latent_hypothesis'):
                raise ValueError('U_must_remain_a_latent_hypothesis')
            if v.role == 'latent_hypothesis' and (v.observed or v.time > variables['A'].time):
                raise ValueError('latent_variable_cannot_be_observed')
            if v.role == 'baseline' and v.time >= variables['A'].time:
                raise ValueError('baseline_must_precede_treatment')
            if v.role == 'mediator' and not variables['A'].time < v.time < variables['Y'].time:
                raise ValueError('invalid_mediator_timing')
            if v.role == 'post_outcome' and v.time <= variables['Y'].time:
                raise ValueError('invalid_post_outcome_timing')
        if len({e.evidence_id for e in self.evidence}) != len(self.evidence):
            raise ValueError('duplicate_evidence_id')
        for e in self.evidence:
            if e.kind == 'edge' and not {e.source, e.target} <= variables.keys():
                raise ValueError('unknown_evidence_variable')
        if self.card_id != 'card-' + sha(card_content(self)):
            raise ValueError('public_content_hash_mismatch')
        return self


def card_content(card) -> dict:
    # Explicit allowlist at EVERY level; a truth object or arbitrary dict is not accepted.
    return {
        'schema_version': card.schema_version, 'title': card.title,
        'variables': [{k: getattr(v, k) for k in ('name', 'role', 'time', 'observed')} for v in card.variables],
        'estimand': {k: getattr(card.estimand, k) for k in (
            'treatment', 'outcome', 'contrast', 'intervention', 'population',
            'observation_target', 'consistency', 'positivity')},
        'assignment_description': card.assignment_description, 'assignment': card.assignment,
        'selection_description': card.selection_description, 'selection': card.selection,
        'structural_exclusion': card.structural_exclusion,
        'evidence': [{'evidence_id': e.evidence_id, **evidence_content(e)} for e in card.evidence],
    }


def public_payload(card: PublicWorldCardV2) -> dict:
    if type(card) is not PublicWorldCardV2:
        raise ValueError('public_card_only')
    payload = {'card_id': card.card_id, **card_content(card)}
    # Revalidate nested mutable contents and their immutable IDs before every export.
    PublicWorldCardV2.model_validate(payload)
    raw = canonical(payload)
    if re.search(rb'true_ate|true_edges|potential_outcomes|mechanism_family|generator_seed|OPENROUTER|mergeid|sk-', raw, re.I):
        raise ValueError('private_marker_in_public_text')
    return payload


def public_bytes(card) -> bytes:
    return canonical(public_payload(card))


def packet_hash(card) -> str:
    return hashlib.sha256(public_bytes(card)).hexdigest()


def prompt_payload(card, variant='base') -> dict:
    if variant not in ('base', 'challenge'):
        raise ValueError('unknown_prompt_variant')
    return {'public_card': public_payload(card), 'answer_schema': 'preflight-graph-2.2',
            'instruction': 'Propose complete causal hypotheses using cited fictional evidence; abstain on ambiguity.',
            'variant': variant}


def public_exports(cards) -> dict[str, bytes]:
    # Opaque content-addressed filenames and sorting eliminate truth/order/seed metadata.
    exports = {f'packet_{packet_hash(c)}.json': public_bytes(c) for c in cards}
    return dict(sorted(exports.items()))
