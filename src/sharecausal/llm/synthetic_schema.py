"""Strict v2 synthetic task contract; separate from the SHARE answer schema."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 'synthetic-graph-v2.0'
TIMES = {'X1': 0, 'X2': 0, 'A': 1, 'M': 2, 'Y': 3, 'C': 4}
MEASURED = frozenset((*TIMES, 'observed'))
Node = Literal['X1', 'X2', 'A', 'M', 'Y', 'C', 'observed', 'U']


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)


class PublicCard(StrictRecord):
    treatment: Literal['A']
    outcome: Literal['Y']
    description: str = Field(min_length=1)
    variables: dict[str, str]
    times: dict[str, int]

    @model_validator(mode='after')
    def fixed_task(self):
        if self.times != TIMES:
            raise ValueError('Incorrect synthetic temporal roles')
        if set(self.variables) != {'X1', 'X2', 'M', 'C', 'observed'}:
            raise ValueError('Incorrect synthetic variables')
        if any(not s.strip() for s in self.variables.values()):
            raise ValueError('Empty variable description')
        return self


PUBLIC_CARD = {
    'treatment': 'A', 'outcome': 'Y',
    'description': 'Hypothetical exposure A and continuous subsequent outcome Y',
    'variables': {'X1': 'pre-exposure factor', 'X2': 'pre-exposure factor',
                  'M': 'post-exposure measure', 'C': 'post-outcome measure',
                  'observed': 'indicator of whether Y was collected'},
    'times': TIMES.copy(),
}
PUBLIC_EVIDENCE = ('Independent synthetic task. Temporal roles are stipulated; '
                   'edge existence is an unverified hypothesis. U, if proposed, is latent. '
                   'Target is the population total effect of A on Y. Never adjust for M or C.')


def prompt_payload(card: dict, variant: str = 'base') -> dict:
    """No DataFrame, generator wrapper, statistics, or secrets are accepted."""
    validated = PublicCard.model_validate(card)
    if variant not in ('base', 'challenge'):
        raise ValueError('Unsupported prompt variant')
    return {'schema_version': SCHEMA_VERSION, 'card': validated.model_dump(),
            'evidence': PUBLIC_EVIDENCE, 'variant': variant,
            'instruction': ('Return SyntheticGraphAnswer JSON with explicit uncertainty and abstention.'
                            + (' Challenge hidden confounding and selection.' if variant == 'challenge' else ''))}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def prompt_hash(card: dict, variant: str = 'base') -> str:
    return digest(prompt_payload(card, variant))


class SyntheticArrow(StrictRecord):
    source: Node
    target: Node
    causal_role: Literal['hypothesis', 'direct', 'confounding', 'mediation', 'selection']
    temporal_rationale: str = Field(min_length=1, pattern=r'\S')
    evidence_reference: str = Field(min_length=1, pattern=r'\S')
    uncertainty: Literal['low', 'medium', 'high']


class SyntheticGraphAnswer(StrictRecord):
    schema_version: Literal['synthetic-graph-v2.0']
    treatment: Literal['A']
    outcome: Literal['Y']
    estimand: Literal['population_total_effect']
    observed_nodes: list[Node]
    edges: list[SyntheticArrow] = Field(max_length=40)
    suggested_adjustment: list[Node]
    rationale: str = Field(min_length=1, pattern=r'\S')
    contested_assumptions: list[str] = Field(min_length=1)
    abstain: bool
    abstention_reason: str

    @model_validator(mode='after')
    def valid_graph(self):
        if set(self.observed_nodes) != MEASURED or len(self.observed_nodes) != len(MEASURED):
            raise ValueError('Measured nodes must match card; U is never observed')
        if len(set(self.suggested_adjustment)) != len(self.suggested_adjustment):
            raise ValueError('Duplicate adjustment variable')
        if not set(self.suggested_adjustment) <= {'X1', 'X2'}:
            raise ValueError('Only pre-exposure measured adjustment is allowed')
        if any(not s.strip() for s in self.contested_assumptions):
            raise ValueError('Empty contested assumption')
        if self.abstain:
            if not self.abstention_reason.strip() or self.edges or self.suggested_adjustment:
                raise ValueError('Abstention needs reason and no proposed graph or adjustment')
        elif self.abstention_reason or not self.edges:
            raise ValueError('Nonabstention needs edges and an empty abstention reason')
        pairs = [(e.source, e.target) for e in self.edges]
        if len(set(pairs)) != len(pairs):
            raise ValueError('Duplicate directed edge')
        if not nx.is_directed_acyclic_graph(nx.DiGraph(pairs)):
            raise ValueError('Cyclic graph')
        for e in self.edges:
            if (e.target == 'U' or e.source == 'observed' or
                    (e.source in TIMES and e.target in TIMES and TIMES[e.source] > TIMES[e.target])):
                raise ValueError('Invalid temporal arrow')
            if e.causal_role == 'confounding' and e.source not in ('X1', 'X2', 'U'):
                raise ValueError('Confounding role must originate before exposure')
            if e.causal_role == 'mediation' and 'M' not in (e.source, e.target):
                raise ValueError('Mediation role must involve M')
            if e.causal_role == 'selection' and e.target != 'observed':
                raise ValueError('Selection role must target observation indicator')
        return self
