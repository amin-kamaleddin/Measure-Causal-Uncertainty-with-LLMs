"""Recorded-response adapter. No transport or generator dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from sharecausal.causal.graphs import GraphSpec, validate_temporal_graph
from .synthetic_schema import (MEASURED, PublicCard, StrictRecord,
                               SyntheticGraphAnswer, prompt_hash)


class FixtureProvenance(StrictRecord):
    source_type: Literal['offline_fixture']
    fixture_id: str = Field(pattern=r'^dummy_[a-z0-9_]{1,80}$')
    requested_model_family: str = Field(pattern=r'^dummy-family-[a-z0-9-]{1,40}$')
    prompt_variant: Literal['base', 'challenge']
    prompt_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    repetition_index: int = Field(ge=0)
    timestamp: str
    schema_version: Literal['synthetic-graph-v2.0']

    @field_validator('timestamp')
    @classmethod
    def timestamp_with_timezone(cls, value):
        if datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError('Timestamp needs timezone')
        return value


@dataclass(frozen=True)
class AdaptedProposal:
    """Only adapt_response should construct this internal graph/audit pair."""
    graph: GraphSpec | None
    audit: dict


_ERROR_TYPES = frozenset({
    'extra_forbidden', 'missing', 'value_error', 'json_invalid', 'model_type',
    'model_attributes_type', 'dict_type', 'list_type', 'literal_error',
    'string_type', 'string_pattern_mismatch', 'string_too_short', 'string_too_long',
    'bool_type', 'int_type', 'greater_than_equal', 'too_long', 'too_short',
})
_SAFE_REASONS = frozenset({
    'Invalid temporal arrow', 'Cyclic graph', 'Duplicate directed edge',
    'Only pre-exposure measured adjustment is allowed',
    'Measured nodes must match card; U is never observed',
})


def _validation_reason(exc: ValidationError, scope: str) -> str:
    """Coarse fixed locations only: Pydantic loc can contain hostile field names."""
    codes = []
    for error in exc.errors(include_input=False, include_url=False)[:8]:
        kind = error['type'] if error['type'] in _ERROR_TYPES else 'schema_error'
        detail = str(error.get('ctx', {}).get('error'))
        suffix = ': ' + detail if detail in _SAFE_REASONS else ''
        codes.append(f'{scope}:{kind}{suffix}')
    return 'validation_failed: ' + '; '.join(codes)


def _new_audit(provenance: object, card: dict) -> dict:
    # Invalid public cards are configuration errors, not source-response errors.
    PublicCard.model_validate(card)
    audit = {'status': 'rejected', 'reason': None, 'provenance': None,
             'response_sha256': None}
    try:
        p = FixtureProvenance.model_validate(provenance)
    except ValidationError as exc:
        audit['reason'] = _validation_reason(exc, 'provenance')
    else:
        audit['provenance'] = p.model_dump()
        if p.prompt_sha256 != prompt_hash(card, p.prompt_variant):
            audit['reason'] = 'prompt_hash_mismatch'
    return audit


def adapt_response(raw_response: str, card: dict, provenance: dict | None) -> AdaptedProposal:
    """Reject hostile input per source; never log raw errors, keys or values."""
    audit = _new_audit(provenance, card)
    if isinstance(raw_response, str):
        try:
            audit['response_sha256'] = hashlib.sha256(raw_response.encode()).hexdigest()
        except UnicodeError:
            audit['reason'] = audit['reason'] or 'response_encoding_failed'
    else:
        audit['reason'] = audit['reason'] or 'response_must_be_json_text'
    if audit['reason']:
        return AdaptedProposal(None, audit)
    try:
        answer = SyntheticGraphAnswer.model_validate_json(raw_response)
    except ValidationError as exc:
        audit['reason'] = _validation_reason(exc, 'response')
        return AdaptedProposal(None, audit)
    audit.update(rationale=answer.rationale, contested_assumptions=answer.contested_assumptions,
                 explicit_abstention=answer.abstain,
                 source_abstention_reason=answer.abstention_reason)
    if answer.abstain:
        audit.update(status='source_abstained', reason='explicit_source_abstention')
        return AdaptedProposal(None, audit)
    graph = GraphSpec(audit['provenance']['fixture_id'],
                      tuple((e.source, e.target) for e in answer.edges),
                      MEASURED, answer.rationale,
                      selection_warning=any(e.target == 'observed' for e in answer.edges))
    validate_temporal_graph(graph, dict(card['times']))
    audit.update(status='accepted', reason='schema_and_temporality_valid_not_truth_verified',
                 admission_basis='format_and_temporal_constraints_only',
                 evidence_support_status='not_adjudicated_offline_hypothesis',
                 edges=[e.model_dump() for e in answer.edges],
                 suggested_adjustment=answer.suggested_adjustment,
                 contested_assumptions=answer.contested_assumptions,
                 estimand=answer.estimand)
    return AdaptedProposal(graph, audit)


def adapt_record(record: object, card: dict) -> AdaptedProposal:
    if not isinstance(record, dict) or set(record) != {'provenance', 'response'}:
        audit = _new_audit(record.get('provenance') if isinstance(record, dict) else None, card)
        audit['reason'] = 'invalid_record_envelope'
        return AdaptedProposal(None, audit)
    raw = record['response']
    if isinstance(raw, dict):
        try:
            raw = json.dumps(raw, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
            # Covers NaN/Infinity, unsupported objects/keys, cycles and deep input.
            # No repr(raw), exception text, or attacker-controlled key is exported.
            audit = _new_audit(record['provenance'], card)
            audit['reason'] = audit['reason'] or 'response_serialization_failed'
            return AdaptedProposal(None, audit)
    return adapt_response(raw, card, record['provenance'])
