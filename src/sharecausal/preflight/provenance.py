"""Future provider-record contract exercised with a LOCAL mock only; no HTTP."""
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator
from .contracts import Strict, packet_hash, prompt_payload, sha
from .comparison import GraphAnswer, graph_audit, no_llm_baseline


class Settings(Strict):
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(gt=0)
    sampling_seed: int | None


class ProviderResponseProvenance(Strict):
    source_type: Literal['provider_response']
    record_origin: Literal['mock_transport', 'recorded_provider']
    requested_model_id: str = Field(pattern=r'^[a-z0-9-]+/[a-z0-9.-]+$')
    returned_model_id: str = Field(pattern=r'^[a-z0-9-]+/[a-z0-9.-]+$')
    route: str = Field(pattern=r'^[a-z0-9-]{1,80}$')
    model_version: str = Field(min_length=1, max_length=80)
    request_timestamp: str
    response_timestamp: str
    request_receipt_id: str = Field(pattern=r'^receipt-[a-f0-9]{64}$')
    prompt_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    schema_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    card_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    evidence_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    response_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    requested_settings: Settings
    effective_settings: Settings
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reserved_input_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(gt=0)
    accounted_cost_usd: float = Field(ge=0)
    retry_count: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    failure: Literal['none', 'schema', 'timeout', 'rate_limit']
    provider_fallback_allowed: Literal[False]

    @model_validator(mode='after')
    def consistent_receipt(self):
        start, end = datetime.fromisoformat(self.request_timestamp), datetime.fromisoformat(self.response_timestamp)
        if start.utcoffset() is None or end.utcoffset() is None or end < start:
            raise ValueError('invalid_receipt_timestamps')
        if (self.input_tokens > self.reserved_input_tokens or self.output_tokens > self.reserved_output_tokens
                or self.requested_settings.max_output_tokens > self.reserved_output_tokens
                or self.retry_count > self.max_retries):
            raise ValueError('usage_or_retry_exceeds_reservation')
        if self.requested_model_id != self.returned_model_id:
            raise ValueError('unapproved_model_fallback')
        if self.requested_settings != self.effective_settings:
            raise ValueError('effective_settings_mismatch')
        return self


class OfflineFixtureProvenance(Strict):
    source_type: Literal['offline_fixture']
    fixture_id: str = Field(pattern=r'^dummy_[a-z0-9_]+$')
    prompt_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    response_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')


def validate_record(card, answer, provenance, policy, variant='base'):
    """Source-neutral graph logic; strict offline consumer rejects real receipts."""
    try:
        source = provenance.get('source_type')
        if source == 'provider_response':
            record = ProviderResponseProvenance.model_validate(provenance)
            if record.record_origin != 'mock_transport' or record.failure != 'none':
                raise ValueError('recorded_provider_not_enabled_in_offline_preflight')
            if (record.card_sha256 != packet_hash(card)
                    or record.schema_sha256 != sha(GraphAnswer.model_json_schema())
                    or record.evidence_sha256 != sha(prompt_payload(card)['public_card']['evidence'])):
                raise ValueError('provenance_payload_binding_failed')
        else:
            record = OfflineFixtureProvenance.model_validate(provenance)
        if record.prompt_sha256 != sha(prompt_payload(card, variant)) or record.response_sha256 != sha(answer):
            raise ValueError('response_or_prompt_digest_mismatch')
    except (ValueError, TypeError, AttributeError):
        return {'format_valid': False, 'admitted': False, 'reason': 'provenance_invalid', 'support': []}, None
    return graph_audit(card, answer, policy)


class MockTransport:
    """No URL, key, environment read, client, retry sleep, or send operation exists."""
    def __init__(self):
        self.calls = 0

    def record(self, card, model_id, variant, repetition, price, budget, input_cap, output_cap, max_retries):
        if price.model_id != model_id:
            raise ValueError('price_model_binding_mismatch')
        prompt = prompt_payload(card, variant)
        reserve = budget.reserve(model_id, price.cost(input_cap, output_cap))
        # Fixed public-only fixture instrument: models vary only by predetermined slot.
        baseline = no_llm_baseline(card)
        graph = baseline[0] if model_id.endswith('a/fixture') else (
            GraphAnswer(schema_version='preflight-graph-2.2', edges=[], adjustment=[],
                        abstain=True, abstention_reason='Offline dummy declines')
            if model_id.endswith('c/fixture') else baseline[-1])
        answer = graph.model_dump()
        # Deliberately conservative byte-count token reservation proxy, not a tokenizer.
        input_used = len(str(prompt).encode())
        output_used = len(str(answer).encode())
        cost = price.cost(input_used, output_used)
        budget.reconcile(reserve, cost)
        self.calls += 1
        settings = Settings(temperature=0.0, max_output_tokens=output_cap, sampling_seed=repetition)
        record = ProviderResponseProvenance(
            source_type='provider_response', record_origin='mock_transport',
            requested_model_id=model_id, returned_model_id=model_id, route='mock-local',
            model_version='fictional-fixture-1', request_timestamp='2026-10-17T00:00:00+00:00',
            response_timestamp='2026-10-17T00:00:00+00:00',
            request_receipt_id='receipt-'+sha([model_id, sha(prompt), repetition]),
            prompt_sha256=sha(prompt), schema_sha256=sha(GraphAnswer.model_json_schema()),
            card_sha256=packet_hash(card), evidence_sha256=sha(prompt['public_card']['evidence']),
            response_sha256=sha(answer), requested_settings=settings, effective_settings=settings,
            input_tokens=input_used, output_tokens=output_used,
            reserved_input_tokens=input_cap, reserved_output_tokens=output_cap,
            accounted_cost_usd=float(cost), retry_count=0, max_retries=max_retries,
            failure='none', provider_fallback_allowed=False)
        return answer, record
