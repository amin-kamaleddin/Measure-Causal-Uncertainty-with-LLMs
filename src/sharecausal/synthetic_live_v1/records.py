"""Typed public-context provider receipts; no HTTP or truth imports."""
from typing import Literal
from pydantic import Field
from sharecausal.preflight.contracts import Strict,sha
from sharecausal.preflight_v23.packets import task_hash
from .public import messages,schema,request_key

class Receipt(Strict):
    version: Literal['synthetic-provider-1']
    source_type: Literal['recorded_provider_response']
    endpoint: Literal['https://openrouter.ai/api/v1/chat/completions']
    variant: Literal['base']
    repetition: Literal[0]
    model_family: str
    catalog_snapshot_sha256: str
    design_commitment_sha256: str | None
    idempotency_key: str = Field(pattern=r'^[a-f0-9]{64}$')
    task_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    prompt_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    schema_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    model_requested: str
    model_returned: str | None
    provider: str | None
    provider_response_id: str | None
    request_timestamp: str
    response_timestamp: str
    settings: dict
    status: Literal['response','http_error','provider_error','transport_error','null_content',
                    'refusal','schema_failure','model_mismatch','cost_unknown']
    transport_outcome: Literal['response','http_error','provider_error','transport_error','null_content',
                              'refusal','schema_failure','model_mismatch']
    http_status: int | None
    reserved_usd: float = Field(ge=0)
    cost_usd: float | None = Field(ge=0)
    usage: dict
    latency_seconds: float = Field(ge=0)
    response_sha256: str
    proposals_wire: str | None
    raw_file: str
    finish_reason: str | None


def validated(task, model, record):
    r=Receipt.model_validate(record)
    chat=messages(task)
    if (r.model_requested!=model or r.task_sha256!=task_hash(task)
        or r.model_family!=model.split('/')[0]
        or r.prompt_sha256!=sha(chat) or r.schema_sha256!=sha(schema())
        or r.idempotency_key!=request_key(task_hash(task),model,chat,r.settings)):
        raise ValueError('provider_receipt_binding_failed')
    if r.status=='response' and (r.model_returned!=model or not r.provider_response_id
                               or not r.provider or r.cost_usd is None):
        raise ValueError('provider_identity_or_cost_incomplete')
    return r

