"""Public-only adapter; must not import truth/scoring/generator modules."""
import json
from pathlib import Path
from pydantic import Field
from typing import Literal
from sharecausal.preflight.contracts import Strict, sha
from sharecausal.preflight_v23.packets import PublicTask, prompt, task_hash, public_task
from sharecausal.preflight_v23.admission import Proposal, audit
from sharecausal.pilot_v24.capture import parse_wire


class SavedResponse(Strict):
    schema_version: Literal['provider-replay-1']
    source_type: Literal['recorded_provider_response', 'synthetic_test_record']
    model_requested: str = Field(min_length=1, max_length=200)
    model_returned: str = Field(min_length=1, max_length=200)
    provider_response_id: str = Field(min_length=1, max_length=200)
    task_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    prompt_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    answer_schema_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    variant: Literal['base','challenge']
    response_text: str = Field(max_length=200000)


def adapt(task, record):
    public_task(task)  # Recheck mutable nested values; no empirical statistics accepted.
    try:
        r=SavedResponse.model_validate(record)
    except (ValueError,TypeError):
        return dict(status='INELIGIBLE_CONTEXT_OR_ENVELOPE',proposals=[],audits=[])
    if (r.task_sha256!=task_hash(task) or r.prompt_sha256!=sha(prompt(task,r.variant))
            or r.answer_schema_sha256!=sha(Proposal.model_json_schema())):
        return dict(status='INELIGIBLE_CONTEXT_OR_ENVELOPE',proposals=[],audits=[])
    parsed=parse_wire(task,r.response_text)
    audits=[audit(task,p,'credible_alternatives')[0] for p in parsed['proposals']]
    return dict(status=parsed['status'],proposals=parsed['proposals'],audits=audits,
                record_sha256=sha(r.model_dump()),response_sha256=sha(r.response_text),
                source_type=r.source_type,model_requested=r.model_requested,
                model_returned=r.model_returned,provider_response_id=r.provider_response_id,
                task_sha256=r.task_sha256,prompt_sha256=r.prompt_sha256)


def seal(tasks, records, output):
    if len(tasks)!=len(records) or not tasks:raise ValueError('One task per recorded response required')
    predictions=[dict(task=public_task(t),prediction=adapt(t,r)) for t,r in zip(tasks,records)]
    payload=dict(version='provider-replay-1',predictions=predictions,real_llm_calls=0,
                 cost_usd=0,interpretation='offline_replay_not_new_model_measurement')
    result=dict(payload=payload,seal_sha256=sha(payload))
    path=Path(output)
    # Exclusive creation: never replace any fixed prediction.
    with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    path.chmod(0o600)
    return result
