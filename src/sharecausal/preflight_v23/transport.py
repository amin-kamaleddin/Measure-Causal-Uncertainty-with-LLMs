"""Isolated deterministic LOCAL fixtures, never a provider transport."""
from typing import Literal
from pydantic import Field
from sharecausal.preflight.contracts import Strict,canonical,sha
from sharecausal.preflight.comparison import Edge
from .packets import task_hash,prompt
from .admission import Proposal,no_llm,proposal,audit


class MockReceipt(Strict):
    source_type:Literal['offline_fixture']
    transport:Literal['local_only_no_sender']
    model_id:str=Field(pattern=r'^mock-family-[a-d]/fixture$')
    variant:Literal['base','challenge']
    repetition:int=Field(ge=0)
    prompt_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    task_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    response_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    schema_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    record_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')


class MockTransport:
    def __init__(self):self.calls=0

    def record(self,task,model,variant,repetition,price,budget,input_cap,output_cap):
        if price.model_id!=model:raise ValueError('model_price_mismatch')
        pp=prompt(task,variant);baseline=no_llm(task)
        if model.endswith('a/fixture'):answers=[baseline[0].model_dump()]
        elif model.endswith('b/fixture'):answers=[baseline[-1].model_dump()]
        elif model.endswith('c/fixture'):
            note=next(e for e in task.card.evidence if e.kind=='general')
            # Deliberately unrelated but well-formed hypothesis: test permissive tier.
            required=[Edge(source=e.source,target=e.target,evidence_id=e.evidence_id)
                      for e in task.card.evidence if e.kind=='edge' and e.relation=='supported']
            answers=[proposal(required+[Edge(source='U',target='Y',evidence_id=note.evidence_id)]).model_dump()]
        elif model.endswith('d/fixture'):
            if len(baseline)>1:answers=[p.model_dump() for p in baseline]
            elif task.card.structural_exclusion or any(v.name=='M' for v in task.card.variables):
                answers=[baseline[0].model_dump()]
            else:answers=[proposal(abstain=True).model_dump()]
        else:raise ValueError('unknown_mock_model')
        reserved=budget.reserve(model,price.cost(input_cap,output_cap))
        used_input=len(canonical(pp));used_output=len(canonical(answers))
        if used_input>input_cap or used_output>output_cap:
            raise ValueError('mock_byte_token_proxy_exceeds_reservation')
        budget.reconcile(reserved,price.cost(used_input,used_output))
        fields={'source_type':'offline_fixture','transport':'local_only_no_sender','model_id':model,
                'variant':variant,'repetition':repetition,'prompt_sha256':sha(pp),'task_sha256':task_hash(task),
                'response_sha256':sha(answers),'schema_sha256':sha(Proposal.model_json_schema())}
        receipt=MockReceipt(**fields,record_sha256=sha(fields));self.calls+=1
        return answers,receipt


def validate_record(task,answers,raw,variant,repetition,model):
    try:
        receipt=MockReceipt.model_validate(raw)
        content=receipt.model_dump(exclude={'record_sha256'})
        if (receipt.record_sha256!=sha(content) or receipt.task_sha256!=task_hash(task)
            or receipt.response_sha256!=sha(answers) or receipt.prompt_sha256!=sha(prompt(task,variant))
            or receipt.schema_sha256!=sha(Proposal.model_json_schema()) or receipt.variant!=variant
            or receipt.repetition!=repetition or receipt.model_id!=model):raise ValueError('receipt_binding')
        if not isinstance(answers,list) or not answers:raise ValueError('response_list_required')
        for raw_answer in answers:
            if not audit(task,raw_answer)[0]['format_valid']:raise ValueError('fixture_format_invalid')
    except (ValueError,TypeError,AttributeError):return False
    return True
