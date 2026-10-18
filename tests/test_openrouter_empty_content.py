"""Safe handling of real-provider 200 responses without visible text, offline only."""
import json
import httpx
import pytest

from sharecausal.llm.openrouter import complete
from sharecausal.llm.budget import Budget

MODEL = {
    'id': 'openai/gpt-6-luna',
    'supported_parameters': ['max_tokens', 'response_format', 'seed', 'reasoning'],
    'pricing': {'prompt': '0.0000001', 'completion': '0.0000005'},
}


def test_null_content_reports_safe_diagnostics_and_does_not_retry():
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'gen-abc123', 'model': MODEL['id'],
            'choices': [{
                'finish_reason': 'length',
                'message': {'role': 'assistant', 'content': None,
                            'reasoning': 'SENSITIVE_PROVIDER_REASONING_SHOULD_NOT_PRINT'},
            }],
            'usage': {
                'completion_tokens': 1200,
                'completion_tokens_details': {'reasoning_tokens': 1200},
                'cost': 0.002,
            },
        })

    budget=Budget(1)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RuntimeError,match='without usable JSON content') as err:
            complete(MODEL,'base',budget,'fake-secret',client=client,seed=0)
    assert len(sent)==1
    assert 'finish_reason=length' in str(err.value)
    assert 'reasoning_tokens=1200' in str(err.value)
    assert 'response_id=gen-abc123' in str(err.value)
    assert 'SENSITIVE_PROVIDER' not in str(err.value)
    assert 'fake-secret' not in str(err.value)
    assert float(budget.spent)==0.002
    assert float(budget.reserved)==0


def test_low_reasoning_effort_is_advertised_and_recorded():
    payloads=[]
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'gen-valid', 'model': MODEL['id'],
            'choices': [{
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps({
                    'edges': [], 'abstain': True,
                    'abstention_reason': 'Insufficient evidence',
                    'contested_assumptions': [],
                })},
            }],
            'usage': {'cost': 0.001, 'completion_tokens': 91,
                      'completion_tokens_details': {'reasoning_tokens': 11}},
        })
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        r=complete(MODEL,'base',Budget(1),'test-key',client=client,
                   max_tokens=4096,seed=0,reasoning_effort='low')
    assert len(payloads)==1
    assert payloads[0]['reasoning']=={'effort':'low'}
    assert payloads[0]['max_tokens']==4096
    assert 'temperature' not in payloads[0]
    assert r['reasoning_effort_sent']=='low'
    assert r['max_tokens_sent']==4096
    assert r['finish_reason']=='stop'
    assert r['graph']['abstain'] is True


def test_reasoning_effort_fails_closed_if_unadvertised():
    unsupported={**MODEL,'supported_parameters':['response_format','max_tokens']}
    budget=Budget(1)
    with pytest.raises(ValueError,match='not advertised'):
        complete(unsupported,'base',budget,'test-key',reasoning_effort='low')
    assert float(budget.reserved)==0
