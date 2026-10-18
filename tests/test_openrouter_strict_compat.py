import json
import httpx
import pytest
from sharecausal.llm.openrouter import _strict_graph_schema, complete
from sharecausal.llm.budget import Budget


def test_generated_strict_schema_has_all_fields_required():
    schema = _strict_graph_schema()
    assert set(schema['required']) == set(schema['properties'])
    assert 'maxItems' not in schema['properties']['edges']
    arrow = schema['$defs']['Arrow']
    assert set(arrow['required']) == set(arrow['properties'])


def test_400_error_is_safe_and_uses_strict_schema():
    requests = []
    def handle(request):
        obj = json.loads(request.content)
        requests.append(obj)
        return httpx.Response(400, json={
            'error': {'code': 400, 'message': 'Schema invalid; secret=very-secret-key',
                      'metadata': {'error_type': 'invalid_request', 'raw': 'ignored provider raw data'}}
        })
    model = {'id': 'google/example','pricing': {'prompt':'0.00000001','completion':'0.000001'}}
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RuntimeError) as exc:
            complete(model, 'base', Budget(1), 'very-secret-key', client=client, seed=None)
    msg = str(exc.value)
    assert 'HTTP 400' in msg and 'google/example' in msg
    assert 'invalid_request' in msg
    assert 'very-secret-key' not in msg and 'ignored provider raw data' not in msg
    assert len(requests) == 1
    assert set(requests[0]['response_format']['json_schema']['schema']['required']) == {'edges', 'abstain', 'abstention_reason', 'contested_assumptions'}
