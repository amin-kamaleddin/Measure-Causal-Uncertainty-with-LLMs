"""Offline transport tests: no billable API calls."""
import json
import httpx
import pytest
from sharecausal.llm.openrouter import complete
from sharecausal.llm.budget import Budget


@pytest.mark.parametrize("advertised,should_send_temperature", [
    (["response_format", "max_tokens", "seed"], False),
    (["response_format", "max_tokens", "seed", "temperature"], True),
])
def test_temperature_only_sent_when_supported(advertised, should_send_temperature):
    captured = []

    def handle(request):
        payload = json.loads(request.content)
        captured.append(payload)
        return httpx.Response(404, json={"error": {"message": "Mock routing error"}})

    model = {
        "id": "openai/gpt-6-luna" if not should_send_temperature else "google/example",
        "supported_parameters": advertised,
        "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
    }
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RuntimeError, match="Mock routing error"):
            complete(model, "base", Budget(1), "fake-test-key", client=client, seed=0)

    assert len(captured) == 1
    sent = captured[0]
    assert ("temperature" in sent) is should_send_temperature
    if should_send_temperature:
        assert sent["temperature"] == 0
    assert sent["seed"] == 0
    assert sent["provider"]["require_parameters"] is True
    assert sent["response_format"]["type"] == "json_schema"
