"""Acceptance tests for the model protocol and its two clients."""

from __future__ import annotations

import httpx
import pytest

from bm25_vfs_ablation.models.client import (
    MockAction,
    MockPolicy,
    MockPolicyError,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleClient,
    ScriptedMockClient,
)


def _request() -> ModelRequest:
    return ModelRequest(
        messages=[{"role": "user", "content": "placeholder"}],
        tools=[],
        max_output_tokens=32,
        temperature=0,
        top_p=1,
        seed=7,
    )


def test_protocol_response_shape() -> None:
    response = ModelResponse(
        content="answer",
        tool_calls=[],
        returned_model_id="model-placeholder",
        provider_usage={"prompt_tokens": 3, "completion_tokens": 2},
        latency_ms=1.5,
        raw_id="response-placeholder",
    )

    assert set(response.model_dump()) == {
        "content",
        "tool_calls",
        "returned_model_id",
        "provider_usage",
        "latency_ms",
        "raw_id",
    }
    assert ModelResponse.model_validate_json(response.model_dump_json()) == response


def test_http_request_and_usage_parse() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "id": "response-placeholder",
                "model": "served-placeholder",
                "choices": [{"message": {"role": "assistant", "content": "answer"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
            request=request,
        )

    client = OpenAICompatibleClient(
        "https://provider.invalid/v1/",
        "placeholder",
        "requested-placeholder",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = client.complete(_request())

    assert seen["url"] == "https://provider.invalid/v1/chat/completions"
    assert seen["authorization"] == "Bearer placeholder"
    assert b'"max_tokens":32' in seen["body"]  # type: ignore[operator]
    assert response.returned_model_id == "served-placeholder"
    assert response.provider_usage["total_tokens"] == 6
    assert response.content == "answer"


def test_http_retry_policy_exact() -> None:
    statuses = iter((500, 429, 200))
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        payload = (
            {"id": "response-placeholder", "model": "served-placeholder", "choices": []}
            if status != 200
            else {
                "id": "response-placeholder",
                "model": "served-placeholder",
                "choices": [{"message": {"content": "answer"}}],
            }
        )
        return httpx.Response(status, json=payload, request=request)

    client = OpenAICompatibleClient(
        "https://provider.invalid",
        "placeholder",
        "model-placeholder",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=sleeps.append,
    )
    assert client.complete(_request()).content == "answer"
    assert sleeps == [0.25, 0.5]


def test_api_key_not_in_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="request rejected", request=request)

    client = OpenAICompatibleClient(
        "https://provider.invalid",
        "placeholder",
        "model-placeholder",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(Exception) as error:
        client.complete(_request())
    assert "placeholder" not in str(error.value)


def test_mock_single_shot_answer() -> None:
    key = ("task-placeholder", "snippets", "primary")
    policy = MockPolicy({key: [MockAction(kind="final", answer="answer")]})
    response = ScriptedMockClient(policy, *key).complete(_request())

    assert response.content == "answer"
    assert response.returned_model_id == "mock-scripted-v1"
    assert response.tool_calls == []


def test_mock_multiturn_tool_sequence() -> None:
    key = ("task-placeholder", "vfs", "primary")
    policy = MockPolicy(
        {
            key: [
                MockAction(
                    kind="tool_call",
                    tool_name="read",
                    arguments={"path": "/doc", "start_line": 1},
                ),
                MockAction(kind="final", answer="answer"),
            ]
        }
    )
    client = ScriptedMockClient(policy, *key)

    tool_response = client.complete(_request())
    final_response = client.complete(_request())

    assert tool_response.tool_calls[0]["function"]["name"] == "read"
    assert tool_response.tool_calls[0]["function"]["arguments"] == (
        '{"path":"/doc","start_line":1}'
    )
    assert final_response.content == "answer"


def test_mock_policy_exhaustion() -> None:
    key = ("task-placeholder", "snippets", "primary")
    client = ScriptedMockClient(
        MockPolicy({key: [MockAction(kind="final", answer="answer")]}),
        *key,
    )
    client.complete(_request())

    with pytest.raises(MockPolicyError, match="MOCK_POLICY_EXHAUSTED"):
        client.complete(_request())
