"""The narrow model boundary used by the experiment harnesses."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field


class ModelRequest(BaseModel):
    """Provider-independent input for one model turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    max_output_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    seed: int | None = None


class ModelResponse(BaseModel):
    """Provider-independent output for one model turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    returned_model_id: str
    provider_usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    raw_id: str | None = None


class ModelClient(Protocol):
    """The model operation shared by real and scripted clients."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one request."""


class ModelClientError(RuntimeError):
    """A sanitized model-client failure."""


class OpenAICompatibleClient:
    """Small OpenAI-compatible ``/chat/completions`` client.

    The HTTP client and sleeper are injectable so acceptance tests never need
    network access and can assert the retry schedule exactly.
    """

    _RETRY_DELAYS = (0.25, 0.5)

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        http_client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_attempts: int = 3,
        timeout_seconds: float | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when set")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        if http_client is not None:
            self._http_client = http_client
        elif timeout_seconds is not None:
            self._http_client = httpx.Client(timeout=timeout_seconds)
        else:
            self._http_client = httpx.Client()
        self._sleeper = sleeper
        self._max_attempts = max_attempts

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Submit *request*, retrying only the frozen transient failures."""

        body: dict[str, Any] = {
            "model": self._model,
            "messages": request.messages,
            "tools": request.tools,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.seed is not None:
            body["seed"] = request.seed

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        for attempt in range(self._max_attempts):
            try:
                response = self._http_client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )
            except httpx.TimeoutException as exc:
                if attempt + 1 < self._max_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise ModelClientError("model request timed out") from exc
            except httpx.RequestError as exc:
                raise ModelClientError("model request transport error") from exc

            if self._is_retryable_status(response.status_code):
                if attempt + 1 < self._max_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise ModelClientError(f"model request failed with HTTP {response.status_code}")
            if response.status_code >= 400:
                raise ModelClientError(f"model request failed with HTTP {response.status_code}")
            break
        else:  # pragma: no cover - the loop either returns or raises
            raise ModelClientError("model request failed")

        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
            tool_calls = message.get("tool_calls") or []
            usage = payload.get("usage") or {}
            returned_model_id = payload["model"]
            raw_id = payload.get("id")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelClientError("invalid model response") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            returned_model_id=returned_model_id,
            provider_usage=usage,
            latency_ms=max(0.0, latency_ms),
            raw_id=raw_id,
        )

    def _sleep_before_retry(self, retry_number: int) -> None:
        delay = self._RETRY_DELAYS[min(retry_number, len(self._RETRY_DELAYS) - 1)]
        self._sleeper(delay)

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code <= 599


class MockAction(BaseModel):
    """One deterministic action in a scripted model policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_call", "final"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None


class MockPolicyError(RuntimeError):
    """A missing or exhausted scripted policy."""

    code = "MOCK_POLICY_EXHAUSTED"

    def __init__(self) -> None:
        super().__init__(self.code)


class MockPolicy:
    """A cursor-based action table keyed by task, condition, and design cell."""

    def __init__(
        self,
        actions: Mapping[tuple[str, str, str], Sequence[MockAction]],
    ) -> None:
        self._actions = {
            key: tuple(
                action if isinstance(action, MockAction) else MockAction.model_validate(action)
                for action in sequence
            )
            for key, sequence in actions.items()
        }
        self._cursors = dict.fromkeys(self._actions, 0)

    def next_action(self, key: tuple[str, str, str]) -> MockAction:
        """Consume and return the next action for *key*."""

        sequence = self._actions.get(key)
        cursor = self._cursors.get(key, 0)
        if sequence is None or cursor >= len(sequence):
            raise MockPolicyError
        self._cursors[key] = cursor + 1
        return sequence[cursor]


class ScriptedMockClient:
    """No-network model client backed by a deterministic ``MockPolicy``."""

    returned_model_id = "mock-scripted-v1"

    def __init__(
        self,
        policy: MockPolicy,
        task_id: str,
        condition: str,
        design_cell: str,
    ) -> None:
        self._policy = policy
        self._key = (task_id, condition, design_cell)

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Consume one scripted action; request content is intentionally opaque."""

        del request
        action = self._policy.next_action(self._key)
        if action.kind == "final":
            return ModelResponse(
                content=action.answer,
                returned_model_id=self.returned_model_id,
                latency_ms=0.0,
            )
        if action.tool_name is None:
            raise ValueError("tool_call action requires tool_name")
        arguments = json.dumps(
            action.arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return ModelResponse(
            content=None,
            tool_calls=[
                {
                    "id": "mock-call",
                    "type": "function",
                    "function": {"name": action.tool_name, "arguments": arguments},
                }
            ],
            returned_model_id=self.returned_model_id,
            latency_ms=0.0,
        )


__all__ = [
    "ModelClient",
    "ModelClientError",
    "ModelRequest",
    "ModelResponse",
    "MockAction",
    "MockPolicy",
    "MockPolicyError",
    "OpenAICompatibleClient",
    "ScriptedMockClient",
]
