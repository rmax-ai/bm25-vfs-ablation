"""Acceptance tests for the deterministic response cache."""

from __future__ import annotations

import json

import pytest

from bm25_vfs_ablation.models.cache import CacheInputs, FileResponseCache, build_cache_key
from bm25_vfs_ablation.models.client import ModelResponse


def _inputs(**overrides: object) -> CacheInputs:
    data: dict[str, object] = {
        "condition": "snippets",
        "model_config": {
            "provider": "mock",
            "base_url": "https://placeholder.invalid/v1",
            "model": "model-placeholder",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "tokenizer": "regex_v1",
        },
        "messages": [
            {"role": "system", "content": "Return the requested answer."},
            {"role": "user", "content": "Which service owns Quartz?"},
        ],
        "tool_schema": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        "corpus_sha256": "a" * 64,
        "experiment_config_sha256": "b" * 64,
    }
    data.update(overrides)
    return CacheInputs.model_validate(data)


def test_cache_key_known_vector() -> None:
    assert build_cache_key(_inputs()) == (
        "101f4229ddb5f437955b489c2f7c49429483e175234008ce9dece4f664c9ff17"
    )


def test_every_required_input_changes_key() -> None:
    base = _inputs()
    variants = (
        _inputs(condition="vfs"),
        _inputs(model_config={**base.model_settings, "model": "other-model-placeholder"}),
        _inputs(
            messages=[
                *base.messages,
                {"role": "user", "content": "Use the evidence."},
            ]
        ),
        _inputs(tool_schema=[]),
        _inputs(corpus_sha256="c" * 64),
        _inputs(experiment_config_sha256="d" * 64),
    )

    assert len({build_cache_key(base), *(build_cache_key(item) for item in variants)}) == 7


def test_api_key_never_affects_key() -> None:
    first = _inputs(
        model_config={
            **_inputs().model_settings,
            "api_key": "api-key-placeholder-a",
            "started_at": "2026-09-19T00:00:00+00:00",
        }
    )
    second = _inputs(
        model_config={
            **_inputs().model_settings,
            "api_key": "api-key-placeholder-b",
            "started_at": "2026-09-20T00:00:00+00:00",
        }
    )

    assert build_cache_key(first) == build_cache_key(second)
    assert "api_key" not in json.dumps(first.model_dump(mode="json"))
    assert "api-key-placeholder" not in json.dumps(first.model_dump(mode="json"))


def test_cache_round_trip(tmp_path) -> None:
    cache = FileResponseCache(tmp_path / "cache")
    key = build_cache_key(_inputs())
    response = ModelResponse(
        content='{"answer":"Quartz"}',
        tool_calls=[],
        returned_model_id="model-placeholder",
        provider_usage={"prompt_tokens": 4, "completion_tokens": 2},
        latency_ms=1.5,
        raw_id="response-placeholder",
    )

    assert cache.get(key) is None
    cache.put(key, response)

    assert cache.get(key) == response
    assert sorted(path.name for path in cache.root.iterdir()) == [f"{key}.json"]


def test_malformed_cache_entry_fails_closed(tmp_path) -> None:
    cache = FileResponseCache(tmp_path / "cache")
    key = build_cache_key(_inputs())
    cache.root.mkdir()
    (cache.root / f"{key}.json").write_text('{"content":\n', encoding="utf-8")

    with pytest.raises(ValueError, match="malformed cached response"):
        cache.get(key)
