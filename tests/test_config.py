from pathlib import Path

import pytest
from pydantic import ValidationError

from bm25_vfs_ablation.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def test_default_config_exact_values() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.model_dump(mode="json") == {
        "schema_version": 1,
        "experiment": {
            "experiment_id": None,
            "seed": 42,
            "conditions": ["snippets", "vfs"],
            "token_ceiling": 4096,
            "concurrency": 1,
            "oracle_mode": False,
            "resume": True,
            "output_runs": "results/runs.jsonl",
        },
        "corpus": {
            "corpus_path": "data/generated/corpus.jsonl",
            "tasks_path": "data/generated/tasks.jsonl",
            "corpus_version": "synthetic-v1",
        },
        "chunking": {
            "chunk_size_tokens": 180,
            "chunk_overlap_tokens": 30,
        },
        "retrieval": {
            "implementation": "rank_bm25_okapi",
            "top_k": 8,
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "query_tokenizer": "unicode_word_v1",
        },
        "harness": {
            "retrieval_context_tokens": 1800,
            "max_answer_tokens": 256,
            "max_tool_calls": 8,
            "invalid_call_limit": 3,
            "tool_result_tokens_per_call": 768,
            "initial_candidate_documents": 5,
        },
        "model": {
            "provider": "mock",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": None,
            "model": "mock-scripted-v1",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "timeout_seconds": 60.0,
            "max_attempts": 2,
            "tokenizer": "regex_v1",
        },
        "evaluation": {
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20250308,
            "confidence_level": 0.95,
            "judge_enabled": False,
            "estimated_input_usd_per_million": 0.0,
            "estimated_output_usd_per_million": 0.0,
        },
        "intervention": {
            "enabled": False,
            "max_tool_calls": [],
        },
    }

    assert isinstance(config.corpus.corpus_path, str)
    assert isinstance(config.corpus.tasks_path, str)
    assert isinstance(config.experiment.output_runs, str)
    assert not Path(config.corpus.corpus_path).is_absolute()
    assert not Path(config.corpus.tasks_path).is_absolute()
    assert not Path(config.experiment.output_runs).is_absolute()
    assert type(config).model_validate_json(config.model_dump_json()) == config


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, {"experiment": {"unknown_setting": True}})


def test_config_cross_field_bounds() -> None:
    invalid_overrides = (
        {"experiment": {"token_ceiling": 0}},
        {"chunking": {"chunk_size_tokens": 30, "chunk_overlap_tokens": 30}},
        {"retrieval": {"top_k": 0}},
        {"harness": {"max_tool_calls": -1}},
        {"intervention": {"max_tool_calls": [0, -1]}},
    )

    for overrides in invalid_overrides:
        with pytest.raises(ValidationError):
            load_config(DEFAULT_CONFIG, overrides)


def test_env_override_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BM25_VFS_BASE_URL", "http://placeholder.invalid/v1")
    monkeypatch.setenv("BM25_VFS_API_KEY", "placeholder-api-key")
    monkeypatch.setenv("BM25_VFS_MODEL", "placeholder-model")
    monkeypatch.setenv("BM25_VFS_TOKEN_CEILING", "1")

    config = load_config(DEFAULT_CONFIG)

    assert config.model.base_url == "http://placeholder.invalid/v1"
    assert config.model.api_key == "placeholder-api-key"
    assert config.model.model == "placeholder-model"
    assert config.experiment.token_ceiling == 4096


def test_api_key_redacted_from_canonical_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BM25_VFS_API_KEY", "placeholder-api-key")

    canonical = load_config(DEFAULT_CONFIG).canonical_dict()

    assert canonical["model"]["api_key"] == "[REDACTED]"
    assert "placeholder-api-key" not in str(canonical)
