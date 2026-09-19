"""Typed configuration loading for the BM25/VFS ablation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bm25_vfs_ablation import Condition


class _StrictModel(BaseModel):
    """Base model for the repository's strict persisted configuration."""

    model_config = ConfigDict(extra="forbid")


def _repo_relative_posix_path(value: str) -> str:
    """Validate a canonical repository-relative POSIX path without resolving it."""

    if not value:
        raise ValueError("path must not be empty")
    if value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError("path must be a repository-relative POSIX string")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("path must use normalized repository-relative components")
    return value


class ExperimentConfig(_StrictModel):
    """Experiment-wide execution settings."""

    experiment_id: str | None = None
    seed: int
    conditions: list[Condition] = Field(min_length=2, max_length=2)
    token_ceiling: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    oracle_mode: bool
    resume: bool
    output_runs: str

    _validate_output_runs = field_validator("output_runs")(_repo_relative_posix_path)

    @field_validator("conditions")
    @classmethod
    def _validate_unique_conditions(cls, value: list[Condition]) -> list[Condition]:
        if len(set(value)) != len(value):
            raise ValueError("conditions must be unique")
        return value


class CorpusConfig(_StrictModel):
    """Repository-relative corpus and task artifact locations."""

    corpus_path: str
    tasks_path: str
    corpus_version: str

    _validate_corpus_path = field_validator("corpus_path")(_repo_relative_posix_path)
    _validate_tasks_path = field_validator("tasks_path")(_repo_relative_posix_path)


class ChunkingConfig(_StrictModel):
    """Token-based chunking settings."""

    chunk_size_tokens: int = Field(gt=0)
    chunk_overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be less than chunk_size_tokens")
        return self


class RetrievalConfig(_StrictModel):
    """Shared BM25 retrieval settings."""

    implementation: Literal["rank_bm25_okapi"]
    top_k: int = Field(gt=0)
    k1: float = Field(ge=0)
    b: float = Field(ge=0, le=1)
    epsilon: float = Field(ge=0)
    query_tokenizer: Literal["unicode_word_v1"]


class HarnessConfig(_StrictModel):
    """Token, call, and initial-context limits for the harnesses."""

    retrieval_context_tokens: int = Field(ge=0)
    max_answer_tokens: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    invalid_call_limit: int = Field(ge=0)
    tool_result_tokens_per_call: int = Field(ge=0)
    initial_candidate_documents: int = Field(ge=0)


class ModelConfig(_StrictModel):
    """Model provider and deterministic sampling settings."""

    provider: str
    base_url: str
    api_key: str | None
    model: str
    temperature: float = Field(ge=0)
    top_p: float = Field(ge=0, le=1)
    seed: int
    timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=0)
    tokenizer: Literal["regex_v1"]


class EvaluationConfig(_StrictModel):
    """Offline evaluation and cost-estimate settings."""

    bootstrap_resamples: int = Field(gt=0)
    bootstrap_seed: int
    confidence_level: float = Field(gt=0, lt=1)
    judge_enabled: bool
    estimated_input_usd_per_million: float = Field(ge=0)
    estimated_output_usd_per_million: float = Field(ge=0)


class InterventionConfig(_StrictModel):
    """Optional assigned maximum-tool-call intervention settings."""

    enabled: bool = False
    max_tool_calls: list[int] = Field(default_factory=list)

    @field_validator("max_tool_calls")
    @classmethod
    def _validate_limits(cls, value: list[int]) -> list[int]:
        if any(limit < 0 for limit in value):
            raise ValueError("max_tool_calls must contain only nonnegative values")
        if any(
            previous >= current
            for previous, current in zip(value, value[1:], strict=False)
        ):
            raise ValueError(
                "max_tool_calls must be strictly increasing with unique values"
            )
        return value

    @model_validator(mode="after")
    def _validate_enabled_grid(self) -> InterventionConfig:
        if self.enabled and not self.max_tool_calls:
            raise ValueError("enabled intervention requires a non-empty max_tool_calls grid")
        return self


class AppConfig(_StrictModel):
    """Complete validated application configuration."""

    schema_version: Literal[1]
    experiment: ExperimentConfig
    corpus: CorpusConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    harness: HarnessConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)

    def canonical_dict(self) -> dict[str, Any]:
        """Return JSON-compatible config data with secrets redacted."""

        payload = self.model_dump(mode="json")
        if payload["model"]["api_key"] is not None:
            payload["model"]["api_key"] = "[REDACTED]"
        return payload


_ENVIRONMENT_OVERRIDES: dict[str, tuple[str, str]] = {
    "BM25_VFS_BASE_URL": ("model", "base_url"),
    "BM25_VFS_API_KEY": ("model", "api_key"),
    "BM25_VFS_MODEL": ("model", "model"),
}


def _apply_environment_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply only the allowlisted environment variables to model settings."""

    result = deepcopy(data)
    for environment_name, (section, field_name) in _ENVIRONMENT_OVERRIDES.items():
        if environment_name in os.environ:
            result[section][field_name] = os.environ[environment_name]
    return result


def _merge_overrides(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Deep-merge CLI-style mappings so nested fields can be overridden."""

    for key, value in overrides.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, Mapping)
        ):
            _merge_overrides(target[key], value)
        else:
            target[key] = deepcopy(value)


def load_config(
    path: Path,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load, validate, and canonicalize a YAML application configuration."""

    config_path = path.expanduser()
    with config_path.open("r", encoding="utf-8") as stream:
        raw_data = yaml.safe_load(stream)

    initial_config = AppConfig.model_validate(raw_data)
    data = _apply_environment_overrides(initial_config.model_dump(mode="json"))

    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be a mapping")
        _merge_overrides(data, overrides)

    return AppConfig.model_validate(data)


__all__ = [
    "AppConfig",
    "ChunkingConfig",
    "CorpusConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "HarnessConfig",
    "InterventionConfig",
    "ModelConfig",
    "RetrievalConfig",
    "load_config",
]
