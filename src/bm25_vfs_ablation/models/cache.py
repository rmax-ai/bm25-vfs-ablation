"""Deterministic, filesystem-backed cache for model responses."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from bm25_vfs_ablation.experiment.artifacts import canonical_json
from bm25_vfs_ablation.models.client import ModelResponse

_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "client_secret",
        "password",
        "secret",
        "token",
    }
)
_RUNTIME_TIMESTAMP_KEYS = frozenset(
    {
        "completed_at",
        "created_at",
        "finished_at",
        "requested_at",
        "recorded_at",
        "started_at",
        "timestamp",
        "timestamps",
        "updated_at",
    }
)


def _normalized_key(key: str) -> str:
    return key.casefold().replace("-", "_")


def _is_excluded_key(key: str) -> bool:
    normalized = _normalized_key(key)
    return (
        normalized in _SECRET_KEYS
        or normalized in _RUNTIME_TIMESTAMP_KEYS
        or normalized.endswith("_timestamp")
        or normalized.endswith("_timestamps")
    )


def _redact_json_value(value: Any) -> Any:
    """Return JSON-compatible data without secrets or runtime timestamps."""

    if isinstance(value, BaseModel):
        return _redact_json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("cache payload object keys must be strings")
            if not _is_excluded_key(key):
                result[key] = _redact_json_value(nested)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _redact_json_value(value.value)
    raise TypeError(f"cache payload contains unsupported value type {type(value).__name__}")


class CacheInputs(BaseModel):
    """The stable inputs that identify one model completion."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    condition: str
    model_settings: dict[str, Any] = Field(
        validation_alias=AliasChoices("model_config", "model_settings"),
        serialization_alias="model_config",
    )
    messages: list[dict[str, Any]]
    tool_schema: list[dict[str, Any]]
    corpus_sha256: str = Field(
        validation_alias=AliasChoices("corpus_sha256", "corpus_hash"),
    )
    experiment_config_sha256: str = Field(
        validation_alias=AliasChoices(
            "experiment_config_sha256",
            "experiment_config_hash",
            "config_sha256",
            "config_hash",
        ),
    )

    @field_validator("model_settings", mode="before")
    @classmethod
    def _redact_model_settings(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        if not isinstance(value, Mapping):
            raise TypeError("model_config must be an object")
        redacted = _redact_json_value(value)
        if not isinstance(redacted, dict):  # pragma: no cover - guarded above
            raise TypeError("model_config must be an object")
        return redacted

    @field_validator("condition", "corpus_sha256", "experiment_config_sha256")
    @classmethod
    def _require_nonblank(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @property
    def corpus_hash(self) -> str:
        """Compatibility name for the corpus digest."""

        return self.corpus_sha256

    @property
    def experiment_config_hash(self) -> str:
        """Compatibility name for the experiment configuration digest."""

        return self.experiment_config_sha256


def _cache_payload(inputs: CacheInputs) -> dict[str, Any]:
    """Build the exact JSON object that is hashed for a cache key."""

    return {
        "condition": inputs.condition,
        "model_config": inputs.model_settings,
        "messages": inputs.messages,
        "tool_schema": inputs.tool_schema,
        "corpus_sha256": inputs.corpus_sha256,
        "experiment_config_sha256": inputs.experiment_config_sha256,
    }


def build_cache_key(inputs: CacheInputs) -> str:
    """Return the SHA-256 digest of the canonical cache-input JSON."""

    if not isinstance(inputs, CacheInputs):
        raise TypeError("inputs must be a CacheInputs instance")
    serialized = canonical_json(_cache_payload(inputs)).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


class FileResponseCache:
    """An atomic one-file-per-key cache for validated model responses."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(os.path.expanduser(os.fspath(root)))

    def _path_for(self, key: str) -> Path:
        if (
            not isinstance(key, str)
            or not key
            or key in {".", ".."}
            or "\x00" in key
            or "/" in key
            or "\\" in key
        ):
            raise ValueError("cache key must be a non-empty filename component")
        return self.root / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        """Return a cached response, or ``None`` only when the file is absent."""

        path = self._path_for(key)
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream, parse_constant=_reject_non_finite)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"malformed cached response at {path}") from error

        if not isinstance(payload, dict):
            raise ValueError(f"malformed cached response at {path}")
        try:
            return ModelResponse.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed cached response at {path}") from error

    def put(self, key: str, response: ModelResponse) -> None:
        """Atomically persist one validated response under *key*."""

        target = self._path_for(key)
        validated = ModelResponse.model_validate(response)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None

        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            payload = canonical_json(validated.model_dump(mode="json")) + "\n"
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
        except BaseException:
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()
            raise


__all__ = ["CacheInputs", "FileResponseCache", "build_cache_key"]
