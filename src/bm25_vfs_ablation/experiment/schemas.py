"""Validated accounting and run-record schemas."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from bm25_vfs_ablation import Condition, FailureClass, TerminationReason, ToolName

_FACT_ID = re.compile(r"^fact-\d{6}$")
_DOC_ID = re.compile(r"^doc-[a-z0-9]+(?:-[a-z0-9]+)*-\d{4}$")
_TASK_ID = re.compile(r"^task-(?:dev|eval)-\d{6}$")
_CHUNK_ID = re.compile(r"^doc-[a-z0-9]+(?:-[a-z0-9]+)*-\d{4}::c\d{4}$")
_EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
_RUN_KEY = re.compile(
    r"^(?P<experiment>[a-z0-9][a-z0-9-]{2,63}):"
    r"(?P<task>task-(?:dev|eval)-\d{6}):"
    r"(?P<condition>snippets|vfs):"
    r"(?P<design>primary|oracle|max_calls_(?:0|1|2|4|8|12))$"
)
_REQUEST_ID = re.compile(r"^tool-\d{3}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_VFS_PATH = re.compile(r"^/[^\s/](?:[^\s]*[^\s/])?$")


class _StrictModel(BaseModel):
    """Base class for strict persisted models."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        validate_assignment=True,
    )


class DesignCell(StrEnum):
    """Frozen run design cells."""

    PRIMARY = "primary"
    ORACLE = "oracle"
    MAX_CALLS_0 = "max_calls_0"
    MAX_CALLS_1 = "max_calls_1"
    MAX_CALLS_2 = "max_calls_2"
    MAX_CALLS_4 = "max_calls_4"
    MAX_CALLS_8 = "max_calls_8"
    MAX_CALLS_12 = "max_calls_12"


def _require_nonblank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _validate_sorted_unique(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    if values != sorted(values):
        raise ValueError(f"{field_name} must be sorted")
    return values


def _validate_ids(
    values: list[str],
    pattern: re.Pattern[str],
    field_name: str,
) -> list[str]:
    if any(pattern.fullmatch(value) is None for value in values):
        raise ValueError(f"{field_name} contains an invalid ID")
    return _validate_sorted_unique(values, field_name)


def _validate_hash(value: str, field_name: str) -> str:
    if _HASH.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _validate_path(value: str, field_name: str = "path") -> str:
    if _VFS_PATH.fullmatch(value) is None or "\\" in value or ".." in value.split("/"):
        raise ValueError(f"{field_name} must be a root-relative POSIX path")
    return value


class ParsedAnswer(_StrictModel):
    """Public model answer envelope."""

    answer: str
    citations: list[str]
    justification: str = Field(max_length=240)

    @field_validator("answer", "justification")
    @classmethod
    def _validate_text(cls, value: str, info: Any) -> str:
        return _require_nonblank(value, info.field_name)

    @field_validator("citations")
    @classmethod
    def _validate_citations(cls, value: list[str]) -> list[str]:
        if any(not citation.strip() for citation in value):
            raise ValueError("citations must not contain blank strings")
        if len(value) != len(set(value)):
            raise ValueError("citations must be deduplicated")
        return value


class SecondaryJudge(_StrictModel):
    """Optional secondary scorer result."""

    enabled: bool
    score: float | None = Field(ge=0, le=1)
    rationale: str | None
    model_id: str | None

    @model_validator(mode="after")
    def _validate_disabled_state(self) -> SecondaryJudge:
        if not self.enabled and any(
            value is not None
            for value in (self.score, self.rationale, self.model_id)
        ):
            raise ValueError("disabled secondary judge fields must be null")
        if self.enabled and self.model_id is None:
            raise ValueError("enabled secondary judge requires model_id")
        return self


class ErrorRecord(_StrictModel):
    """Observable, sanitized run error."""

    stage: str
    code: str
    message: str
    retryable: bool

    @field_validator("stage", "code", "message")
    @classmethod
    def _validate_text(cls, value: str, info: Any) -> str:
        return _require_nonblank(value, info.field_name)


class ModelConfigSnapshot(_StrictModel):
    """Redacted model settings persisted with each run."""

    provider: str
    base_url: str
    model: str = Field(
        validation_alias=AliasChoices("model", "requested_model", "requested_model_id")
    )
    temperature: float = Field(ge=0)
    top_p: float = Field(ge=0, le=1)
    seed: int
    tokenizer: str

    @field_validator("provider", "base_url", "model", "tokenizer")
    @classmethod
    def _validate_text(cls, value: str, info: Any) -> str:
        return _require_nonblank(value, info.field_name)


class LineRange(_StrictModel):
    """One-based inclusive source line range."""

    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("path")
    @classmethod
    def _validate_line_path(cls, value: str) -> str:
        return _validate_path(value)

    @model_validator(mode="after")
    def _validate_range(self) -> LineRange:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class RetrievedDocument(_StrictModel):
    """One ranked initial document result."""

    doc_id: str
    best_score: float = Field(ge=0)
    rank: int = Field(ge=1)

    @field_validator("doc_id")
    @classmethod
    def _validate_doc_id(cls, value: str) -> str:
        if _DOC_ID.fullmatch(value) is None:
            raise ValueError("doc_id has an invalid grammar")
        return value


class RetrievedChunk(_StrictModel):
    """One ranked initial chunk result."""

    chunk_id: str
    doc_id: str
    score: float = Field(ge=0)
    rank: int = Field(ge=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("chunk_id")
    @classmethod
    def _validate_chunk_id(cls, value: str) -> str:
        if _CHUNK_ID.fullmatch(value) is None:
            raise ValueError("chunk_id has an invalid grammar")
        return value

    @field_validator("doc_id")
    @classmethod
    def _validate_doc_id(cls, value: str) -> str:
        if _DOC_ID.fullmatch(value) is None:
            raise ValueError("doc_id has an invalid grammar")
        return value

    @model_validator(mode="after")
    def _validate_chunk(self) -> RetrievedChunk:
        if self.chunk_id.split("::", maxsplit=1)[0] != self.doc_id:
            raise ValueError("chunk_id must belong to doc_id")
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class _EvidenceIdentifiers(_StrictModel):
    """Shared evidence identifier validation."""

    fact_ids: list[str]
    doc_ids: list[str]
    chunk_ids: list[str]

    @field_validator("fact_ids")
    @classmethod
    def _validate_fact_ids(cls, value: list[str]) -> list[str]:
        return _validate_ids(value, _FACT_ID, "fact_ids")

    @field_validator("doc_ids")
    @classmethod
    def _validate_doc_ids(cls, value: list[str]) -> list[str]:
        return _validate_ids(value, _DOC_ID, "doc_ids")

    @field_validator("chunk_ids")
    @classmethod
    def _validate_chunk_ids(cls, value: list[str]) -> list[str]:
        return _validate_ids(value, _CHUNK_ID, "chunk_ids")


class GoldEvidence(_EvidenceIdentifiers):
    """Required evidence identifiers for a task."""

    @model_validator(mode="after")
    def _validate_required_ids(self) -> GoldEvidence:
        if not self.fact_ids or not self.doc_ids or not self.chunk_ids:
            raise ValueError("gold evidence identifiers must not be empty")
        return self


class AccessedEvidence(_EvidenceIdentifiers):
    """Evidence exposed to the model, including source ranges."""

    line_ranges: list[LineRange]

    @field_validator("line_ranges")
    @classmethod
    def _validate_line_ranges(cls, value: list[LineRange]) -> list[LineRange]:
        order = [(item.path, item.start_line, item.end_line) for item in value]
        if order != sorted(order):
            raise ValueError("line_ranges must be sorted")
        if len(order) != len(set(order)):
            raise ValueError("line_ranges must be unique")
        return value


class RetrievalMetrics(_StrictModel):
    """Initial shared-index retrieval metrics."""

    initial_chunk_recall: float = Field(ge=0, le=1)
    initial_document_recall: float = Field(ge=0, le=1)
    k: int = Field(ge=0)


class EvidenceMetrics(_StrictModel):
    """Final evidence exposure and citation metrics."""

    final_recall: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    distinct_gold_facts: int = Field(ge=0)
    all_required_accessed: bool
    citation_precision: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)


class ToolRequest(_StrictModel):
    """Strict request passed to one read-only VFS tool."""

    tool: ToolName = Field(
        validation_alias=AliasChoices("tool", "name", "tool_name"),
        serialization_alias="tool",
    )
    arguments: dict[str, Any]

    @property
    def name(self) -> ToolName:
        """Compatibility accessor for provider-style tool requests."""

        return self.tool


class ToolResult(_StrictModel):
    """Observable result envelope returned by a VFS tool."""

    ok: bool
    tool: ToolName = Field(
        validation_alias=AliasChoices("tool", "name", "tool_name"),
        serialization_alias="tool",
    )
    request_id: str | None = None
    content: str | None = None
    paths: list[str] = Field(default_factory=list)
    line_ranges: list[LineRange] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    truncated: bool = False

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str | None) -> str | None:
        if value is not None and _REQUEST_ID.fullmatch(value) is None:
            raise ValueError("request_id must match tool-{three digits}")
        return value

    @field_validator("paths")
    @classmethod
    def _validate_paths(cls, value: list[str]) -> list[str]:
        for path in value:
            _validate_path(path)
        if value != sorted(set(value)):
            raise ValueError("paths must be sorted and unique")
        return value

    @model_validator(mode="after")
    def _validate_result(self) -> ToolResult:
        if self.ok and (self.error_code is not None or self.error_message is not None):
            raise ValueError("successful tool results cannot contain errors")
        if not self.ok and not self.error_code:
            raise ValueError("failed tool results require error_code")
        return self


class ToolTraceEntry(_StrictModel):
    """One observable VFS request/result pair."""

    request_id: str
    call_ordinal: int = Field(ge=1)
    requested_at: datetime
    completed_at: datetime
    request: ToolRequest
    result: ToolResult
    repeated: bool
    latency_ms: float = Field(ge=0)

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str) -> str:
        if _REQUEST_ID.fullmatch(value) is None:
            raise ValueError("request_id must match tool-{three digits}")
        return value

    @field_validator("requested_at", "completed_at")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_trace(self) -> ToolTraceEntry:
        if self.completed_at < self.requested_at:
            raise ValueError("completed_at must not precede requested_at")
        if self.result.request_id not in (None, self.request_id):
            raise ValueError("result request_id must match trace request_id")
        if self.request.tool != self.result.tool:
            raise ValueError("request and result tools must match")
        return self


class ToolCallCounts(_StrictModel):
    """Counts for each of the four VFS tool names."""

    grep: int = Field(ge=0)
    read: int = Field(ge=0)
    cat: int = Field(ge=0)
    list: int = Field(ge=0)

    def total(self) -> int:
        """Return the number of calls represented by the four counters."""

        return self.grep + self.read + self.cat + self.list


class TurnAccounting(_StrictModel):
    """Accounting for one model request/response turn."""

    turn_index: int = Field(ge=0)
    request_input_tokens_estimated: int = Field(ge=0)
    request_output_cap: int = Field(ge=0)
    response_output_tokens_estimated: int = Field(ge=0)
    snippet_input_tokens: int = Field(ge=0)
    tool_argument_output_tokens: int = Field(ge=0)
    tool_result_input_tokens: int = Field(ge=0)
    provider_input_tokens: int | None = Field(ge=0)
    provider_output_tokens: int | None = Field(ge=0)
    remaining_after_turn: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_components(self) -> TurnAccounting:
        if self.response_output_tokens_estimated > self.request_output_cap:
            raise ValueError("response output cannot exceed request output cap")
        if self.snippet_input_tokens > self.request_input_tokens_estimated:
            raise ValueError("snippet input must be a request-input subset")
        if self.tool_result_input_tokens > self.request_input_tokens_estimated:
            raise ValueError("tool-result input must be a request-input subset")
        if self.tool_argument_output_tokens > self.response_output_tokens_estimated:
            raise ValueError("tool arguments must be an output subset")
        return self


class TokenAccounting(_StrictModel):
    """Per-run estimated and provider token accounting."""

    tokenizer: str
    limit: int = Field(gt=0)
    turns: list[TurnAccounting]
    input_tokens_estimated: int = Field(ge=0)
    output_tokens_estimated: int = Field(ge=0)
    total_tokens_estimated: int = Field(ge=0)
    provider_input_tokens: int | None = Field(ge=0)
    provider_output_tokens: int | None = Field(ge=0)
    provider_total_tokens: int | None = Field(ge=0)
    provider_minus_estimated: dict[str, int]

    @field_validator("tokenizer")
    @classmethod
    def _validate_tokenizer(cls, value: str) -> str:
        return _require_nonblank(value, "tokenizer")

    @model_validator(mode="after")
    def _validate_totals(self) -> TokenAccounting:
        if self.total_tokens_estimated != (
            self.input_tokens_estimated + self.output_tokens_estimated
        ):
            raise ValueError("total_tokens_estimated must equal input plus output tokens")
        turn_input = sum(turn.request_input_tokens_estimated for turn in self.turns)
        turn_output = sum(turn.response_output_tokens_estimated for turn in self.turns)
        if self.input_tokens_estimated != turn_input:
            raise ValueError("input_tokens_estimated must equal the sum of turn inputs")
        if self.output_tokens_estimated != turn_output:
            raise ValueError("output_tokens_estimated must equal the sum of turn outputs")
        if self.total_tokens_estimated > self.limit:
            raise ValueError("estimated token totals must not exceed limit")
        if (
            self.provider_total_tokens is not None
            and self.provider_input_tokens is not None
            and self.provider_output_tokens is not None
            and self.provider_total_tokens
            != self.provider_input_tokens + self.provider_output_tokens
        ):
            raise ValueError("provider_total_tokens must equal provider input plus output")
        for index, turn in enumerate(self.turns):
            if turn.turn_index != index:
                raise ValueError("turns must have consecutive zero-based turn_index values")
        return self


class LatencyBreakdown(_StrictModel):
    """Nonnegative run latency components in milliseconds."""

    wall_ms: float = Field(ge=0)
    model_ms: float = Field(ge=0)
    retrieval_ms: float = Field(ge=0)
    tool_ms: float = Field(ge=0)


class RunRecord(_StrictModel):
    """Complete append-only task-condition run record."""

    schema_version: Literal[1]
    run_key: str
    experiment_id: str
    design_cell: DesignCell
    task_id: str
    condition: Condition
    condition_order: int = Field(ge=0, le=1)
    model_settings: ModelConfigSnapshot = Field(
        validation_alias=AliasChoices("model_config", "model_settings"),
        serialization_alias="model_config",
    )
    requested_model_id: str
    returned_model_id: str | None
    seed: int
    corpus_version: str
    question: str
    final_answer: str | None
    parsed_answer: ParsedAnswer | None
    correctness: bool
    secondary_judge: SecondaryJudge
    scoring_details: dict[str, Any]
    citations: list[str]
    initial_retrieved_documents: list[RetrievedDocument]
    initial_retrieved_chunks: list[RetrievedChunk]
    gold_evidence: GoldEvidence
    accessed_evidence: AccessedEvidence
    retrieval_metrics: RetrievalMetrics
    evidence_metrics: EvidenceMetrics
    tool_trace: list[ToolTraceEntry]
    tool_call_count: int = Field(ge=0)
    max_tool_calls_permitted: int | None = Field(ge=0)
    tool_calls_by_type: ToolCallCounts
    successful_tool_calls: int = Field(ge=0)
    invalid_tool_calls: int = Field(ge=0)
    repeated_tool_calls: int = Field(ge=0)
    unique_files_read: list[str]
    unique_line_ranges_accessed: list[LineRange]
    time_to_first_gold_fact_ms: float | None = Field(ge=0)
    calls_to_complete_gold_coverage: int | None = Field(ge=1)
    token_accounting: TokenAccounting
    total_tokens: int = Field(ge=0)
    latency: LatencyBreakdown
    estimated_cost_usd: float = Field(ge=0)
    termination_reason: TerminationReason
    errors: list[ErrorRecord]
    prompt_sha256: str
    config_sha256: str
    corpus_sha256: str
    cache_key: str | None
    cache_hit: bool
    started_at: datetime
    finished_at: datetime

    @field_validator("experiment_id")
    @classmethod
    def _validate_experiment_id(cls, value: str) -> str:
        if _EXPERIMENT_ID.fullmatch(value) is None:
            raise ValueError("experiment_id has an invalid grammar")
        return value

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        if _TASK_ID.fullmatch(value) is None:
            raise ValueError("task_id has an invalid grammar")
        return value

    @field_validator("requested_model_id", "corpus_version", "question")
    @classmethod
    def _validate_text(cls, value: str, info: Any) -> str:
        return _require_nonblank(value, info.field_name)

    @field_validator("prompt_sha256", "config_sha256", "corpus_sha256")
    @classmethod
    def _validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @field_validator("cache_key")
    @classmethod
    def _validate_cache_key(cls, value: str | None) -> str | None:
        if value is not None:
            return _validate_hash(value, "cache_key")
        return value

    @field_validator("citations")
    @classmethod
    def _validate_run_citations(cls, value: list[str]) -> list[str]:
        if any(not citation.strip() for citation in value):
            raise ValueError("citations must not contain blank strings")
        if len(value) != len(set(value)):
            raise ValueError("citations must be deduplicated")
        return value

    @field_validator("unique_files_read")
    @classmethod
    def _validate_unique_files(cls, value: list[str]) -> list[str]:
        for path in value:
            _validate_path(path, "unique_files_read path")
        if value != sorted(set(value)):
            raise ValueError("unique_files_read must be sorted and unique")
        return value

    @field_validator("unique_line_ranges_accessed")
    @classmethod
    def _validate_unique_ranges(cls, value: list[LineRange]) -> list[LineRange]:
        order = [(item.path, item.start_line, item.end_line) for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("unique_line_ranges_accessed must be sorted and unique")
        return value

    @field_validator("started_at", "finished_at")
    @classmethod
    def _validate_run_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate_run(self) -> RunRecord:
        key_match = _RUN_KEY.fullmatch(self.run_key)
        if key_match is None:
            raise ValueError("run_key has an invalid grammar")
        expected_key = (
            f"{self.experiment_id}:{self.task_id}:{self.condition.value}:"
            f"{self.design_cell.value}"
        )
        if self.run_key != expected_key:
            raise ValueError("run_key components must match the record fields")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.total_tokens != self.token_accounting.total_tokens_estimated:
            raise ValueError("total_tokens must equal token accounting total")
        if self.successful_tool_calls + self.invalid_tool_calls > self.tool_call_count:
            raise ValueError("successful and invalid calls cannot exceed tool_call_count")
        if self.repeated_tool_calls > self.tool_call_count:
            raise ValueError("repeated_tool_calls cannot exceed tool_call_count")
        if self.tool_calls_by_type.total() > self.tool_call_count:
            raise ValueError("tool type counts cannot exceed tool_call_count")

        if self.condition is Condition.SNIPPETS:
            if self.tool_trace:
                raise ValueError("snippet runs must have an empty tool_trace")
            if self.tool_call_count != 0:
                raise ValueError("snippet runs cannot have tool calls")
            if self.max_tool_calls_permitted is not None:
                raise ValueError("snippet runs must not have a tool-call limit")
            if (
                self.successful_tool_calls
                or self.invalid_tool_calls
                or self.repeated_tool_calls
                or self.tool_calls_by_type.total()
            ):
                raise ValueError("snippet tool counters must all be zero")
            if self.unique_files_read or self.unique_line_ranges_accessed:
                raise ValueError("snippet runs cannot have accessed VFS files")
        else:
            if self.max_tool_calls_permitted is None:
                raise ValueError("vfs runs require max_tool_calls_permitted")
            if len(self.tool_trace) != self.tool_call_count:
                raise ValueError("vfs tool_trace length must equal tool_call_count")
            ordinals = [entry.call_ordinal for entry in self.tool_trace]
            request_ids = [entry.request_id for entry in self.tool_trace]
            if ordinals != list(range(1, len(ordinals) + 1)):
                raise ValueError("tool trace call ordinals must be consecutive")
            if request_ids != [f"tool-{ordinal:03d}" for ordinal in ordinals]:
                raise ValueError("tool trace request IDs must match call ordinals")

        if (
            self.condition is Condition.VFS
            and self.design_cell.value.startswith("max_calls_")
        ):
            assigned_limit = int(self.design_cell.value.removeprefix("max_calls_"))
            if self.max_tool_calls_permitted != assigned_limit:
                raise ValueError("design-cell call limit must match max_tool_calls_permitted")
        return self


__all__ = [
    "AccessedEvidence",
    "Condition",
    "DesignCell",
    "ErrorRecord",
    "EvidenceMetrics",
    "FailureClass",
    "GoldEvidence",
    "LatencyBreakdown",
    "LineRange",
    "ModelConfigSnapshot",
    "ParsedAnswer",
    "RetrievedChunk",
    "RetrievedDocument",
    "RetrievalMetrics",
    "RunRecord",
    "SecondaryJudge",
    "TerminationReason",
    "TokenAccounting",
    "ToolCallCounts",
    "ToolName",
    "ToolRequest",
    "ToolResult",
    "ToolTraceEntry",
    "TurnAccounting",
]
