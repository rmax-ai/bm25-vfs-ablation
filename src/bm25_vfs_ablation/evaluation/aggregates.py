"""Stable aggregate tables for validated experiment run records."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from bm25_vfs_ablation import FailureClass
from bm25_vfs_ablation.evaluation.failures import (
    classify_failure,
    export_trace_examples,
)
from bm25_vfs_ablation.evaluation.statistics import (
    paired_summary,
    stratified_summaries,
    within_vfs_regression,
)
from bm25_vfs_ablation.experiment.artifacts import canonical_json

_MISSING = object()
_DESIGN_ORDER = (
    "primary",
    "max_calls_0",
    "max_calls_1",
    "max_calls_2",
    "max_calls_4",
    "max_calls_8",
    "max_calls_12",
    "oracle",
)
_CONDITION_ORDER = ("snippets", "vfs")
_BUCKET_ORDER = ("0-25%", "25-50%", "50-75%", "75-100%", "over-100%", "unknown")

_RUN_FRAME_COLUMNS = (
    "run_key",
    "experiment_id",
    "task_id",
    "split",
    "condition",
    "design_cell",
    "condition_order",
    "question",
    "corpus_version",
    "correctness",
    "success",
    "final_answer",
    "parsed_answer",
    "parsed_answer_json",
    "parsed_answer_text",
    "parsed_answer_justification",
    "citations",
    "citations_json",
    "termination_reason",
    "initial_retrieved_document_count",
    "initial_retrieved_chunk_count",
    "initial_chunk_recall",
    "initial_document_recall",
    "initial_recall",
    "retrieval_k",
    "gold_fact_count",
    "gold_document_count",
    "gold_chunk_count",
    "accessed_fact_count",
    "accessed_document_count",
    "accessed_chunk_count",
    "accessed_line_range_count",
    "final_evidence_recall",
    "final_recall",
    "evidence_precision",
    "distinct_gold_facts",
    "all_required_accessed",
    "citation_precision",
    "citation_recall",
    "tool_call_count",
    "calls",
    "max_tool_calls_permitted",
    "permitted_tool_calls",
    "successful_tool_calls",
    "invalid_tool_calls",
    "repeated_tool_calls",
    "grep_calls",
    "read_calls",
    "cat_calls",
    "list_calls",
    "unique_files_read_count",
    "unique_line_ranges_accessed_count",
    "time_to_first_gold_fact_ms",
    "calls_to_complete_gold_coverage",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tokens",
    "token_limit",
    "wall_ms",
    "model_ms",
    "retrieval_ms",
    "tool_ms",
    "estimated_cost_usd",
    "hop_count",
    "hops",
    "category",
    "task_category",
    "distractor_count",
    "distractors",
    "budget_bucket",
    "token_budget_bucket",
    "initial_retrieved_documents_json",
    "initial_retrieved_chunks_json",
    "gold_evidence_json",
    "accessed_evidence_json",
    "tool_trace_json",
    "token_turns_json",
    "errors_json",
    "status",
)

_METRIC_COLUMNS = (
    "n",
    "successes",
    "success_rate",
    "initial_chunk_recall_mean",
    "initial_document_recall_mean",
    "initial_retrieved_document_count_mean",
    "initial_retrieved_chunk_count_mean",
    "retrieval_k_mean",
    "final_evidence_recall_mean",
    "evidence_precision_mean",
    "distinct_gold_facts_mean",
    "accessed_fact_count_mean",
    "accessed_document_count_mean",
    "accessed_chunk_count_mean",
    "accessed_line_range_count_mean",
    "all_required_accessed_rate",
    "citation_precision_mean",
    "citation_recall_mean",
    "actual_tool_calls",
    "actual_tool_calls_mean",
    "successful_tool_calls_mean",
    "invalid_tool_calls_mean",
    "repeated_tool_calls_mean",
    "grep_calls_mean",
    "read_calls_mean",
    "cat_calls_mean",
    "list_calls_mean",
    "unique_files_read_count_mean",
    "unique_line_ranges_accessed_count_mean",
    "time_to_first_gold_fact_ms_mean",
    "calls_to_complete_gold_coverage_mean",
    "total_tokens",
    "total_tokens_mean",
    "input_tokens",
    "output_tokens",
    "wall_ms_mean",
    "model_ms_mean",
    "retrieval_ms_mean",
    "tool_ms_mean",
    "total_cost_usd",
    "mean_cost_usd",
    "success_per_1000_tokens",
    "status",
)

_CONDITION_COLUMNS = ("condition", "design_cell", *_METRIC_COLUMNS)
_SUBGROUP_COLUMNS = ("subgroup", "level", "condition", "design_cell", *_METRIC_COLUMNS)
_INTERVENTION_COLUMNS = (
    "condition",
    "permitted_call_limit",
    "design_cell",
    *_METRIC_COLUMNS,
)
_EFFICIENCY_COLUMNS = (
    "condition",
    "budget_bucket",
    *_METRIC_COLUMNS,
)
_PAIRED_COLUMNS = (
    "outcome",
    "count",
    "n",
    "rate",
    "snippets_successes",
    "vfs_successes",
    "snippets_success_rate",
    "vfs_success_rate",
    "difference",
    "ci_lower",
    "ci_upper",
    "snippets_only",
    "vfs_only",
    "mcnemar_p_value",
    "status",
)
_TASK_LEVEL_COLUMNS = (
    "design_cell",
    "task_id",
    "split",
    "category",
    "hop_count",
    "distractor_count",
    "snippets_success",
    "vfs_success",
    "paired_outcome",
    "snippets_total_tokens",
    "vfs_total_tokens",
    "snippets_initial_recall",
    "vfs_initial_recall",
    "snippets_final_recall",
    "vfs_final_recall",
    "snippets_actual_tool_calls",
    "vfs_actual_tool_calls",
    "snippets_permitted_tool_calls",
    "vfs_permitted_tool_calls",
    "snippets_wall_ms",
    "vfs_wall_ms",
    "snippets_cost_usd",
    "vfs_cost_usd",
    "status",
)
_STRATIFIED_COLUMNS = (
    "stratum",
    "band",
    "n",
    "snippets_success_rate",
    "vfs_success_rate",
    "difference",
    "mcnemar_p_value",
    "missing_reason",
    "status",
)
_REGRESSION_COLUMNS = (
    "term",
    "coefficient",
    "std_error",
    "ci_lower",
    "ci_upper",
    "p_value",
    "converged",
    "error",
    "associative",
    "covariance_type",
    "n",
    "reference_category",
    "status",
)
_FAILURE_COLUMNS = (
    "condition",
    "design_cell",
    "failure_class",
    "count",
    "n",
    "rate",
    "status",
)


def _value(record: object, *paths: str) -> object:
    """Read the first present value from a mapping or model-like record."""

    for path in paths:
        if isinstance(record, Mapping) and path in record:
            return record[path]
        current: object = record
        found = True
        for part in path.split("."):
            if isinstance(current, Mapping):
                if part not in current:
                    found = False
                    break
                current = current[part]
            else:
                current = getattr(current, part, _MISSING)
                if current is _MISSING:
                    found = False
                    break
        if found:
            return current
    return _MISSING


def _is_missing(value: object) -> bool:
    if value is _MISSING or value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _enum_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    value = getattr(value, "value", value)
    text = str(value).strip()
    if "." in text:
        prefix, suffix = text.split(".", maxsplit=1)
        if prefix.casefold() in {"condition", "designcell", "terminationreason"}:
            text = suffix
    return text.casefold() or None


def _number(value: object) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    result = _number(value)
    if result is None or not result.is_integer():
        return None
    return int(result)


def _boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    number = _number(value)
    if number in {0.0, 1.0}:
        return bool(number)
    return None


def _items(value: object) -> tuple[object, ...]:
    if _is_missing(value):
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)


def _json_value(value: object) -> Any:
    """Convert models and scalar extensions to JSON-safe values."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is _MISSING or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_cell(value: object) -> str | None:
    if _is_missing(value):
        return None
    return canonical_json(_json_value(value))


def _text_value(record: object, *paths: str) -> str | None:
    value = _value(record, *paths)
    if _is_missing(value):
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _count_value(record: object, *paths: str) -> int | None:
    return _integer(_value(record, *paths))


def _list_count(record: object, *paths: str) -> int | None:
    value = _value(record, *paths)
    if _is_missing(value):
        return None
    return len(_items(value))


def _mean(rows: Sequence[Mapping[str, object]], column: str) -> float | None:
    values = [_number(row.get(column)) for row in rows]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _sum(rows: Sequence[Mapping[str, object]], column: str) -> float:
    return sum(value or 0.0 for value in (_number(row.get(column)) for row in rows))


def _design_rank(value: object) -> int:
    text = _enum_text(value) or ""
    try:
        return _DESIGN_ORDER.index(text)
    except ValueError:
        return len(_DESIGN_ORDER)


def _condition_rank(value: object) -> int:
    text = _enum_text(value) or ""
    try:
        return _CONDITION_ORDER.index(text)
    except ValueError:
        return len(_CONDITION_ORDER)


def _bucket_rank(value: object) -> int:
    text = str(value) if not _is_missing(value) else "unknown"
    try:
        return _BUCKET_ORDER.index(text)
    except ValueError:
        return len(_BUCKET_ORDER)


def _normalise_records(records: Iterable[object]) -> tuple[object, ...]:
    if isinstance(records, (str, bytes, bytearray, Mapping)):
        raise TypeError("records must be an iterable of run records")
    return tuple(records)


def _design(record: object) -> str | None:
    return _enum_text(_value(record, "design_cell", "design", "cell")) or "primary"


def _condition(record: object) -> str | None:
    return _enum_text(_value(record, "condition", "condition_name"))


def _scope_records(records: Sequence[object], include_design: str) -> tuple[object, ...]:
    if include_design == "all":
        return tuple(records)
    if include_design == "primary":
        return tuple(record for record in records if _design(record) == "primary")
    if include_design == "intervention":
        return tuple(
            record for record in records if (_design(record) or "").startswith("max_calls_")
        )
    raise ValueError("include_design must be 'primary', 'intervention', or 'all'")


def _task_metadata(record: object) -> tuple[str | None, str | None, int | None, int | None]:
    task_id = _text_value(record, "task_id", "task.task_id", "task_record.task_id")
    split = _enum_text(_value(record, "split", "task.split", "task_record.split"))
    if split is None and task_id is not None:
        parts = task_id.split("-")
        if len(parts) >= 3 and parts[0] == "task":
            split = parts[1]

    category = _text_value(record, "category", "task.category", "task_record.category")
    hop_count = _count_value(record, "hop_count", "task.hop_count", "task_record.hop_count")
    distractor_count = _count_value(
        record,
        "distractor_count",
        "task.distractor_count",
        "task_record.distractor_count",
    )

    if hop_count is None:
        hop_count = _list_count(
            record,
            "gold_evidence.fact_ids",
            "task.required_fact_ids",
            "task_record.required_fact_ids",
        )
    if distractor_count is None:
        distractor_count = _list_count(
            record,
            "distractor_document_ids",
            "task.distractor_document_ids",
            "task_record.distractor_document_ids",
        )
    if category is None:
        gold_documents = _items(_value(record, "gold_evidence.doc_ids"))
        if gold_documents:
            document_id = str(gold_documents[0])
            if document_id.startswith("doc-") and "-" in document_id[4:]:
                category = document_id[4:].rsplit("-", maxsplit=1)[0]
    return split, category, hop_count, distractor_count


def _budget_bucket(total_tokens: int | None, token_limit: int | None) -> str:
    if total_tokens is None or token_limit is None or token_limit <= 0:
        return "unknown"
    ratio = total_tokens / token_limit
    if ratio <= 0.25:
        return "0-25%"
    if ratio <= 0.50:
        return "25-50%"
    if ratio <= 0.75:
        return "50-75%"
    if ratio <= 1.0:
        return "75-100%"
    return "over-100%"


def _flatten_record(record: object) -> dict[str, object]:
    split, category, hop_count, distractor_count = _task_metadata(record)
    design_cell = _design(record)
    condition = _condition(record)
    correctness = _boolean(_value(record, "correctness", "success", "task_success"))
    retrieved_documents = _value(record, "initial_retrieved_documents")
    retrieved_chunks = _value(record, "initial_retrieved_chunks")
    gold_evidence = _value(record, "gold_evidence")
    accessed_evidence = _value(record, "accessed_evidence")
    tool_trace = _value(record, "tool_trace")
    token_accounting = _value(record, "token_accounting")
    parsed_answer = _value(record, "parsed_answer")
    citations = _value(record, "citations")

    total_tokens = _integer(
        _value(
            record,
            "total_tokens",
            "tokens",
            "token_accounting.total_tokens_estimated",
        )
    )
    input_tokens = _integer(
        _value(record, "input_tokens", "token_accounting.input_tokens_estimated")
    )
    output_tokens = _integer(
        _value(record, "output_tokens", "token_accounting.output_tokens_estimated")
    )
    token_limit = _integer(_value(record, "token_limit", "token_accounting.limit", "budget_limit"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    actual_calls = _integer(_value(record, "tool_call_count", "calls", "actual_tool_calls"))
    if actual_calls is None:
        actual_calls = len(_items(tool_trace))
    permitted_calls = _integer(_value(record, "max_tool_calls_permitted", "permitted_tool_calls"))
    if permitted_calls is None and design_cell and design_cell.startswith("max_calls_"):
        permitted_calls = _integer(design_cell.removeprefix("max_calls_"))

    parsed_answer_value = _json_value(parsed_answer) if not _is_missing(parsed_answer) else None
    parsed_answer_text = None
    parsed_answer_justification = None
    if isinstance(parsed_answer_value, Mapping):
        answer = parsed_answer_value.get("answer")
        justification = parsed_answer_value.get("justification")
        parsed_answer_text = str(answer) if answer is not None else None
        parsed_answer_justification = str(justification) if justification is not None else None

    tool_counts = {
        name: _integer(
            _value(
                record,
                f"tool_calls_by_type.{name}",
                f"{name}_calls",
            )
        )
        for name in ("grep", "read", "cat", "list")
    }
    latencies = {
        name: _number(_value(record, f"latency.{name}_ms", f"{name}_ms"))
        for name in ("wall", "model", "retrieval", "tool")
    }
    initial_chunk_recall = _number(
        _value(
            record,
            "initial_chunk_recall",
            "initial_recall",
            "retrieval_metrics.initial_chunk_recall",
        )
    )
    initial_document_recall = _number(
        _value(
            record,
            "initial_document_recall",
            "retrieval_metrics.initial_document_recall",
        )
    )
    final_recall = _number(
        _value(
            record,
            "final_evidence_recall",
            "final_recall",
            "evidence_metrics.final_recall",
        )
    )
    bucket = _budget_bucket(total_tokens, token_limit)

    row: dict[str, object] = {column: None for column in _RUN_FRAME_COLUMNS}
    row.update(
        {
            "run_key": _text_value(record, "run_key"),
            "experiment_id": _text_value(record, "experiment_id"),
            "task_id": _text_value(record, "task_id", "task.task_id"),
            "split": split,
            "condition": condition,
            "design_cell": design_cell,
            "condition_order": _integer(_value(record, "condition_order")),
            "question": _text_value(record, "question", "task.question"),
            "corpus_version": _text_value(record, "corpus_version", "task.corpus_version"),
            "correctness": correctness,
            "success": correctness,
            "final_answer": _value(record, "final_answer"),
            "parsed_answer": _json_cell(parsed_answer),
            "parsed_answer_json": _json_cell(parsed_answer),
            "parsed_answer_text": parsed_answer_text,
            "parsed_answer_justification": parsed_answer_justification,
            "citations": _json_cell(citations),
            "citations_json": _json_cell(citations),
            "termination_reason": _enum_text(_value(record, "termination_reason")),
            "initial_retrieved_document_count": len(_items(retrieved_documents)),
            "initial_retrieved_chunk_count": len(_items(retrieved_chunks)),
            "initial_chunk_recall": initial_chunk_recall,
            "initial_document_recall": initial_document_recall,
            "initial_recall": initial_chunk_recall,
            "retrieval_k": _integer(_value(record, "retrieval_metrics.k", "retrieval_k")),
            "gold_fact_count": _list_count(record, "gold_evidence.fact_ids"),
            "gold_document_count": _list_count(record, "gold_evidence.doc_ids"),
            "gold_chunk_count": _list_count(record, "gold_evidence.chunk_ids"),
            "accessed_fact_count": _list_count(record, "accessed_evidence.fact_ids"),
            "accessed_document_count": _list_count(record, "accessed_evidence.doc_ids"),
            "accessed_chunk_count": _list_count(record, "accessed_evidence.chunk_ids"),
            "accessed_line_range_count": _list_count(record, "accessed_evidence.line_ranges"),
            "final_evidence_recall": final_recall,
            "final_recall": final_recall,
            "evidence_precision": _number(
                _value(record, "evidence_precision", "evidence_metrics.precision")
            ),
            "distinct_gold_facts": _integer(
                _value(record, "distinct_gold_facts", "evidence_metrics.distinct_gold_facts")
            ),
            "all_required_accessed": _boolean(
                _value(record, "all_required_accessed", "evidence_metrics.all_required_accessed")
            ),
            "citation_precision": _number(
                _value(record, "citation_precision", "evidence_metrics.citation_precision")
            ),
            "citation_recall": _number(
                _value(record, "citation_recall", "evidence_metrics.citation_recall")
            ),
            "tool_call_count": actual_calls,
            "calls": actual_calls,
            "max_tool_calls_permitted": permitted_calls,
            "permitted_tool_calls": permitted_calls,
            "successful_tool_calls": _integer(_value(record, "successful_tool_calls")),
            "invalid_tool_calls": _integer(_value(record, "invalid_tool_calls")),
            "repeated_tool_calls": _integer(_value(record, "repeated_tool_calls")),
            "grep_calls": tool_counts["grep"],
            "read_calls": tool_counts["read"],
            "cat_calls": tool_counts["cat"],
            "list_calls": tool_counts["list"],
            "unique_files_read_count": _list_count(record, "unique_files_read"),
            "unique_line_ranges_accessed_count": _list_count(
                record,
                "unique_line_ranges_accessed",
            ),
            "time_to_first_gold_fact_ms": _number(_value(record, "time_to_first_gold_fact_ms")),
            "calls_to_complete_gold_coverage": _integer(
                _value(record, "calls_to_complete_gold_coverage")
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens": total_tokens,
            "token_limit": token_limit,
            "wall_ms": latencies["wall"],
            "model_ms": latencies["model"],
            "retrieval_ms": latencies["retrieval"],
            "tool_ms": latencies["tool"],
            "estimated_cost_usd": _number(_value(record, "estimated_cost_usd", "cost_usd")),
            "hop_count": hop_count,
            "hops": hop_count,
            "category": category,
            "task_category": category,
            "distractor_count": distractor_count,
            "distractors": distractor_count,
            "budget_bucket": bucket,
            "token_budget_bucket": bucket,
            "initial_retrieved_documents_json": _json_cell(retrieved_documents),
            "initial_retrieved_chunks_json": _json_cell(retrieved_chunks),
            "gold_evidence_json": _json_cell(gold_evidence),
            "accessed_evidence_json": _json_cell(accessed_evidence),
            "tool_trace_json": _json_cell(tool_trace),
            "token_turns_json": _json_cell(_value(token_accounting, "turns")),
            "errors_json": _json_cell(_value(record, "errors")),
            "status": "ok",
        }
    )
    return row


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=_RUN_FRAME_COLUMNS)
    result = frame.copy()
    result["__design_rank"] = result["design_cell"].map(_design_rank)
    result["__condition_rank"] = result["condition"].map(_condition_rank)
    result = result.sort_values(
        [
            "task_id",
            "__design_rank",
            "__condition_rank",
            "condition_order",
            "run_key",
        ],
        kind="mergesort",
        na_position="last",
    )
    return result.drop(columns=["__design_rank", "__condition_rank"]).reset_index(drop=True)


def build_run_frame(records: Iterable[object]) -> pd.DataFrame:
    """Flatten run records into a deterministic, analysis-ready DataFrame."""

    rows = [_flatten_record(record) for record in _normalise_records(records)]
    frame = pd.DataFrame(rows, columns=_RUN_FRAME_COLUMNS)
    return _sort_frame(frame)


def _empty_frame(columns: Sequence[str], status: str = "not_run") -> pd.DataFrame:
    row = {column: None for column in columns}
    row["status"] = status
    return pd.DataFrame([row], columns=columns)


def _frame_rows(
    frame: pd.DataFrame | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(frame, pd.DataFrame):
        if frame.empty:
            return []
        return frame.to_dict(orient="records")
    return [dict(row) for row in frame]


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total_tokens = int(_sum(rows, "total_tokens"))
    successes = sum(1 for row in rows if _boolean(row.get("success")) is True)
    n = len(rows)
    total_cost = _sum(rows, "estimated_cost_usd")
    actual_calls = int(_sum(rows, "tool_call_count"))
    required_access = [
        _boolean(row.get("all_required_accessed"))
        for row in rows
        if _boolean(row.get("all_required_accessed")) is not None
    ]
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "initial_chunk_recall_mean": _mean(rows, "initial_chunk_recall"),
        "initial_document_recall_mean": _mean(rows, "initial_document_recall"),
        "initial_retrieved_document_count_mean": _mean(
            rows,
            "initial_retrieved_document_count",
        ),
        "initial_retrieved_chunk_count_mean": _mean(rows, "initial_retrieved_chunk_count"),
        "retrieval_k_mean": _mean(rows, "retrieval_k"),
        "final_evidence_recall_mean": _mean(rows, "final_evidence_recall"),
        "evidence_precision_mean": _mean(rows, "evidence_precision"),
        "distinct_gold_facts_mean": _mean(rows, "distinct_gold_facts"),
        "accessed_fact_count_mean": _mean(rows, "accessed_fact_count"),
        "accessed_document_count_mean": _mean(rows, "accessed_document_count"),
        "accessed_chunk_count_mean": _mean(rows, "accessed_chunk_count"),
        "accessed_line_range_count_mean": _mean(rows, "accessed_line_range_count"),
        "all_required_accessed_rate": (
            sum(required_access) / len(required_access) if required_access else None
        ),
        "citation_precision_mean": _mean(rows, "citation_precision"),
        "citation_recall_mean": _mean(rows, "citation_recall"),
        "actual_tool_calls": actual_calls,
        "actual_tool_calls_mean": actual_calls / n if n else None,
        "successful_tool_calls_mean": _mean(rows, "successful_tool_calls"),
        "invalid_tool_calls_mean": _mean(rows, "invalid_tool_calls"),
        "repeated_tool_calls_mean": _mean(rows, "repeated_tool_calls"),
        "grep_calls_mean": _mean(rows, "grep_calls"),
        "read_calls_mean": _mean(rows, "read_calls"),
        "cat_calls_mean": _mean(rows, "cat_calls"),
        "list_calls_mean": _mean(rows, "list_calls"),
        "unique_files_read_count_mean": _mean(rows, "unique_files_read_count"),
        "unique_line_ranges_accessed_count_mean": _mean(
            rows,
            "unique_line_ranges_accessed_count",
        ),
        "time_to_first_gold_fact_ms_mean": _mean(rows, "time_to_first_gold_fact_ms"),
        "calls_to_complete_gold_coverage_mean": _mean(
            rows,
            "calls_to_complete_gold_coverage",
        ),
        "total_tokens": total_tokens,
        "total_tokens_mean": total_tokens / n if n else None,
        "input_tokens": int(_sum(rows, "input_tokens")),
        "output_tokens": int(_sum(rows, "output_tokens")),
        "wall_ms_mean": _mean(rows, "wall_ms"),
        "model_ms_mean": _mean(rows, "model_ms"),
        "retrieval_ms_mean": _mean(rows, "retrieval_ms"),
        "tool_ms_mean": _mean(rows, "tool_ms"),
        "total_cost_usd": total_cost,
        "mean_cost_usd": total_cost / n if n else None,
        "success_per_1000_tokens": successes * 1000 / total_tokens if total_tokens else None,
        "status": "ok" if rows else "not_run",
    }


def _group_rows(
    frame: pd.DataFrame,
    keys: Sequence[str],
) -> list[tuple[tuple[object, ...], list[dict[str, object]]]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in _frame_rows(frame):
        groups[tuple(row.get(key) for key in keys)].append(row)

    def sort_key(item: tuple[tuple[object, ...], list[dict[str, object]]]) -> tuple[object, ...]:
        values = item[0]
        result: list[object] = []
        for key, value in zip(keys, values, strict=True):
            if key in {"condition"}:
                result.append(_condition_rank(value))
            elif key in {"design_cell"}:
                result.append(_design_rank(value))
            elif key in {"budget_bucket"}:
                result.append(_bucket_rank(value))
            elif key in {"hop_count", "level", "permitted_call_limit"}:
                result.append((0, value) if isinstance(value, (int, float)) else (1, str(value)))
            else:
                result.append("" if _is_missing(value) else str(value))
        return tuple(result)

    return sorted(groups.items(), key=sort_key)


def _condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, design_cell), group in _group_rows(frame, ("condition", "design_cell")):
        rows.append(
            {
                "condition": condition,
                "design_cell": design_cell,
                **_metric_summary(group),
            }
        )
    if not rows:
        return _empty_frame(_CONDITION_COLUMNS)
    return pd.DataFrame(rows, columns=_CONDITION_COLUMNS)


def _subgroup_summary(frame: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for subgroup, source in (("hop_count", "hop_count"), ("category", "category")):
        subgroup_frame = frame[frame[source].notna()]
        for (level, condition, design_cell), group in _group_rows(
            subgroup_frame,
            (source, "condition", "design_cell"),
        ):
            output.append(
                {
                    "subgroup": subgroup,
                    "level": level,
                    "condition": condition,
                    "design_cell": design_cell,
                    **_metric_summary(group),
                }
            )
    if not output:
        return _empty_frame(_SUBGROUP_COLUMNS)
    result = pd.DataFrame(output, columns=_SUBGROUP_COLUMNS)
    result["__subgroup_rank"] = result["subgroup"].map({"hop_count": 0, "category": 1})
    result["__condition_rank"] = result["condition"].map(_condition_rank)
    result = result.sort_values(
        ["__subgroup_rank", "level", "__condition_rank", "design_cell"],
        kind="mergesort",
        na_position="last",
    )
    return result.drop(columns=["__subgroup_rank", "__condition_rank"]).reset_index(drop=True)


def _intervention_summary(frame: pd.DataFrame) -> pd.DataFrame:
    intervention = frame[
        frame["design_cell"].astype("string").str.startswith("max_calls_", na=False)
        & (frame["condition"] == "vfs")
    ].copy()
    if intervention.empty:
        return _empty_frame(_INTERVENTION_COLUMNS)
    intervention["permitted_call_limit"] = intervention["max_tool_calls_permitted"]
    missing_limit = intervention["permitted_call_limit"].isna()
    if missing_limit.any():
        intervention.loc[missing_limit, "permitted_call_limit"] = (
            intervention.loc[
                missing_limit,
                "design_cell",
            ]
            .str.removeprefix("max_calls_")
            .astype(int)
        )
    output: list[dict[str, object]] = []
    for (limit, design_cell), group in _group_rows(
        intervention,
        ("permitted_call_limit", "design_cell"),
    ):
        output.append(
            {
                "condition": "vfs",
                "permitted_call_limit": _integer(limit),
                "design_cell": design_cell,
                **_metric_summary(group),
            }
        )
    return pd.DataFrame(output, columns=_INTERVENTION_COLUMNS)


def _efficiency_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, budget_bucket), group in _group_rows(
        frame,
        ("condition", "budget_bucket"),
    ):
        rows.append(
            {
                "condition": condition,
                "budget_bucket": budget_bucket,
                **_metric_summary(group),
            }
        )
    if not rows:
        return _empty_frame(_EFFICIENCY_COLUMNS)
    return pd.DataFrame(rows, columns=_EFFICIENCY_COLUMNS)


def _primary_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    primary = frame[(frame["design_cell"] == "primary") & frame["condition"].isin(_CONDITION_ORDER)]
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    duplicate = False
    for row in _frame_rows(primary):
        task_id = str(row.get("task_id") or "")
        condition = str(row.get("condition") or "")
        if condition in grouped[task_id]:
            duplicate = True
        else:
            grouped[task_id][condition] = row
    complete = [
        grouped[task_id]
        for task_id in sorted(grouped)
        if set(grouped[task_id]) == set(_CONDITION_ORDER)
    ]
    rows = [row for pair in complete for row in pair.values()]
    status = "duplicate_primary_record" if duplicate else None
    if grouped and len(complete) != len(grouped):
        status = status or "incomplete_pairs"
    return pd.DataFrame(rows, columns=_RUN_FRAME_COLUMNS), status


def _paired_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    paired_frame, pair_status = _primary_pairs(frame)
    if paired_frame.empty:
        return _empty_frame(_PAIRED_COLUMNS)
    pairs: dict[str, dict[str, object]] = defaultdict(dict)
    for row in _frame_rows(paired_frame):
        pairs[str(row["task_id"])][str(row["condition"])] = row

    outcome_counts = {
        "both_succeed": 0,
        "snippets_only": 0,
        "vfs_only": 0,
        "both_fail": 0,
    }
    for pair in pairs.values():
        snippets = _boolean(pair["snippets"].get("success")) is True
        vfs = _boolean(pair["vfs"].get("success")) is True
        if snippets and vfs:
            outcome_counts["both_succeed"] += 1
        elif snippets:
            outcome_counts["snippets_only"] += 1
        elif vfs:
            outcome_counts["vfs_only"] += 1
        else:
            outcome_counts["both_fail"] += 1

    summary: dict[str, object] = {}
    status = pair_status or "ok"
    try:
        stats = paired_summary(paired_frame)
        summary = stats.as_dict()
    except (TypeError, ValueError, KeyError) as error:
        status = f"error: {error}"

    n_pairs = len(pairs)
    rows = []
    for outcome in ("both_succeed", "snippets_only", "vfs_only", "both_fail"):
        rows.append(
            {
                "outcome": outcome,
                "count": outcome_counts[outcome],
                "n": n_pairs,
                "rate": outcome_counts[outcome] / n_pairs if n_pairs else None,
                "snippets_successes": summary.get("snippets_successes"),
                "vfs_successes": summary.get("vfs_successes"),
                "snippets_success_rate": summary.get("snippets_success_rate"),
                "vfs_success_rate": summary.get("vfs_success_rate"),
                "difference": summary.get("difference"),
                "ci_lower": summary.get("ci_lower"),
                "ci_upper": summary.get("ci_upper"),
                "snippets_only": summary.get("snippets_only"),
                "vfs_only": summary.get("vfs_only"),
                "mcnemar_p_value": summary.get("mcnemar_p_value"),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=_PAIRED_COLUMNS)


def _task_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(_TASK_LEVEL_COLUMNS)
    output: list[dict[str, object]] = []
    for (design_cell, task_id), group in _group_rows(frame, ("design_cell", "task_id")):
        by_condition = {
            str(row.get("condition")): row
            for row in _frame_rows(group)
            if row.get("condition") in _CONDITION_ORDER
        }
        snippets = by_condition.get("snippets", {})
        vfs = by_condition.get("vfs", {})
        snippets_success = _boolean(snippets.get("success"))
        vfs_success = _boolean(vfs.get("success"))
        if snippets_success is True and vfs_success is True:
            paired_outcome = "both_succeed"
        elif snippets_success is True and vfs_success is False:
            paired_outcome = "snippets_only"
        elif snippets_success is False and vfs_success is True:
            paired_outcome = "vfs_only"
        elif snippets_success is False and vfs_success is False:
            paired_outcome = "both_fail"
        else:
            paired_outcome = None
        output.append(
            {
                "design_cell": design_cell,
                "task_id": task_id,
                "split": snippets.get("split") or vfs.get("split"),
                "category": snippets.get("category") or vfs.get("category"),
                "hop_count": snippets.get("hop_count") or vfs.get("hop_count"),
                "distractor_count": (
                    snippets.get("distractor_count")
                    if snippets.get("distractor_count") is not None
                    else vfs.get("distractor_count")
                ),
                "snippets_success": snippets_success,
                "vfs_success": vfs_success,
                "paired_outcome": paired_outcome,
                "snippets_total_tokens": snippets.get("total_tokens"),
                "vfs_total_tokens": vfs.get("total_tokens"),
                "snippets_initial_recall": snippets.get("initial_recall"),
                "vfs_initial_recall": vfs.get("initial_recall"),
                "snippets_final_recall": snippets.get("final_recall"),
                "vfs_final_recall": vfs.get("final_recall"),
                "snippets_actual_tool_calls": snippets.get("tool_call_count"),
                "vfs_actual_tool_calls": vfs.get("tool_call_count"),
                "snippets_permitted_tool_calls": snippets.get("max_tool_calls_permitted"),
                "vfs_permitted_tool_calls": vfs.get("max_tool_calls_permitted"),
                "snippets_wall_ms": snippets.get("wall_ms"),
                "vfs_wall_ms": vfs.get("wall_ms"),
                "snippets_cost_usd": snippets.get("estimated_cost_usd"),
                "vfs_cost_usd": vfs.get("estimated_cost_usd"),
                "status": ("ok" if snippets and vfs else "incomplete_pair"),
            }
        )
    return pd.DataFrame(output, columns=_TASK_LEVEL_COLUMNS)


def _stratified_summary(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        result = stratified_summaries(frame)
    except (TypeError, ValueError, KeyError) as error:
        return _empty_frame(_STRATIFIED_COLUMNS, status=f"error: {error}")
    result = result.reindex(columns=_STRATIFIED_COLUMNS[:-1])
    result["status"] = "ok"
    return result.reindex(columns=_STRATIFIED_COLUMNS)


def _regression_summary(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        result = within_vfs_regression(frame)
    except (TypeError, ValueError, KeyError, RuntimeError) as error:
        return _empty_frame(_REGRESSION_COLUMNS, status=f"error: {error}")
    for column in _REGRESSION_COLUMNS:
        if column not in result:
            result[column] = None
    return result.reindex(columns=_REGRESSION_COLUMNS)


def _failure_summary(records: Sequence[object]) -> pd.DataFrame:
    counts: dict[tuple[str | None, str | None, str], int] = defaultdict(int)
    denominators: dict[tuple[str | None, str | None], int] = defaultdict(int)
    for record in records:
        condition = _condition(record)
        design_cell = _design(record)
        denominators[(condition, design_cell)] += 1
        for failure in classify_failure(record):
            counts[(condition, design_cell, str(failure.value))] += 1
    if not counts:
        return _empty_frame(_FAILURE_COLUMNS)
    rows = []
    for (condition, design_cell, failure_class), count in sorted(
        counts.items(),
        key=lambda item: (
            _condition_rank(item[0][0]),
            _design_rank(item[0][1]),
            list(FailureClass).index(FailureClass(item[0][2]))
            if item[0][2] in {failure.value for failure in FailureClass}
            else len(counts),
        ),
    ):
        n = denominators[(condition, design_cell)]
        rows.append(
            {
                "condition": condition,
                "design_cell": design_cell,
                "failure_class": failure_class,
                "count": count,
                "n": n,
                "rate": count / n if n else None,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows, columns=_FAILURE_COLUMNS)


def _write_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> Path:
    output = frame.reindex(columns=columns)
    output.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        na_rep="",
    )
    return path


def _write_trace_json(records: Sequence[object], path: Path) -> Path:
    payload = export_trace_examples(records)
    safe_payload = _json_value(payload)
    path.write_text(
        canonical_json(safe_payload) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_aggregate_tables(
    records: Iterable[object],
    out_dir: Path | str,
    *,
    include_design: str = "primary",
) -> list[Path]:
    """Write all stable analysis tables and return paths in stable order.

    General summaries use the primary design by default.  Intervention rows
    are kept in their own table and are never grouped by actual tool calls.
    Passing ``include_design="all"`` explicitly includes oracle and
    intervention rows in the general descriptive tables.
    """

    record_rows = _normalise_records(records)
    target = Path(os.path.expanduser(os.fspath(out_dir)))
    target.mkdir(parents=True, exist_ok=True)
    scoped_records = _scope_records(record_rows, include_design)
    scoped_frame = build_run_frame(scoped_records)
    primary_frame = build_run_frame(_scope_records(record_rows, "primary"))
    intervention_frame = build_run_frame(_scope_records(record_rows, "intervention"))

    outputs: list[Path] = []
    outputs.append(
        _write_csv(
            _paired_outcomes(primary_frame),
            target / "paired_outcomes.csv",
            _PAIRED_COLUMNS,
        )
    )
    outputs.append(
        _write_csv(
            _condition_summary(scoped_frame),
            target / "condition_summary.csv",
            _CONDITION_COLUMNS,
        )
    )
    outputs.append(
        _write_csv(
            _subgroup_summary(scoped_frame),
            target / "subgroup_summary.csv",
            _SUBGROUP_COLUMNS,
        )
    )
    outputs.append(
        _write_csv(
            _intervention_summary(intervention_frame),
            target / "intervention_summary.csv",
            _INTERVENTION_COLUMNS,
        )
    )
    outputs.append(
        _write_csv(
            _regression_summary(primary_frame),
            target / "within_vfs_regression.csv",
            _REGRESSION_COLUMNS,
        )
    )
    outputs.append(
        _write_csv(
            _efficiency_summary(scoped_frame),
            target / "efficiency_summary.csv",
            _EFFICIENCY_COLUMNS,
        )
    )
    outputs.append(_write_trace_json(scoped_records, target / "trace_examples.json"))
    outputs.append(
        _write_csv(
            _task_level(scoped_frame),
            target / "task_level.csv",
            _TASK_LEVEL_COLUMNS,
        )
    )
    outputs.append(
        _write_csv(
            _stratified_summary(primary_frame),
            target / "stratified_summary.csv",
            _STRATIFIED_COLUMNS,
        )
    )
    outputs.append(
        _write_csv(
            _failure_summary(scoped_records),
            target / "failure_summary.csv",
            _FAILURE_COLUMNS,
        )
    )
    return outputs


__all__ = ["build_run_frame", "write_aggregate_tables"]
