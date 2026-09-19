"""Deterministic failure labels and observable trace-example selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

from bm25_vfs_ablation import FailureClass
from bm25_vfs_ablation.experiment.schemas import RunRecord

_MISSING = object()
_PRIMARY = "primary"
_TRACE_EXAMPLE_KEYS = (
    "vfs_win",
    "snippet_win",
    "both_fail",
    "excessive_tool_use",
)


def _field(value: object, name: str, default: object = _MISSING) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _path_field(value: object, path: str, default: object = _MISSING) -> object:
    current = value
    for part in path.split("."):
        current = _field(current, part, _MISSING)
        if current is _MISSING:
            return default
    return current


def _number(value: object) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _integer(value: object) -> int | None:
    result = _number(value)
    if result is None or result < 0 or not result.is_integer():
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
    return None


def _text(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    enum_value = getattr(value, "value", value)
    return str(enum_value).strip().casefold()


def _first_number(record: object, paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value = _number(_path_field(record, path))
        if value is not None:
            return value
    return None


def _collection(value: object) -> tuple[object, ...]:
    if value is _MISSING or value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)


def _condition(record: object) -> str | None:
    return _text(_field(record, "condition"))


def _design_cell(record: object) -> str:
    return _text(_field(record, "design_cell")) or _PRIMARY


def _task_id(record: object) -> str:
    value = _field(record, "task_id", "")
    return str(value)


def _correct(record: object) -> bool:
    value = _boolean(_field(record, "correctness"))
    return value is True


def _initial_recall(record: object) -> float | None:
    recall = _first_number(
        record,
        (
            "retrieval_metrics.initial_chunk_recall",
            "initial_chunk_recall",
            "initial_retrieval_recall",
            "retrieval_metrics.initial_document_recall",
            "initial_document_recall",
        ),
    )
    if recall is not None:
        return recall

    retrieved = _collection(_field(record, "initial_retrieved_chunks"))
    gold = _collection(_path_field(record, "gold_evidence.chunk_ids"))
    if gold:
        return 0.0 if not retrieved else None
    return None


def _final_recall(record: object) -> float | None:
    return _first_number(
        record,
        (
            "evidence_metrics.final_recall",
            "final_evidence_recall",
            "evidence_recall",
        ),
    )


def _all_required_accessed(record: object) -> bool:
    explicit = _boolean(_path_field(record, "evidence_metrics.all_required_accessed"))
    if explicit is not None:
        return explicit
    recall = _final_recall(record)
    return recall is not None and recall >= 1.0


def _citation_precision(record: object) -> float:
    value = _first_number(
        record,
        (
            "evidence_metrics.citation_precision",
            "citation_precision",
            "scoring_details.citation_metrics.precision",
            "scoring_details.citation_precision",
        ),
    )
    if value is not None:
        return max(0.0, min(1.0, value))

    citations = {
        str(citation)
        for citation in _collection(_field(record, "citations"))
        if isinstance(citation, str)
    }
    gold = _path_field(record, "gold_evidence", _MISSING)
    gold_ids = {
        str(identifier)
        for name in ("doc_ids", "chunk_ids")
        for identifier in _collection(_field(gold, name))
    }
    if not citations:
        return 0.0
    return len(citations & gold_ids) / len(citations)


def _tool_call_count(record: object) -> int:
    value = _integer(_field(record, "tool_call_count"))
    if value is not None:
        return value
    return len(_collection(_field(record, "tool_trace")))


def _repeated_tool_calls(record: object) -> int:
    value = _integer(_field(record, "repeated_tool_calls"))
    if value is not None:
        return value
    return sum(
        1
        for entry in _collection(_field(record, "tool_trace"))
        if _boolean(_field(entry, "repeated")) is True
    )


def _invalid_tool_calls(record: object) -> int:
    value = _integer(_field(record, "invalid_tool_calls"))
    if value is not None:
        return value
    return sum(
        1
        for entry in _collection(_field(record, "tool_trace"))
        if _boolean(_field(_field(entry, "result"), "ok")) is False
    )


def _distinct_gold_facts(record: object) -> int | None:
    return _integer(_path_field(record, "evidence_metrics.distinct_gold_facts"))


def _vfs_no_new_gold_after_calls(record: object) -> bool:
    if _condition(record) != "vfs" or _tool_call_count(record) <= 0:
        return False

    distinct = _distinct_gold_facts(record)
    if distinct is not None:
        return distinct == 0

    accessed = _path_field(record, "accessed_evidence", _MISSING)
    if accessed is not _MISSING:
        accessed_facts = {
            str(fact_id)
            for fact_id in _collection(_field(accessed, "fact_ids"))
        }
        gold_facts = {
            str(fact_id)
            for fact_id in _collection(_path_field(record, "gold_evidence.fact_ids"))
        }
        if gold_facts:
            return not accessed_facts.intersection(gold_facts)
        return not accessed_facts

    final_recall = _final_recall(record)
    return final_recall == 0.0


def _budget_exhausted(record: object) -> bool:
    termination = _text(_field(record, "termination_reason"))
    if termination == "budget_exhausted":
        return True

    for error in _collection(_field(record, "errors")):
        code = _text(_field(error, "code"))
        if code in {"budget_exhausted", "budget_insufficient"}:
            return True
    return False


def _malformed_final_answer(record: object) -> bool:
    parsed = _field(record, "parsed_answer")
    parse_status = _path_field(record, "scoring_details.parse_status")
    if parse_status is _MISSING:
        parse_status = _field(record, "parse_status")
    if isinstance(parse_status, bool):
        return not parse_status
    if parse_status is not _MISSING and parse_status is not None:
        return _text(parse_status) not in {"valid", "ok", "parsed", "true"}
    return parsed is _MISSING or parsed is None


def _excessive_repeated_use(record: object) -> bool:
    calls = _tool_call_count(record)
    repeats = _repeated_tool_calls(record)
    return calls > 0 and (repeats >= 3 or repeats / calls >= 0.5)


def classify_failure(record: RunRecord | Mapping[str, Any]) -> list[FailureClass]:
    """Apply all applicable failure heuristics in frozen enum order.

    The labels describe observable dimensions rather than a mutually exclusive
    root-cause tree.  In particular, an all-evidence wrong answer with an
    unsupported citation receives both composition and unsupported-answer
    labels, while a citation-supported answer receives the wrong-conclusion
    label.
    """

    labels: set[FailureClass] = set()
    initial_recall = _initial_recall(record)
    final_recall = _final_recall(record)
    correct = _correct(record)
    all_evidence = _all_required_accessed(record)
    unsupported_citations = _citation_precision(record) < 1.0

    if initial_recall == 0.0:
        labels.add(FailureClass.INITIAL_RETRIEVAL_MISS)
    if _vfs_no_new_gold_after_calls(record):
        labels.add(FailureClass.FAILED_EXPLORATION)
    if final_recall is not None and final_recall < 1.0:
        labels.add(FailureClass.INCOMPLETE_EVIDENCE_COVERAGE)

    if not correct and all_evidence:
        if unsupported_citations:
            labels.add(FailureClass.INCORRECT_EVIDENCE_COMPOSITION)
        else:
            labels.add(FailureClass.CORRECT_EVIDENCE_WRONG_CONCLUSION)
    if not correct and unsupported_citations:
        labels.add(FailureClass.UNSUPPORTED_ANSWER)

    if _budget_exhausted(record):
        labels.add(FailureClass.BUDGET_EXHAUSTION)
    if _excessive_repeated_use(record):
        labels.add(FailureClass.EXCESSIVE_REPEATED_TOOL_USE)
    if _invalid_tool_calls(record) > 0:
        labels.add(FailureClass.INVALID_TOOL_CALL)
    if _malformed_final_answer(record):
        labels.add(FailureClass.MALFORMED_FINAL_ANSWER)

    return [failure for failure in FailureClass if failure in labels]


def _json_value(value: object) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    enum_value = getattr(value, "value", _MISSING)
    if enum_value is not _MISSING:
        return enum_value
    return value


def trace_example_payload(record: RunRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Return only public, observable fields for one trace example."""

    return {
        "question": _field(record, "question", ""),
        "observable_trace": _json_value(_field(record, "tool_trace", ())),
        "evidence": {
            "initial_retrieved_documents": _json_value(
                _field(record, "initial_retrieved_documents", ())
            ),
            "initial_retrieved_chunks": _json_value(
                _field(record, "initial_retrieved_chunks", ())
            ),
            "gold": _json_value(_field(record, "gold_evidence", {})),
            "accessed": _json_value(_field(record, "accessed_evidence", {})),
        },
        "answers": {
            "final": _field(record, "final_answer", None),
            "parsed": _json_value(_field(record, "parsed_answer", None)),
            "citations": _json_value(_field(record, "citations", ())),
            "correctness": _field(record, "correctness", False),
        },
        "metrics": {
            "retrieval": _json_value(_field(record, "retrieval_metrics", {})),
            "evidence": _json_value(_field(record, "evidence_metrics", {})),
            "tool_call_count": _field(record, "tool_call_count", 0),
            "successful_tool_calls": _field(record, "successful_tool_calls", 0),
            "invalid_tool_calls": _field(record, "invalid_tool_calls", 0),
            "repeated_tool_calls": _field(record, "repeated_tool_calls", 0),
            "total_tokens": _field(record, "total_tokens", 0),
            "latency": _json_value(_field(record, "latency", {})),
            "termination_reason": _json_value(
                _field(record, "termination_reason", None)
            ),
        },
    }


def _normalise_record(record: RunRecord | Mapping[str, Any]) -> RunRecord | Mapping[str, Any]:
    if isinstance(record, RunRecord):
        return record
    try:
        return RunRecord.model_validate(record)
    except (TypeError, ValueError):
        return record


def _primary_scope(
    records: tuple[RunRecord | Mapping[str, Any], ...],
) -> tuple[RunRecord | Mapping[str, Any], ...]:
    primary = tuple(record for record in records if _design_cell(record) == _PRIMARY)
    return primary or records


def _choose_record(
    records: Iterable[RunRecord | Mapping[str, Any]],
) -> RunRecord | Mapping[str, Any] | None:
    values = tuple(records)
    if not values:
        return None
    return min(
        values,
        key=lambda record: (
            _task_id(record),
            str(_field(record, "run_key", "")),
        ),
    )


def select_trace_examples(
    records: Iterable[RunRecord | Mapping[str, Any]],
) -> dict[str, RunRecord]:
    """Select deterministic paired and excessive-use representative records.

    Paired categories use the lexicographically smallest task ID.  The
    excessive-use category uses the greatest repeated-call count first, then
    task ID, and is restricted to unsuccessful VFS runs.
    """

    if isinstance(records, (str, bytes, bytearray, Mapping)):
        raise TypeError("records must be an iterable of run records")

    normalised = tuple(_normalise_record(record) for record in records)
    scoped = _primary_scope(normalised)
    by_task: dict[str, dict[str, RunRecord | Mapping[str, Any]]] = {}
    for record in scoped:
        condition = _condition(record)
        if condition not in {"snippets", "vfs"}:
            continue
        task = by_task.setdefault(_task_id(record), {})
        current = task.get(condition)
        if current is None or str(_field(record, "run_key", "")) < str(
            _field(current, "run_key", "")
        ):
            task[condition] = record

    examples: dict[str, RunRecord] = {}
    vfs_wins = (
        pair["vfs"]
        for pair in by_task.values()
        if "vfs" in pair
        and "snippets" in pair
        and _correct(pair["vfs"])
        and not _correct(pair["snippets"])
    )
    snippets_wins = (
        pair["snippets"]
        for pair in by_task.values()
        if "vfs" in pair
        and "snippets" in pair
        and _correct(pair["snippets"])
        and not _correct(pair["vfs"])
    )
    both_fail = (
        pair["vfs"]
        for pair in by_task.values()
        if "vfs" in pair
        and "snippets" in pair
        and not _correct(pair["vfs"])
        and not _correct(pair["snippets"])
    )

    for category, candidate_records in (
        ("vfs_win", vfs_wins),
        ("snippet_win", snippets_wins),
        ("both_fail", both_fail),
    ):
        selected = _choose_record(candidate_records)
        if selected is not None:
            examples[category] = selected  # type: ignore[assignment]

    excessive_candidates = [
        record
        for record in scoped
        if _condition(record) == "vfs"
        and not _correct(record)
        and _excessive_repeated_use(record)
    ]
    selected_excessive = min(
        excessive_candidates,
        key=lambda record: (
            -_repeated_tool_calls(record),
            _task_id(record),
            str(_field(record, "run_key", "")),
        ),
        default=None,
    )
    if selected_excessive is not None:
        examples["excessive_tool_use"] = selected_excessive  # type: ignore[assignment]

    return {
        category: examples[category]
        for category in _TRACE_EXAMPLE_KEYS
        if category in examples
    }


def export_trace_examples(
    records: Iterable[RunRecord | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select examples and export only their observable public payloads."""

    return {
        category: trace_example_payload(record)
        for category, record in select_trace_examples(records).items()
    }


__all__ = [
    "FailureClass",
    "classify_failure",
    "export_trace_examples",
    "select_trace_examples",
    "trace_example_payload",
]
