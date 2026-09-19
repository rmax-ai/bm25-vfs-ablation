from __future__ import annotations

from copy import deepcopy

from bm25_vfs_ablation import FailureClass
from bm25_vfs_ablation.evaluation.failures import (
    classify_failure,
    export_trace_examples,
    select_trace_examples,
)


def _record(
    task_id: str = "task-eval-000001",
    condition: str = "vfs",
    *,
    correctness: bool = False,
    initial_recall: float = 1.0,
    final_recall: float = 1.0,
    all_required_accessed: bool | None = None,
    citation_precision: float = 1.0,
    calls: int = 0,
    repeats: int = 0,
    invalid: int = 0,
    termination: str = "answered",
    parsed: bool = True,
) -> dict[str, object]:
    if all_required_accessed is None:
        all_required_accessed = final_recall == 1.0
    return {
        "task_id": task_id,
        "condition": condition,
        "design_cell": "primary",
        "question": f"Question for {task_id}.",
        "correctness": correctness,
        "initial_retrieved_documents": [],
        "initial_retrieved_chunks": [],
        "initial_recall": initial_recall,
        "retrieval_metrics": {
            "initial_chunk_recall": initial_recall,
            "initial_document_recall": initial_recall,
            "k": 8,
        },
        "evidence_metrics": {
            "final_recall": final_recall,
            "precision": 1.0,
            "distinct_gold_facts": int(final_recall > 0),
            "all_required_accessed": all_required_accessed,
            "citation_precision": citation_precision,
            "citation_recall": citation_precision,
        },
        "gold_evidence": {
            "fact_ids": ["fact-000001"],
            "doc_ids": ["doc-services-0001"],
            "chunk_ids": ["doc-services-0001::c0000"],
        },
        "accessed_evidence": {
            "fact_ids": ["fact-000001"] if final_recall > 0 else [],
            "doc_ids": ["doc-services-0001"] if final_recall > 0 else [],
            "chunk_ids": ["doc-services-0001::c0000"] if final_recall > 0 else [],
            "line_ranges": [],
        },
        "tool_trace": [],
        "tool_call_count": calls,
        "repeated_tool_calls": repeats,
        "invalid_tool_calls": invalid,
        "successful_tool_calls": max(0, calls - invalid),
        "total_tokens": 20,
        "latency": {"wall_ms": 1.0},
        "termination_reason": termination,
        "final_answer": '{"answer":"wrong"}' if parsed else "not-json",
        "parsed_answer": (
            {"answer": "wrong", "citations": [], "justification": "Public summary."}
            if parsed
            else None
        ),
        "citations": ["doc-services-0001"] if citation_precision == 1.0 else [],
        "scoring_details": {"parse_status": "valid" if parsed else "malformed"},
    }


def _with(record: dict[str, object], **updates: object) -> dict[str, object]:
    result = deepcopy(record)
    result.update(updates)
    return result


def test_initial_retrieval_miss_rule() -> None:
    labels = classify_failure(_record(initial_recall=0.0, final_recall=0.0))

    assert FailureClass.INITIAL_RETRIEVAL_MISS in labels


def test_incomplete_vs_composition_rules() -> None:
    incomplete = classify_failure(_record(final_recall=0.5, citation_precision=1.0))
    unsupported_composition = classify_failure(
        _record(final_recall=1.0, citation_precision=0.0)
    )
    supported_composition = classify_failure(
        _record(final_recall=1.0, citation_precision=1.0)
    )

    assert FailureClass.INCOMPLETE_EVIDENCE_COVERAGE in incomplete
    assert FailureClass.INCORRECT_EVIDENCE_COMPOSITION in unsupported_composition
    assert FailureClass.UNSUPPORTED_ANSWER in unsupported_composition
    assert FailureClass.CORRECT_EVIDENCE_WRONG_CONCLUSION in supported_composition


def test_budget_and_invalid_rules() -> None:
    labels = classify_failure(
        _record(
            calls=2,
            invalid=1,
            termination="budget_exhausted",
        )
    )

    assert FailureClass.BUDGET_EXHAUSTION in labels
    assert FailureClass.INVALID_TOOL_CALL in labels


def test_excessive_repeat_threshold() -> None:
    below_threshold = classify_failure(_record(calls=6, repeats=2))
    at_ratio_threshold = classify_failure(_record(calls=4, repeats=2))
    at_count_threshold = classify_failure(_record(calls=5, repeats=3))

    assert FailureClass.EXCESSIVE_REPEATED_TOOL_USE not in below_threshold
    assert FailureClass.EXCESSIVE_REPEATED_TOOL_USE in at_ratio_threshold
    assert FailureClass.EXCESSIVE_REPEATED_TOOL_USE in at_count_threshold


def test_failure_labels_stable_order() -> None:
    labels = classify_failure(
        _record(
            initial_recall=0.0,
            final_recall=0.0,
            calls=6,
            repeats=3,
            invalid=1,
            termination="budget_exhausted",
            parsed=False,
        )
    )
    order = {failure: index for index, failure in enumerate(FailureClass)}

    assert labels == sorted(labels, key=order.__getitem__)
    assert {
        FailureClass.INITIAL_RETRIEVAL_MISS,
        FailureClass.FAILED_EXPLORATION,
        FailureClass.INCOMPLETE_EVIDENCE_COVERAGE,
        FailureClass.BUDGET_EXHAUSTION,
        FailureClass.EXCESSIVE_REPEATED_TOOL_USE,
        FailureClass.INVALID_TOOL_CALL,
        FailureClass.MALFORMED_FINAL_ANSWER,
    }.issubset(labels)


def test_trace_example_categories() -> None:
    records = [
        _record("task-eval-000004", "vfs", correctness=False, calls=6, repeats=4),
        _record("task-eval-000004", "snippets", correctness=False),
        _record("task-eval-000002", "vfs", correctness=True, calls=1),
        _record("task-eval-000002", "snippets", correctness=False),
        _record("task-eval-000001", "vfs", correctness=False),
        _record("task-eval-000001", "snippets", correctness=False),
        _record("task-eval-000003", "vfs", correctness=False, calls=8, repeats=5),
        _record("task-eval-000003", "snippets", correctness=True),
    ]

    examples = select_trace_examples(records)

    assert set(examples) == {
        "vfs_win",
        "snippet_win",
        "both_fail",
        "excessive_tool_use",
    }
    assert examples["vfs_win"]["task_id"] == "task-eval-000002"
    assert examples["snippet_win"]["task_id"] == "task-eval-000003"
    assert examples["both_fail"]["task_id"] == "task-eval-000001"
    assert examples["excessive_tool_use"]["task_id"] == "task-eval-000003"


def test_examples_exclude_hidden_reasoning() -> None:
    record = _with(
        _record("task-eval-000001", "vfs", calls=2, repeats=1),
        scoring_details={
            "parse_status": "valid",
            "hidden_reasoning": "must not be exported",
        },
    )

    exported = export_trace_examples([record])

    assert set(exported) == {"excessive_tool_use"}
    payload = exported["excessive_tool_use"]
    assert set(payload) == {
        "question",
        "observable_trace",
        "evidence",
        "answers",
        "metrics",
    }
    assert "scoring_details" not in payload
    assert "hidden_reasoning" not in str(payload)
