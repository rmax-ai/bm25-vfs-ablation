from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bm25_vfs_ablation.evaluation.aggregates import (
    build_run_frame,
    write_aggregate_tables,
)


def _record(
    task_id: str,
    condition: str,
    *,
    correctness: bool,
    design_cell: str = "primary",
    tokens: int = 100,
    actual_calls: int = 0,
    permitted_calls: int | None = None,
    repeats: int = 0,
) -> dict[str, object]:
    return {
        "run_key": f"exp-test-42:{task_id}:{condition}:{design_cell}",
        "experiment_id": "exp-test-42",
        "task_id": task_id,
        "condition": condition,
        "design_cell": design_cell,
        "condition_order": 0 if condition == "snippets" else 1,
        "question": f"Question for {task_id}.",
        "corpus_version": "synthetic-v1",
        "correctness": correctness,
        "final_answer": "answer",
        "parsed_answer": {
            "answer": "answer",
            "citations": ["doc-ops-0001"],
            "justification": "Public justification.",
        },
        "citations": ["doc-ops-0001"],
        "initial_retrieved_documents": [{"doc_id": "doc-ops-0001"}],
        "initial_retrieved_chunks": [{"chunk_id": "doc-ops-0001::c0000"}],
        "gold_evidence": {
            "fact_ids": ["fact-000001", "fact-000002"],
            "doc_ids": ["doc-ops-0001"],
            "chunk_ids": ["doc-ops-0001::c0000"],
        },
        "accessed_evidence": {
            "fact_ids": ["fact-000001"],
            "doc_ids": ["doc-ops-0001"],
            "chunk_ids": ["doc-ops-0001::c0000"],
            "line_ranges": [],
        },
        "retrieval_metrics": {
            "initial_chunk_recall": 0.5,
            "initial_document_recall": 1.0,
            "k": 8,
        },
        "evidence_metrics": {
            "final_recall": 0.5,
            "precision": 1.0,
            "distinct_gold_facts": 1,
            "all_required_accessed": False,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
        },
        "tool_trace": [],
        "tool_call_count": actual_calls,
        "max_tool_calls_permitted": permitted_calls,
        "tool_calls_by_type": {"grep": 0, "read": 0, "cat": 0, "list": 0},
        "successful_tool_calls": actual_calls,
        "invalid_tool_calls": 0,
        "repeated_tool_calls": repeats,
        "unique_files_read": [],
        "unique_line_ranges_accessed": [],
        "time_to_first_gold_fact_ms": None,
        "calls_to_complete_gold_coverage": None,
        "token_accounting": {
            "limit": 1000,
            "input_tokens": tokens // 2,
            "output_tokens": tokens - tokens // 2,
            "total_tokens_estimated": tokens,
            "turns": [],
        },
        "total_tokens": tokens,
        "latency": {
            "wall_ms": 10.0,
            "model_ms": 8.0,
            "retrieval_ms": 1.0,
            "tool_ms": 1.0,
        },
        "estimated_cost_usd": 0.001,
        "termination_reason": "answered",
        "errors": [],
    }


def _records() -> list[dict[str, object]]:
    outcomes = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    rows: list[dict[str, object]] = []
    for index, (snippets, vfs) in enumerate(outcomes, start=1):
        task_id = f"task-eval-{index:06d}"
        rows.extend(
            [
                _record(task_id, "snippets", correctness=snippets, tokens=100 + index),
                _record(
                    task_id,
                    "vfs",
                    correctness=vfs,
                    tokens=200 + index,
                    actual_calls=index,
                    permitted_calls=12,
                    repeats=3 if index == 3 else 0,
                ),
            ]
        )
    rows.append(
        _record(
            "task-eval-000001",
            "vfs",
            correctness=True,
            design_cell="max_calls_2",
            tokens=120,
            actual_calls=9,
            permitted_calls=2,
        )
    )
    return rows


def _read_csv(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(directory / name)


def test_run_frame_flattens_complete_schema() -> None:
    frame = build_run_frame([_record("task-eval-000001", "vfs", correctness=True, tokens=250)])

    assert frame.loc[0, "condition"] == "vfs"
    assert frame.loc[0, "initial_chunk_recall"] == 0.5
    assert frame.loc[0, "final_evidence_recall"] == 0.5
    assert frame.loc[0, "tool_call_count"] == 0
    assert frame.loc[0, "total_tokens"] == 250
    assert frame.loc[0, "hop_count"] == 2
    assert frame.loc[0, "category"] == "ops"


def test_paired_outcome_counts(tmp_path: Path) -> None:
    write_aggregate_tables(_records(), tmp_path)
    table = _read_csv(tmp_path, "paired_outcomes.csv")

    assert dict(zip(table["outcome"], table["count"], strict=True)) == {
        "both_succeed": 1,
        "snippets_only": 1,
        "vfs_only": 1,
        "both_fail": 1,
    }


def test_success_per_thousand_tokens(tmp_path: Path) -> None:
    records = [
        _record("task-eval-000001", "snippets", correctness=True, tokens=100),
        _record("task-eval-000002", "snippets", correctness=False, tokens=100),
    ]
    write_aggregate_tables(records, tmp_path)
    table = _read_csv(tmp_path, "efficiency_summary.csv")
    row = table[table["condition"] == "snippets"].iloc[0]

    assert row["success_per_1000_tokens"] == 5.0


def test_intervention_uses_permitted_not_actual_calls(tmp_path: Path) -> None:
    records = [
        _record(
            "task-eval-000001",
            "vfs",
            correctness=True,
            design_cell="max_calls_2",
            actual_calls=9,
            permitted_calls=2,
        ),
        _record(
            "task-eval-000002",
            "vfs",
            correctness=False,
            design_cell="max_calls_2",
            actual_calls=1,
            permitted_calls=2,
        ),
    ]
    write_aggregate_tables(records, tmp_path)
    table = _read_csv(tmp_path, "intervention_summary.csv")

    assert table["permitted_call_limit"].tolist() == [2]
    assert table.loc[0, "actual_tool_calls"] == 10


def test_csv_order_and_newlines(tmp_path: Path) -> None:
    paths = write_aggregate_tables(_records(), tmp_path)
    names = [path.name for path in paths]

    assert names == [
        "paired_outcomes.csv",
        "condition_summary.csv",
        "subgroup_summary.csv",
        "intervention_summary.csv",
        "within_vfs_regression.csv",
        "efficiency_summary.csv",
        "trace_examples.json",
        "task_level.csv",
        "stratified_summary.csv",
        "failure_summary.csv",
    ]
    csv_bytes = (tmp_path / "condition_summary.csv").read_bytes()
    assert b"\r" not in csv_bytes
    assert csv_bytes.startswith(b"condition,design_cell,n,successes,success_rate,")


def test_empty_optional_analysis_writes_status(tmp_path: Path) -> None:
    write_aggregate_tables([], tmp_path)

    for name in (
        "paired_outcomes.csv",
        "intervention_summary.csv",
        "within_vfs_regression.csv",
        "stratified_summary.csv",
        "failure_summary.csv",
    ):
        table = _read_csv(tmp_path, name)
        assert "status" in table.columns
        assert table.loc[0, "status"]


def test_trace_examples_json_safe(tmp_path: Path) -> None:
    write_aggregate_tables(_records(), tmp_path)
    payload = json.loads((tmp_path / "trace_examples.json").read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    assert "vfs_win" in payload
    assert "snippet_win" in payload
    assert "both_fail" in payload
    json.dumps(payload, allow_nan=False)
