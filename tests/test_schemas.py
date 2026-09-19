from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from bm25_vfs_ablation.corpus.schema import DocumentRecord, FactSpan, TaskRecord
from bm25_vfs_ablation.experiment.schemas import RunRecord, ToolTraceEntry


def _documents_and_task() -> tuple[DocumentRecord, DocumentRecord, TaskRecord]:
    first = DocumentRecord(
        schema_version=1,
        corpus_version="synthetic-v1",
        doc_id="doc-services-0001",
        path="/services/doc-services-0001.md",
        category="services",
        title="Atlas service",
        content="# Atlas\nAtlas depends on Quartz.\n",
        facts=[
            FactSpan(
                fact_id="fact-000001",
                subject="Atlas",
                predicate="depends_on",
                object="Quartz",
                line_start=2,
                line_end=2,
            )
        ],
        generator_seed=42,
    )
    second = DocumentRecord(
        schema_version=1,
        corpus_version="synthetic-v1",
        doc_id="doc-policies-0002",
        path="/policies/doc-policies-0002.md",
        category="policies",
        title="Security policy",
        content="# Review policy\nQuartz usage requires a security review.\n",
        facts=[
            FactSpan(
                fact_id="fact-000002",
                subject="Quartz",
                predicate="requires",
                object="security review",
                line_start=2,
                line_end=2,
            )
        ],
        generator_seed=42,
    )
    task = TaskRecord(
        schema_version=1,
        task_id="task-eval-000001",
        split="eval",
        question="Does Atlas require a security review?",
        canonical_answer="yes",
        acceptable_answer_variants=["yes, Atlas requires a security review"],
        required_fact_ids=["fact-000001", "fact-000002"],
        gold_document_ids=["doc-policies-0002", "doc-services-0001"],
        gold_chunk_ids=[
            "doc-policies-0002::c0000",
            "doc-services-0001::c0000",
        ],
        hop_count=2,
        task_template="dependency-policy",
        category="security",
        distractor_document_ids=["doc-services-0003"],
        distractor_count=1,
        generator_seed=42,
        corpus_version="synthetic-v1",
    )
    return first, second, task


def _valid_run() -> dict:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return {
        "schema_version": 1,
        "run_key": "exp-aaaaaaaaaaaa-42:task-eval-000001:snippets:primary",
        "experiment_id": "exp-aaaaaaaaaaaa-42",
        "design_cell": "primary",
        "task_id": "task-eval-000001",
        "condition": "snippets",
        "condition_order": 0,
        "model_config": {
            "provider": "mock",
            "base_url": "http://placeholder.invalid/v1",
            "model": "placeholder-model",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "tokenizer": "regex_v1",
        },
        "requested_model_id": "placeholder-model",
        "returned_model_id": "placeholder-model",
        "seed": 42,
        "corpus_version": "synthetic-v1",
        "question": "Does Atlas require a security review?",
        "final_answer": '{"answer":"yes","citations":[],"justification":"linked facts"}',
        "parsed_answer": {
            "answer": "yes",
            "citations": [],
            "justification": "linked facts",
        },
        "correctness": True,
        "secondary_judge": {
            "enabled": False,
            "score": None,
            "rationale": None,
            "model_id": None,
        },
        "scoring_details": {"matched_variant": "yes"},
        "citations": [],
        "initial_retrieved_documents": [
            {"doc_id": "doc-policies-0002", "best_score": 1.0, "rank": 1}
        ],
        "initial_retrieved_chunks": [
            {
                "chunk_id": "doc-policies-0002::c0000",
                "doc_id": "doc-policies-0002",
                "score": 1.0,
                "rank": 1,
                "start_line": 1,
                "end_line": 2,
            }
        ],
        "gold_evidence": {
            "fact_ids": ["fact-000001", "fact-000002"],
            "doc_ids": ["doc-policies-0002", "doc-services-0001"],
            "chunk_ids": [
                "doc-policies-0002::c0000",
                "doc-services-0001::c0000",
            ],
        },
        "accessed_evidence": {
            "fact_ids": ["fact-000002"],
            "doc_ids": ["doc-policies-0002"],
            "chunk_ids": ["doc-policies-0002::c0000"],
            "line_ranges": [
                {"path": "/policies/doc-policies-0002.md", "start_line": 1, "end_line": 2}
            ],
        },
        "retrieval_metrics": {
            "initial_chunk_recall": 0.5,
            "initial_document_recall": 0.5,
            "k": 8,
        },
        "evidence_metrics": {
            "final_recall": 0.5,
            "precision": 1.0,
            "distinct_gold_facts": 1,
            "all_required_accessed": False,
            "citation_precision": 1.0,
            "citation_recall": 0.0,
        },
        "tool_trace": [],
        "tool_call_count": 0,
        "max_tool_calls_permitted": None,
        "tool_calls_by_type": {"grep": 0, "read": 0, "cat": 0, "list": 0},
        "successful_tool_calls": 0,
        "invalid_tool_calls": 0,
        "repeated_tool_calls": 0,
        "unique_files_read": [],
        "unique_line_ranges_accessed": [],
        "time_to_first_gold_fact_ms": None,
        "calls_to_complete_gold_coverage": None,
        "token_accounting": {
            "tokenizer": "regex_v1",
            "limit": 100,
            "turns": [
                {
                    "turn_index": 0,
                    "request_input_tokens_estimated": 12,
                    "request_output_cap": 8,
                    "response_output_tokens_estimated": 3,
                    "snippet_input_tokens": 4,
                    "tool_argument_output_tokens": 0,
                    "tool_result_input_tokens": 0,
                    "provider_input_tokens": None,
                    "provider_output_tokens": None,
                    "remaining_after_turn": 85,
                }
            ],
            "input_tokens_estimated": 12,
            "output_tokens_estimated": 3,
            "total_tokens_estimated": 15,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_total_tokens": None,
            "provider_minus_estimated": {"input": 0, "output": 0, "total": 0},
        },
        "total_tokens": 15,
        "latency": {"wall_ms": 1.0, "model_ms": 0.5, "retrieval_ms": 0.25, "tool_ms": 0.0},
        "estimated_cost_usd": 0.0,
        "termination_reason": "answered",
        "errors": [],
        "prompt_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "corpus_sha256": "c" * 64,
        "cache_key": None,
        "cache_hit": False,
        "started_at": timestamp,
        "finished_at": timestamp,
    }


def test_document_and_task_round_trip() -> None:
    first, second, task = _documents_and_task()

    assert DocumentRecord.model_validate_json(first.model_dump_json()) == first
    assert TaskRecord.model_validate_json(task.model_dump_json()) == task
    assert second.model_dump(mode="json")["path"] == "/policies/doc-policies-0002.md"


def test_id_grammars() -> None:
    first, _, task = _documents_and_task()

    with pytest.raises(ValidationError):
        FactSpan(
            fact_id="fact-1",
            subject="Atlas",
            predicate="depends_on",
            object="Quartz",
            line_start=1,
            line_end=1,
        )
    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(first.model_dump(mode="python") | {"doc_id": "document"})
    with pytest.raises(ValidationError):
        TaskRecord.model_validate(task.model_dump(mode="python") | {"task_id": "task-eval-1"})
    with pytest.raises(ValidationError):
        TaskRecord.model_validate(
            task.model_dump(mode="python")
            | {"gold_chunk_ids": ["chunk-without-a-document"]}
        )

    run = _valid_run()
    with pytest.raises(ValidationError):
        RunRecord.model_validate(run | {"experiment_id": "experiment"})
    with pytest.raises(ValidationError):
        RunRecord.model_validate(run | {"run_key": "not-a-run-key"})
    with pytest.raises(ValidationError):
        RunRecord.model_validate(run | {"condition": "invalid-condition"})
    with pytest.raises(ValidationError):
        ToolTraceEntry.model_validate(
            {
                "request_id": "request-1",
                "call_ordinal": 1,
                "requested_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:00Z",
                "request": {"tool": "grep", "arguments": {"query": "Atlas"}},
                "result": {"ok": True, "tool": "grep", "content": "Atlas"},
                "repeated": False,
                "latency_ms": 0,
            }
        )


def test_fact_span_is_one_based() -> None:
    with pytest.raises(ValidationError, match="greater than or equal"):
        FactSpan(
            fact_id="fact-000001",
            subject="Atlas",
            predicate="depends_on",
            object="Quartz",
            line_start=2,
            line_end=1,
        )
    with pytest.raises(ValidationError):
        FactSpan(
            fact_id="fact-000001",
            subject="Atlas",
            predicate="depends_on",
            object="Quartz",
            line_start=0,
            line_end=1,
        )


def test_run_record_requires_complete_shape() -> None:
    payload = _valid_run()
    valid = RunRecord.model_validate(payload)
    assert valid.model_dump(mode="json")["model_config"]["provider"] == "mock"
    assert RunRecord.model_validate_json(valid.model_dump_json()) == valid

    incomplete = payload.copy()
    incomplete.pop("run_key")
    with pytest.raises(ValidationError, match="run_key"):
        RunRecord.model_validate(incomplete)


def test_token_totals_are_consistent() -> None:
    payload = _valid_run()
    payload["total_tokens"] = 16

    with pytest.raises(ValidationError, match="total_tokens"):
        RunRecord.model_validate(payload)

    payload = _valid_run()
    payload["token_accounting"]["total_tokens_estimated"] = 14
    with pytest.raises(ValidationError, match="total_tokens_estimated"):
        RunRecord.model_validate(payload)


def test_condition_specific_trace_invariant() -> None:
    payload = _valid_run()
    payload["tool_trace"] = [
        {
            "request_id": "tool-001",
            "call_ordinal": 1,
            "requested_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:00:00Z",
            "request": {"tool": "grep", "arguments": {"query": "Atlas"}},
            "result": {
                "ok": True,
                "tool": "grep",
                "request_id": "tool-001",
                "content": "Atlas",
                "paths": ["/services/doc-services-0001.md"],
                "line_ranges": [],
                "token_count": 1,
                "latency_ms": 0.1,
            },
            "repeated": False,
            "latency_ms": 0.1,
        }
    ]

    with pytest.raises(ValidationError, match="empty tool_trace"):
        RunRecord.model_validate(payload)
