from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from bm25_vfs_ablation import Condition
from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.corpus.loader import CorpusBundle
from bm25_vfs_ablation.corpus.schema import DocumentRecord, FactSpan, TaskRecord
from bm25_vfs_ablation.evaluation.statistics import within_vfs_regression
from bm25_vfs_ablation.experiment.designs import condition_order, expand_design
from bm25_vfs_ablation.experiment.runner import ExperimentRunner
from bm25_vfs_ablation.experiment.schemas import (
    ParsedAnswer,
    RunRecord,
    TerminationReason,
)
from bm25_vfs_ablation.harnesses.base import HarnessContext, HarnessResult
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.harnesses.vfs_agent import VfsHarness
from bm25_vfs_ablation.models.client import ModelRequest, ModelResponse
from bm25_vfs_ablation.retrieval.bm25 import BM25Index
from bm25_vfs_ablation.retrieval.chunking import Chunker, ChunkRecord

_FINAL_ANSWER = json.dumps(
    {
        "answer": "yes, a security review is required",
        "citations": [
            "doc-policies-0001::c0000",
            "doc-services-0001::c0000",
        ],
        "justification": "The service dependency and policy records establish the requirement.",
    },
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
)


@dataclass(frozen=True, slots=True)
class _TinyFixture:
    config: AppConfig
    bundle: CorpusBundle
    task: TaskRecord
    chunks: tuple[ChunkRecord, ...]
    index: BM25Index


def _config(
    *,
    token_ceiling: int = 4096,
    top_k: int = 1,
    max_tool_calls: int = 8,
    intervention: bool = False,
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "experiment_id": None,
                "seed": 42,
                "conditions": ["snippets", "vfs"],
                "token_ceiling": token_ceiling,
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
                "top_k": top_k,
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
                "query_tokenizer": "unicode_word_v1",
            },
            "harness": {
                "retrieval_context_tokens": 1800,
                "max_answer_tokens": 256,
                "max_tool_calls": max_tool_calls,
                "invalid_call_limit": 3,
                "tool_result_tokens_per_call": 768,
                "initial_candidate_documents": 5,
            },
            "model": {
                "provider": "mock",
                "base_url": "http://placeholder.invalid/v1",
                "api_key": None,
                "model": "model-placeholder",
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "timeout_seconds": 60.0,
                "max_attempts": 2,
                "tokenizer": "regex_v1",
            },
            "evaluation": {
                "bootstrap_resamples": 10,
                "bootstrap_seed": 20250308,
                "confidence_level": 0.95,
                "judge_enabled": False,
                "estimated_input_usd_per_million": 0.0,
                "estimated_output_usd_per_million": 0.0,
            },
            "intervention": {
                "enabled": intervention,
                "max_tool_calls": [0, 1, 2, 4, 8, 12] if intervention else [],
            },
        }
    )


def _fixture(
    *,
    config: AppConfig | None = None,
) -> _TinyFixture:
    documents = (
        DocumentRecord(
            schema_version=1,
            corpus_version="synthetic-v1",
            doc_id="doc-policies-0001",
            path="/policies/doc-policies-0001.md",
            category="policies",
            title="Security policy",
            content="# Review policy\nQuartz requires a security review.\n",
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
        ),
        DocumentRecord(
            schema_version=1,
            corpus_version="synthetic-v1",
            doc_id="doc-services-0001",
            path="/services/doc-services-0001.md",
            category="services",
            title="Atlas service",
            content="# Atlas service\nAtlas depends on Quartz.\n",
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
        ),
    )
    task = TaskRecord(
        schema_version=1,
        task_id="task-eval-000001",
        split="eval",
        question="Does Atlas require a security review?",
        canonical_answer="yes, a security review is required",
        acceptable_answer_variants=["yes", "security review required"],
        required_fact_ids=["fact-000001", "fact-000002"],
        gold_document_ids=["doc-policies-0001", "doc-services-0001"],
        gold_chunk_ids=[
            "doc-policies-0001::c0000",
            "doc-services-0001::c0000",
        ],
        hop_count=2,
        task_template="dependency-policy",
        category="security",
        distractor_document_ids=[],
        distractor_count=0,
        generator_seed=42,
        corpus_version="synthetic-v1",
    )
    selected_config = config or _config()
    chunks = tuple(
        Chunker(
            selected_config.chunking.chunk_size_tokens,
            selected_config.chunking.chunk_overlap_tokens,
        ).chunk(documents)
    )
    index = BM25Index(
        chunks,
        k1=selected_config.retrieval.k1,
        b=selected_config.retrieval.b,
        epsilon=selected_config.retrieval.epsilon,
    )
    bundle = CorpusBundle(
        documents=documents,
        tasks=(task,),
        corpus_sha256="a" * 64,
        tasks_sha256="b" * 64,
        corpus_version="synthetic-v1",
    )
    return _TinyFixture(
        config=selected_config,
        bundle=bundle,
        task=task,
        chunks=chunks,
        index=index,
    )


class _ReadBothModel:
    """Return both gold lines to VFS, then the public answer."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        tool_messages = [
            message for message in request.messages if message.get("role") == "tool"
        ]
        if not request.tools or len(tool_messages) >= 2:
            return ModelResponse(
                content=_FINAL_ANSWER,
                returned_model_id="model-placeholder",
                latency_ms=0.0,
            )

        path = (
            "/services/doc-services-0001.md"
            if not tool_messages
            else "/policies/doc-policies-0001.md"
        )
        arguments = json.dumps(
            {"end_line": 2, "path": path, "start_line": 2},
            separators=(",", ":"),
            sort_keys=True,
        )
        return ModelResponse(
            content=None,
            tool_calls=[
                {
                    "id": f"read-{len(tool_messages) + 1}",
                    "type": "function",
                    "function": {"name": "read", "arguments": arguments},
                }
            ],
            returned_model_id="model-placeholder",
            latency_ms=0.0,
        )


def _context(fixture: _TinyFixture, model: object) -> HarnessContext:
    return HarnessContext(
        bundle=fixture.bundle,
        index=fixture.index,
        model=model,  # type: ignore[arg-type]
        cache=None,
        config=fixture.config,
        budget_factory=lambda: BudgetController(fixture.config.experiment.token_ceiling),
    )


def _stub_result(fixture: _TinyFixture) -> HarnessResult:
    return HarnessResult(
        raw_answer=_FINAL_ANSWER,
        initial_hits=tuple(
            fixture.index.search(
                fixture.task.question,
                fixture.config.retrieval.top_k,
            )
        ),
        accounting=BudgetController(
            fixture.config.experiment.token_ceiling
        ).accounting(),
        termination=TerminationReason.ANSWERED,
    )


def _run_primary(
    tmp_path: Path,
    fixture: _TinyFixture,
    model: object,
) -> tuple[ExperimentRunner, tuple[RunRecord, ...]]:
    runner = ExperimentRunner(
        fixture.config,
        fixture.bundle,
        fixture.index,
        model,  # type: ignore[arg-type]
        None,
        output_runs=tmp_path / "runs.jsonl",
    )
    return runner, runner.run().records


def test_conditions_share_exact_index_and_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    contexts: list[HarnessContext] = []

    def capture_context(
        self: object,
        task: TaskRecord,
        context: HarnessContext,
    ) -> HarnessResult:
        del self, task
        contexts.append(context)
        return _stub_result(fixture)

    monkeypatch.setattr(
        "bm25_vfs_ablation.experiment.runner.SnippetHarness.run",
        capture_context,
    )
    monkeypatch.setattr(
        "bm25_vfs_ablation.experiment.runner.VfsHarness.run",
        capture_context,
    )
    _, records = _run_primary(tmp_path, fixture, _ReadBothModel())

    assert len(contexts) == 2
    assert {id(context.bundle) for context in contexts} == {id(fixture.bundle)}
    assert {id(context.index) for context in contexts} == {id(fixture.index)}
    assert all(context.config is fixture.config for context in contexts)
    assert contexts[0].index.corpus_fingerprint == contexts[1].index.corpus_fingerprint
    assert contexts[0].bundle.documents == contexts[1].bundle.documents == fixture.bundle.documents
    assert contexts[0].config.canonical_dict() == contexts[1].config.canonical_dict()
    assert {record.corpus_sha256 for record in records} == {fixture.bundle.corpus_sha256}
    assert len({record.config_sha256 for record in records}) == 1


def test_conditions_receive_equal_token_ceiling(tmp_path: Path) -> None:
    fixture = _fixture(config=_config(token_ceiling=777))
    _, records = _run_primary(tmp_path, fixture, _ReadBothModel())

    assert {record.condition for record in records} == {Condition.SNIPPETS, Condition.VFS}
    assert {
        record.token_accounting.limit
        for record in records
    } == {fixture.config.experiment.token_ceiling}
    assert all(
        record.total_tokens <= fixture.config.experiment.token_ceiling
        for record in records
    )


def test_seed_and_config_reproduce_assignments() -> None:
    fixture = _fixture(config=_config(intervention=True))
    equivalent_config = AppConfig.model_validate(
        fixture.config.model_dump(mode="json")
    )

    first = expand_design(fixture.config, [fixture.task])
    second = expand_design(equivalent_config, [fixture.task])
    expected_order = condition_order(42, fixture.task.task_id)

    assert fixture.config.canonical_dict() == equivalent_config.canonical_dict()
    assert first == second
    assert condition_order(42, fixture.task.task_id) == expected_order
    assert condition_order(42, fixture.task.task_id) == expected_order
    for design_cell in {assignment.design_cell for assignment in first}:
        cell_rows = [row for row in first if row.design_cell is design_cell]
        assert tuple(row.condition for row in cell_rows) == expected_order
        assert [row.condition_order for row in cell_rows] == [0, 1]


def test_retrieval_evidence_success_are_separate(tmp_path: Path) -> None:
    fixture = _fixture()
    _, records = _run_primary(tmp_path, fixture, _ReadBothModel())
    by_condition = {record.condition: record for record in records}
    vfs_record = by_condition[Condition.VFS]

    assert set(RunRecord.model_fields) >= {
        "retrieval_metrics",
        "evidence_metrics",
        "correctness",
    }
    assert vfs_record.retrieval_metrics.initial_chunk_recall == pytest.approx(0.5)
    assert vfs_record.evidence_metrics.final_recall == pytest.approx(1.0)
    assert vfs_record.correctness is True
    assert (
        vfs_record.retrieval_metrics.initial_chunk_recall
        != vfs_record.evidence_metrics.final_recall
    )
    assert "correctness" not in type(vfs_record.retrieval_metrics).model_fields
    assert "correctness" not in type(vfs_record.evidence_metrics).model_fields


def test_every_vfs_interaction_is_observable() -> None:
    fixture = _fixture()
    result = VfsHarness(max_tool_calls=8).run(
        fixture.task,
        _context(fixture, _ReadBothModel()),
    )

    assert result.termination is TerminationReason.ANSWERED
    assert result.tool_call_count == len(result.trace) == 2
    assert result.successful_tool_calls == 2
    assert result.invalid_tool_calls == 0
    assert result.tool_calls_by_type.read == 2
    assert result.tool_calls_by_type.total() == result.tool_call_count
    for ordinal, entry in enumerate(result.trace, start=1):
        assert entry.call_ordinal == ordinal
        assert entry.request_id == f"tool-{ordinal:03d}"
        assert entry.result.request_id == entry.request_id
        assert entry.request.tool == entry.result.tool
        assert entry.result.ok is True
        assert entry.result.line_ranges
        assert isinstance(entry.requested_at, datetime)
        assert isinstance(entry.completed_at, datetime)
        assert entry.requested_at.tzinfo is not None
        assert entry.completed_at.tzinfo is not None
        assert entry.model_dump(mode="json")["request"]
        assert entry.model_dump(mode="json")["result"]


def test_tool_call_analysis_is_associative() -> None:
    rows: list[dict[str, object]] = []
    for ordinal in range(20):
        rows.append(
            {
                "task_id": f"task-eval-{ordinal + 1:06d}",
                "condition": "vfs",
                "design_cell": "primary",
                "correctness": ordinal % 2 == 0,
                "tool_call_count": ordinal % 7,
                "max_tool_calls_permitted": 12,
                "initial_chunk_recall": 0.1 + (ordinal % 9) / 10,
                "hop_count": 2 + ordinal % 3,
                "category": "security",
                "total_tokens": 100 + ordinal * 13,
                "distractor_count": ordinal % 6,
            }
        )

    regression = within_vfs_regression(rows)

    assert regression.attrs["associative"] is True
    assert regression.attrs["covariance_type"] == "HC3"
    assert set(regression["associative"].dropna()) == {True}
    assert set(regression["covariance_type"].dropna()) == {"HC3"}
    assert "calls" in set(regression["term"])
    assert "calls_squared" in set(regression["term"])
    assert "max_tool_calls_permitted" not in set(regression["term"])


def test_intervention_separates_permitted_and_actual() -> None:
    fixture = _fixture(config=_config(intervention=True))
    assignments = expand_design(fixture.config, [fixture.task])
    zero_call_assignment = next(
        assignment
        for assignment in assignments
        if assignment.condition is Condition.VFS
        and assignment.design_cell.value == "max_calls_0"
    )
    result = VfsHarness(
        max_tool_calls=zero_call_assignment.max_tool_calls_permitted
    ).run(
        fixture.task,
        _context(fixture, _ReadBothModel()),
    )

    assert zero_call_assignment.max_tool_calls_permitted == 0
    assert result.max_tool_calls_permitted == 0
    assert result.tool_call_count == 1
    assert result.tool_call_count != result.max_tool_calls_permitted
    assert result.trace[0].result.error_code == "MAX_TOOL_CALLS"
    assert result.termination is TerminationReason.MAX_TOOL_CALLS


def _schema_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                names.update(str(name) for name in properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return names


def test_run_schema_has_no_chain_of_thought_field() -> None:
    property_names = _schema_property_names(RunRecord.model_json_schema())
    property_names.update(_schema_property_names(ParsedAnswer.model_json_schema()))
    normalized_names = {name.casefold().replace("-", "_") for name in property_names}
    forbidden_names = {
        "analysis",
        "chain_of_thought",
        "chainofthought",
        "cot",
        "hidden_reasoning",
        "private_reasoning",
        "reasoning",
        "thoughts",
    }

    assert normalized_names.isdisjoint(forbidden_names)
    assert "justification" in normalized_names
