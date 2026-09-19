"""Acceptance tests for retrieval and accessed-evidence scoring."""

from __future__ import annotations

from datetime import UTC, datetime

from bm25_vfs_ablation.corpus.schema import DocumentRecord, FactSpan, TaskRecord
from bm25_vfs_ablation.evaluation.evidence_scoring import (
    complete_coverage_call_ordinal,
    derive_accessed_evidence,
    score_evidence,
    score_retrieval,
)
from bm25_vfs_ablation.experiment.schemas import ToolRequest, ToolResult, ToolTraceEntry
from bm25_vfs_ablation.retrieval.bm25 import SearchHit
from bm25_vfs_ablation.retrieval.chunking import ChunkRecord


def _task() -> TaskRecord:
    return TaskRecord(
        schema_version=1,
        task_id="task-eval-000001",
        split="eval",
        question="Does the service need a review?",
        canonical_answer="yes",
        acceptable_answer_variants=["yes, a review is required"],
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
        corpus_version="test-v1",
    )


def _documents() -> list[DocumentRecord]:
    return [
        DocumentRecord(
            schema_version=1,
            corpus_version="test-v1",
            doc_id="doc-services-0001",
            path="/services/doc-services-0001.md",
            category="services",
            title="Service record",
            content="# Service\nService depends on Quartz.\nNotes\nBackground note.\n",
            facts=[
                FactSpan(
                    fact_id="fact-000001",
                    subject="Service",
                    predicate="depends_on",
                    object="Quartz",
                    line_start=2,
                    line_end=2,
                ),
                FactSpan(
                    fact_id="fact-000003",
                    subject="Service",
                    predicate="has_note",
                    object="Background note",
                    line_start=4,
                    line_end=4,
                ),
            ],
            generator_seed=42,
        ),
        DocumentRecord(
            schema_version=1,
            corpus_version="test-v1",
            doc_id="doc-policies-0002",
            path="/policies/doc-policies-0002.md",
            category="policies",
            title="Policy record",
            content="# Policy\nQuartz requires a security review.\n",
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
    ]


def _chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="doc-services-0001::c0000",
            doc_id="doc-services-0001",
            path="/services/doc-services-0001.md",
            chunk_index=0,
            start_line=2,
            end_line=2,
            text="Service depends on Quartz.",
            token_count=5,
            fact_ids=("fact-000001",),
        ),
        ChunkRecord(
            chunk_id="doc-services-0001::c0001",
            doc_id="doc-services-0001",
            path="/services/doc-services-0001.md",
            chunk_index=1,
            start_line=4,
            end_line=4,
            text="Background note.",
            token_count=3,
            fact_ids=("fact-000003",),
        ),
        ChunkRecord(
            chunk_id="doc-policies-0002::c0000",
            doc_id="doc-policies-0002",
            path="/policies/doc-policies-0002.md",
            chunk_index=0,
            start_line=2,
            end_line=2,
            text="Quartz requires a security review.",
            token_count=6,
            fact_ids=("fact-000002",),
        ),
    ]


def _trace(
    ordinal: int,
    tool: str,
    *,
    ranges: list[dict[str, int | str]] | None = None,
    content: str = "visible content",
    ok: bool = True,
) -> ToolTraceEntry:
    timestamp = datetime(2026, 1, 1, 0, 0, ordinal, tzinfo=UTC)
    request = ToolRequest(tool=tool, arguments={"path": "/services/doc-services-0001.md"})
    result = ToolResult(
        ok=ok,
        tool=tool,
        request_id=f"tool-{ordinal:03d}",
        content=content if ok else None,
        line_ranges=ranges or [],
        error_code=None if ok else "NOT_FOUND",
        error_message=None if ok else "missing path",
    )
    return ToolTraceEntry(
        request_id=f"tool-{ordinal:03d}",
        call_ordinal=ordinal,
        requested_at=timestamp,
        completed_at=timestamp,
        request=request,
        result=result,
        repeated=False,
        latency_ms=1.0,
    )


def test_initial_chunk_and_document_recall_separate() -> None:
    task = _task()
    hits = [
        SearchHit(
            chunk_id="doc-policies-0002::c0000",
            doc_id="doc-policies-0002",
            path="/policies/doc-policies-0002.md",
            score=2.0,
            start_line=2,
            end_line=2,
            text="policy",
        ),
        SearchHit(
            chunk_id="doc-services-0001::c0001",
            doc_id="doc-services-0001",
            path="/services/doc-services-0001.md",
            score=1.0,
            start_line=4,
            end_line=4,
            text="note",
        ),
    ]

    metrics = score_retrieval(task, hits, k=2)

    assert metrics.initial_chunk_recall == 0.5
    assert metrics.initial_document_recall == 1.0
    assert metrics.k == 2


def test_snippet_content_counts_as_accessed() -> None:
    accessed = derive_accessed_evidence(_task(), [_chunks()[0]], [])

    assert accessed.fact_ids == ["fact-000001"]
    assert accessed.doc_ids == ["doc-services-0001"]
    assert accessed.chunk_ids == ["doc-services-0001::c0000"]
    assert [(item.start_line, item.end_line) for item in accessed.line_ranges] == [(2, 2)]


def test_grep_preview_counts_by_line_overlap() -> None:
    task = _task()
    trace = [
        _trace(
            1,
            "grep",
            ranges=[
                {
                    "path": "/policies/doc-policies-0002.md",
                    "start_line": 1,
                    "end_line": 3,
                }
            ],
        )
    ]

    accessed = derive_accessed_evidence(task, _chunks(), trace)

    assert accessed.fact_ids == ["fact-000002"]
    assert accessed.chunk_ids == ["doc-policies-0002::c0000"]


def test_manifest_does_not_count_as_evidence() -> None:
    trace = [
        _trace(
            1,
            "list",
            ranges=[],
            content="/services/\n/policies/",
        )
    ]

    accessed = derive_accessed_evidence(_task(), _chunks(), trace)

    assert accessed.fact_ids == []
    assert accessed.doc_ids == []
    assert accessed.chunk_ids == []
    assert accessed.line_ranges == []


def test_accessed_ranges_are_merged() -> None:
    trace = [
        _trace(
            1,
            "read",
            ranges=[
                {
                    "path": "/services/doc-services-0001.md",
                    "start_line": 1,
                    "end_line": 3,
                }
            ],
        ),
        _trace(
            2,
            "grep",
            ranges=[
                {
                    "path": "/services/doc-services-0001.md",
                    "start_line": 2,
                    "end_line": 4,
                }
            ],
        ),
    ]

    accessed = derive_accessed_evidence(_task(), _chunks(), trace)

    assert [(item.path, item.start_line, item.end_line) for item in accessed.line_ranges] == [
        ("/services/doc-services-0001.md", 1, 4)
    ]


def test_evidence_precision_and_recall() -> None:
    trace = [
        _trace(
            1,
            "read",
            ranges=[
                {
                    "path": "/services/doc-services-0001.md",
                    "start_line": 2,
                    "end_line": 4,
                }
            ],
        )
    ]
    accessed = derive_accessed_evidence(_task(), _chunks(), trace)
    metrics = score_evidence(
        _task(),
        accessed,
        citation_precision=0.5,
        citation_recall=0.25,
    )

    assert metrics.precision == 0.5
    assert metrics.final_recall == 0.5
    assert metrics.distinct_gold_facts == 1
    assert metrics.all_required_accessed is False
    assert metrics.citation_precision == 0.5
    assert metrics.citation_recall == 0.25


def test_complete_coverage_call_ordinal() -> None:
    trace = [
        _trace(
            1,
            "read",
            ranges=[
                {
                    "path": "/services/doc-services-0001.md",
                    "start_line": 2,
                    "end_line": 2,
                }
            ],
        ),
        _trace(
            2,
            "read",
            ranges=[
                {
                    "path": "/policies/doc-policies-0002.md",
                    "start_line": 2,
                    "end_line": 2,
                }
            ],
        ),
    ]

    assert complete_coverage_call_ordinal(_task(), _chunks(), trace) == 2
