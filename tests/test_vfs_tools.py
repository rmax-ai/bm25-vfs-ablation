"""Acceptance tests for the bounded VFS tool executor."""

from __future__ import annotations

from pathlib import Path

from bm25_vfs_ablation.corpus.schema import DocumentRecord
from bm25_vfs_ablation.experiment.schemas import ToolRequest
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.retrieval.bm25 import BM25Index
from bm25_vfs_ablation.retrieval.chunking import Chunker, estimate_tokens
from bm25_vfs_ablation.vfs.filesystem import VirtualFilesystem, materialize_corpus
from bm25_vfs_ablation.vfs.tools import VfsToolExecutor, tool_schema


def _document(doc_id: str, category: str, content: str) -> DocumentRecord:
    return DocumentRecord(
        schema_version=1,
        corpus_version="test-v1",
        doc_id=doc_id,
        path=f"/{category}/{doc_id}.md",
        category=category,
        title=f"{category} document",
        content=content,
        facts=[],
        generator_seed=42,
    )


def _executor(
    tmp_path: Path,
    *,
    documents: list[DocumentRecord] | None = None,
    budget_limit: int = 500,
    per_call_limit: int = 50,
) -> VfsToolExecutor:
    records = documents or [
        _document(
            "doc-services-0001",
            "services",
            "service alpha\nquartz is stable\nservice footer\n",
        ),
        _document(
            "doc-services-0002",
            "services",
            "service beta\nquartz quartz is current\nservice footer\n",
        ),
        _document(
            "doc-notes-0001",
            "notes",
            "note header\nreference only\nnote footer\n",
        ),
    ]
    root = tmp_path / "materialized"
    materialize_corpus(root, records)
    filesystem = VirtualFilesystem(root, records)
    chunks = Chunker().chunk(records)
    index = BM25Index(chunks)
    return VfsToolExecutor(
        filesystem,
        index,
        BudgetController(budget_limit),
        per_call_limit,
    )


def test_tool_schema_exact_names() -> None:
    schema = tool_schema()

    assert [item["function"]["name"] for item in schema] == ["grep", "read", "cat", "list"]
    assert all(item["type"] == "function" for item in schema)
    assert tool_schema() == schema


def test_grep_rank_and_preview_bounds(tmp_path: Path) -> None:
    executor = _executor(tmp_path)

    result = executor.execute(
        ToolRequest(
            tool="grep",
            arguments={"query": "quartz", "max_results": 2},
        )
    )

    assert result.ok is True
    assert result.paths == [
        "/services/doc-services-0001.md",
        "/services/doc-services-0002.md",
    ]
    assert [line_range.path for line_range in result.line_ranges] == [
        "/services/doc-services-0002.md",
        "/services/doc-services-0001.md",
    ]
    assert all(line_range.start_line >= 1 for line_range in result.line_ranges)
    assert all(
        line_range.end_line - line_range.start_line <= 2
        for line_range in result.line_ranges
    )
    assert "quartz quartz" in (result.content or "")


def test_read_tool_envelope(tmp_path: Path) -> None:
    executor = _executor(tmp_path)

    result = executor.execute(
        ToolRequest(
            tool="read",
            arguments={
                "path": "/services/doc-services-0001.md",
                "start_line": 2,
                "end_line": 3,
            },
        )
    )

    assert result.ok is True
    assert result.request_id is None
    assert result.content == "quartz is stable\nservice footer\n"
    assert result.paths == ["/services/doc-services-0001.md"]
    assert result.line_ranges[0].start_line == 2
    assert result.line_ranges[0].end_line == 3
    assert result.token_count == estimate_tokens(result.content or "")
    assert result.error_code is None
    assert result.latency_ms >= 0


def test_cat_refuses_partial_result(tmp_path: Path) -> None:
    executor = _executor(tmp_path, per_call_limit=2)

    result = executor.execute(
        ToolRequest(
            tool="cat",
            arguments={"path": "/services/doc-services-0001.md"},
        )
    )

    assert result.ok is False
    assert result.error_code == "PER_CALL_LIMIT"
    assert result.content is None
    assert result.token_count == 0
    assert result.paths == []
    assert result.line_ranges == []


def test_list_tool_sorted(tmp_path: Path) -> None:
    executor = _executor(tmp_path)

    result = executor.execute(
        ToolRequest(tool="list", arguments={"path": "/"})
    )

    assert result.ok is True
    assert result.paths == []
    assert result.content == "/notes/\n/services/"
    assert result.token_count == estimate_tokens(result.content or "")


def test_invalid_arguments_exact_codes(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    requests_and_codes = [
        (
            ToolRequest(
                tool="grep",
                arguments={"query": " ", "max_results": 1},
            ),
            "RESULT_LIMIT_INVALID",
        ),
        (
            ToolRequest(
                tool="grep",
                arguments={"query": "quartz", "max_results": 51},
            ),
            "RESULT_LIMIT_INVALID",
        ),
        (
            ToolRequest(
                tool="read",
                arguments={
                    "path": "/services/doc-services-0001.md",
                    "start_line": 0,
                    "end_line": 1,
                },
            ),
            "RANGE_INVALID",
        ),
        (
            ToolRequest(tool="cat", arguments={"path": "/missing.md"}),
            "NOT_FOUND",
        ),
    ]

    for request, expected_code in requests_and_codes:
        result = executor.execute(request)
        assert result.ok is False
        assert result.error_code == expected_code


def test_tool_result_token_count_exact(tmp_path: Path) -> None:
    executor = _executor(tmp_path, per_call_limit=3)

    result = executor.execute(
        ToolRequest(
            tool="read",
            arguments={
                "path": "/services/doc-services-0001.md",
                "start_line": 1,
                "end_line": 3,
            },
        )
    )

    assert result.ok is True
    assert result.truncated is True
    assert result.content is not None
    assert result.token_count == estimate_tokens(result.content)
    assert result.token_count <= 3
