"""Acceptance tests for bounded read-only virtual filesystem operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from bm25_vfs_ablation.corpus.schema import DocumentRecord
from bm25_vfs_ablation.retrieval.chunking import estimate_tokens
from bm25_vfs_ablation.vfs.filesystem import VirtualFilesystem, materialize_corpus
from bm25_vfs_ablation.vfs.security import VfsError


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


def _filesystem(tmp_path: Path) -> tuple[Path, list[DocumentRecord], VirtualFilesystem]:
    documents = [
        _document("doc-services-0003", "services", "service three\n"),
        _document("doc-ops-0002", "ops", "ops two\n"),
        _document("doc-services-0001", "services", "service one\n"),
    ]
    root = tmp_path / "materialized"
    materialize_corpus(root, documents)
    return root, documents, VirtualFilesystem(root, documents)


def test_list_is_shallow_and_sorted(tmp_path: Path) -> None:
    _, _, filesystem = _filesystem(tmp_path)

    assert filesystem.list() == ["/ops/", "/services/"]
    assert filesystem.list("/services/") == [
        "/services/doc-services-0001.md",
        "/services/doc-services-0003.md",
    ]


def test_read_inclusive_bounded_lines(tmp_path: Path) -> None:
    content = "header\nline two\r\nline three\nfinal line"
    document = _document("doc-ops-0001", "ops", content)
    root = tmp_path / "materialized"
    materialize_corpus(root, [document])
    filesystem = VirtualFilesystem(root, [document])

    bounded = filesystem.read(document.path, 2, 3)
    clamped = filesystem.read(document.path, 3, 99)

    assert bounded.path == document.path
    assert bounded.start_line == 2
    assert bounded.end_line == 3
    assert bounded.text == "line two\r\nline three\n"
    assert bounded.token_count == estimate_tokens(bounded.text)
    assert clamped.end_line == 4
    assert clamped.text == "line three\nfinal line"


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    ((0, 1), (1, 0), (3, 2), (5, 9)),
)
def test_read_rejects_invalid_range(
    tmp_path: Path,
    start_line: int,
    end_line: int,
) -> None:
    _, documents, filesystem = _filesystem(tmp_path)

    with pytest.raises(VfsError) as caught:
        filesystem.read(documents[1].path, start_line, end_line)

    assert caught.value.code == "RANGE_INVALID"


def test_cat_returns_exact_file(tmp_path: Path) -> None:
    _, documents, filesystem = _filesystem(tmp_path)
    document = documents[0]

    result = filesystem.cat(document.path)

    assert result.path == document.path
    assert result.start_line == 1
    assert result.end_line == 1
    assert result.text == document.content
    assert result.token_count == estimate_tokens(document.content)


def test_materialized_content_must_match(tmp_path: Path) -> None:
    root, documents, _ = _filesystem(tmp_path)
    target = root / "services" / "doc-services-0003.md"
    target.write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="materialized content mismatch"):
        VirtualFilesystem(root, documents)


def test_filesystem_exposes_no_write_method(tmp_path: Path) -> None:
    _, _, filesystem = _filesystem(tmp_path)

    assert not hasattr(filesystem, "write")
    assert not hasattr(filesystem, "write_text")
    assert not hasattr(filesystem, "unlink")
