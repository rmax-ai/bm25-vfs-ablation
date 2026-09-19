"""Acceptance tests for B08 normalization, tokenization, and chunking."""

from __future__ import annotations

import json
from dataclasses import asdict

from bm25_vfs_ablation.corpus.generator import GenerationConfig, generate_dataset
from bm25_vfs_ablation.models.tokenization import RegexTokenizer
from bm25_vfs_ablation.retrieval.chunking import (
    Chunker,
    normalize_terms,
)


def _document(doc_id: str, content: str, facts: list[dict] | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "path": f"/ops/{doc_id}.md",
        "content": content,
        "facts": facts or [],
    }


def test_unicode_normalization_and_casefold() -> None:
    assert normalize_terms("Ｆoo STRAẞE café") == ["foo", "strasse", "café"]


def test_stopwords_are_retained() -> None:
    assert normalize_terms("the and of") == ["the", "and", "of"]


def test_chunk_boundaries_and_overlap() -> None:
    document = _document("doc-ops-0001", "\n".join(f"line {i}" for i in range(1, 101)))
    chunks = Chunker(size_tokens=10, overlap_tokens=2).chunk([document])
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5
    assert chunks[1].start_line == 5
    assert chunks[1].end_line == 9
    assert chunks[0].text.endswith("line 5")
    assert chunks[1].text.startswith("line 5")
    assert all(chunk.token_count <= 10 for chunk in chunks)


def test_long_line_split_is_bounded() -> None:
    words = " ".join(f"word{i}" for i in range(401))
    chunks = Chunker(size_tokens=180, overlap_tokens=30).chunk([_document("doc-ops-0001", words)])
    assert [chunk.token_count for chunk in chunks] == [180, 180, 101]
    assert all(chunk.start_line == chunk.end_line == 1 for chunk in chunks)
    assert all(chunk.token_count <= 180 for chunk in chunks)
    assert "word0" in chunks[0].text
    assert "word400" in chunks[-1].text


def test_chunk_ids_stable() -> None:
    documents = [
        _document("doc-ops-0002", "beta gamma"),
        _document("doc-ops-0001", "alpha"),
    ]
    forward = Chunker(size_tokens=2, overlap_tokens=0).chunk(documents)
    reverse = Chunker(size_tokens=2, overlap_tokens=0).chunk(reversed(documents))
    assert forward == reverse
    assert [chunk.chunk_id for chunk in forward] == [
        "doc-ops-0001::c0000",
        "doc-ops-0002::c0000",
    ]


def test_regex_tokenizer_never_exceeds_limit() -> None:
    tokenizer = RegexTokenizer()
    text = "one, two three! four"
    for limit in range(0, 8):
        truncated = tokenizer.truncate_text(text, limit)
        assert tokenizer.count_text(truncated) <= limit
    assert tokenizer.count_messages([{"role": "user", "content": "hello"}]) >= 2


def test_chunk_record_json_round_trip_and_facts() -> None:
    document = _document(
        "doc-ops-0001",
        "alpha beta\ngamma delta",
        [{"fact_id": "fact-000001", "line_start": 2, "line_end": 2}],
    )
    chunk = Chunker(size_tokens=2, overlap_tokens=0).chunk([document])[1]
    restored = json.loads(json.dumps(asdict(chunk)))
    assert restored["chunk_id"] == chunk.chunk_id
    assert restored["fact_ids"] == ["fact-000001"]


def test_generated_tasks_have_no_single_complete_chunk() -> None:
    documents, tasks = generate_dataset(
        GenerationConfig(corpus_size=4, dev_tasks=2, eval_tasks=0, seed=42)
    )
    chunks = Chunker().chunk(documents)
    by_doc = {
        document.doc_id: [chunk for chunk in chunks if chunk.doc_id == document.doc_id]
        for document in documents
    }
    for task in tasks:
        required = set(task.required_fact_ids)
        task_chunks = [
            chunk
            for doc_id in task.gold_document_ids
            for chunk in by_doc[doc_id]
            if required.intersection(chunk.fact_ids)
        ]
        assert required <= {fact_id for chunk in task_chunks for fact_id in chunk.fact_ids}
        assert not any(required.issubset(chunk.fact_ids) for chunk in task_chunks)
