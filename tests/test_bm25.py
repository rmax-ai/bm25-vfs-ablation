"""Acceptance tests for the stable shared BM25 index."""

from __future__ import annotations

import json

from bm25_vfs_ablation.retrieval.bm25 import BM25Index
from bm25_vfs_ablation.retrieval.chunking import ChunkRecord


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str | None = None,
    path: str | None = None,
    chunk_index: int = 0,
) -> ChunkRecord:
    resolved_doc_id = doc_id or chunk_id.split("::", maxsplit=1)[0]
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=resolved_doc_id,
        path=path or f"/services/{resolved_doc_id}.md",
        chunk_index=chunk_index,
        start_line=chunk_index + 1,
        end_line=chunk_index + 1,
        text=text,
        token_count=len(text.split()),
        fact_ids=(),
    )


def test_bm25_known_ranking() -> None:
    index = BM25Index(
        [
            _chunk("doc-ops-0002::c0000", "atlas service"),
            _chunk("doc-ops-0001::c0000", "quartz quartz atlas"),
            _chunk("doc-ops-0003::c0000", "unrelated deployment notes"),
            _chunk("doc-ops-0004::c0000", "unrelated policy notes"),
        ]
    )

    hits = index.search("quartz", top_k=4)
    print([(hit.chunk_id, hit.score) for hit in hits])

    assert [hit.chunk_id for hit in hits[:2]] == [
        "doc-ops-0001::c0000",
        "doc-ops-0002::c0000",
    ]
    assert hits[0].score > 0
    assert hits[1].score == 0
    assert json.loads(json.dumps(hits[0].to_dict()))["score"] == round(hits[0].score, 12)


def test_bm25_tie_breaks_by_chunk_id() -> None:
    index = BM25Index(
        [
            _chunk("doc-ops-0003::c0000", "alpha"),
            _chunk("doc-ops-0001::c0000", "beta"),
            _chunk("doc-ops-0002::c0000", "gamma"),
        ]
    )

    hits = index.search("missing", top_k=2)

    assert [hit.chunk_id for hit in hits] == [
        "doc-ops-0001::c0000",
        "doc-ops-0002::c0000",
    ]
    assert [hit.score for hit in hits] == [0.0, 0.0]


def test_allowed_path_filter() -> None:
    chunks = [
        _chunk(
            "doc-services-0001::c0000",
            "atlas service",
            path="/services/atlas.md",
            doc_id="doc-services-0001",
        ),
        _chunk(
            "doc-services-0002::c0000",
            "quartz service",
            path="/services/quartz.md",
            doc_id="doc-services-0002",
        ),
        _chunk(
            "doc-notes-0001::c0000",
            "service note",
            path="/notes/readme.md",
            doc_id="doc-notes-0001",
        ),
    ]
    index = BM25Index(chunks)

    exact = index.search("atlas", top_k=10, allowed_paths=["/services/atlas.md"])
    prefix = index.search("atlas", top_k=10, allowed_paths=["/services/"])
    near_miss = index.search("atlas", top_k=10, allowed_paths=["/services"])

    assert [hit.path for hit in exact] == ["/services/atlas.md"]
    assert {hit.path for hit in prefix} == {"/services/atlas.md", "/services/quartz.md"}
    assert near_miss == []


def test_document_hit_aggregation() -> None:
    index = BM25Index(
        [
            _chunk(
                "doc-ops-0001::c0000",
                "rare rare atlas",
                doc_id="doc-ops-0001",
                path="/ops/atlas.md",
                chunk_index=0,
            ),
            _chunk(
                "doc-ops-0001::c0001",
                "rare atlas",
                doc_id="doc-ops-0001",
                path="/ops/atlas.md",
                chunk_index=1,
            ),
            _chunk(
                "doc-ops-0002::c0000",
                "rare quartz",
                doc_id="doc-ops-0002",
                path="/ops/quartz.md",
            ),
            _chunk("doc-ops-0003::c0000", "ordinary notes"),
            _chunk("doc-ops-0004::c0000", "ordinary policy"),
        ]
    )

    chunk_hits = index.search("rare", top_k=3)
    document_hits = index.document_hits(chunk_hits)
    atlas_chunks = [hit for hit in chunk_hits if hit.doc_id == "doc-ops-0001"]

    assert [hit.doc_id for hit in document_hits] == ["doc-ops-0001", "doc-ops-0002"]
    assert document_hits[0].best_score == max(hit.score for hit in atlas_chunks)
    assert document_hits[0].best_chunk_id == atlas_chunks[0].chunk_id
    assert document_hits[0].chunk_ids == tuple(hit.chunk_id for hit in atlas_chunks)
    assert json.loads(json.dumps(document_hits[0].to_dict()))["best_score"] == round(
        document_hits[0].best_score, 12
    )


def test_index_fingerprint_stable() -> None:
    chunks = [
        _chunk("doc-ops-0002::c0000", "beta"),
        _chunk("doc-ops-0001::c0000", "alpha"),
    ]

    first = BM25Index(chunks)
    reordered = BM25Index(list(reversed(chunks)))
    changed_text = BM25Index([_chunk("doc-ops-0002::c0000", "changed"), chunks[1]])
    changed_parameters = BM25Index(chunks, k1=1.6)

    assert first.corpus_fingerprint == reordered.corpus_fingerprint
    assert len(first.corpus_fingerprint) == 64
    assert first.corpus_fingerprint != changed_text.corpus_fingerprint
    assert first.corpus_fingerprint != changed_parameters.corpus_fingerprint


def test_same_index_object_can_serve_both_conditions() -> None:
    index = BM25Index(
        [
            _chunk(
                "doc-services-0001::c0000",
                "atlas depends on quartz",
                doc_id="doc-services-0001",
                path="/services/atlas.md",
            ),
            _chunk(
                "doc-records-0001::c0000",
                "deployment record",
                doc_id="doc-records-0001",
                path="/records/quartz.md",
            ),
            _chunk(
                "doc-notes-0001::c0000",
                "unrelated notes",
                doc_id="doc-notes-0001",
                path="/notes/readme.md",
            ),
        ]
    )
    snippets_index = index
    vfs_index = index

    snippet_hits = snippets_index.search("quartz", top_k=2)
    vfs_hits = vfs_index.search("quartz", top_k=2, allowed_paths=["/services/"])

    assert snippets_index is vfs_index
    assert index.corpus_fingerprint == snippets_index.corpus_fingerprint
    assert snippet_hits[0].chunk_id == "doc-services-0001::c0000"
    assert [hit.chunk_id for hit in vfs_hits] == ["doc-services-0001::c0000"]
