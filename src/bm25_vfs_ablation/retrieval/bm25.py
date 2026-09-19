"""Deterministic shared BM25 retrieval over fixed corpus chunks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from bm25_vfs_ablation.retrieval.chunking import ChunkRecord, normalize_terms

_TOKENIZER_NAME = "unicode_word_v1"
_SCORE_DECIMAL_PLACES = 12


def _chunk_value(chunk: ChunkRecord, name: str) -> Any:
    return getattr(chunk, name)


def _rounded_score(score: float) -> float:
    """Return the canonical JSON score representation."""

    rounded = round(score, _SCORE_DECIMAL_PLACES)
    return 0.0 if rounded == 0.0 else rounded


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked chunk result with its unrounded in-memory BM25 score."""

    chunk_id: str
    doc_id: str
    path: str
    score: float
    start_line: int
    end_line: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable serialized representation of this hit."""

        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "path": self.path,
            "score": _rounded_score(self.score),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class DocumentHit:
    """A document-level result aggregated from one or more chunk hits."""

    doc_id: str
    best_score: float
    path: str
    chunk_ids: tuple[str, ...]
    best_chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable serialized representation of this document hit."""

        return {
            "doc_id": self.doc_id,
            "best_score": _rounded_score(self.best_score),
            "path": self.path,
            "chunk_ids": list(self.chunk_ids),
            "best_chunk_id": self.best_chunk_id,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class _DocumentAggregate:
    best_score: float
    best_chunk: SearchHit
    chunk_ids: tuple[str, ...]


class BM25Index:
    """One immutable, shared Okapi BM25 index over canonicalized chunks."""

    __slots__ = (
        "_bm25",
        "_chunks",
        "_corpus_fingerprint",
        "_epsilon",
        "_k1",
        "_b",
    )

    def __init__(
        self,
        chunks: Sequence[ChunkRecord],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        if not math.isfinite(k1) or k1 < 0:
            raise ValueError("k1 must be a finite non-negative number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be a finite non-negative number")

        ordered_chunks = tuple(sorted(chunks, key=lambda chunk: str(chunk.chunk_id)))
        chunk_ids = [str(chunk.chunk_id) for chunk in ordered_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunks must have unique chunk_id values")

        self._chunks = ordered_chunks
        self._k1 = float(k1)
        self._b = float(b)
        self._epsilon = float(epsilon)
        tokenized_corpus = [
            normalize_terms(str(_chunk_value(chunk, "text"))) for chunk in ordered_chunks
        ]
        self._bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self._k1,
            b=self._b,
            epsilon=self._epsilon,
        )
        self._corpus_fingerprint = self._build_fingerprint()

    @property
    def chunks(self) -> tuple[ChunkRecord, ...]:
        """Return the canonical chunk sequence used by the index."""

        return self._chunks

    @property
    def k1(self) -> float:
        """Return the frozen BM25 ``k1`` parameter."""

        return self._k1

    @property
    def b(self) -> float:
        """Return the frozen BM25 ``b`` parameter."""

        return self._b

    @property
    def epsilon(self) -> float:
        """Return the frozen BM25 epsilon parameter."""

        return self._epsilon

    @property
    def corpus_fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of corpus and scoring inputs."""

        return self._corpus_fingerprint

    def _build_fingerprint(self) -> str:
        payload = {
            "chunks": [
                [
                    str(_chunk_value(chunk, "chunk_id")),
                    str(_chunk_value(chunk, "text")),
                ]
                for chunk in self._chunks
            ],
            "bm25": {
                "b": self._b,
                "epsilon": self._epsilon,
                "k1": self._k1,
            },
            "tokenizer": _TOKENIZER_NAME,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _path_allowed(path: str, allowed_paths: tuple[str, ...] | None) -> bool:
        if allowed_paths is None:
            return True
        return any(
            path.startswith(allowed_path) if allowed_path.endswith("/") else path == allowed_path
            for allowed_path in allowed_paths
        )

    def search(
        self,
        query: str,
        top_k: int,
        allowed_paths: Iterable[str] | str | None = None,
    ) -> list[SearchHit]:
        """Return deterministic non-negative BM25 hits for *query*."""

        if top_k <= 0:
            return []
        if isinstance(allowed_paths, str):
            allowed = (allowed_paths,)
        elif allowed_paths is None:
            allowed = None
        else:
            allowed = tuple(str(path) for path in allowed_paths)

        raw_scores = self._bm25.get_scores(normalize_terms(query))
        ranked: list[tuple[float, ChunkRecord]] = []
        for chunk, score in zip(self._chunks, raw_scores, strict=True):
            raw_score = float(score)
            if raw_score < 0.0:
                continue
            if not self._path_allowed(str(chunk.path), allowed):
                continue
            ranked.append((raw_score, chunk))

        ranked.sort(key=lambda item: (-item[0], str(item[1].chunk_id)))
        return [
            SearchHit(
                chunk_id=str(chunk.chunk_id),
                doc_id=str(chunk.doc_id),
                path=str(chunk.path),
                score=raw_score,
                start_line=int(chunk.start_line),
                end_line=int(chunk.end_line),
                text=str(chunk.text),
            )
            for raw_score, chunk in ranked[:top_k]
        ]

    def document_hits(self, hits: Iterable[SearchHit]) -> list[DocumentHit]:
        """Aggregate chunk hits by document using each document's best score."""

        aggregates: dict[str, _DocumentAggregate] = {}
        for hit in hits:
            current = aggregates.get(hit.doc_id)
            if current is None:
                aggregates[hit.doc_id] = _DocumentAggregate(
                    best_score=hit.score,
                    best_chunk=hit,
                    chunk_ids=(hit.chunk_id,),
                )
                continue

            chunk_ids = tuple(sorted((*current.chunk_ids, hit.chunk_id)))
            if hit.score > current.best_score or (
                hit.score == current.best_score and hit.chunk_id < current.best_chunk.chunk_id
            ):
                aggregates[hit.doc_id] = _DocumentAggregate(
                    best_score=hit.score,
                    best_chunk=hit,
                    chunk_ids=chunk_ids,
                )
            else:
                aggregates[hit.doc_id] = _DocumentAggregate(
                    best_score=current.best_score,
                    best_chunk=current.best_chunk,
                    chunk_ids=chunk_ids,
                )

        ordered = sorted(
            aggregates.items(),
            key=lambda item: (-item[1].best_score, item[0]),
        )
        return [
            DocumentHit(
                doc_id=doc_id,
                best_score=aggregate.best_score,
                path=aggregate.best_chunk.path,
                chunk_ids=aggregate.chunk_ids,
                best_chunk_id=aggregate.best_chunk.chunk_id,
            )
            for doc_id, aggregate in ordered
        ]


__all__ = ["BM25Index", "DocumentHit", "SearchHit"]
