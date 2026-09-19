"""Deterministic initial-retrieval and accessed-evidence scoring."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.experiment.schemas import (
    AccessedEvidence,
    EvidenceMetrics,
    LineRange,
    RetrievalMetrics,
)

_FACT_ID = re.compile(r"^fact-\d{6}$")
_DOC_ID = re.compile(r"^doc-[a-z0-9]+(?:-[a-z0-9]+)*-\d{4}$")
_CHUNK_ID = re.compile(r"^doc-[a-z0-9]+(?:-[a-z0-9]+)*-\d{4}::c\d{4}$")

_MISSING = object()
_CONTENT_TOOLS = frozenset({"grep", "read", "cat"})


def _value(item: object, name: str, default: Any = None) -> Any:
    """Read a field from either a persisted mapping or a model/dataclass."""

    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _task_value(task: TaskRecord | Mapping[str, Any], name: str) -> Any:
    value = _value(task, name, _MISSING)
    if value is _MISSING:
        raise TypeError(f"task is missing {name}")
    return value


def _items(value: object) -> tuple[object, ...]:
    """Normalize one optional collection without treating strings as collections."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("evidence values must be collections, not strings")
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)


def _canonical_path(value: object) -> str | None:
    if value is None:
        return None
    path = str(value).replace("\\", "/")
    if not path:
        return None
    if not path.startswith("/"):
        path = "/" + path
    return PurePosixPath(path).as_posix()


def _valid_id(value: object, pattern: re.Pattern[str]) -> str | None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        return None
    return value


def _line_value(item: object, *names: str) -> int | None:
    for name in names:
        value = _value(item, name, _MISSING)
        if value is not _MISSING:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return None
            return value
    return None


def _overlaps(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return first_start <= second_end and second_start <= first_end


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class _FactSpan:
    fact_id: str
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class _Chunk:
    chunk_id: str
    doc_id: str | None
    path: str
    start_line: int
    end_line: int
    fact_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class _CorpusEvidence:
    """Internal path/fact index used to resolve hidden provenance."""

    facts: dict[str, list[_FactSpan]] = field(default_factory=dict)
    chunks: dict[str, _Chunk] = field(default_factory=dict)
    path_to_doc: dict[str, str] = field(default_factory=dict)

    def add_fact(self, span: _FactSpan) -> None:
        entries = self.facts.setdefault(span.fact_id, [])
        if span not in entries:
            entries.append(span)

    def add_chunk(self, chunk: _Chunk) -> None:
        current = self.chunks.get(chunk.chunk_id)
        if current is None:
            self.chunks[chunk.chunk_id] = chunk
            return
        fact_ids = tuple(sorted(set((*current.fact_ids, *chunk.fact_ids))))
        self.chunks[chunk.chunk_id] = _Chunk(
            chunk_id=current.chunk_id,
            doc_id=current.doc_id or chunk.doc_id,
            path=current.path,
            start_line=min(current.start_line, chunk.start_line),
            end_line=max(current.end_line, chunk.end_line),
            fact_ids=fact_ids,
        )


def _document_like(item: object) -> bool:
    return (
        _value(item, "path", _MISSING) is not _MISSING
        and _value(item, "facts", _MISSING) is not _MISSING
        and _value(item, "content", _MISSING) is not _MISSING
    )


def _add_document(index: _CorpusEvidence, document: object) -> None:
    path = _canonical_path(_value(document, "path"))
    doc_id = _valid_id(_value(document, "doc_id"), _DOC_ID)
    if path is None:
        return
    if doc_id is not None:
        index.path_to_doc[path] = doc_id

    for fact in _items(_value(document, "facts", ())):
        fact_id = _valid_id(_value(fact, "fact_id"), _FACT_ID)
        start_line = _line_value(fact, "line_start", "start_line")
        end_line = _line_value(fact, "line_end", "end_line")
        if fact_id is None or start_line is None or end_line is None:
            continue
        index.add_fact(_FactSpan(fact_id, path, start_line, end_line))


def _chunk_like(item: object) -> bool:
    return (
        _value(item, "chunk_id", _MISSING) is not _MISSING
        and _value(item, "path", _MISSING) is not _MISSING
        and _line_value(item, "start_line") is not None
        and _line_value(item, "end_line") is not None
    )


def _add_chunk(index: _CorpusEvidence, item: object) -> _Chunk | None:
    chunk_id = _valid_id(_value(item, "chunk_id"), _CHUNK_ID)
    path = _canonical_path(_value(item, "path"))
    start_line = _line_value(item, "start_line")
    end_line = _line_value(item, "end_line")
    if chunk_id is None or path is None or start_line is None or end_line is None:
        return None

    doc_id = _valid_id(_value(item, "doc_id"), _DOC_ID)
    if doc_id is not None:
        index.path_to_doc.setdefault(path, doc_id)

    fact_ids = tuple(
        sorted(
            {
                fact_id
                for fact_id in (
                    _valid_id(fact, _FACT_ID) for fact in _items(_value(item, "fact_ids", ()))
                )
                if fact_id is not None
            }
        )
    )
    chunk = _Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        path=path,
        start_line=start_line,
        end_line=end_line,
        fact_ids=fact_ids,
    )
    index.add_chunk(chunk)
    return chunk


def _add_source(index: _CorpusEvidence, source: object) -> None:
    """Collect documents and chunks from common corpus/container shapes."""

    if source is None:
        return
    if _document_like(source):
        _add_document(index, source)
        return
    if _chunk_like(source):
        _add_chunk(index, source)
        return

    nested_documents = _value(source, "documents", _MISSING)
    nested_chunks = _value(source, "chunks", _MISSING)
    if nested_documents is not _MISSING or nested_chunks is not _MISSING:
        if nested_documents is not _MISSING:
            for item in _items(nested_documents):
                _add_source(index, item)
        if nested_chunks is not _MISSING:
            for item in _items(nested_chunks):
                _add_source(index, item)
        return

    if isinstance(source, Mapping):
        for key in ("documents", "chunks"):
            if key in source:
                for item in _items(source[key]):
                    _add_source(index, item)
        return

    if isinstance(source, (str, bytes, bytearray)):
        return
    try:
        for item in source:  # type: ignore[operator]
            _add_source(index, item)
    except TypeError:
        return


def _complete_index(sources: Iterable[object]) -> _CorpusEvidence:
    index = _CorpusEvidence()
    for source in sources:
        _add_source(index, source)

    # Chunk records normally carry fact_ids. For imported or hand-built records
    # without them, exact document spans provide the same hidden provenance.
    for chunk_id, chunk in tuple(index.chunks.items()):
        inferred = {
            fact_id
            for fact_id, spans in index.facts.items()
            if any(
                span.path == chunk.path
                and _overlaps(
                    span.start_line,
                    span.end_line,
                    chunk.start_line,
                    chunk.end_line,
                )
                for span in spans
            )
        }
        fact_ids = tuple(sorted(set((*chunk.fact_ids, *inferred))))
        if fact_ids != chunk.fact_ids:
            index.chunks[chunk_id] = _Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id or index.path_to_doc.get(chunk.path),
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                fact_ids=fact_ids,
            )

    # A chunk-only input still contains useful span provenance through its
    # fact_ids. Use the chunk range as the conservative hidden span when no
    # exact FactSpan was supplied.
    for chunk in index.chunks.values():
        for fact_id in chunk.fact_ids:
            if fact_id not in index.facts:
                index.add_fact(
                    _FactSpan(
                        fact_id=fact_id,
                        path=chunk.path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                    )
                )

    for chunk in index.chunks.values():
        if chunk.doc_id is not None:
            index.path_to_doc.setdefault(chunk.path, chunk.doc_id)
    return index


def _range_from(value: object) -> tuple[str, int, int] | None:
    path = _canonical_path(_value(value, "path"))
    start_line = _line_value(value, "start_line")
    end_line = _line_value(value, "end_line")
    if path is None or start_line is None or end_line is None or end_line < start_line:
        return None
    return path, start_line, end_line


def _trace_result(entry: object) -> object:
    result = _value(entry, "result", _MISSING)
    return entry if result is _MISSING else result


def _trace_tool(entry: object, result: object) -> str | None:
    tool = _value(result, "tool", _MISSING)
    if tool is _MISSING:
        request = _value(entry, "request", None)
        tool = _value(request, "tool", None)
    if tool is None:
        return None
    return getattr(tool, "value", str(tool))


def _successful_content_ranges(entry: object) -> tuple[tuple[str, int, int], ...]:
    result = _trace_result(entry)
    if _value(result, "ok", False) is not True:
        return ()
    if _trace_tool(entry, result) not in _CONTENT_TOOLS:
        return ()
    if not isinstance(_value(result, "content", None), str):
        return ()
    return tuple(
        line_range
        for value in _items(_value(result, "line_ranges", ()))
        if (line_range := _range_from(value)) is not None
    )


def _ordered_trace(trace: Sequence[object] | Iterable[object]) -> tuple[object, ...]:
    entries = tuple(trace)
    if not entries:
        return ()
    if all(_line_value(entry, "call_ordinal") is not None for entry in entries):
        return tuple(
            sorted(
                entries,
                key=lambda entry: _line_value(entry, "call_ordinal") or 0,
            )
        )
    return entries


def _merge_ranges(ranges: Iterable[tuple[str, int, int]]) -> tuple[LineRange, ...]:
    by_path: dict[str, list[tuple[int, int]]] = {}
    for path, start_line, end_line in ranges:
        by_path.setdefault(path, []).append((start_line, end_line))

    merged: list[LineRange] = []
    for path in sorted(by_path):
        current_start: int | None = None
        current_end: int | None = None
        for start_line, end_line in sorted(by_path[path]):
            if current_start is None:
                current_start, current_end = start_line, end_line
            elif start_line <= (current_end or start_line):
                current_end = max(current_end or end_line, end_line)
            else:
                merged.append(
                    LineRange(
                        path=path,
                        start_line=current_start,
                        end_line=current_end or current_start,
                    )
                )
                current_start, current_end = start_line, end_line
        if current_start is not None:
            merged.append(
                LineRange(
                    path=path,
                    start_line=current_start,
                    end_line=current_end or current_start,
                )
            )
    return tuple(merged)


def _facts_for_ranges(
    index: _CorpusEvidence,
    ranges: Iterable[tuple[str, int, int]],
) -> set[str]:
    exposed: set[str] = set()
    for path, start_line, end_line in ranges:
        for fact_id, spans in index.facts.items():
            if any(
                span.path == path
                and _overlaps(span.start_line, span.end_line, start_line, end_line)
                for span in spans
            ):
                exposed.add(fact_id)
    return exposed


def _chunks_for_ranges(
    index: _CorpusEvidence,
    ranges: Iterable[tuple[str, int, int]],
) -> tuple[set[str], set[str]]:
    chunk_ids: set[str] = set()
    doc_ids: set[str] = set()
    for path, start_line, end_line in ranges:
        if path in index.path_to_doc:
            doc_ids.add(index.path_to_doc[path])
        for chunk in index.chunks.values():
            if chunk.path != path or not _overlaps(
                chunk.start_line,
                chunk.end_line,
                start_line,
                end_line,
            ):
                continue
            chunk_ids.add(chunk.chunk_id)
            if chunk.doc_id is not None:
                doc_ids.add(chunk.doc_id)
    return chunk_ids, doc_ids


def _chunk_from_exposed(index: _CorpusEvidence, item: object) -> _Chunk | None:
    chunk_id = _valid_id(_value(item, "chunk_id"), _CHUNK_ID)
    if chunk_id is not None and chunk_id in index.chunks:
        return index.chunks[chunk_id]
    if _chunk_like(item):
        path = _canonical_path(_value(item, "path"))
        start_line = _line_value(item, "start_line")
        end_line = _line_value(item, "end_line")
        if path is not None and start_line is not None and end_line is not None:
            return _Chunk(
                chunk_id=chunk_id or "",
                doc_id=_valid_id(_value(item, "doc_id"), _DOC_ID),
                path=path,
                start_line=start_line,
                end_line=end_line,
                fact_ids=tuple(
                    sorted(
                        fact_id
                        for fact_id in (
                            _valid_id(fact, _FACT_ID)
                            for fact in _items(_value(item, "fact_ids", ()))
                        )
                        if fact_id is not None
                    )
                ),
            )
    return None


def derive_accessed_evidence(
    task: TaskRecord | Mapping[str, Any],
    chunks: Sequence[object] | Iterable[object] = (),
    trace: Sequence[object] | Iterable[object] = (),
    *,
    snippet_chunks: Sequence[object] | Iterable[object] | None = None,
    corpus_chunks: Sequence[object] | Iterable[object] | None = None,
) -> AccessedEvidence:
    """Derive model-visible evidence from snippets and successful VFS results.

    ``chunks`` is the exposed snippet sequence for a snippet run. For VFS
    runs, callers can pass the complete corpus chunk sequence so trace ranges
    can be mapped back to hidden fact spans; VFS candidate paths are not
    passed as snippet chunks. ``corpus_chunks`` and ``snippet_chunks`` are
    explicit forms for callers that have both sequences available.
    """

    chunk_values = tuple(chunks)
    trace_values = tuple(trace)
    source_values: list[object] = list(chunk_values)
    if corpus_chunks is not None:
        source_values.extend(tuple(corpus_chunks))
    if snippet_chunks is not None:
        source_values.extend(tuple(snippet_chunks))

    # A mapping/container may carry the documents alongside its chunks.
    source_values.append(task)
    index = _complete_index(source_values)

    direct_values = (
        tuple(snippet_chunks)
        if snippet_chunks is not None
        else chunk_values
        if not trace_values
        else ()
    )
    direct_ranges: list[tuple[str, int, int]] = []
    direct_fact_ids: set[str] = set()
    direct_chunk_ids: set[str] = set()
    direct_doc_ids: set[str] = set()
    for item in direct_values:
        chunk = _chunk_from_exposed(index, item)
        if chunk is None:
            continue
        direct_ranges.append((chunk.path, chunk.start_line, chunk.end_line))
        direct_fact_ids.update(chunk.fact_ids)
        if _valid_id(chunk.chunk_id, _CHUNK_ID) is not None:
            direct_chunk_ids.add(chunk.chunk_id)
        if chunk.doc_id is not None:
            direct_doc_ids.add(chunk.doc_id)

    trace_ranges = [
        line_range
        for entry in _ordered_trace(trace_values)
        for line_range in _successful_content_ranges(entry)
    ]
    all_ranges = [*direct_ranges, *trace_ranges]
    range_chunk_ids, range_doc_ids = _chunks_for_ranges(index, all_ranges)
    facts = _facts_for_ranges(index, all_ranges)
    facts.update(direct_fact_ids)

    return AccessedEvidence(
        fact_ids=sorted(fact_id for fact_id in facts if _FACT_ID.fullmatch(fact_id)),
        doc_ids=sorted(
            {doc_id for doc_id in (*direct_doc_ids, *range_doc_ids) if _DOC_ID.fullmatch(doc_id)}
        ),
        chunk_ids=sorted(
            {
                chunk_id
                for chunk_id in (*direct_chunk_ids, *range_chunk_ids)
                if _CHUNK_ID.fullmatch(chunk_id)
            }
        ),
        line_ranges=list(_merge_ranges(all_ranges)),
    )


def _looks_like_task(value: object) -> bool:
    return _value(value, "gold_chunk_ids", _MISSING) is not _MISSING


def _retrieval_id(value: object, name: str) -> str | None:
    if isinstance(value, str):
        if name == "doc_id" and _DOC_ID.fullmatch(value):
            return value
        if name == "chunk_id" and _CHUNK_ID.fullmatch(value):
            return value
    result = _value(value, name, None)
    if result is not None:
        return str(result)
    if name == "doc_id":
        chunk_id = _value(value, "chunk_id", None)
        if isinstance(chunk_id, str) and "::" in chunk_id:
            return chunk_id.split("::", maxsplit=1)[0]
    return None


def score_retrieval(
    task: TaskRecord | Mapping[str, Any] | Sequence[object],
    chunks: Sequence[object] | TaskRecord | Mapping[str, Any] = (),
    documents: Sequence[object] | int | None = None,
    k: int | None = None,
    *,
    top_k: int | None = None,
    initial_chunks: Sequence[object] | None = None,
    initial_documents: Sequence[object] | None = None,
    initial_retrieved_chunks: Sequence[object] | None = None,
    initial_retrieved_documents: Sequence[object] | None = None,
) -> RetrievalMetrics:
    """Score unique initial chunk and document recall separately."""

    if not _looks_like_task(task) and _looks_like_task(chunks):
        task, chunks = chunks, task  # type: ignore[assignment]

    if initial_chunks is not None:
        chunks = initial_chunks
    if initial_retrieved_chunks is not None:
        chunks = initial_retrieved_chunks
    if initial_documents is not None:
        documents = initial_documents
    if initial_retrieved_documents is not None:
        documents = initial_retrieved_documents

    if top_k is not None:
        if k is not None:
            raise TypeError("retrieval k was supplied twice")
        k = top_k
    if isinstance(documents, int):
        if k is not None:
            raise TypeError("retrieval k was supplied twice")
        k = documents
        documents = None
    explicit_k = k is not None
    if k is None:
        k = len(tuple(chunks)) if not isinstance(chunks, Mapping) else 1
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")

    chunk_values = tuple(chunks) if not isinstance(chunks, Mapping) else (chunks,)
    if explicit_k:
        chunk_values = chunk_values[:k]
    gold_chunks = set(_task_value(task, "gold_chunk_ids"))  # type: ignore[arg-type]
    gold_documents = set(_task_value(task, "gold_document_ids"))  # type: ignore[arg-type]

    retrieved_chunks = {
        chunk_id
        for chunk_id in (_retrieval_id(item, "chunk_id") for item in chunk_values)
        if chunk_id is not None
    }
    chunk_doc_ids = {
        doc_id
        for doc_id in (_retrieval_id(item, "doc_id") for item in chunk_values)
        if doc_id is not None
    }

    if documents is None:
        retrieved_documents = chunk_doc_ids
    else:
        document_values = _items(documents)
        if explicit_k:
            document_values = document_values[:k]
        retrieved_documents = {
            doc_id
            for doc_id in (_retrieval_id(item, "doc_id") for item in document_values)
            if doc_id is not None
        }

    return RetrievalMetrics(
        initial_chunk_recall=_ratio(len(retrieved_chunks & gold_chunks), len(gold_chunks)),
        initial_document_recall=_ratio(
            len(retrieved_documents & gold_documents),
            len(gold_documents),
        ),
        k=k,
    )


def _evidence_ids(accessed: object) -> set[str]:
    return {
        fact_id
        for fact_id in (
            _valid_id(value, _FACT_ID) for value in _items(_value(accessed, "fact_ids", ()))
        )
        if fact_id is not None
    }


def _citation_values(
    task: TaskRecord | Mapping[str, Any],
    citation_score: object,
    citations: Sequence[str] | None,
) -> tuple[float, float]:
    if citations is not None:
        from bm25_vfs_ablation.evaluation.answer_scoring import score_citations

        score = score_citations(citations, task)
        return score.precision, score.recall

    if citation_score is None:
        return 0.0, 0.0
    if isinstance(citation_score, Sequence) and not isinstance(
        citation_score, (str, bytes, bytearray)
    ):
        from bm25_vfs_ablation.evaluation.answer_scoring import score_citations

        score = score_citations(citation_score, task)
        return score.precision, score.recall
    if isinstance(citation_score, Mapping):
        precision = citation_score.get(
            "precision",
            citation_score.get("citation_precision", 0.0),
        )
        recall = citation_score.get(
            "recall",
            citation_score.get("citation_recall", 0.0),
        )
    else:
        precision = getattr(
            citation_score,
            "precision",
            getattr(citation_score, "citation_precision", 0.0),
        )
        recall = getattr(
            citation_score,
            "recall",
            getattr(citation_score, "citation_recall", 0.0),
        )
    if not isinstance(precision, (int, float)) or isinstance(precision, bool):
        precision = 0.0
    if not isinstance(recall, (int, float)) or isinstance(recall, bool):
        recall = 0.0
    return float(precision), float(recall)


def score_evidence(
    task: TaskRecord | Mapping[str, Any] | object,
    accessed: AccessedEvidence | Mapping[str, Any] | object,
    citation_score: object = None,
    citations: Sequence[str] | None = None,
    citation_precision: float | None = None,
    citation_recall: float | None = None,
) -> EvidenceMetrics:
    """Score required-fact recall and exposed-fact precision."""

    if not _looks_like_task(task) and _looks_like_task(accessed):
        task, accessed = accessed, task

    required = {
        fact_id
        for fact_id in (
            _valid_id(value, _FACT_ID) for value in _items(_task_value(task, "required_fact_ids"))
        )
        if fact_id is not None
    }
    exposed = _evidence_ids(accessed)
    exposed_required = required & exposed

    if citation_precision is None or citation_recall is None:
        score_precision, score_recall = _citation_values(task, citation_score, citations)
        if citation_precision is None:
            citation_precision = score_precision
        if citation_recall is None:
            citation_recall = score_recall

    return EvidenceMetrics(
        final_recall=_ratio(len(exposed_required), len(required)),
        precision=_ratio(len(exposed_required), len(exposed)),
        distinct_gold_facts=len(exposed_required),
        all_required_accessed=required <= exposed,
        citation_precision=max(0.0, min(1.0, float(citation_precision))),
        citation_recall=max(0.0, min(1.0, float(citation_recall))),
    )


def complete_coverage_call_ordinal(
    task: TaskRecord | Mapping[str, Any],
    chunks: Sequence[object] | Iterable[object],
    trace: Sequence[object] | Iterable[object],
    *,
    corpus_chunks: Sequence[object] | Iterable[object] | None = None,
) -> int | None:
    """Return the first ordered tool-call ordinal exposing every required fact."""

    sources: list[object] = list(chunks)
    if corpus_chunks is not None:
        sources.extend(tuple(corpus_chunks))
    sources.append(task)
    index = _complete_index(sources)
    required = set(_task_value(task, "required_fact_ids"))
    exposed: set[str] = set()

    for fallback_ordinal, entry in enumerate(_ordered_trace(tuple(trace)), start=1):
        ranges = _successful_content_ranges(entry)
        if not ranges:
            continue
        exposed.update(_facts_for_ranges(index, ranges))
        if required <= exposed:
            ordinal = _line_value(entry, "call_ordinal")
            return ordinal if ordinal is not None else fallback_ordinal
    return None


coverage_call_ordinal = complete_coverage_call_ordinal
calls_to_complete_gold_coverage = complete_coverage_call_ordinal


__all__ = [
    "calls_to_complete_gold_coverage",
    "complete_coverage_call_ordinal",
    "coverage_call_ordinal",
    "derive_accessed_evidence",
    "score_evidence",
    "score_retrieval",
]
