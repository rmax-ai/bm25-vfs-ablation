"""Shared normalization, token approximation, and deterministic chunking."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

_TERM_PATTERN = re.compile(
    r"[^\W_]+(?:['-][^\W_]+)*",
    flags=re.UNICODE,
)
_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:['-][^\W_]+)*|[^\w\s]",
    flags=re.UNICODE,
)


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def normalize_terms(text: str) -> list[str]:
    """Return normalized retrieval terms, retaining stopwords."""

    return _TERM_PATTERN.findall(_normalized(text))


def estimate_tokens(text: str) -> int:
    """Estimate tokens with the frozen word-plus-punctuation regex."""

    return len(_TOKEN_PATTERN.findall(_normalized(text)))


def _long_line_tokens(line: str) -> list[str]:
    """Return display tokens whose normalized count matches the source line."""

    display_tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(line):
        normalized_tokens = _TOKEN_PATTERN.findall(_normalized(match.group()))
        if len(normalized_tokens) == 1:
            display_tokens.append(match.group())
        else:
            display_tokens.extend(normalized_tokens)

    if len(display_tokens) != estimate_tokens(line):
        return _TOKEN_PATTERN.findall(_normalized(line))
    return display_tokens


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _path(value: Any) -> str:
    # Paths in persisted documents are already root-relative.  Replacing
    # separators here also keeps imported records platform-independent.
    return PurePosixPath(str(value).replace("\\", "/")).as_posix()


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One deterministic retrieval chunk and its source-line provenance."""

    chunk_id: str
    doc_id: str
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    text: str
    token_count: int
    fact_ids: tuple[str, ...]


class Chunker:
    """Build line-aware, bounded chunks with deterministic source overlap."""

    def __init__(self, size_tokens: int = 180, overlap_tokens: int = 30) -> None:
        if size_tokens <= 0:
            raise ValueError("size_tokens must be greater than zero")
        if overlap_tokens < 0 or overlap_tokens >= size_tokens:
            raise ValueError("overlap_tokens must satisfy 0 <= overlap < size")
        self.size_tokens = size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, documents: Iterable[Any]) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        ordered = sorted(documents, key=lambda document: str(_value(document, "doc_id", "")))
        for document in ordered:
            records.extend(self._chunk_document(document))
        return records

    def _chunk_document(self, document: Any) -> list[ChunkRecord]:
        doc_id = str(_value(document, "doc_id"))
        path = _path(_value(document, "path", ""))
        content = str(_value(document, "content", ""))
        lines = content.splitlines()
        if not lines:
            return []

        line_tokens = [_TOKEN_PATTERN.findall(_normalized(line)) for line in lines]
        source_line_tokens = [_long_line_tokens(line) for line in lines]
        chunks: list[ChunkRecord] = []
        start = 0
        while start < len(lines):
            tokens = line_tokens[start]
            if len(tokens) > self.size_tokens:
                step = self.size_tokens - self.overlap_tokens
                segment_start = 0
                while segment_start < len(tokens):
                    segment_end = min(segment_start + self.size_tokens, len(tokens))
                    self._add_chunk(
                        chunks,
                        document,
                        doc_id,
                        path,
                        start,
                        start,
                        " ".join(source_line_tokens[start][segment_start:segment_end]),
                        segment_end - segment_start,
                    )
                    if segment_end == len(tokens):
                        break
                    segment_start += step
                start += 1
                continue

            end = start
            total = 0
            while end < len(lines):
                line_count = len(line_tokens[end])
                if end > start and total + line_count > self.size_tokens:
                    break
                total += line_count
                end += 1
                if total >= self.size_tokens:
                    break
            if end == start:
                end += 1

            if total:
                text = "\n".join(lines[start:end])
                self._add_chunk(
                    chunks,
                    document,
                    doc_id,
                    path,
                    start,
                    end - 1,
                    text,
                    estimate_tokens(text),
                )

            if total <= self.overlap_tokens:
                start = end
                continue

            remaining = self.overlap_tokens
            overlap_start = end - 1
            while overlap_start >= start and remaining > 0:
                remaining -= len(line_tokens[overlap_start])
                overlap_start -= 1
            next_start = overlap_start + 1
            start = next_start if next_start > start else end

        return chunks

    def _add_chunk(
        self,
        chunks: list[ChunkRecord],
        document: Any,
        doc_id: str,
        path: str,
        start: int,
        end: int,
        text: str,
        token_count: int,
    ) -> None:
        if token_count <= 0:
            return
        start_line = start + 1
        end_line = end + 1
        fact_ids = tuple(
            sorted(
                str(_value(fact, "fact_id"))
                for fact in (_value(document, "facts", ()) or ())
                if int(_value(fact, "line_start", 0)) <= end_line
                and int(_value(fact, "line_end", 0)) >= start_line
            )
        )
        index = len(chunks)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc_id}::c{index:04d}",
                doc_id=doc_id,
                path=path,
                chunk_index=index,
                start_line=start_line,
                end_line=end_line,
                text=text,
                token_count=token_count,
                fact_ids=fact_ids,
            )
        )


__all__ = ["ChunkRecord", "Chunker", "estimate_tokens", "normalize_terms"]
