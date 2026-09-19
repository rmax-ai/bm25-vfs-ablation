"""Deterministic scoring for the public model answer envelope."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.experiment.schemas import ParsedAnswer, SecondaryJudge


class AnswerParseError(ValueError):
    """Raised internally when a response is not a valid public answer."""


def normalize_answer(value: str) -> str:
    """Apply the frozen answer normalization rules in order."""

    if not isinstance(value, str):
        raise TypeError("answer must be a string")

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.strip().split())
    return normalized.rstrip(".?!")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnswerParseError("answer JSON contains duplicate keys")
        result[key] = value
    return result


def _deduplicate_citations(citations: Iterable[str]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        if citation not in seen:
            seen.add(citation)
            unique.append(citation)
    return tuple(unique)


def parse_answer(raw: str) -> ParsedAnswer | None:
    """Parse one answer object, returning ``None`` for malformed responses.

    The raw response must contain only one JSON object, apart from surrounding
    whitespace. Citation duplicates are accepted in the public response and
    normalized in first-seen order for the persisted parsed answer.
    """

    if not isinstance(raw, str):
        return None

    try:
        payload = json.loads(raw.strip(), object_pairs_hook=_reject_duplicate_keys)
    except (AnswerParseError, json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    answer = payload.get("answer")
    citations = payload.get("citations")
    justification = payload.get("justification")
    if (
        not isinstance(answer, str)
        or not isinstance(citations, list)
        or any(not isinstance(citation, str) for citation in citations)
        or not isinstance(justification, str)
    ):
        return None

    normalized_payload = {
        "answer": answer,
        "citations": list(_deduplicate_citations(citations)),
        "justification": justification,
    }
    try:
        return ParsedAnswer.model_validate(normalized_payload)
    except (TypeError, ValueError):
        return None


def _task_value(task: TaskRecord | Mapping[str, Any], field_name: str) -> Any:
    if isinstance(task, Mapping):
        return task[field_name]
    return getattr(task, field_name)


def _as_strings(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) for value in result):
        raise TypeError(f"{field_name} must contain strings")
    return result


@dataclass(frozen=True, slots=True)
class CitationScore:
    """Deterministic document- and chunk-level citation metrics."""

    citations: tuple[str, ...]
    correct_citations: tuple[str, ...]
    document_citations: tuple[str, ...]
    chunk_citations: tuple[str, ...]
    correct_document_citations: tuple[str, ...]
    correct_chunk_citations: tuple[str, ...]
    covered_document_ids: tuple[str, ...]
    precision: float
    recall: float
    document_precision: float
    document_recall: float
    chunk_precision: float
    chunk_recall: float

    @property
    def unique_citations(self) -> tuple[str, ...]:
        """Return the deduplicated citations in first-seen order."""

        return self.citations

    @property
    def citation_precision(self) -> float:
        """Compatibility name for the overall citation precision."""

        return self.precision

    @property
    def citation_recall(self) -> float:
        """Compatibility name for the overall citation recall."""

        return self.recall

    @property
    def document_citation_precision(self) -> float:
        """Return precision among document-shaped citations."""

        return self.document_precision

    @property
    def document_citation_recall(self) -> float:
        """Return gold-document coverage from document or chunk citations."""

        return self.document_recall

    @property
    def chunk_citation_precision(self) -> float:
        """Return precision among chunk-shaped citations."""

        return self.chunk_precision

    @property
    def chunk_citation_recall(self) -> float:
        """Return recall among required chunk IDs."""

        return self.chunk_recall

    def to_dict(self) -> dict[str, Any]:
        """Return stable fields suitable for run-record scoring details."""

        return {
            "citations": list(self.citations),
            "correct_citations": list(self.correct_citations),
            "document_citations": list(self.document_citations),
            "chunk_citations": list(self.chunk_citations),
            "correct_document_citations": list(self.correct_document_citations),
            "correct_chunk_citations": list(self.correct_chunk_citations),
            "covered_document_ids": list(self.covered_document_ids),
            "precision": self.precision,
            "recall": self.recall,
            "document_precision": self.document_precision,
            "document_recall": self.document_recall,
            "chunk_precision": self.chunk_precision,
            "chunk_recall": self.chunk_recall,
        }

    as_dict = to_dict


def score_citations(
    citations: Sequence[str] | TaskRecord | Mapping[str, Any],
    gold_document_ids: Sequence[str] | TaskRecord | Mapping[str, Any],
    gold_chunk_ids: Sequence[str] | None = None,
) -> CitationScore:
    """Score unique citations against gold document and chunk identifiers.

    The usual call is ``score_citations(citations, task)`` or
    ``score_citations(citations, gold_document_ids, gold_chunk_ids)``. The
    reversed ``score_citations(task, citations)`` form is accepted as a small
    convenience for callers constructing a score from a task first.
    """

    task: TaskRecord | Mapping[str, Any] | None = None
    if isinstance(citations, (TaskRecord, Mapping)):
        task = citations
        citations = gold_document_ids
        gold_document_ids = _task_value(task, "gold_document_ids")
        if gold_chunk_ids is None:
            gold_chunk_ids = _task_value(task, "gold_chunk_ids")
    elif isinstance(gold_document_ids, (TaskRecord, Mapping)):
        task = gold_document_ids
        gold_document_ids = _task_value(task, "gold_document_ids")
        if gold_chunk_ids is None:
            gold_chunk_ids = _task_value(task, "gold_chunk_ids")

    if task is not None and gold_chunk_ids is None:
        gold_chunk_ids = _task_value(task, "gold_chunk_ids")

    if isinstance(citations, str):
        raise TypeError("citations must be a sequence of strings")
    if gold_chunk_ids is None:
        raise TypeError("gold_chunk_ids is required")

    unique_citations = _deduplicate_citations(_as_strings(citations, "citations"))
    gold_documents = _as_strings(gold_document_ids, "gold_document_ids")
    gold_chunks = _as_strings(gold_chunk_ids, "gold_chunk_ids")
    gold_document_set = set(gold_documents)
    gold_chunk_set = set(gold_chunks)

    document_citations = tuple(
        citation for citation in unique_citations if "::" not in citation
    )
    chunk_citations = tuple(citation for citation in unique_citations if "::" in citation)
    correct_document_citations = tuple(
        citation for citation in document_citations if citation in gold_document_set
    )
    correct_chunk_citations = tuple(
        citation for citation in chunk_citations if citation in gold_chunk_set
    )
    correct_citations = tuple(
        citation
        for citation in unique_citations
        if citation in gold_document_set or citation in gold_chunk_set
    )

    covered_documents = set(correct_document_citations)
    covered_documents.update(
        citation.split("::", maxsplit=1)[0] for citation in correct_chunk_citations
    )
    covered_document_ids = tuple(
        document_id for document_id in gold_documents if document_id in covered_documents
    )

    precision = _ratio(len(correct_citations), len(unique_citations))
    recall = _ratio(len(covered_document_ids), len(gold_documents))
    document_precision = _ratio(
        len(correct_document_citations), len(document_citations)
    )
    document_recall = recall
    chunk_precision = _ratio(len(correct_chunk_citations), len(chunk_citations))
    chunk_recall = _ratio(
        len(set(correct_chunk_citations)), len(gold_chunks)
    )

    return CitationScore(
        citations=unique_citations,
        correct_citations=correct_citations,
        document_citations=document_citations,
        chunk_citations=chunk_citations,
        correct_document_citations=correct_document_citations,
        correct_chunk_citations=correct_chunk_citations,
        covered_document_ids=covered_document_ids,
        precision=precision,
        recall=recall,
        document_precision=document_precision,
        document_recall=document_recall,
        chunk_precision=chunk_precision,
        chunk_recall=chunk_recall,
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class AnswerScore:
    """Complete deterministic primary answer score."""

    raw: str
    parsed_answer: ParsedAnswer | None
    normalized_answer: str | None
    normalized_canonical_answer: str
    normalized_variants: tuple[str, ...]
    matched_variant: str | None
    correctness: bool
    citation_score: CitationScore
    secondary_judge: SecondaryJudge
    parse_status: str

    @property
    def parsed(self) -> ParsedAnswer | None:
        """Return the parsed public envelope."""

        return self.parsed_answer

    @property
    def correct(self) -> bool:
        """Return primary deterministic correctness."""

        return self.correctness

    @property
    def is_correct(self) -> bool:
        """Return primary deterministic correctness."""

        return self.correctness

    @property
    def citations(self) -> tuple[str, ...]:
        """Return deduplicated citations in first-seen order."""

        return self.citation_score.citations

    @property
    def raw_answer(self) -> str:
        """Return the exact unmodified model response."""

        return self.raw

    @property
    def scoring_details(self) -> dict[str, Any]:
        """Return the observable scoring fields for a run record."""

        return {
            "parse_status": self.parse_status,
            "normalized_answer": self.normalized_answer,
            "normalized_canonical_answer": self.normalized_canonical_answer,
            "normalized_variants": list(self.normalized_variants),
            "matched_variant": self.matched_variant,
            "citation_metrics": self.citation_score.to_dict(),
            "secondary_judge": self.secondary_judge.model_dump(mode="json"),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible score representation."""

        return {
            "raw": self.raw,
            "parsed_answer": (
                self.parsed_answer.model_dump(mode="json")
                if self.parsed_answer is not None
                else None
            ),
            "normalized_answer": self.normalized_answer,
            "normalized_canonical_answer": self.normalized_canonical_answer,
            "normalized_variants": list(self.normalized_variants),
            "matched_variant": self.matched_variant,
            "correctness": self.correctness,
            "citation_score": self.citation_score.to_dict(),
            "secondary_judge": self.secondary_judge.model_dump(mode="json"),
            "parse_status": self.parse_status,
        }

    as_dict = to_dict


def score_answer(
    task: TaskRecord | Mapping[str, Any],
    raw: str,
) -> AnswerScore:
    """Score one raw response using exact normalized answer equality."""

    canonical_answer = _task_value(task, "canonical_answer")
    variants = _task_value(task, "acceptable_answer_variants")
    normalized_canonical = normalize_answer(canonical_answer)
    normalized_variants = tuple(normalize_answer(variant) for variant in variants)

    parsed = parse_answer(raw)
    if parsed is None:
        citation_score = score_citations(
            (),
            _task_value(task, "gold_document_ids"),
            _task_value(task, "gold_chunk_ids"),
        )
        return AnswerScore(
            raw=raw,
            parsed_answer=None,
            normalized_answer=None,
            normalized_canonical_answer=normalized_canonical,
            normalized_variants=normalized_variants,
            matched_variant=None,
            correctness=False,
            citation_score=citation_score,
            secondary_judge=_disabled_secondary_judge(),
            parse_status="malformed",
        )

    normalized_answer = normalize_answer(parsed.answer)
    expected_answers = (canonical_answer, *variants)
    normalized_expected = (normalized_canonical, *normalized_variants)
    matched_variant: str | None = None
    for expected, normalized_expected_value in zip(
        expected_answers, normalized_expected, strict=True
    ):
        if normalized_answer == normalized_expected_value:
            matched_variant = expected
            break

    citation_score = score_citations(
        parsed.citations,
        _task_value(task, "gold_document_ids"),
        _task_value(task, "gold_chunk_ids"),
    )
    return AnswerScore(
        raw=raw,
        parsed_answer=parsed,
        normalized_answer=normalized_answer,
        normalized_canonical_answer=normalized_canonical,
        normalized_variants=normalized_variants,
        matched_variant=matched_variant,
        correctness=matched_variant is not None,
        citation_score=citation_score,
        secondary_judge=_disabled_secondary_judge(),
        parse_status="valid",
    )


def _disabled_secondary_judge() -> SecondaryJudge:
    return SecondaryJudge(
        enabled=False,
        score=None,
        rationale=None,
        model_id=None,
    )


__all__ = [
    "AnswerParseError",
    "AnswerScore",
    "CitationScore",
    "normalize_answer",
    "parse_answer",
    "score_answer",
    "score_citations",
]
