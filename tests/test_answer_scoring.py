"""Acceptance tests for deterministic answer and citation scoring."""

from __future__ import annotations

import json

from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.evaluation.answer_scoring import (
    normalize_answer,
    parse_answer,
    score_answer,
)


def _task() -> TaskRecord:
    return TaskRecord(
        schema_version=1,
        task_id="task-eval-000001",
        split="eval",
        question="Does the service need a review?",
        canonical_answer="yes",
        acceptable_answer_variants=["yes, a review is required"],
        required_fact_ids=["fact-000001", "fact-000002"],
        gold_document_ids=["doc-policies-0001", "doc-services-0001"],
        gold_chunk_ids=[
            "doc-policies-0001::c0000",
            "doc-services-0001::c0000",
        ],
        hop_count=2,
        task_template="policy-review",
        category="security",
        distractor_document_ids=[],
        distractor_count=0,
        generator_seed=42,
        corpus_version="synthetic-v1",
    )


def _raw(
    *,
    answer: str = "yes",
    citations: list[str] | None = None,
    justification: str = "The linked records require review.",
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": citations if citations is not None else [],
            "justification": justification,
        }
    )


def test_answer_normalization_exact() -> None:
    assert normalize_answer("  ＹＥＳ\t\nrequired?!  ") == "yes required"
    assert normalize_answer("A.B, really!") == "a.b, really"


def test_variant_exact_match() -> None:
    score = score_answer(
        _task(),
        _raw(answer="  YES, A REVIEW IS REQUIRED!!! "),
    )

    assert score.correctness is True
    assert score.matched_variant == "yes, a review is required"
    assert score.normalized_answer == "yes, a review is required"


def test_substring_is_not_correct() -> None:
    score = score_answer(
        _task(),
        _raw(answer="yes, a review is required because of the policy"),
    )

    assert score.correctness is False
    assert score.matched_variant is None


def test_malformed_answer_is_incorrect() -> None:
    raw = '{"answer":"yes","citations":[]}'
    score = score_answer(_task(), raw)

    assert score.correctness is False
    assert score.parsed_answer is None
    assert score.raw == raw
    assert score.parse_status == "malformed"
    assert score.secondary_judge.enabled is False
    assert score.secondary_judge.score is None


def test_markdown_fence_rejected() -> None:
    raw = f"```json\n{_raw()}\n```"

    assert parse_answer(raw) is None
    assert score_answer(_task(), raw).correctness is False


def test_citation_precision_and_recall() -> None:
    score = score_answer(
        _task(),
        _raw(
            citations=[
                "doc-services-0001::c0000",
                "doc-policies-0001",
                "doc-services-0001::c0000",
                "doc-missing-0001",
                "doc-policies-0001::c0001",
            ]
        ),
    )

    assert score.citations == (
        "doc-services-0001::c0000",
        "doc-policies-0001",
        "doc-missing-0001",
        "doc-policies-0001::c0001",
    )
    assert score.citation_score.precision == 0.5
    assert score.citation_score.recall == 1.0
    assert score.citation_score.document_precision == 0.5
    assert score.citation_score.document_recall == 1.0
    assert score.citation_score.chunk_precision == 0.5
    assert score.citation_score.chunk_recall == 0.5


def test_justification_length_bound() -> None:
    raw = _raw(justification="x" * 241)
    score = score_answer(_task(), raw)

    assert parse_answer(raw) is None
    assert score.correctness is False
    assert score.parsed_answer is None
