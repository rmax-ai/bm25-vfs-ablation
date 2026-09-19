"""Validated corpus and task records."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_FACT_ID = re.compile(r"^fact-\d{6}$")
_DOC_ID = re.compile(rf"^doc-(?P<category>{_SLUG})-\d{{4}}$")
_TASK_ID = re.compile(r"^task-(?P<split>dev|eval)-\d{6}$")
_CHUNK_ID = re.compile(rf"^doc-(?:{_SLUG})-\d{{4}}::c\d{{4}}$")
_DOCUMENT_PATH = re.compile(rf"^/(?P<category>{_SLUG})/(?P<doc_id>doc-(?:{_SLUG})-\d{{4}})\.md$")


class _StrictModel(BaseModel):
    """Base class for persisted records."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Split(StrEnum):
    """Dataset split names."""

    DEV = "dev"
    EVAL = "eval"


def _require_nonblank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _validate_slug(value: str, field_name: str) -> str:
    if not re.fullmatch(_SLUG, value):
        raise ValueError(f"{field_name} must be a lowercase ASCII slug")
    return value


def _validate_sorted_unique(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    if values != sorted(values):
        raise ValueError(f"{field_name} must be sorted")
    return values


class FactSpan(_StrictModel):
    """A hidden, one-based inclusive fact span in a document."""

    fact_id: str
    subject: str
    predicate: str
    object: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @field_validator("fact_id")
    @classmethod
    def _validate_fact_id(cls, value: str) -> str:
        if _FACT_ID.fullmatch(value) is None:
            raise ValueError("fact_id must match fact-{six digits}")
        return value

    @field_validator("subject", "predicate", "object")
    @classmethod
    def _validate_fact_text(cls, value: str, info: Any) -> str:
        return _require_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_order(self) -> FactSpan:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class DocumentRecord(_StrictModel):
    """One versioned corpus document and its hidden fact provenance."""

    schema_version: Literal[1]
    corpus_version: str
    doc_id: str
    path: str
    category: str
    title: str
    content: str
    facts: list[FactSpan]
    generator_seed: int

    @field_validator("corpus_version", "title")
    @classmethod
    def _validate_required_text(cls, value: str, info: Any) -> str:
        return _require_nonblank(value, info.field_name)

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        return _validate_slug(value, "category")

    @field_validator("doc_id")
    @classmethod
    def _validate_doc_id(cls, value: str) -> str:
        if _DOC_ID.fullmatch(value) is None:
            raise ValueError("doc_id must match doc-{category}-{four digits}")
        return value

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        return _require_nonblank(value, "content")

    @model_validator(mode="after")
    def _validate_document(self) -> DocumentRecord:
        match = _DOC_ID.fullmatch(self.doc_id)
        assert match is not None
        if match.group("category") != self.category:
            raise ValueError("doc_id category must match category")

        path_match = _DOCUMENT_PATH.fullmatch(self.path)
        if path_match is None:
            raise ValueError("path must match /{category}/{doc_id}.md")
        if (
            path_match.group("category") != self.category
            or path_match.group("doc_id") != self.doc_id
        ):
            raise ValueError("path must identify this document")

        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("facts must contain unique fact IDs")

        line_count = len(self.content.splitlines())
        if line_count == 0:
            raise ValueError("content must contain at least one line")
        for fact in self.facts:
            if fact.line_end > line_count:
                raise ValueError("fact span must be within document content")
        return self


class TaskRecord(_StrictModel):
    """One deterministic multi-hop question and its gold provenance."""

    schema_version: Literal[1]
    task_id: str
    split: Split
    question: str
    canonical_answer: str
    acceptable_answer_variants: list[str]
    required_fact_ids: list[str]
    gold_document_ids: list[str]
    gold_chunk_ids: list[str]
    hop_count: int = Field(ge=2, le=4)
    task_template: str
    category: str
    distractor_document_ids: list[str]
    distractor_count: int = Field(ge=0)
    generator_seed: int
    corpus_version: str

    @field_validator("question", "canonical_answer", "task_template", "corpus_version")
    @classmethod
    def _validate_required_text(cls, value: str, info: Any) -> str:
        return _require_nonblank(value, info.field_name)

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        return _validate_slug(value, "category")

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str) -> str:
        if _TASK_ID.fullmatch(value) is None:
            raise ValueError("task_id must match task-{split}-{six digits}")
        return value

    @field_validator("acceptable_answer_variants")
    @classmethod
    def _validate_variants(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("acceptable_answer_variants must not be empty")
        for variant in value:
            _require_nonblank(variant, "acceptable_answer_variants item")
        if len(value) != len(set(value)):
            raise ValueError("acceptable_answer_variants must be unique")
        return value

    @field_validator(
        "required_fact_ids",
        "gold_document_ids",
        "gold_chunk_ids",
        "distractor_document_ids",
    )
    @classmethod
    def _validate_sorted_lists(cls, value: list[str], info: Any) -> list[str]:
        if not value and info.field_name != "distractor_document_ids":
            raise ValueError(f"{info.field_name} must not be empty")
        return _validate_sorted_unique(value, info.field_name)

    @field_validator("required_fact_ids")
    @classmethod
    def _validate_fact_ids(cls, value: list[str]) -> list[str]:
        if any(_FACT_ID.fullmatch(item) is None for item in value):
            raise ValueError("required_fact_ids must contain fact-{six digits} IDs")
        return value

    @field_validator("gold_document_ids", "distractor_document_ids")
    @classmethod
    def _validate_document_ids(cls, value: list[str], info: Any) -> list[str]:
        if any(_DOC_ID.fullmatch(item) is None for item in value):
            raise ValueError(f"{info.field_name} must contain document IDs")
        return value

    @field_validator("gold_chunk_ids")
    @classmethod
    def _validate_chunk_ids(cls, value: list[str]) -> list[str]:
        if any(_CHUNK_ID.fullmatch(item) is None for item in value):
            raise ValueError("gold_chunk_ids must contain chunk IDs")
        return value

    @model_validator(mode="after")
    def _validate_task(self) -> TaskRecord:
        task_match = _TASK_ID.fullmatch(self.task_id)
        assert task_match is not None
        if task_match.group("split") != self.split.value:
            raise ValueError("task_id split must match split")
        if len(self.required_fact_ids) != self.hop_count:
            raise ValueError("hop_count must equal the number of required facts")
        if len(self.distractor_document_ids) != self.distractor_count:
            raise ValueError("distractor_count must equal distractor_document_ids length")
        if set(self.gold_document_ids) & set(self.distractor_document_ids):
            raise ValueError("gold and distractor document IDs must be disjoint")
        gold_documents = set(self.gold_document_ids)
        for chunk_id in self.gold_chunk_ids:
            if chunk_id.split("::", maxsplit=1)[0] not in gold_documents:
                raise ValueError("gold_chunk_ids must belong to gold_document_ids")
        return self


def validate_dataset(
    documents: Sequence[DocumentRecord],
    tasks: Sequence[TaskRecord],
    chunker: Any,
) -> None:
    """Validate cross-record references and, when supplied, chunk provenance."""

    document_ids = [document.doc_id for document in documents]
    document_paths = [document.path for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("documents contain duplicate doc_id values")
    if len(document_paths) != len(set(document_paths)):
        raise ValueError("documents contain duplicate paths")

    documents_by_id = {document.doc_id: document for document in documents}
    fact_to_document: dict[str, str] = {}
    corpus_versions = {document.corpus_version for document in documents}
    if len(corpus_versions) > 1:
        raise ValueError("documents must use one corpus_version")
    for document in documents:
        for fact in document.facts:
            if fact.fact_id in fact_to_document:
                raise ValueError(f"duplicate fact_id: {fact.fact_id}")
            fact_to_document[fact.fact_id] = document.doc_id

    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("tasks contain duplicate task_id values")
    for task in tasks:
        if not corpus_versions or task.corpus_version not in corpus_versions:
            raise ValueError("task corpus_version must match the corpus version")
        if any(
            fact_id not in fact_to_document for fact_id in task.required_fact_ids
        ):
            raise ValueError("task references an unknown fact_id")
        if any(doc_id not in documents_by_id for doc_id in task.gold_document_ids):
            raise ValueError("task references an unknown gold document")
        if any(
            doc_id not in documents_by_id for doc_id in task.distractor_document_ids
        ):
            raise ValueError("task references an unknown distractor document")
        if any(
            fact_to_document[fact_id] not in task.gold_document_ids
            for fact_id in task.required_fact_ids
        ):
            raise ValueError("required facts must belong to gold documents")

    if chunker is not None:
        chunks = chunker.chunk(list(documents))
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        for task in tasks:
            if any(chunk_id not in chunk_ids for chunk_id in task.gold_chunk_ids):
                raise ValueError("task references an unknown gold chunk")


__all__ = [
    "DocumentRecord",
    "FactSpan",
    "Split",
    "TaskRecord",
    "validate_dataset",
]
