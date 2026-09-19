"""Load and validate corpus and task JSONL artifacts."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import ValidationError

from bm25_vfs_ablation.corpus.schema import (
    DocumentRecord,
    TaskRecord,
    validate_dataset,
)
from bm25_vfs_ablation.experiment.artifacts import read_jsonl, sha256_file


def _expanded_path(path: Path) -> Path:
    """Return a path with a user-home prefix expanded."""

    return Path(os.path.expanduser(os.fspath(path)))


def _ensure_unique(values: Sequence[str], field_name: str, path: Path) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {field_name} in {path}: {value}")
        seen.add(value)


def _ensure_sorted(values: Sequence[str], field_name: str, path: Path) -> None:
    if list(values) != sorted(values):
        raise ValueError(f"{field_name} records in {path} must be sorted")


def _ensure_one_version(
    records: Sequence[DocumentRecord] | Sequence[TaskRecord],
    record_name: str,
    path: Path,
) -> str | None:
    versions = {record.corpus_version for record in records}
    if len(versions) > 1:
        raise ValueError(f"{record_name} in {path} must use one corpus_version")
    return next(iter(versions), None)


def _load_records[RecordT: DocumentRecord | TaskRecord](
    path: Path,
    record_type: type[RecordT],
    record_name: str,
) -> tuple[list[RecordT], str]:
    target = _expanded_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{record_name} file does not exist: {target}")

    records: list[RecordT] = []
    for line_number, row in enumerate(read_jsonl(target), start=1):
        try:
            record = record_type.model_validate(row)
        except ValidationError as error:
            raise ValueError(
                f"invalid {record_name} record in {target} at line {line_number}: {error}"
            ) from error
        records.append(record)

    return records, sha256_file(target)


def _load_corpus_records(path: Path) -> tuple[list[DocumentRecord], str]:
    documents, source_hash = _load_records(path, DocumentRecord, "corpus")
    target = _expanded_path(path)

    _ensure_unique([document.doc_id for document in documents], "doc_id", target)
    _ensure_unique([document.path for document in documents], "path", target)
    _ensure_unique(
        [fact.fact_id for document in documents for fact in document.facts],
        "fact_id",
        target,
    )
    _ensure_sorted([document.doc_id for document in documents], "corpus", target)
    _ensure_one_version(documents, "corpus records", target)
    return documents, source_hash


def _load_task_records(path: Path) -> tuple[list[TaskRecord], str]:
    tasks, source_hash = _load_records(path, TaskRecord, "task")
    target = _expanded_path(path)

    _ensure_unique([task.task_id for task in tasks], "task_id", target)
    task_order = [(0 if task.split.value == "dev" else 1, task.task_id) for task in tasks]
    if task_order != sorted(task_order):
        raise ValueError(f"task records in {target} must be sorted")
    _ensure_one_version(tasks, "task records", target)
    return tasks, source_hash


@dataclass(frozen=True, slots=True)
class CorpusBundle:
    """One validated corpus/task pair and the exact source byte hashes."""

    documents: tuple[DocumentRecord, ...]
    tasks: tuple[TaskRecord, ...]
    corpus_sha256: str
    tasks_sha256: str
    corpus_version: str

    @property
    def documents_by_id(self) -> Mapping[str, DocumentRecord]:
        """Return a read-only document index."""

        return MappingProxyType({document.doc_id: document for document in self.documents})

    @property
    def tasks_by_id(self) -> Mapping[str, TaskRecord]:
        """Return a read-only task index."""

        return MappingProxyType({task.task_id: task for task in self.tasks})


def load_corpus(path: Path) -> list[DocumentRecord]:
    """Load and validate one sorted corpus JSONL file."""

    documents, _ = _load_corpus_records(path)
    return documents


def load_tasks(path: Path) -> list[TaskRecord]:
    """Load and validate one sorted task JSONL file."""

    tasks, _ = _load_task_records(path)
    return tasks


def load_bundle(corpus_path: Path, tasks_path: Path) -> CorpusBundle:
    """Load, cross-validate, and hash a corpus/task JSONL pair."""

    documents, corpus_sha256 = _load_corpus_records(corpus_path)
    tasks, tasks_sha256 = _load_task_records(tasks_path)
    if not documents:
        raise ValueError("corpus must contain at least one document")

    validate_dataset(documents, tasks, chunker=None)
    corpus_version = documents[0].corpus_version
    if any(task.corpus_version != corpus_version for task in tasks):
        raise ValueError("task corpus_version must match the corpus version")

    return CorpusBundle(
        documents=tuple(documents),
        tasks=tuple(tasks),
        corpus_sha256=corpus_sha256,
        tasks_sha256=tasks_sha256,
        corpus_version=corpus_version,
    )


__all__ = ["CorpusBundle", "load_bundle", "load_corpus", "load_tasks"]
