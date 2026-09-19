import json
from pathlib import Path

import pytest

from bm25_vfs_ablation.corpus.loader import load_bundle, load_corpus, load_tasks
from bm25_vfs_ablation.corpus.schema import DocumentRecord, FactSpan, TaskRecord
from bm25_vfs_ablation.experiment.artifacts import sha256_file, write_jsonl_atomic


def _records() -> tuple[list[DocumentRecord], list[TaskRecord]]:
    documents = [
        DocumentRecord(
            schema_version=1,
            corpus_version="user-v1",
            doc_id="doc-policies-0002",
            path="/policies/doc-policies-0002.md",
            category="policies",
            title="Security policy",
            content="# Review policy\nQuartz usage requires a security review.\n",
            facts=[
                FactSpan(
                    fact_id="fact-000002",
                    subject="Quartz",
                    predicate="requires",
                    object="security review",
                    line_start=2,
                    line_end=2,
                )
            ],
            generator_seed=11,
        ),
        DocumentRecord(
            schema_version=1,
            corpus_version="user-v1",
            doc_id="doc-services-0001",
            path="/services/doc-services-0001.md",
            category="services",
            title="Atlas service",
            content="# Atlas\nAtlas depends on Quartz.\n",
            facts=[
                FactSpan(
                    fact_id="fact-000001",
                    subject="Atlas",
                    predicate="depends_on",
                    object="Quartz",
                    line_start=2,
                    line_end=2,
                )
            ],
            generator_seed=11,
        ),
        DocumentRecord(
            schema_version=1,
            corpus_version="user-v1",
            doc_id="doc-services-0003",
            path="/services/doc-services-0003.md",
            category="services",
            title="Borealis service",
            content="# Borealis\nBorealis uses a different library.\n",
            facts=[],
            generator_seed=11,
        ),
    ]
    tasks = [
        TaskRecord(
            schema_version=1,
            task_id="task-eval-000001",
            split="eval",
            question="Does Atlas require a security review?",
            canonical_answer="yes",
            acceptable_answer_variants=["yes, Atlas requires a security review"],
            required_fact_ids=["fact-000001", "fact-000002"],
            gold_document_ids=["doc-policies-0002", "doc-services-0001"],
            gold_chunk_ids=[
                "doc-policies-0002::c0000",
                "doc-services-0001::c0000",
            ],
            hop_count=2,
            task_template="dependency-policy",
            category="security",
            distractor_document_ids=["doc-services-0003"],
            distractor_count=1,
            generator_seed=11,
            corpus_version="user-v1",
        )
    ]
    return documents, tasks


def _write_bundle(
    tmp_path: Path,
    documents: list[DocumentRecord],
    tasks: list[TaskRecord],
) -> tuple[Path, Path]:
    corpus_path = tmp_path / "corpus.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl_atomic(
        corpus_path,
        (document.model_dump(mode="json") for document in documents),
    )
    write_jsonl_atomic(tasks_path, (task.model_dump(mode="json") for task in tasks))
    return corpus_path, tasks_path


def test_load_valid_bundle(tmp_path: Path) -> None:
    documents, tasks = _records()
    corpus_path, tasks_path = _write_bundle(tmp_path, documents, tasks)

    bundle = load_bundle(corpus_path, tasks_path)

    assert bundle.documents == tuple(documents)
    assert bundle.tasks == tuple(tasks)
    assert bundle.corpus_version == "user-v1"
    assert bundle.corpus_sha256 == sha256_file(corpus_path)
    assert bundle.tasks_sha256 == sha256_file(tasks_path)
    assert bundle.documents_by_id["doc-services-0001"] == documents[1]
    assert bundle.tasks_by_id["task-eval-000001"] == tasks[0]
    with pytest.raises(TypeError):
        bundle.documents_by_id["new-document"] = documents[0]  # type: ignore[index]


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    documents, tasks = _records()
    corpus_path, tasks_path = _write_bundle(
        tmp_path,
        documents + [documents[0]],
        tasks,
    )

    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_corpus(corpus_path)

    unsorted_corpus, _ = _write_bundle(tmp_path / "unsorted", list(reversed(documents)), tasks)
    with pytest.raises(ValueError, match="must be sorted"):
        load_corpus(unsorted_corpus)


def test_loader_rejects_dangling_gold(tmp_path: Path) -> None:
    documents, tasks = _records()
    dangling_gold = tasks[0].model_copy(
        update={
            "gold_document_ids": [
                "doc-policies-0002",
                "doc-services-9999",
            ],
            "gold_chunk_ids": [
                "doc-policies-0002::c0000",
                "doc-services-9999::c0000",
            ],
        }
    )
    corpus_path, tasks_path = _write_bundle(tmp_path, documents, [dangling_gold])

    with pytest.raises(ValueError, match="unknown gold document"):
        load_bundle(corpus_path, tasks_path)

    dangling_distractor = tasks[0].model_copy(
        update={"distractor_document_ids": ["doc-services-9999"]}
    )
    corpus_path, tasks_path = _write_bundle(
        tmp_path / "distractor",
        documents,
        [dangling_distractor],
    )
    with pytest.raises(ValueError, match="unknown distractor document"):
        load_bundle(corpus_path, tasks_path)


def test_loader_rejects_mixed_versions(tmp_path: Path) -> None:
    documents, tasks = _records()
    mixed_document = documents[1].model_copy(update={"corpus_version": "user-v2"})
    corpus_path, tasks_path = _write_bundle(
        tmp_path,
        [documents[0], mixed_document, documents[2]],
        tasks,
    )

    with pytest.raises(ValueError, match="one corpus_version"):
        load_corpus(corpus_path)

    mixed_task = tasks[0].model_copy(
        update={"task_id": "task-eval-000002", "corpus_version": "user-v2"}
    )
    _, mixed_tasks_path = _write_bundle(
        tmp_path / "mixed-tasks",
        documents,
        [tasks[0], mixed_task],
    )
    with pytest.raises(ValueError, match="one corpus_version"):
        load_tasks(mixed_tasks_path)


def test_user_import_uses_same_schema(tmp_path: Path) -> None:
    documents, tasks = _records()
    corpus_path = tmp_path / "user-corpus.jsonl"
    tasks_path = tmp_path / "user-tasks.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps(document.model_dump(mode="json")) for document in documents) + "\n",
        encoding="utf-8",
    )
    tasks_path.write_text(
        "\n".join(json.dumps(task.model_dump(mode="json")) for task in tasks) + "\n",
        encoding="utf-8",
    )

    assert load_corpus(corpus_path) == documents
    assert load_tasks(tasks_path) == tasks

    invalid_line = tmp_path / "invalid.jsonl"
    invalid_line.write_bytes(b"{}\n\n")
    with pytest.raises(ValueError, match=r"line 1"):
        load_corpus(invalid_line)

    blank_line = tmp_path / "blank-line.jsonl"
    blank_line.write_bytes(
        json.dumps(documents[0].model_dump(mode="json")).encode("utf-8") + b"\n\n"
    )
    with pytest.raises(ValueError, match=r"line 2"):
        load_corpus(blank_line)

    non_utf8 = tmp_path / "non-utf8.jsonl"
    non_utf8.write_bytes(b'{"schema_version":1,\xff}\n')
    with pytest.raises(ValueError, match=r"line 1"):
        load_tasks(non_utf8)

    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "absent.jsonl")
