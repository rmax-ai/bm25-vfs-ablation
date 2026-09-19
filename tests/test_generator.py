import random
import re
import unicodedata

from bm25_vfs_ablation.corpus.generator import GenerationConfig, generate_dataset
from bm25_vfs_ablation.corpus.schema import Split


def _config(seed: int = 42) -> GenerationConfig:
    return GenerationConfig(
        seed=seed,
        dev_tasks=4,
        eval_tasks=8,
        corpus_size=12,
        distractors=2,
        min_hops=2,
        max_hops=4,
    )


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _models(documents: list, tasks: list) -> tuple[list[dict], list[dict]]:
    return (
        [document.model_dump(mode="json") for document in documents],
        [task.model_dump(mode="json") for task in tasks],
    )


def test_generation_same_seed_same_models() -> None:
    first = generate_dataset(_config())
    second = generate_dataset(_config())

    assert _models(*first) == _models(*second)


def test_generation_different_seed_changes_content() -> None:
    first = generate_dataset(_config(42))
    second = generate_dataset(_config(43))

    assert _models(*first) != _models(*second)


def test_no_question_text_leakage() -> None:
    documents, tasks = generate_dataset(_config())
    document_text = [_normalise(document.content) for document in documents]

    for task in tasks:
        question = _normalise(task.question)
        assert question
        assert all(question not in content for content in document_text)
        assert all("fact-" not in content for content in document_text)


def test_tasks_require_multiple_documents() -> None:
    documents, tasks = generate_dataset(_config())
    facts = {fact.fact_id: document.doc_id for document in documents for fact in document.facts}

    for task in tasks:
        required_documents = {facts[fact_id] for fact_id in task.required_fact_ids}
        assert len(required_documents) >= 2
        assert len(task.required_fact_ids) == task.hop_count
        assert 2 <= task.hop_count <= 4
        assert task.acceptable_answer_variants


def test_split_templates_or_entities_are_disjoint() -> None:
    _, tasks = generate_dataset(_config())
    dev = [task for task in tasks if task.split is Split.DEV]
    evaluation = [task for task in tasks if task.split is Split.EVAL]

    assert dev and evaluation
    assert all(int(task.task_template[-2:]) % 2 == 0 for task in dev)
    assert all(int(task.task_template[-2:]) % 2 == 1 for task in evaluation)
    dev_documents = {
        document_id
        for task in dev
        for document_id in task.gold_document_ids + task.distractor_document_ids
    }
    eval_documents = {
        document_id
        for task in evaluation
        for document_id in task.gold_document_ids + task.distractor_document_ids
    }
    assert dev_documents.isdisjoint(eval_documents)


def test_generator_uses_no_global_random() -> None:
    state = random.getstate()

    generate_dataset(_config())

    assert random.getstate() == state
