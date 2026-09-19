from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bm25_vfs_ablation.config import load_config
from bm25_vfs_ablation.corpus.generator import (
    GenerationConfig,
    generate_dataset,
    write_dataset,
)
from bm25_vfs_ablation.corpus.loader import load_bundle
from bm25_vfs_ablation.corpus.schema import DocumentRecord, TaskRecord
from bm25_vfs_ablation.experiment.artifacts import (
    canonical_json,
    read_jsonl,
    sha256_file,
)
from bm25_vfs_ablation.experiment.runner import ExperimentRunner
from bm25_vfs_ablation.experiment.schemas import RunRecord
from bm25_vfs_ablation.models.client import (
    MockAction,
    MockPolicy,
    ModelRequest,
    ModelResponse,
    ScriptedMockClient,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_RUNTIME_ONLY_KEYS = frozenset(
    {
        "completed_at",
        "finished_at",
        "latency",
        "latency_ms",
        "model_ms",
        "requested_at",
        "retrieval_ms",
        "started_at",
        "time_to_first_gold_fact_ms",
        "tool_ms",
        "wall_ms",
    }
)

_REQUIRED_ARTIFACTS = (
    "Dockerfile",
    "Makefile",
    "PLAN.md",
    "README.md",
    "SPEC.md",
    "pyproject.toml",
    "uv.lock",
    "configs/default.yaml",
    "configs/tool_call_intervention.yaml",
    "data/.gitkeep",
    "reports/.gitkeep",
    "results/.gitkeep",
    "src/bm25_vfs_ablation/__init__.py",
    "src/bm25_vfs_ablation/__main__.py",
    "src/bm25_vfs_ablation/cli.py",
    "src/bm25_vfs_ablation/config.py",
    "src/bm25_vfs_ablation/corpus/generator.py",
    "src/bm25_vfs_ablation/corpus/loader.py",
    "src/bm25_vfs_ablation/corpus/schema.py",
    "src/bm25_vfs_ablation/evaluation/aggregates.py",
    "src/bm25_vfs_ablation/evaluation/answer_scoring.py",
    "src/bm25_vfs_ablation/evaluation/evidence_scoring.py",
    "src/bm25_vfs_ablation/evaluation/failures.py",
    "src/bm25_vfs_ablation/evaluation/plots.py",
    "src/bm25_vfs_ablation/evaluation/statistics.py",
    "src/bm25_vfs_ablation/experiment/artifacts.py",
    "src/bm25_vfs_ablation/experiment/designs.py",
    "src/bm25_vfs_ablation/experiment/reporting.py",
    "src/bm25_vfs_ablation/experiment/runner.py",
    "src/bm25_vfs_ablation/experiment/schemas.py",
    "src/bm25_vfs_ablation/harnesses/base.py",
    "src/bm25_vfs_ablation/harnesses/budget.py",
    "src/bm25_vfs_ablation/harnesses/snippets.py",
    "src/bm25_vfs_ablation/harnesses/vfs_agent.py",
    "src/bm25_vfs_ablation/models/cache.py",
    "src/bm25_vfs_ablation/models/client.py",
    "src/bm25_vfs_ablation/models/tokenization.py",
    "src/bm25_vfs_ablation/retrieval/bm25.py",
    "src/bm25_vfs_ablation/retrieval/chunking.py",
    "src/bm25_vfs_ablation/vfs/filesystem.py",
    "src/bm25_vfs_ablation/vfs/security.py",
    "src/bm25_vfs_ablation/vfs/tools.py",
)


def _write_seed42_dataset(root: Path):
    documents, tasks = generate_dataset(GenerationConfig(seed=42))
    return write_dataset(root / "data" / "generated", documents, tasks)


class _FourTaskMock:
    """Route each public runner request to a deterministic scripted policy."""

    def __init__(
        self,
        tasks: Sequence[TaskRecord],
        documents: Sequence[DocumentRecord],
    ) -> None:
        facts = {
            fact.fact_id: (document, fact)
            for document in documents
            for fact in document.facts
        }
        actions: dict[tuple[str, str, str], tuple[MockAction, ...]] = {}
        for task in tasks:
            answer = canonical_json(
                {
                    "answer": task.canonical_answer,
                    "citations": list(task.gold_chunk_ids),
                    "justification": "Acceptance mock uses generated ground truth.",
                }
            )
            actions[(task.task_id, "snippets", "primary")] = (
                MockAction(kind="final", answer=answer),
            )
            vfs_actions = [
                MockAction(
                    kind="tool_call",
                    tool_name="read",
                    arguments={
                        "end_line": facts[fact_id][1].line_end,
                        "path": facts[fact_id][0].path,
                        "start_line": facts[fact_id][1].line_start,
                    },
                )
                for fact_id in task.required_fact_ids
            ]
            actions[(task.task_id, "vfs", "primary")] = (
                *vfs_actions,
                MockAction(kind="final", answer=answer),
            )
        self._policy = MockPolicy(actions)
        self._tasks = tuple(tasks)

    def complete(self, request: ModelRequest) -> ModelResponse:
        question = next(
            (
                task.question
                for task in self._tasks
                if any(
                    isinstance(message.get("content"), str)
                    and task.question in message["content"]
                    for message in request.messages
                )
            ),
            None,
        )
        if question is None:
            raise AssertionError("acceptance mock could not identify the task")
        task = next(task for task in self._tasks if task.question == question)
        condition = "vfs" if request.tools else "snippets"
        return ScriptedMockClient(
            self._policy,
            task.task_id,
            condition,
            "primary",
        ).complete(request)


def _without_runtime_values(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _without_runtime_values(child)
            for key, child in value.items()
            if key not in _RUNTIME_ONLY_KEYS
        }
    if isinstance(value, list):
        return [_without_runtime_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_runtime_values(item) for item in value)
    return value


def _stable_run_projection(records: Iterable[RunRecord]) -> str:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.run_key):
        payload = record.model_dump(mode="json", by_alias=True)
        stable_payload = _without_runtime_values(payload)
        assert isinstance(stable_payload, dict)
        rows.append(stable_payload)
    return canonical_json(rows)


def _run_four_tasks(root: Path) -> tuple[str, str]:
    generation_config = GenerationConfig(
        seed=42,
        dev_tasks=2,
        eval_tasks=2,
        corpus_size=8,
        distractors=2,
        min_hops=2,
        max_hops=4,
    )
    documents, tasks = generate_dataset(generation_config)
    hashes = write_dataset(root / "data" / "generated", documents, tasks)
    bundle = load_bundle(hashes.corpus_path, hashes.tasks_path)
    config = load_config(
        REPOSITORY_ROOT / "configs" / "default.yaml",
        overrides={"model": {"provider": "mock", "api_key": None}},
    )
    runs_path = root / "results" / "runs.jsonl"
    summary = ExperimentRunner(
        config,
        bundle,
        model=_FourTaskMock(bundle.tasks, bundle.documents),
        output_runs=runs_path,
    ).run()
    assert summary.completed == 8

    records = tuple(RunRecord.model_validate(row) for row in read_jsonl(runs_path))
    assert len(records) == 8
    assert len({record.task_id for record in records}) == 4
    assert {record.condition.value for record in records} == {"snippets", "vfs"}
    assert all(record.correctness for record in records)
    return hashes.corpus_sha256, _stable_run_projection(records)


def test_seed42_corpus_and_tasks_sha256_equal_on_regeneration(tmp_path: Path) -> None:
    first = _write_seed42_dataset(tmp_path / "first")
    second = _write_seed42_dataset(tmp_path / "second")

    assert first.corpus_path.read_bytes() == second.corpus_path.read_bytes()
    assert first.tasks_path.read_bytes() == second.tasks_path.read_bytes()
    assert first.corpus_sha256 == second.corpus_sha256
    assert first.tasks_sha256 == second.tasks_sha256
    assert first.corpus_sha256 == sha256_file(first.corpus_path)
    assert first.tasks_sha256 == sha256_file(first.tasks_path)


def test_stable_run_projection_equal_on_rerun(tmp_path: Path) -> None:
    first_hash, first_projection = _run_four_tasks(tmp_path / "first")
    second_hash, second_projection = _run_four_tasks(tmp_path / "second")

    assert first_hash == second_hash
    assert first_projection == second_projection


def test_required_repository_artifacts_exist() -> None:
    missing = [
        relative_path
        for relative_path in _REQUIRED_ARTIFACTS
        if not (REPOSITORY_ROOT / relative_path).is_file()
    ]
    assert not missing, f"missing required repository artifacts: {missing}"
