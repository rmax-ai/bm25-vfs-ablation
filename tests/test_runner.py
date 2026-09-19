from __future__ import annotations

import json
from pathlib import Path

import pytest

from bm25_vfs_ablation import Condition
from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.corpus.loader import CorpusBundle
from bm25_vfs_ablation.corpus.schema import DocumentRecord, FactSpan, TaskRecord
from bm25_vfs_ablation.experiment.runner import (
    ExperimentRunner,
    condition_order,
    load_completed_run_keys,
)
from bm25_vfs_ablation.experiment.schemas import (
    LatencyBreakdown,
    RunRecord,
    TerminationReason,
)
from bm25_vfs_ablation.harnesses.base import HarnessResult
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.models.client import ModelRequest, ModelResponse
from bm25_vfs_ablation.retrieval.bm25 import BM25Index
from bm25_vfs_ablation.retrieval.chunking import Chunker


def _config(*, output_runs: str = "results/runs.jsonl", token_ceiling: int = 4096) -> AppConfig:
    return AppConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "experiment_id": "exp-runner-placeholder",
                "seed": 42,
                "conditions": ["snippets", "vfs"],
                "token_ceiling": token_ceiling,
                "concurrency": 1,
                "oracle_mode": False,
                "resume": True,
                "output_runs": output_runs,
            },
            "corpus": {
                "corpus_path": "data/generated/corpus.jsonl",
                "tasks_path": "data/generated/tasks.jsonl",
                "corpus_version": "synthetic-v1",
            },
            "chunking": {
                "chunk_size_tokens": 180,
                "chunk_overlap_tokens": 30,
            },
            "retrieval": {
                "implementation": "rank_bm25_okapi",
                "top_k": 4,
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
                "query_tokenizer": "unicode_word_v1",
            },
            "harness": {
                "retrieval_context_tokens": 180,
                "max_answer_tokens": 64,
                "max_tool_calls": 2,
                "invalid_call_limit": 2,
                "tool_result_tokens_per_call": 64,
                "initial_candidate_documents": 5,
            },
            "model": {
                "provider": "mock",
                "base_url": "http://placeholder.invalid/v1",
                "api_key": None,
                "model": "model-placeholder",
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "timeout_seconds": 60.0,
                "max_attempts": 2,
                "tokenizer": "regex_v1",
            },
            "evaluation": {
                "bootstrap_resamples": 10,
                "bootstrap_seed": 20250308,
                "confidence_level": 0.95,
                "judge_enabled": False,
                "estimated_input_usd_per_million": 0.0,
                "estimated_output_usd_per_million": 0.0,
            },
        }
    )


def _bundle() -> CorpusBundle:
    documents = (
        DocumentRecord(
            schema_version=1,
            corpus_version="synthetic-v1",
            doc_id="doc-ops-0001",
            path="/ops/doc-ops-0001.md",
            category="ops",
            title="Operations placeholder",
            content="Owner: Quartz.\nPolicy: approved.\n",
            facts=(
                FactSpan(
                    fact_id="fact-000001",
                    subject="service",
                    predicate="owner",
                    object="Quartz",
                    line_start=1,
                    line_end=1,
                ),
                FactSpan(
                    fact_id="fact-000002",
                    subject="service",
                    predicate="policy",
                    object="approved",
                    line_start=2,
                    line_end=2,
                ),
            ),
            generator_seed=42,
        ),
    )
    tasks = (
        TaskRecord(
            schema_version=1,
            task_id="task-eval-000001",
            split="eval",
            question="Which policy applies to Quartz?",
            canonical_answer="approved",
            acceptable_answer_variants=["approved"],
            required_fact_ids=["fact-000001", "fact-000002"],
            gold_document_ids=["doc-ops-0001"],
            gold_chunk_ids=["doc-ops-0001::c0000"],
            hop_count=2,
            task_template="policy-owner",
            category="ops",
            distractor_document_ids=[],
            distractor_count=0,
            generator_seed=42,
            corpus_version="synthetic-v1",
        ),
    )
    return CorpusBundle(
        documents=documents,
        tasks=tasks,
        corpus_sha256="a" * 64,
        tasks_sha256="b" * 64,
        corpus_version="synthetic-v1",
    )


class _RecordingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=(
                '{"answer":"approved","citations":["doc-ops-0001::c0000"],'
                '"justification":"The policy record says approved."}'
            ),
            returned_model_id="served-model-placeholder",
            latency_ms=0.0,
        )


def _runner(
    tmp_path: Path,
    model: _RecordingModel | None = None,
) -> tuple[ExperimentRunner, _RecordingModel]:
    bundle = _bundle()
    chunks = Chunker().chunk(bundle.documents)
    index = BM25Index(chunks)
    selected_model = model or _RecordingModel()
    runner = ExperimentRunner(
        _config(),
        bundle,
        index,
        selected_model,
        None,
        output_runs=tmp_path / "runs.jsonl",
    )
    return runner, selected_model


def test_runner_shares_corpus_and_index_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[tuple[object, object]] = []

    def run(self: object, task: TaskRecord, context: object) -> HarnessResult:
        del self, task
        contexts.append((context.bundle, context.index))
        return HarnessResult(
            raw_answer='{"answer":"approved","citations":[],"justification":"ok"}',
            initial_hits=(),
            trace=(),
            accounting=BudgetController(512).accounting(),
            latencies=LatencyBreakdown(
                wall_ms=0.0,
                model_ms=0.0,
                retrieval_ms=0.0,
                tool_ms=0.0,
            ),
            termination=TerminationReason.ANSWERED,
            errors=(),
        )

    monkeypatch.setattr("bm25_vfs_ablation.experiment.runner.SnippetHarness.run", run)
    monkeypatch.setattr("bm25_vfs_ablation.experiment.runner.VfsHarness.run", run)
    runner, _ = _runner(tmp_path)

    runner.run()
    assert contexts == [(runner.bundle, runner.index), (runner.bundle, runner.index)]


def test_runner_randomized_condition_order(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    summary = runner.run()
    assert len(summary.records) == 2
    expected = condition_order(42, "task-eval-000001")
    assert tuple(record.condition for record in summary.records) == expected
    assert [record.condition_order for record in summary.records] == [0, 1]


def test_runner_writes_complete_records(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    runner.run()
    rows = [json.loads(line) for line in runner.output_runs.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        record = RunRecord.model_validate(row)
        assert set(row) == set(record.model_dump(mode="json", by_alias=True))
        assert row["started_at"] and row["finished_at"]
        assert row["token_accounting"]["limit"] == 4096


def test_runner_resume_skips_completed_calls(tmp_path: Path) -> None:
    runner, model = _runner(tmp_path)
    first = runner.run()
    assert first.completed == 2
    call_count = len(model.requests)

    second = ExperimentRunner(
        runner.config,
        runner.bundle,
        runner.index,
        model,
        None,
        output_runs=runner.output_runs,
    ).run()
    assert second.completed == 0
    assert second.skipped == 2
    assert len(model.requests) == call_count


def test_runner_rejects_duplicate_run_key(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    runner.run()
    with runner.output_runs.open("a", encoding="utf-8") as stream:
        stream.write(runner.output_runs.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate run_key"):
        load_completed_run_keys(runner.output_runs)


def test_runner_records_returned_model_id(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    summary = runner.run()
    assert {record.returned_model_id for record in summary.records} == {"served-model-placeholder"}


def test_runner_token_ceiling_equal_by_condition(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    summary = runner.run()
    assert {
        record.token_accounting.limit
        for record in summary.records
        if record.condition in {Condition.SNIPPETS, Condition.VFS}
    } == {4096}
