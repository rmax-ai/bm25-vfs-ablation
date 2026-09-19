"""Operator-authored tests: §8 gold-chunk fingerprint guard on imports (AIR-2 fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.corpus.loader import CorpusBundle
from bm25_vfs_ablation.corpus.schema import DocumentRecord, FactSpan, TaskRecord
from bm25_vfs_ablation.experiment.runner import ExperimentRunner


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "experiment_id": "exp-air-placeholder",
                "seed": 42,
                "conditions": ["snippets", "vfs"],
                "token_ceiling": 512,
                "concurrency": 1,
                "oracle_mode": False,
                "resume": True,
                "output_runs": "results/runs.jsonl",
            },
            "corpus": {
                "corpus_path": "data/generated/corpus.jsonl",
                "tasks_path": "data/generated/tasks.jsonl",
                "corpus_version": "synthetic-v1",
            },
            "chunking": {"chunk_size_tokens": 180, "chunk_overlap_tokens": 30},
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


def _bundle(gold_chunk_ids: list[str]) -> CorpusBundle:
    document = DocumentRecord(
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
    )
    task = TaskRecord(
        schema_version=1,
        task_id="task-eval-000001",
        split="eval",
        question="Which policy applies to Quartz?",
        canonical_answer="approved",
        acceptable_answer_variants=["approved"],
        required_fact_ids=["fact-000001", "fact-000002"],
        gold_document_ids=["doc-ops-0001"],
        gold_chunk_ids=gold_chunk_ids,
        hop_count=2,
        task_template="policy-owner",
        category="ops",
        distractor_document_ids=[],
        distractor_count=0,
        generator_seed=42,
        corpus_version="synthetic-v1",
    )
    return CorpusBundle(
        documents=(document,),
        tasks=(task,),
        corpus_sha256="a" * 64,
        tasks_sha256="b" * 64,
        corpus_version="synthetic-v1",
    )


class _NoCallModel:
    def complete(self, request: object) -> object:  # pragma: no cover - must not run
        raise AssertionError("no model call expected during runner construction")


def test_mismatched_gold_chunks_rejected(tmp_path: Path) -> None:
    bundle = _bundle(["doc-ops-0001::c0099"])  # absent from any index built here

    with pytest.raises(ValueError) as excinfo:
        ExperimentRunner(
            _config(),
            bundle,
            None,
            _NoCallModel(),
            None,
            output_runs=tmp_path / "runs.jsonl",
        )

    message = str(excinfo.value)
    assert "gold chunk" in message
    assert "c0099" in message
    assert "regenerate" in message


def test_consistent_gold_chunks_accepted(tmp_path: Path) -> None:
    bundle = _bundle(["doc-ops-0001::c0000"])

    runner = ExperimentRunner(
        _config(),
        bundle,
        None,
        _NoCallModel(),
        None,
        output_runs=tmp_path / "runs.jsonl",
    )

    assert runner.index is not None
