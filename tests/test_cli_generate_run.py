from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from bm25_vfs_ablation.cli import app
from bm25_vfs_ablation.corpus.generator import (
    GenerationConfig,
    generate_dataset,
    write_dataset,
)

RUNNER = CliRunner()


def _write_config(
    root: Path,
    *,
    provider: str = "mock",
    api_key: str | None = None,
    output_runs: str = "results/runs.jsonl",
) -> Path:
    config = {
        "schema_version": 1,
        "experiment": {
            "experiment_id": "exp-cli-placeholder",
            "seed": 42,
            "conditions": ["snippets", "vfs"],
            "token_ceiling": 4096,
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
            "top_k": 8,
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "query_tokenizer": "unicode_word_v1",
        },
        "harness": {
            "retrieval_context_tokens": 1800,
            "max_answer_tokens": 256,
            "max_tool_calls": 8,
            "invalid_call_limit": 3,
            "tool_result_tokens_per_call": 768,
            "initial_candidate_documents": 5,
        },
        "model": {
            "provider": provider,
            "base_url": "http://placeholder.invalid/v1",
            "api_key": api_key,
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
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_dataset(root: Path, *, dev_tasks: int = 1, eval_tasks: int = 2) -> None:
    documents, tasks = generate_dataset(
        GenerationConfig(
            seed=42,
            dev_tasks=dev_tasks,
            eval_tasks=eval_tasks,
            corpus_size=8,
            distractors=2,
            min_hops=2,
            max_hops=2,
        )
    )
    write_dataset(root / "data" / "generated", documents, tasks)


def test_generate_cli_defaults_and_hashes(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    first = RUNNER.invoke(
        app,
        [
            "generate",
            "--output-dir",
            str(output_dir),
            "--dev-tasks",
            "2",
            "--tasks",
            "4",
            "--corpus-size",
            "8",
            "--distractors",
            "2",
        ],
    )
    assert first.exit_code == 0, first.stdout
    corpus = output_dir / "corpus.jsonl"
    tasks = output_dir / "tasks.jsonl"
    assert corpus.is_file()
    assert tasks.is_file()
    assert f"corpus_sha256={hashlib.sha256(corpus.read_bytes()).hexdigest()}" in first.stdout
    assert f"tasks_sha256={hashlib.sha256(tasks.read_bytes()).hexdigest()}" in first.stdout
    corpus_bytes = corpus.read_bytes()
    tasks_bytes = tasks.read_bytes()

    second = RUNNER.invoke(
        app,
        [
            "generate",
            "--output-dir",
            str(output_dir),
            "--dev-tasks",
            "2",
            "--tasks",
            "4",
            "--corpus-size",
            "8",
            "--distractors",
            "2",
            "--force",
        ],
    )
    assert second.exit_code == 0, second.stdout
    assert corpus.read_bytes() == corpus_bytes
    assert tasks.read_bytes() == tasks_bytes


def test_generate_cli_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    first = RUNNER.invoke(
        app,
        [
            "generate",
            "--output-dir",
            str(output_dir),
            "--dev-tasks",
            "1",
            "--tasks",
            "1",
            "--corpus-size",
            "8",
            "--distractors",
            "2",
        ],
    )
    assert first.exit_code == 0, first.stdout

    second = RUNNER.invoke(
        app,
        ["generate", "--output-dir", str(output_dir)],
    )
    assert second.exit_code == 2
    assert "refusing to overwrite" in second.output


def test_run_cli_mock_writes_two_records_per_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dataset(tmp_path)
    config = _write_config(tmp_path)
    runs = tmp_path / "runs.jsonl"

    result = RUNNER.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--runs",
            str(runs),
            "--task-limit",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(row["scoring_details"]["mock_oracle_policy"] is True for row in rows)
    assert "run_records=4" in result.stdout


def test_run_cli_flag_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dataset(tmp_path, eval_tasks=1)
    config = _write_config(tmp_path, provider="openai_compatible")
    runs = tmp_path / "override-runs.jsonl"

    result = RUNNER.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--runs",
            str(runs),
            "--split",
            "all",
            "--model-provider",
            "mock",
            "--no-resume",
        ],
    )
    assert result.exit_code == 0, result.stdout
    rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {row["model_config"]["provider"] for row in rows} == {"mock"}
    assert {row["returned_model_id"] for row in rows} == {"mock-scripted-v1"}


def test_cli_schema_error_exit_two(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "invalid.yaml"
    config.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")

    result = RUNNER.invoke(app, ["run", "--config", str(config)])

    assert result.exit_code == 2
    assert "error:" in result.output


def test_live_provider_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dataset(tmp_path, eval_tasks=1)
    config = _write_config(tmp_path, provider="openai_compatible")

    result = RUNNER.invoke(
        app,
        ["run", "--config", str(config), "--split", "eval"],
    )

    assert result.exit_code == 2
    assert "requires a configured API key" in result.output
