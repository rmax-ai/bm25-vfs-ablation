from __future__ import annotations

import copy
import csv
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
from bm25_vfs_ablation.evaluation.plots import PLOT_FILENAMES
from bm25_vfs_ablation.experiment.artifacts import write_jsonl_atomic

RUNNER = CliRunner()
AGGREGATE_FILENAMES = (
    "paired_outcomes.csv",
    "condition_summary.csv",
    "subgroup_summary.csv",
    "intervention_summary.csv",
    "within_vfs_regression.csv",
    "efficiency_summary.csv",
    "trace_examples.json",
    "task_level.csv",
    "stratified_summary.csv",
    "failure_summary.csv",
)


def _write_config(root: Path) -> Path:
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
            "output_runs": "results/runs.jsonl",
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
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_dataset(root: Path) -> None:
    documents, tasks = generate_dataset(
        GenerationConfig(
            seed=42,
            dev_tasks=1,
            eval_tasks=1,
            corpus_size=8,
            distractors=2,
            min_hops=2,
            max_hops=2,
        )
    )
    write_dataset(root / "data" / "generated", documents, tasks)


def _write_runs(root: Path) -> Path:
    _write_dataset(root)
    config = _write_config(root)
    runs = root / "runs.jsonl"
    result = RUNNER.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--runs",
            str(runs),
            "--task-limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    return runs


def _add_design_cells(runs: Path) -> None:
    rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    extra: list[dict[str, object]] = []
    for row in rows:
        if row["condition"] == "snippets":
            oracle = copy.deepcopy(row)
            oracle["design_cell"] = "oracle"
            oracle["run_key"] = (
                f'{oracle["experiment_id"]}:{oracle["task_id"]}:'
                f'{oracle["condition"]}:oracle'
            )
            extra.append(oracle)
        elif row["condition"] == "vfs":
            intervention = copy.deepcopy(row)
            intervention["design_cell"] = "max_calls_2"
            intervention["max_tool_calls_permitted"] = 2
            intervention["run_key"] = (
                f'{intervention["experiment_id"]}:{intervention["task_id"]}:'
                f'{intervention["condition"]}:max_calls_2'
            )
            extra.append(intervention)
    write_jsonl_atomic(runs, [*rows, *extra])


def _csv_values(path: Path, column: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row[column] for row in csv.DictReader(stream)]


def test_evaluate_cli_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = _write_runs(tmp_path)
    aggregates = tmp_path / "outputs" / "aggregates"
    plots = tmp_path / "outputs" / "plots"

    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            "--runs",
            str(runs),
            "--output-dir",
            str(aggregates),
            "--plots-dir",
            str(plots),
        ],
    )

    assert result.exit_code == 0, result.output
    assert {path.name for path in aggregates.iterdir()} == set(AGGREGATE_FILENAMES)
    assert {path.name for path in plots.iterdir()} == set(PLOT_FILENAMES)
    assert all((plots / filename).stat().st_size > 0 for filename in PLOT_FILENAMES)


def test_evaluate_primary_excludes_design_cells(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = _write_runs(tmp_path)
    _add_design_cells(runs)
    aggregates = tmp_path / "aggregates"
    plots = tmp_path / "plots"

    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            "--runs",
            str(runs),
            "--output-dir",
            str(aggregates),
            "--plots-dir",
            str(plots),
        ],
    )

    assert result.exit_code == 0, result.output
    assert set(_csv_values(aggregates / "condition_summary.csv", "design_cell")) == {"primary"}
    assert set(_csv_values(aggregates / "intervention_summary.csv", "design_cell")) == {
        "max_calls_2"
    }
    assert "oracle" not in (aggregates / "condition_summary.csv").read_text(encoding="utf-8")


def test_evaluate_invalid_jsonl_exit_two(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "invalid.jsonl"
    runs.write_text("{not-json}\n", encoding="utf-8")
    aggregates = tmp_path / "aggregates"
    plots = tmp_path / "plots"

    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            "--runs",
            str(runs),
            "--output-dir",
            str(aggregates),
            "--plots-dir",
            str(plots),
        ],
    )

    assert result.exit_code == 2
    assert "invalid JSON" in result.output
    assert not aggregates.exists()
    assert not plots.exists()


def test_report_cli_writes_markdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = _write_runs(tmp_path)
    aggregates = tmp_path / "aggregates"
    plots = tmp_path / "plots"
    evaluated = RUNNER.invoke(
        app,
        [
            "evaluate",
            "--runs",
            str(runs),
            "--output-dir",
            str(aggregates),
            "--plots-dir",
            str(plots),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output

    output = tmp_path / "reports" / "experiment.md"
    result = RUNNER.invoke(
        app,
        [
            "report",
            "--runs",
            str(runs),
            "--aggregates-dir",
            str(aggregates),
            "--plots-dir",
            str(plots),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# BM25 VFS Workspace vs BM25 Snippet Ablation\n")
    assert "## Primary" in text
    assert "01_success_by_condition.png" in text


def test_report_cli_missing_aggregates_regenerates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = _write_runs(tmp_path)
    aggregates = tmp_path / "generated" / "aggregates"
    plots = tmp_path / "generated" / "plots"
    output = tmp_path / "generated" / "reports" / "experiment.md"

    first = RUNNER.invoke(
        app,
        [
            "report",
            "--runs",
            str(runs),
            "--aggregates-dir",
            str(aggregates),
            "--plots-dir",
            str(plots),
            "--output",
            str(output),
        ],
    )

    assert first.exit_code == 0, first.output
    first_bytes = output.read_bytes()
    assert {path.name for path in aggregates.iterdir()} == set(AGGREGATE_FILENAMES)
    assert not plots.exists()

    second = RUNNER.invoke(
        app,
        [
            "report",
            "--runs",
            str(runs),
            "--aggregates-dir",
            str(aggregates),
            "--plots-dir",
            str(plots),
            "--output",
            str(output),
        ],
    )

    assert second.exit_code == 0, second.output
    assert output.read_bytes() == first_bytes
