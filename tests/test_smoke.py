from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import httpx
from typer.testing import CliRunner

from bm25_vfs_ablation.cli import app
from bm25_vfs_ablation.evaluation.plots import PLOT_FILENAMES
from bm25_vfs_ablation.experiment.schemas import RunRecord

RUNNER = CliRunner()
AGGREGATE_FILENAMES = {
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
}


def _run_smoke(workdir: Path, seed: int = 42):
    result = RUNNER.invoke(
        app,
        ["smoke", "--workdir", str(workdir), "--seed", str(seed)],
    )
    assert result.exit_code == 0, result.output
    return result


def _output_value(output: str, key: str) -> str:
    prefix = f"{key}="
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    raise AssertionError(f"{key}= was not printed:\n{output}")


def _run_root(result) -> Path:
    return Path(_output_value(result.output, "smoke_run_dir"))


def _records(result) -> list[RunRecord]:
    runs_path = Path(_output_value(result.output, "runs_path"))
    rows = [
        json.loads(line)
        for line in runs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [RunRecord.model_validate(row) for row in rows]


def test_offline_smoke_end_to_end(tmp_path: Path) -> None:
    workdir = tmp_path / "smoke"
    workdir.mkdir()
    sentinel = workdir / "preexisting.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    result = _run_smoke(workdir)
    run_root = _run_root(result)
    records = _records(result)

    assert run_root == workdir / "run-42"
    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert len(records) == 16
    assert (
        len((run_root / "data/generated/tasks.jsonl").read_text(encoding="utf-8").splitlines())
        == 12
    )
    assert (
        _output_value(result.output, "corpus_sha256")
        == hashlib.sha256((run_root / "data/generated/corpus.jsonl").read_bytes()).hexdigest()
    )
    assert (
        _output_value(result.output, "tasks_sha256")
        == hashlib.sha256((run_root / "data/generated/tasks.jsonl").read_bytes()).hexdigest()
    )
    assert "run_records=16" in result.output


def test_smoke_two_runs_have_equal_generation_hashes(tmp_path: Path) -> None:
    first = _run_smoke(tmp_path / "first")
    second = _run_smoke(tmp_path / "second")
    first_records = _records(first)
    second_records = _records(second)

    assert _output_value(first.output, "corpus_sha256") == _output_value(
        second.output,
        "corpus_sha256",
    )
    assert _output_value(first.output, "tasks_sha256") == _output_value(
        second.output,
        "tasks_sha256",
    )
    assert len(first_records) == len(second_records) == 16
    assert all(record.design_cell.value == "primary" for record in first_records + second_records)


def test_smoke_records_are_paired_and_bounded(tmp_path: Path) -> None:
    records = _records(_run_smoke(tmp_path / "smoke"))
    pairs: defaultdict[str, set[str]] = defaultdict(set)

    for record in records:
        assert record.design_cell.value == "primary"
        pairs[record.task_id].add(record.condition.value)
        assert record.total_tokens <= record.token_accounting.limit
        assert record.errors == []
        if record.condition.value == "snippets":
            assert record.tool_call_count == 0
            assert record.tool_trace == []
        else:
            assert record.tool_call_count <= record.max_tool_calls_permitted
            assert record.tool_trace

    assert len(pairs) == 8
    assert all(conditions == {"snippets", "vfs"} for conditions in pairs.values())


def test_smoke_outputs_report_and_ten_plots(tmp_path: Path) -> None:
    result = _run_smoke(tmp_path / "smoke")
    run_root = _run_root(result)
    plots_dir = run_root / "results/plots"
    aggregates_dir = run_root / "results/aggregates"
    report_path = run_root / "reports/experiment.md"

    assert {path.name for path in plots_dir.iterdir()} == set(PLOT_FILENAMES)
    assert all(path.stat().st_size > 0 for path in plots_dir.iterdir())
    assert {path.name for path in aggregates_dir.iterdir()} == AGGREGATE_FILENAMES
    assert report_path.is_file()
    assert report_path.read_text(encoding="utf-8").startswith(
        "# BM25 VFS Workspace vs BM25 Snippet Ablation\n"
    )
    assert "plot_files=10" in result.output


def test_smoke_makes_no_network_request(tmp_path: Path, monkeypatch) -> None:
    def fail_request(*args, **kwargs):
        raise AssertionError("smoke attempted a network request")

    monkeypatch.setattr(httpx.Client, "request", fail_request)
    result = _run_smoke(tmp_path / "smoke")
    assert "run_records=16" in result.output
