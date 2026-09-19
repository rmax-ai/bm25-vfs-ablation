"""Typer commands for deterministic dataset generation and experiment runs."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, NoReturn

import typer
import yaml

from bm25_vfs_ablation.config import load_config
from bm25_vfs_ablation.corpus.generator import (
    GenerationConfig,
    generate_dataset,
    write_dataset,
)
from bm25_vfs_ablation.corpus.loader import CorpusBundle, load_bundle
from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.evaluation.aggregates import write_aggregate_tables
from bm25_vfs_ablation.evaluation.plots import produce_all_plots
from bm25_vfs_ablation.experiment.artifacts import read_jsonl, write_jsonl_atomic
from bm25_vfs_ablation.experiment.reporting import write_report
from bm25_vfs_ablation.experiment.runner import ExperimentRunner
from bm25_vfs_ablation.experiment.schemas import RunRecord, TerminationReason
from bm25_vfs_ablation.models.client import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleClient,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic BM25/VFS ablation commands.",
)

_PROVIDERS = frozenset({"mock", "openai_compatible"})
_SPLITS = frozenset({"dev", "eval", "all"})
_INCLUDE_DESIGNS = frozenset({"primary", "all"})
_AGGREGATE_FILENAMES = (
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


def _repository_root(*paths: Path) -> Path:
    """Find the repository root without making configured paths cwd-relative."""

    starts = [path if path.is_dir() else path.parent for path in paths]
    if not starts:
        starts.append(Path.cwd())
    for start in starts:
        current = start
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "SPEC.md").is_file():
                return candidate
    for path in paths:
        if path != path.parent:
            return path if path.is_dir() else path.parent
    return Path.cwd()


def _resolve_repo_path(root: Path, path: Path | str) -> Path:
    """Resolve an absolute path or repository-relative POSIX path."""

    expanded = Path(os.path.expanduser(os.fspath(path)))
    if expanded.is_absolute():
        return expanded
    return root / expanded


def _user_error(error: BaseException) -> NoReturn:
    """Report a user, configuration, or schema error with exit code two."""

    message = str(error).strip() or error.__class__.__name__
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=2)


def _runtime_error(error: BaseException) -> NoReturn:
    """Report an execution failure with exit code one."""

    message = str(error).strip() or error.__class__.__name__
    typer.echo(f"runtime error: {message}", err=True)
    raise typer.Exit(code=1)


def _load_validated_runs(path: Path) -> tuple[RunRecord, ...]:
    """Read and validate every run record before any output is produced."""

    records: list[RunRecord] = []
    for line_number, row in enumerate(read_jsonl(path), start=1):
        try:
            records.append(RunRecord.model_validate(row))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid run record in {path} at line {line_number}: {error}"
            ) from error
    return tuple(records)


def _missing_aggregate_files(path: Path) -> bool:
    return any(not (path / filename).is_file() for filename in _AGGREGATE_FILENAMES)


def _validate_provider(provider: str) -> str:
    if provider not in _PROVIDERS:
        raise ValueError("model provider must be one of: mock, openai_compatible")
    return provider


def _final_answer(task: TaskRecord) -> str:
    """Build the only public answer emitted by the explicit mock policy."""

    return json.dumps(
        {
            "answer": task.canonical_answer,
            "citations": list(task.gold_chunk_ids),
            "justification": "Explicit mock oracle policy used.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _task_from_request(
    request: ModelRequest,
    tasks: tuple[TaskRecord, ...],
) -> TaskRecord:
    """Match a model request to its task using the deterministic user prompt."""

    user_messages = [
        message.get("content") for message in request.messages if message.get("role") == "user"
    ]
    prompt = next((content for content in reversed(user_messages) if isinstance(content, str)), "")
    matches = [task for task in tasks if f"Question:\n{task.question}\n\n" in prompt]
    if not matches:
        raise RuntimeError("mock policy could not identify the requested task")
    return max(matches, key=lambda task: len(task.question))


class _GoldMockClient:
    """Explicit no-network oracle policy for CLI smoke and acceptance runs."""

    returned_model_id = "mock-scripted-v1"

    def __init__(self, bundle: CorpusBundle) -> None:
        self._tasks = tuple(bundle.tasks)
        self._facts = {
            fact.fact_id: (document, fact)
            for document in bundle.documents
            for fact in document.facts
        }

    def complete(self, request: ModelRequest) -> ModelResponse:
        task = _task_from_request(request, self._tasks)
        is_vfs = bool(request.tools)
        if not is_vfs:
            return ModelResponse(
                content=_final_answer(task),
                returned_model_id=self.returned_model_id,
                latency_ms=0.0,
            )

        completed_reads = sum(message.get("role") == "tool" for message in request.messages)
        required_facts = sorted(task.required_fact_ids)
        if completed_reads < len(required_facts):
            document, fact = self._facts[required_facts[completed_reads]]
            return ModelResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "mock-call",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps(
                                {
                                    "end_line": fact.line_end,
                                    "path": document.path,
                                    "start_line": fact.line_start,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        },
                    }
                ],
                returned_model_id=self.returned_model_id,
                latency_ms=0.0,
            )
        return ModelResponse(
            content=_final_answer(task),
            returned_model_id=self.returned_model_id,
            latency_ms=0.0,
        )


def _model_client(provider: str, config: Any, bundle: CorpusBundle) -> ModelClient:
    if provider == "mock":
        return _GoldMockClient(bundle)
    api_key = config.model.api_key
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("openai_compatible provider requires a configured API key")
    return OpenAICompatibleClient(
        config.model.base_url,
        api_key,
        config.model.model,
        max_attempts=config.model.max_attempts,
    )


def _mark_mock_records(path: Path, run_keys: set[str]) -> None:
    """Mark newly written mock records so they cannot be read as performance."""

    if not run_keys or not path.is_file():
        return
    rows = list(read_jsonl(path))
    changed = False
    for row in rows:
        if row.get("run_key") not in run_keys:
            continue
        details = row.get("scoring_details")
        if not isinstance(details, dict):
            details = {}
        if details.get("mock_oracle_policy") is True:
            continue
        row["scoring_details"] = {
            **details,
            "mock_oracle_policy": True,
        }
        changed = True
    if changed:
        write_jsonl_atomic(path, rows)


@app.command()
def generate(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("data/generated"),
        "--output-dir",
        help="Directory for corpus.jsonl and tasks.jsonl.",
    ),
    dev_tasks: int = typer.Option(50, "--dev-tasks", help="Number of development tasks."),
    tasks: int = typer.Option(200, "--tasks", help="Number of evaluation tasks."),
    seed: int = typer.Option(42, "--seed", help="Deterministic generator seed."),
    corpus_size: int = typer.Option(80, "--corpus-size", help="Number of synthetic entities."),
    distractors: int = typer.Option(6, "--distractors", help="Distractor documents per task."),
    min_hops: int = typer.Option(2, "--min-hops", help="Minimum task hop count."),
    max_hops: int = typer.Option(4, "--max-hops", help="Maximum task hop count."),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Replace existing generated targets.",
    ),
) -> None:
    """Generate deterministic corpus and task JSONL artifacts."""

    try:
        root = _repository_root()
        target_dir = _resolve_repo_path(root, output_dir)
        targets = (target_dir / "corpus.jsonl", target_dir / "tasks.jsonl")
        existing = [path for path in targets if path.exists()]
        if existing and not force:
            names = ", ".join(str(path) for path in existing)
            raise ValueError(f"refusing to overwrite existing generated targets: {names}")

        generation_config = GenerationConfig(
            seed=seed,
            dev_tasks=dev_tasks,
            eval_tasks=tasks,
            corpus_size=corpus_size,
            distractors=distractors,
            min_hops=min_hops,
            max_hops=max_hops,
        )
        documents, task_records = generate_dataset(generation_config)
        hashes = write_dataset(target_dir, documents, task_records)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        _user_error(error)
    except Exception as error:
        _runtime_error(error)

    typer.echo(f"corpus_path={hashes.corpus_path}")
    typer.echo(f"corpus_sha256={hashes.corpus_sha256}")
    typer.echo(f"tasks_path={hashes.tasks_path}")
    typer.echo(f"tasks_sha256={hashes.tasks_sha256}")


@app.command()
def run(
    config: Path = typer.Option(  # noqa: B008
        ...,
        "--config",
        help="Repository-relative YAML config path.",
    ),
    runs: Path | None = typer.Option(  # noqa: B008
        None,
        "--runs",
        help="Optional repository-relative run-record output path.",
    ),
    task_limit: int | None = typer.Option(
        None,
        "--task-limit",
        help="Optional maximum number of selected tasks.",
    ),
    split: Literal["dev", "eval", "all"] = typer.Option(  # noqa: B008
        "eval",
        "--split",
        help="Task split: dev, eval, or all.",
    ),
    model_provider: Literal["mock", "openai_compatible"] | None = typer.Option(  # noqa: B008
        None,
        "--model-provider",
        help="Provider override: mock or openai_compatible.",
    ),
    resume: bool | None = typer.Option(
        None,
        "--resume/--no-resume",
        help="Override the configured resume behavior.",
    ),
) -> None:
    """Run the configured experiment over a selected task split."""

    try:
        if split not in _SPLITS:
            raise ValueError("split must be one of: dev, eval, all")
        if task_limit is not None and task_limit < 1:
            raise ValueError("task-limit must be positive")
        if model_provider is not None:
            _validate_provider(model_provider)

        root = _repository_root()
        config_path = _resolve_repo_path(root, config)
        root = _repository_root(config_path)
        overrides: dict[str, dict[str, object]] = {}
        if model_provider is not None:
            overrides["model"] = {"provider": model_provider}
        if resume is not None:
            overrides["experiment"] = {"resume": resume}
        effective_config = load_config(config_path, overrides or None)
        provider = _validate_provider(effective_config.model.provider)
        if provider == "openai_compatible" and not (
            isinstance(effective_config.model.api_key, str)
            and effective_config.model.api_key.strip()
        ):
            raise ValueError("openai_compatible provider requires a configured API key")

        corpus_path = _resolve_repo_path(root, effective_config.corpus.corpus_path)
        tasks_path = _resolve_repo_path(root, effective_config.corpus.tasks_path)
        bundle = load_bundle(corpus_path, tasks_path)
        selected_tasks = tuple(
            task for task in bundle.tasks if split == "all" or task.split.value == split
        )
        if task_limit is not None:
            selected_tasks = selected_tasks[:task_limit]
        selected_bundle = replace(bundle, tasks=selected_tasks)

        output_runs = (
            _resolve_repo_path(root, runs)
            if runs is not None
            else _resolve_repo_path(root, effective_config.experiment.output_runs)
        )
        client = _model_client(provider, effective_config, selected_bundle)
        summary = ExperimentRunner(
            effective_config,
            selected_bundle,
            client,
            output_runs=output_runs,
        ).run()
        if provider == "mock":
            _mark_mock_records(
                output_runs,
                {record.run_key for record in summary.records},
            )
        if any(
            record.termination_reason is TerminationReason.MODEL_ERROR for record in summary.records
        ):
            raise RuntimeError("one or more model calls failed")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        _user_error(error)
    except Exception as error:
        _runtime_error(error)

    record_count = sum(1 for _ in read_jsonl(output_runs)) if output_runs.is_file() else 0
    typer.echo(f"completed={summary.completed}")
    typer.echo(f"skipped={summary.skipped}")
    typer.echo(f"run_records={record_count}")
    typer.echo(f"runs_path={output_runs}")


@app.command()
def evaluate(
    runs: Path = typer.Option(  # noqa: B008
        ...,
        "--runs",
        help="Path to validated experiment run records.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("results/aggregates"),
        "--output-dir",
        help="Directory for aggregate CSV and JSON artifacts.",
    ),
    plots_dir: Path = typer.Option(  # noqa: B008
        Path("results/plots"),
        "--plots-dir",
        help="Directory for the ten required PNG plots.",
    ),
    include_design: Literal["primary", "all"] = typer.Option(  # noqa: B008
        "primary",
        "--include-design",
        help="Include only primary records or all design cells.",
    ),
) -> None:
    """Validate runs, write aggregate tables, and produce all plots."""

    try:
        if include_design not in _INCLUDE_DESIGNS:
            raise ValueError("include-design must be one of: primary, all")
        root = _repository_root()
        runs_path = _resolve_repo_path(root, runs)
        aggregate_path = _resolve_repo_path(root, output_dir)
        plot_path = _resolve_repo_path(root, plots_dir)
        records = _load_validated_runs(runs_path)
        aggregate_outputs = write_aggregate_tables(
            records,
            aggregate_path,
            include_design=include_design,
        )
        plot_outputs = produce_all_plots(records, plot_path)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        _user_error(error)
    except Exception as error:
        _runtime_error(error)

    typer.echo(f"aggregates_dir={aggregate_path}")
    typer.echo(f"aggregate_files={len(aggregate_outputs)}")
    typer.echo(f"plots_dir={plot_path}")
    typer.echo(f"plot_files={len(plot_outputs)}")


@app.command()
def report(
    runs: Path = typer.Option(  # noqa: B008
        ...,
        "--runs",
        help="Path to validated experiment run records.",
    ),
    aggregates_dir: Path = typer.Option(  # noqa: B008
        Path("results/aggregates"),
        "--aggregates-dir",
        help="Directory containing aggregate tables.",
    ),
    plots_dir: Path = typer.Option(  # noqa: B008
        Path("results/plots"),
        "--plots-dir",
        help="Directory containing plot artifacts.",
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("reports/experiment.md"),
        "--output",
        help="Markdown report output path.",
    ),
) -> None:
    """Validate runs, regenerate missing aggregates, and write the report."""

    try:
        root = _repository_root()
        runs_path = _resolve_repo_path(root, runs)
        aggregate_path = _resolve_repo_path(root, aggregates_dir)
        plot_path = _resolve_repo_path(root, plots_dir)
        report_path = _resolve_repo_path(root, output)
        records = _load_validated_runs(runs_path)
        if _missing_aggregate_files(aggregate_path):
            write_aggregate_tables(
                records,
                aggregate_path,
                include_design="primary",
            )
        written_report = write_report(
            records,
            aggregate_path,
            plot_path,
            report_path,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        _user_error(error)
    except Exception as error:
        _runtime_error(error)

    typer.echo(f"report_path={written_report}")


__all__ = ["app"]
