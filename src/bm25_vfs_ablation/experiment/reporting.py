"""Evidence-disciplined Markdown reporting for experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bm25_vfs_ablation.evaluation.plots import PLOT_FILENAMES

_MISSING = object()

_TABLE_FILENAMES = (
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

_NOT_RUN_STATUSES = {
    "missing",
    "not-run",
    "not_run",
    "unavailable",
}


@dataclass(frozen=True, slots=True)
class _ArtifactTable:
    """A small, dependency-free representation of one aggregate artifact."""

    name: str
    path: Path
    rows: tuple[dict[str, str], ...]
    available: bool
    reason: str | None = None


def _expanded_path(path: Path | str) -> Path:
    return Path(os.path.expanduser(os.fspath(path)))


def _field(value: object, path: str, default: object = _MISSING) -> object:
    """Read a dotted field from mappings, Pydantic models, or simple objects."""

    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                if part == "model_config" and "model_settings" in current:
                    current = current["model_settings"]
                    continue
                return default
            current = current[part]
            continue

        attribute = getattr(current, part, _MISSING)
        if attribute is _MISSING and part == "model_config":
            attribute = getattr(current, "model_settings", _MISSING)
        if attribute is _MISSING:
            return default
        current = attribute
    return current


def _enum_text(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    value = getattr(value, "value", value)
    text = str(value).strip()
    return text or None


def _text(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = _enum_text(value)
    return text if text else None


def _number(value: object) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _row_number(row: Mapping[str, str], *names: str) -> float | None:
    for name in names:
        value = _number(row.get(name, _MISSING))
        if value is not None:
            return value
    return None


def _row_text(row: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = _text(row.get(name, _MISSING))
        if value is not None:
            return value
    return None


def _status(row: Mapping[str, str]) -> str | None:
    status = _row_text(row, "status")
    return status.casefold() if status else None


def _error_reason(row: Mapping[str, str]) -> str | None:
    for name in ("error", "missing_reason"):
        value = _row_text(row, name)
        if value:
            return value
    status = _status(row)
    if status and status not in {"ok", "coefficient", "converged"}:
        return status
    return None


def _read_csv_table(directory: Path, name: str) -> _ArtifactTable:
    path = directory / name
    if not path.is_file():
        return _ArtifactTable(name, path, (), False, f"{name} is not available")

    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                return _ArtifactTable(name, path, (), False, f"{name} has no header")
            rows = tuple(
                {
                    str(key): "" if value is None else value
                    for key, value in row.items()
                    if key is not None
                }
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error):
        return _ArtifactTable(name, path, (), False, f"{name} could not be read")

    if not rows:
        return _ArtifactTable(name, path, (), False, f"{name} has no data rows")

    statuses = {_status(row) for row in rows}
    statuses.discard(None)
    unavailable = statuses and statuses.issubset(
        _NOT_RUN_STATUSES
        | {
            "error",
        }
    )
    if unavailable:
        reason = next((_error_reason(row) for row in rows if _error_reason(row)), None)
        return _ArtifactTable(
            name,
            path,
            rows,
            False,
            reason or f"{name} reports that the analysis was not run",
        )
    return _ArtifactTable(name, path, rows, True)


def _read_artifact_tables(directory: Path | str) -> dict[str, _ArtifactTable]:
    target = _expanded_path(directory)
    return {
        name: _read_csv_table(target, name) for name in _TABLE_FILENAMES if name.endswith(".csv")
    }


def _read_trace_table(directory: Path | str) -> _ArtifactTable:
    target = _expanded_path(directory)
    name = "trace_examples.json"
    path = target / name
    if not path.is_file():
        return _ArtifactTable(name, path, (), False, f"{name} is not available")
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _ArtifactTable(name, path, (), False, f"{name} could not be read")
    if not isinstance(payload, Mapping) or not payload:
        return _ArtifactTable(name, path, (), False, f"{name} contains no trace examples")
    return _ArtifactTable(name, path, ({"examples": str(len(payload))},), True)


def _read_tables(directory: Path | str) -> dict[str, _ArtifactTable]:
    tables = _read_artifact_tables(directory)
    tables["trace_examples.json"] = _read_trace_table(directory)
    return tables


def _normalise_records(records: Iterable[object]) -> tuple[object, ...]:
    if isinstance(records, (str, bytes, bytearray, Mapping)):
        raise TypeError("records must be an iterable of run records")
    return tuple(records)


def _unique_record_values(records: Iterable[object], path: str) -> tuple[str, ...]:
    values = {
        value
        for record in records
        for raw_value in (_field(record, path),)
        if (value := _text(raw_value)) is not None
    }
    return tuple(sorted(values))


def _record_condition(record: object) -> str | None:
    return _enum_text(_field(record, "condition"))


def _record_design(record: object) -> str | None:
    return _enum_text(_field(record, "design_cell"))


def _record_task_id(record: object) -> str | None:
    return _text(_field(record, "task_id"))


def _record_hop(record: object) -> int | None:
    return _integer(_field(record, "hop_count"))


def _record_category(record: object) -> str | None:
    return _text(_field(record, "category"))


def _record_split(record: object) -> str | None:
    split = _text(_field(record, "split"))
    if split:
        return split
    task_id = _record_task_id(record)
    if task_id and task_id.startswith("task-"):
        parts = task_id.split("-")
        if len(parts) > 1:
            return parts[1]
    return None


def _record_token_limit(record: object) -> int | None:
    return _integer(
        _field(
            record,
            "token_accounting.limit",
            _field(record, "token_limit"),
        )
    )


def _judge_state(records: Iterable[object]) -> str:
    states: list[bool] = []
    for record in records:
        value = _field(record, "secondary_judge.enabled")
        if isinstance(value, bool):
            states.append(value)
    if states and not any(states):
        return "disabled"
    if any(states):
        return "enabled"
    return "not recorded"


def _fmt_number(value: float | None, *, digits: int = 3) -> str | None:
    if value is None:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def _fmt_rate(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.3f}"


def _fmt_percent(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.1f}%"


def _tag(tag: str, text: str) -> str:
    return f"{tag} {text}"


def _not_run(table: _ArtifactTable | None, detail: str) -> str:
    reason = table.reason if table is not None else None
    return f"Not run — {reason or detail}."


def _usable_rows(table: _ArtifactTable | None) -> tuple[dict[str, str], ...]:
    if table is None or not table.available:
        return ()
    return table.rows


def _condition_rows(
    table: _ArtifactTable | None,
    condition: str,
    *,
    design_cell: str = "primary",
) -> tuple[dict[str, str], ...]:
    return tuple(
        row
        for row in _usable_rows(table)
        if _row_text(row, "condition") == condition
        and (_row_text(row, "design_cell") or design_cell) == design_cell
    )


def _first_row(table: _ArtifactTable | None) -> dict[str, str] | None:
    rows = _usable_rows(table)
    return rows[0] if rows else None


def _relative_link(
    directory: Path | str,
    filename: str,
    report_dir: Path | str,
) -> str:
    target_directory = _expanded_path(directory)
    start_directory = _expanded_path(report_dir)
    if target_directory.is_absolute() != start_directory.is_absolute():
        target_directory = Path(target_directory.name)
        start_directory = Path(".")
    try:
        relative = os.path.relpath(
            target_directory / filename,
            start=start_directory,
        )
    except ValueError:
        relative = f"{target_directory.name}/{filename}"
    return Path(relative).as_posix()


def _default_report_dir(aggregates_dir: Path, plots_dir: Path) -> Path:
    if aggregates_dir.is_absolute() and plots_dir.is_absolute():
        try:
            common = Path(os.path.commonpath((aggregates_dir, plots_dir)))
        except ValueError:
            return Path("reports")
        return common / "reports"
    return Path("reports")


def _plot_links(plots_dir: Path | str, report_dir: Path | str) -> str:
    links = [
        f"[{filename}]({_relative_link(plots_dir, filename, report_dir)})"
        for filename in PLOT_FILENAMES
    ]
    return _tag(
        "[OBSERVED]",
        "Required plot artifacts: " + ", ".join(links) + ".",
    )


def _table_links(aggregates_dir: Path | str, report_dir: Path | str) -> str:
    links = [
        f"[{filename}]({_relative_link(aggregates_dir, filename, report_dir)})"
        for filename in _TABLE_FILENAMES
    ]
    return _tag(
        "[OBSERVED]",
        "Aggregate tables and trace artifacts: " + ", ".join(links) + ".",
    )


def _paired_observation(table: _ArtifactTable) -> str | None:
    row = _first_row(table)
    if row is None:
        return None
    n = _row_number(row, "n")
    snippets_rate = _row_number(row, "snippets_success_rate")
    vfs_rate = _row_number(row, "vfs_success_rate")
    difference = _row_number(row, "difference")
    parts: list[str] = []
    if n is not None:
        parts.append(f"{_fmt_number(n)} paired tasks")
    if snippets_rate is not None:
        parts.append(f"snippets success rate {_fmt_rate(snippets_rate)}")
    if vfs_rate is not None:
        parts.append(f"VFS success rate {_fmt_rate(vfs_rate)}")
    if difference is not None:
        parts.append(f"VFS minus snippets {_fmt_rate(difference)}")
    if not parts:
        return None
    return ", ".join(parts)


def _paired_inference(table: _ArtifactTable) -> str | None:
    row = _first_row(table)
    if row is None:
        return None
    lower = _row_number(row, "ci_lower")
    upper = _row_number(row, "ci_upper")
    p_value = _row_number(row, "mcnemar_p_value")
    parts: list[str] = []
    if lower is not None and upper is not None:
        parts.append(f"paired interval [{_fmt_rate(lower)}, {_fmt_rate(upper)}]")
    if p_value is not None:
        parts.append(f"exact McNemar p-value {_fmt_number(p_value, digits=4)}")
    return ", ".join(parts) if parts else None


def _condition_observations(table: _ArtifactTable) -> str | None:
    parts: list[str] = []
    for condition in ("snippets", "vfs"):
        rows = _condition_rows(table, condition)
        row = rows[0] if rows else None
        if row is None:
            continue
        metrics = [
            ("initial chunk recall", _row_number(row, "initial_chunk_recall_mean")),
            ("initial document recall", _row_number(row, "initial_document_recall_mean")),
            ("final evidence recall", _row_number(row, "final_evidence_recall_mean")),
            ("evidence precision", _row_number(row, "evidence_precision_mean")),
        ]
        rendered = [f"{label} {_fmt_rate(value)}" for label, value in metrics if value is not None]
        if rendered:
            parts.append(f"{condition}: " + ", ".join(rendered))
    return "; ".join(parts) if parts else None


def _tool_observations(table: _ArtifactTable) -> str | None:
    rows = _condition_rows(table, "vfs")
    row = rows[0] if rows else None
    if row is None:
        return None
    metrics = [
        ("actual calls mean", _row_number(row, "actual_tool_calls_mean")),
        ("successful calls mean", _row_number(row, "successful_tool_calls_mean")),
        ("invalid calls mean", _row_number(row, "invalid_tool_calls_mean")),
        ("repeated calls mean", _row_number(row, "repeated_tool_calls_mean")),
        ("unique files read mean", _row_number(row, "unique_files_read_count_mean")),
    ]
    rendered = [f"{label} {_fmt_number(value)}" for label, value in metrics if value is not None]
    return ", ".join(rendered) if rendered else None


def _intervention_rows(table: _ArtifactTable) -> tuple[dict[str, str], ...]:
    rows = _usable_rows(table)
    result: list[dict[str, str]] = []
    for row in rows:
        if _row_text(row, "condition") not in {None, "vfs"}:
            continue
        design = _row_text(row, "design_cell")
        limit = _row_number(row, "permitted_call_limit", "max_tool_calls_permitted")
        if limit is None and design and design.startswith("max_calls_"):
            limit = _number(design.removeprefix("max_calls_"))
        if limit is not None:
            result.append(row)
    return tuple(result)


def _intervention_observation(table: _ArtifactTable) -> str | None:
    rows = _intervention_rows(table)
    if not rows:
        return None
    levels: list[str] = []
    rates: list[str] = []
    for row in rows:
        limit = _row_number(row, "permitted_call_limit", "max_tool_calls_permitted")
        if limit is None:
            design = _row_text(row, "design_cell")
            if design:
                limit = _number(design.removeprefix("max_calls_"))
        if limit is not None:
            rendered_limit = _fmt_number(limit)
            if rendered_limit not in levels:
                levels.append(rendered_limit or "")
        rate = _row_number(row, "success_rate")
        if rate is not None:
            rates.append(f"{_fmt_rate(rate)} at assigned limit")
    text = f"assigned maximum-call levels {', '.join(levels)}"
    if rates:
        text += "; observed success rates include " + ", ".join(rates)
    return text


def _failure_observation(table: _ArtifactTable) -> str | None:
    rows = _usable_rows(table)
    if not rows:
        return None
    counts: list[tuple[str, float]] = []
    for row in rows:
        label = _row_text(row, "failure_class")
        count = _row_number(row, "count")
        if label and count is not None:
            counts.append((label, count))
    if not counts:
        return None
    label, count = max(counts, key=lambda item: (-item[1], item[0]))
    return (
        f"{_fmt_number(float(len(rows)))} failure-summary rows; "
        f"largest recorded label {label} count {_fmt_number(count)}"
    )


def _efficiency_observation(table: _ArtifactTable) -> str | None:
    parts: list[str] = []
    for condition in ("snippets", "vfs"):
        rows = _condition_rows(table, condition)
        row = rows[0] if rows else None
        if row is None:
            continue
        metrics = [
            ("tokens mean", _row_number(row, "total_tokens_mean")),
            ("wall latency mean ms", _row_number(row, "wall_ms_mean")),
            ("estimated cost mean", _row_number(row, "mean_cost_usd")),
            ("success per 1000 tokens", _row_number(row, "success_per_1000_tokens")),
        ]
        rendered = [
            f"{label} {_fmt_number(value)}" for label, value in metrics if value is not None
        ]
        if rendered:
            parts.append(f"{condition}: " + ", ".join(rendered))
    return "; ".join(parts) if parts else None


def _regression_is_available(table: _ArtifactTable) -> bool:
    return any(
        row.get("term", "").strip() not in {"", "__status__"}
        and _row_number(row, "coefficient") is not None
        for row in _usable_rows(table)
    )


def _regression_observation(table: _ArtifactTable) -> str | None:
    if not _regression_is_available(table):
        return None
    rows = [
        row for row in _usable_rows(table) if row.get("term", "").strip() not in {"", "__status__"}
    ]
    n = _row_number(rows[0], "n") if rows else None
    return f"{_fmt_number(float(len(rows)))} coefficient rows" + (
        f" across {_fmt_number(n)} VFS records" if n is not None else ""
    )


def _metadata_paragraphs(records: tuple[object, ...]) -> list[str]:
    if not records:
        return ["Not run — no run records were supplied for provenance fields."]

    task_ids = {task_id for record in records if (task_id := _record_task_id(record)) is not None}
    conditions = {
        condition for record in records if (condition := _record_condition(record)) is not None
    }
    designs = {design for record in records if (design := _record_design(record)) is not None}
    versions = _unique_record_values(records, "corpus_version")
    splits = {split for record in records if (split := _record_split(record)) is not None}
    tokens = {limit for record in records if (limit := _record_token_limit(record)) is not None}
    paragraphs = [
        _tag(
            "[OBSERVED]",
            f"Supplied run records cover {_fmt_number(float(len(task_ids)))} unique task IDs, "
            f"{_fmt_number(float(len(conditions)))} conditions, and "
            f"{_fmt_number(float(len(designs)))} design cells.",
        )
    ]
    details: list[str] = []
    if versions:
        details.append("corpus version(s): " + ", ".join(f"`{value}`" for value in versions))
    if splits:
        details.append("split(s): " + ", ".join(f"`{value}`" for value in sorted(splits)))
    if tokens:
        details.append(
            "recorded token ceiling(s): "
            + ", ".join(_fmt_number(float(value)) for value in sorted(tokens))
        )
    if details:
        paragraphs.append(_tag("[OBSERVED]", "; ".join(details) + "."))
    return paragraphs


def _hash_paragraph(records: tuple[object, ...]) -> str:
    config_hashes = _unique_record_values(records, "config_sha256")
    corpus_hashes = _unique_record_values(records, "corpus_sha256")
    if not config_hashes and not corpus_hashes:
        return "Not run — configuration and corpus hashes were not present in the run records."
    config_text = ", ".join(f"`{value}`" for value in config_hashes) or "not recorded"
    corpus_text = ", ".join(f"`{value}`" for value in corpus_hashes) or "not recorded"
    return _tag(
        "[OBSERVED]",
        f"Recorded configuration SHA-256 value(s): {config_text}; "
        f"recorded corpus SHA-256 value(s): {corpus_text}.",
    )


def _judge_paragraph(records: tuple[object, ...]) -> str:
    state = _judge_state(records)
    if state == "disabled":
        return _tag(
            "[OBSERVED]",
            "Judge disabled: the secondary judge is disabled; deterministic answer scoring "
            "is the primary scorer.",
        )
    if state == "enabled":
        return _tag(
            "[OBSERVED]",
            "The run records include an enabled secondary judge; its result remains secondary.",
        )
    return "Not run — secondary-judge state was not recorded."


def _section_links(
    lines: list[str],
    aggregates_dir: Path,
    plots_dir: Path,
    report_dir: Path,
) -> None:
    lines.append(_plot_links(plots_dir, report_dir))
    lines.append("")
    lines.append(_table_links(aggregates_dir, report_dir))


def generate_report(
    records: Iterable[object],
    aggregates_dir: Path | str,
    plots_dir: Path | str,
    *,
    report_dir: Path | str | None = None,
) -> str:
    """Generate a deterministic Markdown report from records and aggregate artifacts."""

    record_rows = _normalise_records(records)
    aggregate_path = _expanded_path(aggregates_dir)
    plot_path = _expanded_path(plots_dir)
    report_path = (
        _expanded_path(report_dir)
        if report_dir is not None
        else _default_report_dir(aggregate_path, plot_path)
    )
    tables = _read_tables(aggregate_path)
    paired = tables["paired_outcomes.csv"]
    condition = tables["condition_summary.csv"]
    intervention = tables["intervention_summary.csv"]
    regression = tables["within_vfs_regression.csv"]
    efficiency = tables["efficiency_summary.csv"]
    failure = tables["failure_summary.csv"]
    traces = tables["trace_examples.json"]

    lines: list[str] = [
        "# BM25 VFS Workspace vs BM25 Snippet Ablation",
        "",
        "Evidence-disciplined report generated from recorded runs and aggregate artifacts.",
        "",
        "## Executive summary",
        "",
    ]
    paired_text = _paired_observation(paired)
    if paired_text:
        lines.append(_tag("[OBSERVED]", f"The primary paired outcome table reports {paired_text}."))
    else:
        lines.append(_not_run(paired, "the primary paired outcome table is unavailable"))
    paired_inference = _paired_inference(paired)
    if paired_inference:
        lines.append(
            _tag(
                "[INFERENCE]",
                f"The paired uncertainty summary reports {paired_inference}; "
                "this is statistical inference about the recorded pairs, not a mechanism claim.",
            )
        )
    else:
        lines.append(
            "[INFERENCE] No primary paired inference is stated because the required uncertainty "
            "fields were not available."
        )
    lines.append(_judge_paragraph(record_rows))
    lines.extend(["", "## Hypothesis/design", ""])
    lines.append(
        _tag(
            "[SPECULATION]",
            "The prespecified hypothesis is that BM25-backed VFS navigation may improve "
            "multi-hop task success relative to BM25 snippets; this expectation is not itself "
            "an observed result.",
        )
    )
    lines.append(
        _tag(
            "[OBSERVED]",
            "The experimental design compares the snippets and VFS conditions on the same "
            "question set, corpus, shared BM25 index, answer scorer, and configured token ceiling.",
        )
    )
    lines.append(
        _tag(
            "[OBSERVED]",
            "Assigned maximum-call cells, when present, are separate from the primary cell; "
            "actual calls and permitted calls are reported as different quantities.",
        )
    )

    lines.extend(["", "## Controls", ""])
    lines.append(
        _tag(
            "[OBSERVED]",
            "Controlled variables are model settings, sampling settings, corpus, tasks, "
            "retrieval implementation, chunking, answer format, retry policy, concurrency, "
            "and the per-task estimated token ceiling.",
        )
    )
    lines.append(_hash_paragraph(record_rows))
    lines.append(
        _tag(
            "[OBSERVED]",
            "Condition order is recorded per task so paired analyses can preserve the "
            "preassigned execution order.",
        )
    )

    lines.extend(["", "## Dataset", ""])
    lines.extend(_metadata_paragraphs(record_rows))
    lines.append(
        _tag(
            "[OBSERVED]",
            "Dataset provenance is taken from the run records; this report does not infer "
            "unrecorded corpus contents or ground-truth evidence.",
        )
    )

    lines.extend(["", "## Primary", ""])
    if paired_text:
        lines.append(
            _tag(
                "[OBSERVED]",
                "Primary success is task-level correctness, with snippets and VFS "
                "paired by task ID.",
            )
        )
    else:
        lines.append(_not_run(paired, "primary success analysis is unavailable"))
    lines.append(
        _tag(
            "[INFERENCE]",
            "The paired estimate should be interpreted with its reported interval and exact "
            "paired test; it does not by itself identify why conditions differ.",
        )
    )
    lines.append(
        f"Key primary table: [{paired.name}]("
        f"{_relative_link(aggregate_path, paired.name, report_path)})."
    )

    lines.extend(["", "## Retrieval/evidence", ""])
    condition_text = _condition_observations(condition)
    if condition_text:
        lines.append(
            _tag(
                "[OBSERVED]",
                "Retrieval and evidence metrics are reported separately by condition: "
                f"{condition_text}.",
            )
        )
    else:
        lines.append(
            _not_run(
                condition,
                "condition-level retrieval and evidence metrics are unavailable",
            )
        )
    lines.append(
        _tag(
            "[INFERENCE]",
            "Initial retrieval recall, accessed evidence recall, evidence precision, and "
            "answer correctness are distinct measurements; retrieval recall alone is not "
            "treated as proof of successful multi-hop synthesis.",
        )
    )
    for name in ("subgroup_summary.csv", "stratified_summary.csv"):
        lines.append(
            f"Supporting table: [{name}]({_relative_link(aggregate_path, name, report_path)})."
        )

    lines.extend(["", "## Tool use", ""])
    tool_text = _tool_observations(condition)
    if tool_text:
        lines.append(_tag("[OBSERVED]", f"Within the primary VFS condition, {tool_text}."))
    else:
        lines.append(_not_run(condition, "primary VFS tool-use metrics are unavailable"))
    regression_text = _regression_observation(regression)
    if regression_text:
        lines.append(
            _tag(
                "[INFERENCE]",
                f"The within-VFS calls regression contains {regression_text}. "
                "It is associative; observed calls are post-treatment variables and are "
                "not interpreted as an assigned treatment.",
            )
        )
    else:
        lines.append(
            _not_run(
                regression,
                "the within-VFS calls regression is unavailable",
            )
            + " The specified calls regression is associative and treats observed calls as "
            "post-treatment variables."
        )
    lines.append(
        f"Regression table: [within_vfs_regression.csv]("
        f"{_relative_link(aggregate_path, 'within_vfs_regression.csv', report_path)})."
    )

    lines.extend(["", "## Intervention", ""])
    intervention_text = _intervention_observation(intervention)
    if intervention_text:
        lines.append(
            _tag(
                "[OBSERVED]",
                f"The intervention table reports {intervention_text}; permitted limits remain "
                "separate from actual calls.",
            )
        )
        lines.append(
            _tag(
                "[CAUSAL-INTERVENTION]",
                "Because maximum permitted calls were assigned as a design factor, this "
                "comparison is the report's causal-intervention evidence; it does not assign "
                "the observed number of calls.",
            )
        )
    else:
        lines.append(
            _not_run(
                intervention,
                "no intervention design cell with a permitted-call limit was available",
            )
            + " No intervention-based conclusion is made."
        )
    lines.append(
        f"Intervention table: [intervention_summary.csv]("
        f"{_relative_link(aggregate_path, 'intervention_summary.csv', report_path)})."
    )

    lines.extend(["", "## Efficiency", ""])
    efficiency_text = _efficiency_observation(condition)
    if efficiency_text is None:
        efficiency_text = _efficiency_observation(efficiency)
    if efficiency_text:
        lines.append(
            _tag(
                "[OBSERVED]",
                f"Efficiency summaries report {efficiency_text}.",
            )
        )
    else:
        lines.append(_not_run(efficiency, "efficiency metrics are unavailable"))
    lines.append(
        _tag(
            "[INFERENCE]",
            "Latency, cost, and success per token are trade-off measures and should be "
            "read alongside the common token-accounting boundary.",
        )
    )
    lines.append(
        f"Efficiency table: [efficiency_summary.csv]("
        f"{_relative_link(aggregate_path, 'efficiency_summary.csv', report_path)})."
    )

    lines.extend(["", "## Failures", ""])
    failure_text = _failure_observation(failure)
    if failure_text:
        lines.append(_tag("[OBSERVED]", f"Failure classification reports {failure_text}."))
    else:
        lines.append(_not_run(failure, "failure classification is unavailable"))
    if traces.available:
        lines.append(
            _tag(
                "[OBSERVED]",
                "Trace examples are limited to observable messages, tool calls, accessed "
                "evidence, answers, and concise justifications.",
            )
        )
    else:
        lines.append(_not_run(traces, "observable trace examples are unavailable"))
    lines.append(
        _tag(
            "[INFERENCE]",
            "Failure labels are reproducible, nonexclusive classifications; they are not "
            "claims about hidden model reasoning.",
        )
    )
    lines.append(
        f"Failure table: [failure_summary.csv]("
        f"{_relative_link(aggregate_path, 'failure_summary.csv', report_path)}); "
        f"trace table: [trace_examples.json]("
        f"{_relative_link(aggregate_path, 'trace_examples.json', report_path)})."
    )

    lines.extend(["", "## Threats", ""])
    lines.append(
        _tag(
            "[INFERENCE]",
            "Threats to validity are documented below and limit the scope of any interpretation.",
        )
    )
    threats = (
        (
            "Synthetic-task realism",
            "Synthetic multi-hop tasks may not represent production questions, documents, "
            "noise, or stakes.",
        ),
        (
            "Prompt sensitivity",
            "Small changes to system instructions, retrieved context, tool descriptions, "
            "or answer formatting may change outcomes.",
        ),
        (
            "Model/provider drift",
            "A different served model, provider behavior, or later model version may not "
            "reproduce these observations.",
        ),
        (
            "Tokenizer/accounting error",
            "Approximate token counts and provider reconciliation can misstate the effective "
            "budget even when enforcement is deterministic.",
        ),
        (
            "VFS interface quality",
            "Tool names, search ranking, previews, limits, and path navigation may favor "
            "or hinder the VFS condition.",
        ),
        (
            "Different prompt overhead between conditions",
            "Condition-specific instructions and tool schemas can create unequal fixed "
            "prompt overhead even under a common total ceiling.",
        ),
        (
            "Tool-call count as a post-treatment variable",
            "Observed call count is produced during execution and can reflect difficulty, "
            "early stopping, or failure; it is not a randomized mediator.",
        ),
        (
            "Repeated observations over shared templates",
            "Tasks sharing templates or entities can reduce effective independence and make "
            "simple row counts overstate information.",
        ),
        (
            "Judge-model bias, if an LLM judge is used",
            "A secondary judge can add model, rubric, and preference bias; the deterministic "
            "scorer remains primary when the judge is disabled.",
        ),
    )
    for title, explanation in threats:
        lines.append(_tag("[INFERENCE]", f"**{title}.** {explanation}"))

    lines.extend(["", "## Conclusions", ""])
    if paired_text:
        lines.append(
            _tag(
                "[INFERENCE]",
                "The defensible conclusion is limited to the recorded paired comparison and "
                "its uncertainty; it should not be generalized beyond this corpus, task set, "
                "model configuration, and accounting boundary.",
            )
        )
    else:
        lines.append(
            "[INFERENCE] No primary conclusion is reported because the required paired "
            "analysis was not run or was unavailable."
        )
    if intervention_text:
        lines.append(
            _tag(
                "[CAUSAL-INTERVENTION]",
                "The assigned-call comparison may support an intervention-specific conclusion "
                "only for the recorded assignment and design cell; it does not turn observed "
                "calls into a randomized exposure.",
            )
        )
    else:
        lines.append(
            "[INFERENCE] No intervention conclusion is reported because the intervention "
            "cell was not run."
        )
    lines.append(
        _tag(
            "[SPECULATION]",
            "Any broader explanation that adaptive navigation is the mechanism, rather than "
            "an association compatible with the observations, remains speculation without "
            "additional controlled evidence.",
        )
    )

    lines.extend(["", "### Reproduction commands", ""])
    lines.append(
        _tag(
            "[OBSERVED]",
            "The exact offline reproduction sequence is:",
        )
    )
    lines.extend(
        [
            "```bash",
            "uv run python -m bm25_vfs_ablation generate --tasks 200 --seed 42",
            "uv run python -m bm25_vfs_ablation run --config configs/default.yaml",
            "uv run python -m bm25_vfs_ablation evaluate --runs results/runs.jsonl",
            "uv run python -m bm25_vfs_ablation report --runs results/runs.jsonl "
            "--output reports/experiment.md",
            "uv run pytest -q",
            "uv run ruff check .",
            "```",
        ]
    )
    lines.append("")
    _section_links(lines, aggregate_path, plot_path, report_path)
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    records: Iterable[object],
    aggregates_dir: Path | str,
    plots_dir: Path | str,
    output: Path | str,
) -> Path:
    """Generate and write a UTF-8 Markdown report at ``output``."""

    target = _expanded_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(
        records,
        aggregates_dir,
        plots_dir,
        report_dir=target.parent,
    )
    target.write_text(report, encoding="utf-8", newline="\n")
    return target


__all__ = ["generate_report", "write_report"]
