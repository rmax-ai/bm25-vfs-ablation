"""Deterministic, headless plots for the experiment aggregates."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bm25_vfs_ablation import FailureClass
from bm25_vfs_ablation.evaluation.aggregates import build_run_frame
from bm25_vfs_ablation.evaluation.failures import classify_failure
from bm25_vfs_ablation.evaluation.statistics import paired_summary

FIGSIZE = (8.0, 5.0)
DPI = 120
CONDITION_ORDER = ("snippets", "vfs")
CONDITION_COLORS = {
    "snippets": "#4472C4",
    "vfs": "#C44E52",
}
PLOT_FILENAMES = (
    "01_success_by_condition.png",
    "02_success_by_hop.png",
    "03_success_by_actual_tool_calls.png",
    "04_success_by_max_tool_calls.png",
    "05_initial_recall_vs_success.png",
    "06_evidence_recall_vs_success.png",
    "07_tokens_vs_success.png",
    "08_latency_cost_by_condition.png",
    "09_failure_modes.png",
    "10_paired_outcomes.png",
)

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"

_MISSING = object()


def _is_missing(value: object) -> bool:
    if value is _MISSING or value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _text(value: object) -> str:
    if _is_missing(value):
        return ""
    value = getattr(value, "value", value)
    return str(value).strip().casefold()


def _number(value: object) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _success(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return 1.0
        if normalized in {"false", "no", "0"}:
            return 0.0
    number = _number(value)
    if number in {0.0, 1.0}:
        return number
    return None


def _as_frame(data: pd.DataFrame | Iterable[object] | Mapping[str, object]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        return build_run_frame([data])
    return build_run_frame(data)


def _primary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "design_cell" not in frame.columns:
        return frame.copy()
    design = frame["design_cell"].map(_text)
    return frame.loc[design.eq("primary")].copy()


def _condition_frame(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    if "condition" not in frame.columns:
        return frame.iloc[0:0].copy()
    values = frame["condition"].map(_text)
    return frame.loc[values.eq(condition)].copy()


def _numeric_column(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[name], errors="coerce")
        result = result.where(result.notna(), candidate)
    return result


def _success_column(frame: pd.DataFrame) -> pd.Series:
    names = (
        "success",
        "correctness",
        "task_success",
    )
    for name in names:
        if name in frame.columns:
            return frame[name].map(_success)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _new_figure() -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    _style_axis(axis)
    return figure, axis


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#808080")


def _target(out_dir: Path | str, filename: str) -> Path:
    target = Path(os.path.expanduser(os.fspath(out_dir)))
    if target.suffix.casefold() == ".png":
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    directory = target
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _save(figure: plt.Figure, path: Path) -> Path:
    try:
        figure.tight_layout()
        figure.savefig(
            path,
            format="png",
            dpi=DPI,
            facecolor="white",
            edgecolor="white",
        )
    finally:
        plt.close(figure)
    return path


def _no_data(axis: plt.Axes, message: str, title: str) -> None:
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="#555555",
    )


def _rate_points(
    frame: pd.DataFrame,
    x_values: pd.Series,
    *,
    conditions: Sequence[str] = CONDITION_ORDER,
    primary_only: bool = True,
) -> dict[str, list[tuple[float, float, int]]]:
    working = _primary_frame(frame) if primary_only else frame.copy()
    if "condition" not in working.columns:
        return {}
    success = _success_column(working)
    x_values = x_values.reindex(working.index)
    output: dict[str, list[tuple[float, float, int]]] = {}
    condition_values = working["condition"].map(_text)
    for condition in conditions:
        subset = working.loc[condition_values.eq(condition)]
        points: dict[float, list[float]] = {}
        for index in subset.index:
            x_value = _number(x_values.loc[index])
            success_value = _number(success.loc[index])
            if x_value is None or success_value is None:
                continue
            points.setdefault(x_value, []).append(success_value)
        output[condition] = [
            (x_value, sum(values) / len(values), len(values))
            for x_value, values in sorted(points.items())
        ]
    return output


def _draw_rate_points(
    axis: plt.Axes,
    points: Mapping[str, Sequence[tuple[float, float, int]]],
    *,
    xlabel: str,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    plotted = False
    for condition in CONDITION_ORDER:
        condition_points = points.get(condition, ())
        if not condition_points:
            continue
        plotted = True
        x_values = [point[0] for point in condition_points]
        rates = [point[1] for point in condition_points]
        axis.plot(
            x_values,
            rates,
            marker="o",
            linewidth=1.8,
            color=CONDITION_COLORS[condition],
            label=condition,
        )
        for x_value, rate, count in condition_points:
            axis.annotate(
                f"mean={rate:.2f}\n(n={count})",
                (x_value, rate),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=CONDITION_COLORS[condition],
            )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Success rate (mean)")
    axis.set_ylim(-0.05, 1.05)
    if xlim is not None:
        axis.set_xlim(*xlim)
    if plotted:
        axis.legend(frameon=False)
    else:
        _no_data(axis, "No data", title)


def _paired_stat_value(stats: object, names: Sequence[str]) -> float | None:
    for name in names:
        if isinstance(stats, Mapping):
            value = stats.get(name, _MISSING)
        else:
            value = getattr(stats, name, _MISSING)
        number = _number(value)
        if number is not None:
            return number
    return None


def _paired_interval(frame: pd.DataFrame) -> tuple[float, float, float] | None:
    try:
        stats = paired_summary(frame)
    except (KeyError, TypeError, ValueError):
        return None
    difference = _paired_stat_value(stats, ("difference", "vfs_minus_snippets"))
    lower = _paired_stat_value(stats, ("ci_lower", "ci_low"))
    upper = _paired_stat_value(stats, ("ci_upper", "ci_high"))
    if difference is None or lower is None or upper is None:
        return None
    return difference, lower, upper


def _condition_success_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
) -> tuple[list[str], list[float], list[int]]:
    labels: list[str] = []
    means: list[float] = []
    counts: list[int] = []
    for condition in CONDITION_ORDER:
        values = _success_column(_condition_frame(frame, condition)).dropna()
        if values.empty:
            continue
        labels.append(condition)
        means.append(float(values.mean()))
        counts.append(int(values.size))
    if means:
        positions = np.arange(len(means), dtype=float)
        bars = axis.bar(
            positions,
            means,
            color=[CONDITION_COLORS[label] for label in labels],
            width=0.62,
        )
        for bar, mean, count in zip(bars, means, counts, strict=True):
            axis.annotate(
                f"mean={mean:.2f}\n(n={count})",
                (bar.get_x() + bar.get_width() / 2, mean),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1.1)
        axis.set_ylabel("Success rate (mean)")
    return labels, means, counts


def plot_success_by_condition(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot primary success rates and the paired VFS-minus-snippets interval."""

    working = _primary_frame(_as_frame(frame))
    figure, axis = _new_figure()
    labels, _, _ = _condition_success_bars(axis, working)
    if not labels:
        _no_data(axis, "No primary condition data", "Success rate by condition")
    else:
        axis.set_title("Success rate by condition")
        interval = _paired_interval(working)
        if interval is None:
            interval_text = "Paired bootstrap interval unavailable"
        else:
            difference, lower, upper = interval
            interval_text = (
                f"Paired VFS - snippets: {difference:+.3f} "
                f"(95% CI [{lower:+.3f}, {upper:+.3f}])"
            )
        axis.text(
            0.5,
            0.98,
            interval_text,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )
    return _save(figure, _target(out_dir, PLOT_FILENAMES[0]))


def plot_success_by_hop(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot mean binary success by task hop count."""

    working = _primary_frame(_as_frame(frame))
    x_values = _numeric_column(working, ("hop_count", "hops"))
    points = _rate_points(working, x_values)
    figure, axis = _new_figure()
    _draw_rate_points(
        axis,
        points,
        xlabel="Hop count",
        title="Success rate by hop count",
    )
    return _save(figure, _target(out_dir, PLOT_FILENAMES[1]))


def plot_success_by_actual_calls(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot VFS success by observed, actual tool-call count."""

    working = _primary_frame(_as_frame(frame))
    working = _condition_frame(working, "vfs")
    x_values = _numeric_column(
        working,
        ("tool_call_count", "actual_tool_calls", "calls"),
    )
    points = _rate_points(
        working,
        x_values,
        conditions=("vfs",),
        primary_only=False,
    )
    figure, axis = _new_figure()
    _draw_rate_points(
        axis,
        points,
        xlabel="Actual VFS tool calls (observed)",
        title="Success rate versus actual VFS tool calls",
    )
    return _save(figure, _target(out_dir, PLOT_FILENAMES[2]))


def _intervention_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "condition" in frame.columns:
        condition = frame["condition"].map(_text)
        frame = frame.loc[condition.eq("vfs")].copy()
    else:
        frame = frame.copy()
    if "design_cell" in frame.columns:
        design = frame["design_cell"].map(_text)
        frame = frame.loc[design.str.startswith("max_calls_", na=False)].copy()
    permitted = _numeric_column(
        frame,
        ("max_tool_calls_permitted", "permitted_tool_calls"),
    )
    if "design_cell" in frame.columns:
        design = frame["design_cell"].map(_text)
        parsed = pd.to_numeric(
            design.str.removeprefix("max_calls_"),
            errors="coerce",
        )
        permitted = permitted.where(permitted.notna(), parsed)
    frame["__plot_permitted_calls"] = permitted
    return frame


def plot_intervention_dose_response(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot success by assigned maximum permitted VFS calls."""

    working = _intervention_frame(_as_frame(frame))
    x_values = _numeric_column(working, ("__plot_permitted_calls",))
    points = _rate_points(
        working,
        x_values,
        conditions=("vfs",),
        primary_only=False,
    )
    figure, axis = _new_figure()
    if not points.get("vfs"):
        _no_data(axis, "No intervention data", "Intervention dose response")
    else:
        _draw_rate_points(
            axis,
            points,
            xlabel="Maximum permitted VFS tool calls (assigned)",
            title="Success rate versus maximum permitted VFS tool calls",
        )
    return _save(figure, _target(out_dir, PLOT_FILENAMES[3]))


def plot_initial_recall_vs_success(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot success against initial retrieval recall."""

    working = _primary_frame(_as_frame(frame))
    x_values = _numeric_column(
        working,
        ("initial_chunk_recall", "initial_recall", "initial_document_recall"),
    )
    points = _rate_points(working, x_values)
    figure, axis = _new_figure()
    _draw_rate_points(
        axis,
        points,
        xlabel="Initial retrieval recall",
        title="Success rate versus initial retrieval recall",
        xlim=(0.0, 1.0),
    )
    return _save(figure, _target(out_dir, PLOT_FILENAMES[4]))


def plot_evidence_recall_vs_success(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot success against final accessed-evidence recall."""

    working = _primary_frame(_as_frame(frame))
    x_values = _numeric_column(
        working,
        ("final_evidence_recall", "final_recall", "evidence_recall"),
    )
    points = _rate_points(working, x_values)
    figure, axis = _new_figure()
    _draw_rate_points(
        axis,
        points,
        xlabel="Final evidence recall",
        title="Success rate versus final evidence recall",
        xlim=(0.0, 1.0),
    )
    return _save(figure, _target(out_dir, PLOT_FILENAMES[5]))


def plot_tokens_vs_success(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot mean success by consumed token count."""

    working = _primary_frame(_as_frame(frame))
    x_values = _numeric_column(
        working,
        ("total_tokens", "tokens", "token_count"),
    )
    points = _rate_points(working, x_values)
    figure, axis = _new_figure()
    _draw_rate_points(
        axis,
        points,
        xlabel="Total tokens consumed",
        title="Success rate versus tokens consumed",
    )
    return _save(figure, _target(out_dir, PLOT_FILENAMES[6]))


def _condition_metric(
    frame: pd.DataFrame,
    names: Sequence[str],
) -> dict[str, tuple[float, int]]:
    output: dict[str, tuple[float, int]] = {}
    for condition in CONDITION_ORDER:
        values = _numeric_column(_condition_frame(frame, condition), names).dropna()
        if not values.empty:
            output[condition] = (float(values.mean()), int(values.size))
    return output


def _draw_metric_panel(
    axis: plt.Axes,
    metrics: Mapping[str, tuple[float, int]],
    *,
    title: str,
    ylabel: str,
    missing_message: str,
) -> None:
    positions = np.arange(len(CONDITION_ORDER), dtype=float)
    available_positions: list[float] = []
    available_values: list[float] = []
    available_labels: list[str] = []
    available_counts: list[int] = []
    for position, condition in zip(positions, CONDITION_ORDER, strict=True):
        metric = metrics.get(condition)
        if metric is None:
            continue
        available_positions.append(position)
        available_values.append(metric[0])
        available_labels.append(condition)
        available_counts.append(metric[1])
    if available_values:
        bars = axis.bar(
            available_positions,
            available_values,
            color=[CONDITION_COLORS[label] for label in available_labels],
            width=0.62,
        )
        for bar, value, count in zip(
            bars,
            available_values,
            available_counts,
            strict=True,
        ):
            axis.annotate(
                f"mean={value:.4g}\n(n={count})",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xticks(positions, CONDITION_ORDER)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    else:
        _no_data(axis, missing_message, title)


def plot_efficiency(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot wall-clock latency and estimated cost in separate panels."""

    working = _primary_frame(_as_frame(frame))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        dpi=DPI,
        facecolor="white",
    )
    latency_axis, cost_axis = axes
    _style_axis(latency_axis)
    _style_axis(cost_axis)
    _draw_metric_panel(
        latency_axis,
        _condition_metric(working, ("wall_ms", "latency_ms")),
        title="Latency",
        ylabel="Mean wall-clock latency (ms)",
        missing_message="No latency data",
    )
    _draw_metric_panel(
        cost_axis,
        _condition_metric(working, ("estimated_cost_usd", "cost_usd")),
        title="Estimated cost",
        ylabel="Mean estimated cost (USD)",
        missing_message="No cost data",
    )
    figure.suptitle("Latency and estimated cost by condition")
    return _save(figure, _target(out_dir, PLOT_FILENAMES[7]))


def _failure_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    working = _primary_frame(frame)
    counts: dict[str, dict[str, int]] = {
        condition: {} for condition in CONDITION_ORDER
    }
    if "failure_class" in working.columns:
        condition_values = (
            working["condition"].map(_text)
            if "condition" in working.columns
            else pd.Series("all", index=working.index)
        )
        for index in working.index:
            label = _text(working.loc[index, "failure_class"])
            if not label:
                continue
            condition = _text(condition_values.loc[index])
            if condition not in counts:
                continue
            count = _number(working.loc[index, "count"]) if "count" in working.columns else 1.0
            counts[condition][label] = counts[condition].get(label, 0) + int(count or 0)
        return counts

    for index in working.index:
        row = working.loc[index].to_dict()
        condition = _text(row.get("condition"))
        if condition not in counts:
            continue
        for failure in classify_failure(row):
            label = failure.value
            counts[condition][label] = counts[condition].get(label, 0) + 1
    return counts


def _failure_label(value: str) -> str:
    return value.replace("_", " ").title()


def plot_failure_modes(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot deterministic failure-mode counts for the primary conditions."""

    counts = _failure_counts(_as_frame(frame))
    labels = [
        failure.value
        for failure in FailureClass
        if any(failure.value in counts[condition] for condition in CONDITION_ORDER)
    ]
    figure, axis = _new_figure()
    if not labels:
        _no_data(axis, "No failure data", "Failure-mode distribution")
    else:
        positions = np.arange(len(labels), dtype=float)
        width = 0.8 / len(CONDITION_ORDER)
        for offset, condition in enumerate(CONDITION_ORDER):
            values = [counts[condition].get(label, 0) for label in labels]
            bars = axis.bar(
                positions + (offset - 0.5) * width,
                values,
                width=width,
                color=CONDITION_COLORS[condition],
                label=condition,
            )
            for bar, value in zip(bars, values, strict=True):
                if value:
                    axis.annotate(
                        str(value),
                        (bar.get_x() + bar.get_width() / 2, value),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        axis.set_xticks(positions, [_failure_label(label) for label in labels], rotation=35)
        axis.set_ylabel("Failure count")
        axis.set_title("Failure-mode distribution")
        axis.legend(frameon=False)
    return _save(figure, _target(out_dir, PLOT_FILENAMES[8]))


def _paired_outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
    outcomes = ("both_succeed", "snippets_only", "vfs_only", "both_fail")
    if {"outcome", "count"}.issubset(frame.columns):
        counts = {outcome: 0 for outcome in outcomes}
        for index in frame.index:
            outcome = _text(frame.loc[index, "outcome"])
            if outcome not in counts:
                continue
            value = _number(frame.loc[index, "count"])
            counts[outcome] = int(value or 0)
        return counts

    working = _primary_frame(frame)
    if not {"task_id", "condition"}.issubset(working.columns):
        return {outcome: 0 for outcome in outcomes}
    task_rows: dict[str, dict[str, float]] = {}
    success = _success_column(working)
    conditions = working["condition"].map(_text)
    for index in working.index:
        task_id = str(working.loc[index, "task_id"])
        condition = conditions.loc[index]
        value = _number(success.loc[index])
        if condition not in CONDITION_ORDER or value is None:
            continue
        task_rows.setdefault(task_id, {}).setdefault(condition, value)

    counts = {outcome: 0 for outcome in outcomes}
    for pair in task_rows.values():
        if set(pair) != set(CONDITION_ORDER):
            continue
        snippets = bool(pair["snippets"])
        vfs = bool(pair["vfs"])
        if snippets and vfs:
            counts["both_succeed"] += 1
        elif snippets:
            counts["snippets_only"] += 1
        elif vfs:
            counts["vfs_only"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def plot_paired_outcomes(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> Path:
    """Plot the four complete paired task outcomes."""

    counts = _paired_outcome_counts(_as_frame(frame))
    labels = ("both_succeed", "snippets_only", "vfs_only", "both_fail")
    figure, axis = _new_figure()
    if sum(counts.values()) == 0:
        _no_data(axis, "No paired outcome data", "Paired task outcomes")
    else:
        positions = np.arange(len(labels), dtype=float)
        bars = axis.bar(
            positions,
            [counts[label] for label in labels],
            color=["#70AD47", "#4472C4", "#C44E52", "#A5A5A5"],
            width=0.62,
        )
        total = sum(counts.values())
        for bar, label in zip(bars, labels, strict=True):
            count = counts[label]
            rate = count / total
            axis.annotate(
                f"{count}\n({rate:.1%})",
                (bar.get_x() + bar.get_width() / 2, count),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xticks(positions, [_failure_label(label) for label in labels], rotation=20)
        axis.set_ylabel("Task count")
        axis.set_title("Paired task outcomes")
    return _save(figure, _target(out_dir, PLOT_FILENAMES[9]))


def produce_all_plots(
    frame: pd.DataFrame | Iterable[object] | Mapping[str, object],
    out_dir: Path | str,
) -> list[Path]:
    """Produce all ten required PNGs in the frozen order."""

    working = _as_frame(frame)
    producers = (
        plot_success_by_condition,
        plot_success_by_hop,
        plot_success_by_actual_calls,
        plot_intervention_dose_response,
        plot_initial_recall_vs_success,
        plot_evidence_recall_vs_success,
        plot_tokens_vs_success,
        plot_efficiency,
        plot_failure_modes,
        plot_paired_outcomes,
    )
    return [producer(working, out_dir) for producer in producers]


__all__ = [
    "DPI",
    "FIGSIZE",
    "PLOT_FILENAMES",
    "plot_evidence_recall_vs_success",
    "plot_efficiency",
    "plot_failure_modes",
    "plot_initial_recall_vs_success",
    "plot_intervention_dose_response",
    "plot_paired_outcomes",
    "plot_success_by_actual_calls",
    "plot_success_by_condition",
    "plot_success_by_hop",
    "plot_tokens_vs_success",
    "produce_all_plots",
]
