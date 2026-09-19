from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from bm25_vfs_ablation.evaluation import plots


def _frame(*, include_intervention: bool = True) -> pd.DataFrame:
    rows = [
        {
            "task_id": "task-eval-000001",
            "condition": "snippets",
            "design_cell": "primary",
            "correctness": True,
            "hop_count": 2,
            "initial_recall": 0.5,
            "final_recall": 0.5,
            "tool_call_count": 0,
            "max_tool_calls_permitted": None,
            "total_tokens": 100,
            "wall_ms": 10.0,
            "estimated_cost_usd": 0.001,
            "category": "ops",
        },
        {
            "task_id": "task-eval-000001",
            "condition": "vfs",
            "design_cell": "primary",
            "correctness": False,
            "hop_count": 2,
            "initial_recall": 0.5,
            "final_recall": 1.0,
            "tool_call_count": 1,
            "max_tool_calls_permitted": 8,
            "total_tokens": 120,
            "wall_ms": 15.0,
            "estimated_cost_usd": 0.002,
            "category": "ops",
        },
        {
            "task_id": "task-eval-000002",
            "condition": "snippets",
            "design_cell": "primary",
            "correctness": False,
            "hop_count": 3,
            "initial_recall": 1.0,
            "final_recall": 0.0,
            "tool_call_count": 0,
            "max_tool_calls_permitted": None,
            "total_tokens": 110,
            "wall_ms": 11.0,
            "estimated_cost_usd": 0.0011,
            "category": "security",
        },
        {
            "task_id": "task-eval-000002",
            "condition": "vfs",
            "design_cell": "primary",
            "correctness": True,
            "hop_count": 3,
            "initial_recall": 1.0,
            "final_recall": 1.0,
            "tool_call_count": 3,
            "max_tool_calls_permitted": 8,
            "total_tokens": 140,
            "wall_ms": 18.0,
            "estimated_cost_usd": 0.0025,
            "category": "security",
        },
    ]
    if include_intervention:
        rows.extend(
            [
                {
                    "task_id": "task-eval-000001",
                    "condition": "vfs",
                    "design_cell": "max_calls_0",
                    "correctness": False,
                    "hop_count": 2,
                    "initial_recall": 0.5,
                    "final_recall": 0.0,
                    "tool_call_count": 0,
                    "max_tool_calls_permitted": 0,
                    "total_tokens": 90,
                    "wall_ms": 12.0,
                    "estimated_cost_usd": 0.0015,
                    "category": "ops",
                },
                {
                    "task_id": "task-eval-000002",
                    "condition": "vfs",
                    "design_cell": "max_calls_2",
                    "correctness": True,
                    "hop_count": 3,
                    "initial_recall": 1.0,
                    "final_recall": 1.0,
                    "tool_call_count": 2,
                    "max_tool_calls_permitted": 2,
                    "total_tokens": 130,
                    "wall_ms": 17.0,
                    "estimated_cost_usd": 0.0022,
                    "category": "security",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_produce_all_ten_filenames(tmp_path: Path) -> None:
    paths = plots.produce_all_plots(_frame(), tmp_path)

    assert [path.name for path in paths] == list(plots.PLOT_FILENAMES)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert all(plt.imread(path).shape[:2] == (600, 960) for path in paths)


def test_plot_backend_is_agg() -> None:
    assert matplotlib.get_backend().casefold() == "agg"


def test_success_plot_uses_paired_interval(tmp_path: Path, monkeypatch) -> None:
    calls: list[pd.DataFrame] = []
    figures = []

    def fake_paired_summary(frame: pd.DataFrame):
        calls.append(frame)
        return {
            "difference": 0.25,
            "ci_lower": -0.10,
            "ci_upper": 0.55,
        }

    def fake_save(figure, path):
        figures.append(figure)
        return path

    monkeypatch.setattr(plots, "paired_summary", fake_paired_summary)
    monkeypatch.setattr(plots, "_save", fake_save)
    path = plots.plot_success_by_condition(_frame(include_intervention=False), tmp_path)

    assert len(calls) == 1
    assert len(figures) == 1
    assert "Paired VFS - snippets" in "\n".join(
        text.get_text() for text in figures[0].axes[0].texts
    )
    plt.close(figures[0])
    assert not path.exists()


def test_actual_and_permitted_call_axes_differ(tmp_path: Path, monkeypatch) -> None:
    labels: list[str] = []

    def capture_draw(axis, points, *, xlabel, title, xlim=None):
        del axis, points, title, xlim
        labels.append(xlabel)

    monkeypatch.setattr(plots, "_draw_rate_points", capture_draw)
    plots.plot_success_by_actual_calls(_frame(), tmp_path)
    plots.plot_intervention_dose_response(_frame(), tmp_path)

    assert labels == [
        "Actual VFS tool calls (observed)",
        "Maximum permitted VFS tool calls (assigned)",
    ]


def test_missing_intervention_plot_is_valid(tmp_path: Path) -> None:
    path = plots.plot_intervention_dose_response(
        _frame(include_intervention=False),
        tmp_path,
    )

    assert path.exists()
    assert plt.imread(path).shape[:2] == (600, 960)


def test_all_figures_are_closed(tmp_path: Path) -> None:
    plots.produce_all_plots(_frame(), tmp_path)

    assert plt.get_fignums() == []
