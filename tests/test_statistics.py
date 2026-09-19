"""Acceptance tests for paired inference and associative VFS analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bm25_vfs_ablation.evaluation.statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    exact_mcnemar,
    paired_summary,
    stratified_summaries,
    within_vfs_regression,
)


def _paired_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = {
        "task-000001": (False, True, 10, 12, 0.0, 2, "alpha"),
        "task-000002": (True, False, 20, 22, 0.5, 2, "alpha"),
        "task-000003": (True, True, 30, 30, 1.0, 3, "beta"),
        "task-000004": (False, False, 40, 41, 0.5, 3, "beta"),
    }
    for task_id, (
        snippets,
        vfs,
        snippets_tokens,
        vfs_tokens,
        recall,
        hops,
        category,
    ) in outcomes.items():
        common = {
            "task_id": task_id,
            "design_cell": "primary",
            "initial_recall": recall,
            "hop_count": hops,
            "category": category,
            "distractor_count": 2,
        }
        rows.append(
            common
            | {
                "condition": "snippets",
                "correctness": snippets,
                "total_tokens": snippets_tokens,
                "tool_call_count": 0,
            }
        )
        rows.append(
            common
            | {
                "condition": "vfs",
                "correctness": vfs,
                "total_tokens": vfs_tokens,
                "tool_call_count": hops,
            }
        )
    return pd.DataFrame(rows)


def test_paired_difference_and_percentile_ci_known_seed() -> None:
    frame = _paired_frame()
    result = paired_summary(frame)

    differences = np.asarray([1.0, -1.0, 0.0, 0.0])
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(differences),
        size=(DEFAULT_BOOTSTRAP_RESAMPLES, len(differences)),
    )
    expected_ci = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])

    assert result.n == 4
    assert result.difference == pytest.approx(0.0, abs=1e-12)
    assert result.ci_lower == pytest.approx(expected_ci[0], abs=1e-12)
    assert result.ci_upper == pytest.approx(expected_ci[1], abs=1e-12)
    assert result.mcnemar_p_value == pytest.approx(1.0, abs=1e-12)
    assert result.resamples == 10_000


def test_exact_mcnemar_known_table() -> None:
    assert exact_mcnemar(3, 1) == pytest.approx(0.625, abs=1e-12)


def test_mcnemar_no_discordance() -> None:
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar([True, False], [True, False]) == 1.0


def test_unpaired_rows_rejected() -> None:
    frame = _paired_frame().query("task_id != 'task-000004'").copy()
    frame = frame[frame["condition"] != "vfs"].reset_index(drop=True)

    with pytest.raises(ValueError, match="paired condition"):
        paired_summary(frame)


def test_stratified_pairing() -> None:
    frame = _paired_frame()
    frame.loc[frame["task_id"] != "task-000004", "total_tokens"] = [
        10,
        10,
        20,
        20,
        30,
        30,
    ]
    frame.loc[frame["task_id"] == "task-000004", "total_tokens"] = [40, 25]
    result = stratified_summaries(frame)

    token = result[result["stratum"] == "token_quartile"]
    assert token["n"].sum() == 3
    assert token.loc[token["band"] == "q1", "n"].item() == 1
    assert pd.isna(token.loc[token["band"] == "q1", "missing_reason"].item())
    assert result[result["stratum"] == "recall_band"].set_index("band").loc["0", "n"] == 1
    assert result["stratum"].tolist() == [
        "token_quartile",
        "token_quartile",
        "token_quartile",
        "token_quartile",
        "recall_band",
        "recall_band",
        "recall_band",
        "hop",
        "hop",
        "category",
        "category",
    ]


def test_within_vfs_regression_columns_and_hc3() -> None:
    frame = _paired_frame()
    rows = []
    for index in range(40):
        row = frame.iloc[index % len(frame)].to_dict()
        row.update(
            {
                "task_id": f"task-reg-{index:06d}",
                "condition": "vfs",
                "correctness": (index * 7 + index // 3) % 5 < 2,
                "total_tokens": 20 + index,
                "tool_call_count": index % 6,
                "initial_recall": ((index * 2) % 11) / 10,
                "hop_count": 2 + (index % 3),
                "category": "alpha" if index % 2 else "beta",
                "distractor_count": (index * 5) % 7,
            }
        )
        rows.append(row)

    result = within_vfs_regression(pd.DataFrame(rows))

    assert {
        "intercept",
        "calls",
        "calls_squared",
        "initial_recall",
        "hops",
        "tokens",
        "distractors",
        "category_beta",
    }.issubset(set(result["term"]))
    assert set(result["covariance_type"]) == {"HC3"}
    assert result["associative"].all()
    assert result.iloc[-1]["status"] == "converged"
    assert result.iloc[-1]["converged"]


def test_regression_failure_is_reported() -> None:
    frame = _paired_frame()
    frame = frame[frame["condition"] == "vfs"].copy()
    frame["correctness"] = True

    result = within_vfs_regression(frame)

    assert len(result) == 1
    assert result.iloc[0]["term"] == "__status__"
    assert result.iloc[0]["status"] == "error"
    assert result.iloc[0]["error"] == "binomial outcome requires both success classes"
    assert result.iloc[0]["associative"]
