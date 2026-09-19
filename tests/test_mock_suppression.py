"""Operator-authored tests: §8 mock-policy suppression in generated reports (AIR-1 fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bm25_vfs_ablation.experiment.reporting import generate_report

_PAIRED_HEADER = (
    "outcome,count,n,rate,snippets_successes,vfs_successes,snippets_success_rate,"
    "vfs_success_rate,difference,ci_lower,ci_upper,snippets_only,vfs_only,"
    "mcnemar_p_value,status"
)
_PAIRED_ROW = "both_succeed,8,8,1.0,8,8,1.0,1.0,0.0,0.0,0.0,0,0,1.0,ok"


def _record(*, mock: bool, task_id: str = "task-eval-000001") -> dict[str, object]:
    return {
        "run_key": f"exp:test:{task_id}:vfs:primary",
        "task_id": task_id,
        "condition": "vfs",
        "scoring_details": {"mock_oracle_policy": True} if mock else {},
        "secondary_judge": {"enabled": False},
    }


def _report(tmp_path: Path, records: list[dict[str, object]]) -> str:
    aggregates = tmp_path / "aggregates"
    plots = tmp_path / "plots"
    aggregates.mkdir()
    plots.mkdir()
    (aggregates / "paired_outcomes.csv").write_text(
        f"{_PAIRED_HEADER}\n{_PAIRED_ROW}\n", encoding="utf-8", newline="\n"
    )
    return generate_report(records, aggregates, plots, report_dir=tmp_path)


def test_all_mock_report_is_labelled_device_check(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        [_record(mock=True), _record(mock=True, task_id="task-eval-000002")],
    )

    assert "Mock device-check" in report
    assert "mock_oracle_policy=true" in report
    assert "not a scientific result" in report
    # the headline itself must carry the qualifier, not just a distant note
    headline = next(
        line for line in report.splitlines() if "primary paired outcome table reports" in line
    )
    assert "Mock device-check" in headline


def test_live_report_has_no_mock_notice(tmp_path: Path) -> None:
    report = _report(tmp_path, [_record(mock=False)])

    assert "Mock device-check" not in report
    assert "Mock/live contamination" not in report


def test_mixed_records_fail_report_generation(tmp_path: Path) -> None:
    report_dir = tmp_path / "aggregates"
    plots = tmp_path / "plots"
    report_dir.mkdir()
    plots.mkdir()
    records = [_record(mock=True), _record(mock=False, task_id="task-eval-000002")]

    with pytest.raises(ValueError, match="mixes mock-stamped"):
        generate_report(records, report_dir, plots, report_dir=tmp_path)
