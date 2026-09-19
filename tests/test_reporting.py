from __future__ import annotations

import hashlib
from pathlib import Path

from bm25_vfs_ablation.evaluation.plots import PLOT_FILENAMES
from bm25_vfs_ablation.experiment.reporting import generate_report


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records() -> list[dict[str, object]]:
    return [
        {
            "task_id": "task-eval-000001",
            "condition": "snippets",
            "design_cell": "primary",
            "corpus_version": "synthetic-v1",
            "config_sha256": _digest("config"),
            "corpus_sha256": _digest("corpus"),
            "secondary_judge": {"enabled": False},
            "token_accounting": {"limit": 4096},
        },
        {
            "task_id": "task-eval-000001",
            "condition": "vfs",
            "design_cell": "primary",
            "corpus_version": "synthetic-v1",
            "config_sha256": _digest("config"),
            "corpus_sha256": _digest("corpus"),
            "secondary_judge": {"enabled": False},
            "token_accounting": {"limit": 4096},
        },
    ]


def _write_csv(directory: Path, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8", newline="\n")


def _write_base_tables(directory: Path, *, include_intervention: bool = True) -> None:
    _write_csv(
        directory,
        "paired_outcomes.csv",
        (
            "outcome,count,n,rate,snippets_success_rate,vfs_success_rate,difference,"
            "ci_lower,ci_upper,mcnemar_p_value,status\n"
            "both_succeed,1,2,0.5,0.5,0.5,0.0,-0.5,0.5,1.0,ok\n"
        ),
    )
    _write_csv(
        directory,
        "condition_summary.csv",
        (
            "condition,design_cell,n,successes,success_rate,initial_chunk_recall_mean,"
            "initial_document_recall_mean,final_evidence_recall_mean,evidence_precision_mean,"
            "actual_tool_calls_mean,successful_tool_calls_mean,invalid_tool_calls_mean,"
            "repeated_tool_calls_mean,total_tokens_mean,wall_ms_mean,mean_cost_usd,"
            "success_per_1000_tokens,status\n"
            "snippets,primary,2,1,0.5,0.5,1.0,0.5,1.0,0,0,0,0,100,10,0.001,5,ok\n"
            "vfs,primary,2,1,0.5,0.5,1.0,1.0,1.0,2,2,0,0,120,15,0.002,4.167,ok\n"
        ),
    )
    _write_csv(
        directory,
        "subgroup_summary.csv",
        "subgroup,level,condition,design_cell,status\nhop_count,2,snippets,primary,ok\n",
    )
    _write_csv(
        directory,
        "within_vfs_regression.csv",
        (
            "term,coefficient,std_error,ci_lower,ci_upper,p_value,converged,error,"
            "associative,covariance_type,n,reference_category,status\n"
            "calls,0.1,0.2,-0.3,0.5,0.6,True,,True,HC3,2,ops,coefficient\n"
            "__status__,,,,,,, ,True,HC3,2,ops,converged\n"
        ),
    )
    _write_csv(
        directory,
        "efficiency_summary.csv",
        (
            "condition,budget_bucket,total_tokens_mean,wall_ms_mean,mean_cost_usd,"
            "success_per_1000_tokens,status\n"
            "snippets,75-100%,100,10,0.001,5,ok\n"
        ),
    )
    _write_csv(
        directory,
        "task_level.csv",
        "design_cell,task_id,status\nprimary,task-eval-000001,ok\n",
    )
    _write_csv(
        directory,
        "stratified_summary.csv",
        "stratum,band,n,status\nhop,2,1,ok\n",
    )
    _write_csv(
        directory,
        "failure_summary.csv",
        "condition,design_cell,failure_class,count,n,rate,status\nvfs,primary,budget_exhaustion,1,2,0.5,ok\n",
    )
    (directory / "trace_examples.json").write_text(
        '{"vfs_win":{"answers":{"correctness":true}}}\n',
        encoding="utf-8",
        newline="\n",
    )
    if include_intervention:
        _write_csv(
            directory,
            "intervention_summary.csv",
            (
                "condition,permitted_call_limit,design_cell,n,success_rate,status\n"
                "vfs,2,max_calls_2,2,0.5,ok\n"
            ),
        )


def _report(tmp_path: Path, *, include_intervention: bool = True) -> str:
    aggregates = tmp_path / "aggregates"
    plots = tmp_path / "plots"
    aggregates.mkdir()
    plots.mkdir()
    _write_base_tables(aggregates, include_intervention=include_intervention)
    return generate_report(_records(), aggregates, plots)


def test_report_has_all_required_sections(tmp_path: Path) -> None:
    report = _report(tmp_path)
    headings = {
        line.removeprefix("## ").strip().casefold()
        for line in report.splitlines()
        if line.startswith("## ")
    }
    assert headings == {
        "executive summary",
        "hypothesis/design",
        "controls",
        "dataset",
        "primary",
        "retrieval/evidence",
        "tool use",
        "intervention",
        "efficiency",
        "failures",
        "threats",
        "conclusions",
    }


def test_report_has_all_validity_threats(tmp_path: Path) -> None:
    report = _report(tmp_path).casefold()
    for threat in (
        "synthetic-task realism",
        "prompt sensitivity",
        "model/provider drift",
        "tokenizer/accounting error",
        "vfs interface quality",
        "different prompt overhead between conditions",
        "tool-call count as a post-treatment variable",
        "repeated observations over shared templates",
        "judge-model bias, if an llm judge is used",
    ):
        assert threat in report


def test_report_labels_claim_types(tmp_path: Path) -> None:
    report = _report(tmp_path)
    for tag in ("[OBSERVED]", "[INFERENCE]", "[CAUSAL-INTERVENTION]", "[SPECULATION]"):
        assert tag in report


def test_report_never_calls_observed_calls_causal(tmp_path: Path) -> None:
    report = _report(tmp_path).casefold()
    assert "observed calls are post-treatment variables" in report
    assert "observed calls are causal" not in report
    assert "observed tool-call count causes" not in report
    assert "associative" in report


def test_report_handles_missing_intervention(tmp_path: Path) -> None:
    report = _report(tmp_path, include_intervention=False)
    intervention_section = report.split("## Intervention", maxsplit=1)[1]
    assert "Not run" in intervention_section
    assert "No intervention-based conclusion is made." in intervention_section
    assert "[CAUSAL-INTERVENTION]" not in intervention_section


def test_report_links_ten_plots(tmp_path: Path) -> None:
    report = _report(tmp_path)
    for filename in PLOT_FILENAMES:
        assert f"]({filename})" in report or f"/{filename})" in report


def test_report_contains_reproduction_commands(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert "uv run python -m bm25_vfs_ablation generate --tasks 200 --seed 42" in report
    assert "uv run python -m bm25_vfs_ablation run --config configs/default.yaml" in report
    assert "uv run python -m bm25_vfs_ablation evaluate --runs results/runs.jsonl" in report
    assert "uv run python -m bm25_vfs_ablation report --runs results/runs.jsonl" in report
