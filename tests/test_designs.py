from pathlib import Path

from bm25_vfs_ablation import Condition
from bm25_vfs_ablation.config import load_config
from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.experiment.designs import (
    DesignCell,
    condition_order,
    expand_design,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
INTERVENTION_CONFIG = ROOT / "configs" / "tool_call_intervention.yaml"


def _tasks() -> list[TaskRecord]:
    common = {
        "schema_version": 1,
        "split": "eval",
        "question": "Which control applies?",
        "canonical_answer": "approved",
        "acceptable_answer_variants": ["approved"],
        "required_fact_ids": ["fact-000001", "fact-000002"],
        "gold_document_ids": ["doc-policies-0001", "doc-services-0001"],
        "gold_chunk_ids": [
            "doc-policies-0001::c0000",
            "doc-services-0001::c0000",
        ],
        "hop_count": 2,
        "task_template": "policy-control",
        "category": "security",
        "distractor_document_ids": [],
        "distractor_count": 0,
        "generator_seed": 42,
        "corpus_version": "synthetic-v1",
    }
    return [
        TaskRecord(task_id="task-eval-000002", **common),
        TaskRecord(
            task_id="task-eval-000001",
            question="Which control applies first?",
            **{key: value for key, value in common.items() if key != "question"},
        ),
    ]


def test_primary_expands_two_conditions() -> None:
    assignments = expand_design(load_config(DEFAULT_CONFIG), _tasks())

    assert len(assignments) == 4
    assert {assignment.design_cell for assignment in assignments} == {DesignCell.PRIMARY}
    assert {(assignment.task_id, assignment.condition) for assignment in assignments} == {
        ("task-eval-000001", Condition.SNIPPETS),
        ("task-eval-000001", Condition.VFS),
        ("task-eval-000002", Condition.SNIPPETS),
        ("task-eval-000002", Condition.VFS),
    }


def test_intervention_expands_paired_cells() -> None:
    assignments = expand_design(load_config(INTERVENTION_CONFIG), _tasks())

    assert len(assignments) == 2 * (1 + 6) * 2
    for task_id in {"task-eval-000001", "task-eval-000002"}:
        task_assignments = [item for item in assignments if item.task_id == task_id]
        assert {item.design_cell for item in task_assignments} == {
            DesignCell.PRIMARY,
            DesignCell.MAX_CALLS_0,
            DesignCell.MAX_CALLS_1,
            DesignCell.MAX_CALLS_2,
            DesignCell.MAX_CALLS_4,
            DesignCell.MAX_CALLS_8,
            DesignCell.MAX_CALLS_12,
        }
        for item in task_assignments:
            if item.condition is Condition.SNIPPETS:
                assert item.max_tool_calls is None
            elif item.design_cell is DesignCell.PRIMARY:
                assert item.max_tool_calls == 8
            else:
                assert item.max_tool_calls == int(item.design_cell.value.removeprefix("max_calls_"))


def test_condition_order_is_seeded_and_stable() -> None:
    first = condition_order(42, "task-eval-000001")
    second = condition_order(42, "task-eval-000001")
    assignments = expand_design(load_config(INTERVENTION_CONFIG), _tasks())

    assert first == second
    assert set(first) == {Condition.SNIPPETS, Condition.VFS}
    for task_id in {"task-eval-000001", "task-eval-000002"}:
        for cell in {item.design_cell for item in assignments}:
            rows = [
                item for item in assignments if item.task_id == task_id and item.design_cell is cell
            ]
            assert tuple(item.condition for item in rows) == condition_order(42, task_id)
            assert [item.condition_order for item in rows] == [0, 1]


def test_oracle_cell_is_separate() -> None:
    config = load_config(
        DEFAULT_CONFIG,
        {"experiment": {"oracle_mode": True}},
    )
    assignments = expand_design(config, _tasks())

    assert {item.design_cell for item in assignments} == {
        DesignCell.PRIMARY,
        DesignCell.ORACLE,
    }
    assert all(item.oracle_mode is (item.design_cell is DesignCell.ORACLE) for item in assignments)


def test_design_run_keys_unique() -> None:
    config = load_config(
        INTERVENTION_CONFIG,
        {"experiment": {"oracle_mode": True}},
    )
    assignments = expand_design(config, _tasks())

    run_keys = [item.run_key for item in assignments]
    assert len(run_keys) == len(set(run_keys))
    assert all(
        run_key.endswith(f":{item.condition.value}:{item.design_cell.value}")
        for item, run_key in zip(assignments, run_keys, strict=True)
    )
