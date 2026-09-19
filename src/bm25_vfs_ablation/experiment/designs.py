"""Deterministic expansion of experimental design cells."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bm25_vfs_ablation import Condition
from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.experiment.artifacts import canonical_json
from bm25_vfs_ablation.experiment.schemas import DesignCell

_EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
_INTERVENTION_CELLS = {
    0: DesignCell.MAX_CALLS_0,
    1: DesignCell.MAX_CALLS_1,
    2: DesignCell.MAX_CALLS_2,
    4: DesignCell.MAX_CALLS_4,
    8: DesignCell.MAX_CALLS_8,
    12: DesignCell.MAX_CALLS_12,
}
_PRIMARY_CONDITIONS = (Condition.SNIPPETS, Condition.VFS)


def condition_order(seed: int, task_id: str) -> tuple[Condition, Condition]:
    """Return the reproducible within-task condition order.

    The low bit of the SHA-256 integer derived from ``(seed, task_id)`` chooses
    between the two possible condition tuples.  The same order is reused for
    every design cell for a task so intervention rows remain paired.
    """

    payload = f"{seed}:{task_id}".encode()
    digest = hashlib.sha256(payload).digest()
    low_bit = int.from_bytes(digest, byteorder="big") & 1
    if low_bit:
        return _PRIMARY_CONDITIONS[::-1]
    return _PRIMARY_CONDITIONS


@dataclass(frozen=True, slots=True)
class DesignAssignment:
    """One task-condition assignment in a frozen design cell."""

    experiment_id: str
    task_id: str
    design_cell: DesignCell
    condition: Condition
    condition_order: int
    max_tool_calls: int | None
    oracle_mode: bool

    @property
    def run_key(self) -> str:
        """Return the stable run identity for this assignment."""

        return (
            f"{self.experiment_id}:{self.task_id}:{self.condition.value}:{self.design_cell.value}"
        )

    @property
    def max_tool_calls_permitted(self) -> int | None:
        """Return the effective VFS call limit used by the runner."""

        return self.max_tool_calls

    @property
    def cell(self) -> DesignCell:
        """Compatibility alias for callers that refer to a design cell as ``cell``."""

        return self.design_cell


def _config_payload(config: AppConfig) -> dict[str, Any]:
    if hasattr(config, "canonical_dict"):
        return config.canonical_dict()
    return config.model_dump(mode="json")


def _experiment_id(config: AppConfig) -> str:
    configured = config.experiment.experiment_id
    if configured is not None:
        if _EXPERIMENT_ID.fullmatch(configured) is None:
            raise ValueError("experiment_id has an invalid grammar")
        return configured

    config_hash = hashlib.sha256(
        canonical_json(_config_payload(config)).encode("utf-8")
    ).hexdigest()
    return f"exp-{config_hash[:12]}-{config.experiment.seed}"


def _design_cells(config: AppConfig) -> tuple[DesignCell, ...]:
    cells = [DesignCell.PRIMARY]
    if config.intervention.enabled:
        for limit in config.intervention.max_tool_calls:
            try:
                cells.append(_INTERVENTION_CELLS[limit])
            except KeyError as error:
                raise ValueError(
                    "intervention.max_tool_calls must use the frozen grid [0, 1, 2, 4, 8, 12]"
                ) from error
    if config.experiment.oracle_mode:
        cells.append(DesignCell.ORACLE)
    return tuple(cells)


def _assignment_limit(config: AppConfig, cell: DesignCell, condition: Condition) -> int | None:
    if condition is Condition.SNIPPETS:
        return None
    if cell is DesignCell.PRIMARY or cell is DesignCell.ORACLE:
        return config.harness.max_tool_calls
    return int(cell.value.removeprefix("max_calls_"))


def expand_design(
    config: AppConfig,
    tasks: Sequence[TaskRecord],
) -> list[DesignAssignment]:
    """Expand tasks into sorted, paired primary/intervention/oracle assignments."""

    task_list = list(tasks)
    task_ids = [task.task_id for task in task_list]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("tasks must have unique task_id values")

    experiment_id = _experiment_id(config)
    cells = _design_cells(config)
    assignments: list[DesignAssignment] = []

    for task in sorted(task_list, key=lambda item: item.task_id):
        ordered_conditions = condition_order(config.experiment.seed, task.task_id)
        for cell in cells:
            for order, condition in enumerate(ordered_conditions):
                assignments.append(
                    DesignAssignment(
                        experiment_id=experiment_id,
                        task_id=task.task_id,
                        design_cell=cell,
                        condition=condition,
                        condition_order=order,
                        max_tool_calls=_assignment_limit(config, cell, condition),
                        oracle_mode=cell is DesignCell.ORACLE,
                    )
                )

    assignments.sort(key=lambda item: (item.task_id, item.design_cell.value, item.condition_order))
    run_keys = [assignment.run_key for assignment in assignments]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("design assignments must have unique run keys")
    return assignments


__all__ = [
    "DesignAssignment",
    "DesignCell",
    "condition_order",
    "expand_design",
]
