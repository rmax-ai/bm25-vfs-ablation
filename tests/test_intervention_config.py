from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from bm25_vfs_ablation.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
INTERVENTION_CONFIG = ROOT / "configs" / "tool_call_intervention.yaml"


def test_shipped_intervention_diff_is_exact() -> None:
    default = load_config(DEFAULT_CONFIG).model_dump(mode="json")
    intervention = load_config(INTERVENTION_CONFIG).model_dump(mode="json")

    assert intervention["experiment"]["output_runs"] == "results/intervention_runs.jsonl"
    assert intervention["intervention"] == {
        "enabled": True,
        "max_tool_calls": [0, 1, 2, 4, 8, 12],
    }

    normalized_default = deepcopy(default)
    normalized_intervention = deepcopy(intervention)
    normalized_default["experiment"]["output_runs"] = normalized_intervention[
        "experiment"
    ]["output_runs"]
    normalized_default.pop("intervention")
    normalized_intervention.pop("intervention")

    assert normalized_intervention == normalized_default


@pytest.mark.parametrize(
    "grid",
    (
        [0, 0, 1],
        [-1, 0],
        [2, 1],
    ),
)
def test_intervention_grid_validation(grid: list[int]) -> None:
    with pytest.raises(ValidationError):
        load_config(
            DEFAULT_CONFIG,
            {"intervention": {"enabled": True, "max_tool_calls": grid}},
        )


def test_absent_intervention_defaults_disabled() -> None:
    intervention = load_config(DEFAULT_CONFIG).intervention

    assert intervention.enabled is False
    assert intervention.max_tool_calls == []
