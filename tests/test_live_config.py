"""Operator contracts for the live-model experiment config (issue #49 gates)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bm25_vfs_ablation.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "default.yaml"
INTERVENTION_CONFIG = REPOSITORY_ROOT / "configs" / "tool_call_intervention.yaml"
LIVE_CONFIG = REPOSITORY_ROOT / "configs" / "live.yaml"

ENVIRONMENT_NAMES = ("BM25_VFS_BASE_URL", "BM25_VFS_API_KEY", "BM25_VFS_MODEL")


def test_live_config_operator_gates() -> None:
    config = load_config(LIVE_CONFIG)

    # AIR-4: retries stay off for the first live run; billed failed attempts
    # would be invisible in records when retries are enabled.
    assert config.model.max_attempts == 1
    assert config.model.provider == "openai_compatible"
    # Primary cells only: no oracle, no tool-call intervention.
    assert config.experiment.oracle_mode is False
    assert config.intervention.enabled is False
    # 4096 cannot complete most 4-hop VFS tasks under honest schema charging
    # (AIR-1); 8192 keeps the hardest cells observable.
    assert config.experiment.token_ceiling == 8192
    # Interrupted runs must resume, never restart, within the same output path.
    assert config.experiment.resume is True


def test_live_config_is_fair_with_default() -> None:
    """Controlled variables must match the mock/default experiment exactly."""

    default = load_config(DEFAULT_CONFIG)
    live = load_config(LIVE_CONFIG)

    assert live.chunking == default.chunking
    assert live.retrieval == default.retrieval
    assert live.harness == default.harness
    assert live.corpus == default.corpus
    assert live.experiment.seed == default.experiment.seed
    assert live.experiment.conditions == default.experiment.conditions
    assert live.model.temperature == default.model.temperature
    assert live.model.top_p == default.model.top_p
    assert live.model.seed == default.model.seed
    assert live.model.tokenizer == default.model.tokenizer
    assert live.evaluation == default.evaluation
    # Deliberate deltas: provider mode, attempts, timeout, ceiling, output path.
    assert live.model.provider != default.model.provider
    assert live.experiment.token_ceiling != default.experiment.token_ceiling
    assert live.experiment.output_runs != default.experiment.output_runs


def test_live_config_fails_closed_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model fields stay empty/null in the committed config; env supplies them."""

    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)

    config = load_config(LIVE_CONFIG)

    assert config.model.api_key is None
    assert config.model.base_url == ""
    assert config.model.model == ""


def test_live_config_outputs_never_collide_with_mock_runs() -> None:
    default = load_config(DEFAULT_CONFIG)
    intervention = load_config(INTERVENTION_CONFIG)
    live = load_config(LIVE_CONFIG)

    assert live.experiment.output_runs == "results/live/runs.jsonl"
    assert live.experiment.output_runs != default.experiment.output_runs
    assert live.experiment.output_runs != intervention.experiment.output_runs


def test_live_config_runbook_documented() -> None:
    text = LIVE_CONFIG.read_text(encoding="utf-8")

    for fragment in (
        "BM25_VFS_BASE_URL",
        "BM25_VFS_API_KEY",
        "BM25_VFS_MODEL",
        "make generate",
        "results/live-pilot/runs.jsonl",
        "results/live/runs.jsonl",
        "--output-dir results/live/aggregates",
        "max_attempts: 1",
        "issue #49",
    ):
        assert fragment in text, f"live.yaml runbook is missing: {fragment}"
