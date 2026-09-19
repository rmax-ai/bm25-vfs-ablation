from dataclasses import fields
from typing import cast

from bm25_vfs_ablation.experiment.schemas import LatencyBreakdown
from bm25_vfs_ablation.harnesses.base import (
    COMMON_SYSTEM_PROMPT,
    HarnessContext,
    HarnessResult,
    build_answer_instruction,
    build_common_system_prompt,
)


def test_answer_instruction_exact_schema() -> None:
    instruction = build_answer_instruction()
    assert '"answer"' in instruction
    assert '"citations"' in instruction
    assert '"justification"' in instruction
    assert "JSON object" in instruction
    assert "chain-of-thought" in instruction


def test_common_system_prompt_has_no_condition_language() -> None:
    prompt = build_common_system_prompt()
    assert prompt == COMMON_SYSTEM_PROMPT
    assert prompt == build_common_system_prompt()
    prompt = prompt.casefold()
    assert "snippet" not in prompt
    assert "vfs" not in prompt
    assert "tool" not in prompt


def test_harness_context_preserves_index_identity() -> None:
    index = object()
    context = HarnessContext(
        bundle=cast(object, object()),
        index=cast(object, index),
        model=cast(object, object()),
        cache=cast(object, object()),
        config=cast(object, object()),
        budget_factory=lambda: cast(object, object()),
    )
    assert context.index is index


def test_harness_result_defaults_are_complete() -> None:
    result = HarnessResult()
    assert result.raw_answer is None
    assert result.initial_hits == ()
    assert result.trace == ()
    assert result.accounting is None
    assert result.latencies == LatencyBreakdown(
        wall_ms=0.0,
        model_ms=0.0,
        retrieval_ms=0.0,
        tool_ms=0.0,
    )
    assert result.termination is None
    assert result.errors == ()
    assert {field.name for field in fields(result)} == {
        "raw_answer",
        "initial_hits",
        "trace",
        "accounting",
        "latencies",
        "termination",
        "errors",
    }
