"""Shared contracts used by the two experiment harnesses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.corpus.loader import CorpusBundle
from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.experiment.schemas import (
    ErrorRecord,
    LatencyBreakdown,
    TerminationReason,
    TokenAccounting,
    ToolTraceEntry,
)
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.models.cache import FileResponseCache
from bm25_vfs_ablation.models.client import ModelClient
from bm25_vfs_ablation.retrieval.bm25 import BM25Index, SearchHit

COMMON_SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. Use the available evidence "
    "to answer the user's question. Follow the requested response format exactly."
)


def build_common_system_prompt() -> str:
    """Return the condition-independent system message."""

    return COMMON_SYSTEM_PROMPT


def build_answer_instruction() -> str:
    """Return the shared, structured answer-format instruction."""

    return (
        'Return exactly one JSON object with these keys: "answer" (a string), '
        '"citations" (an array of stable document or chunk IDs), and '
        '"justification" (a short string of at most 240 characters). '
        "Do not include chain-of-thought or hidden reasoning."
    )


@dataclass(frozen=True, slots=True)
class HarnessContext:
    """The immutable shared inputs supplied to a harness run."""

    bundle: CorpusBundle
    index: BM25Index
    model: ModelClient
    cache: FileResponseCache | None
    config: AppConfig
    budget_factory: Callable[[], BudgetController]

    def new_budget(self) -> BudgetController:
        """Create the per-task budget controller."""

        return self.budget_factory()


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """Observable output of one harness execution."""

    raw_answer: str | None = None
    initial_hits: tuple[SearchHit, ...] = ()
    trace: tuple[ToolTraceEntry, ...] = ()
    accounting: TokenAccounting | None = None
    latencies: LatencyBreakdown = field(
        default_factory=lambda: LatencyBreakdown(
            wall_ms=0.0,
            model_ms=0.0,
            retrieval_ms=0.0,
            tool_ms=0.0,
        )
    )
    termination: TerminationReason | None = None
    errors: tuple[ErrorRecord, ...] = ()

    @property
    def tool_trace(self) -> tuple[ToolTraceEntry, ...]:
        """Compatibility name for callers describing the trace explicitly."""

        return self.trace

    @property
    def latency(self) -> LatencyBreakdown:
        """Compatibility singular alias for the latency breakdown."""

        return self.latencies

    @property
    def termination_reason(self) -> TerminationReason | None:
        """Compatibility name matching the persisted run-record field."""

        return self.termination


class Harness(Protocol):
    """Minimal interface implemented by each primary condition."""

    def run(self, task: TaskRecord, context: HarnessContext) -> HarnessResult:
        """Run one task with the shared context."""

        ...


__all__ = [
    "Harness",
    "HarnessContext",
    "HarnessResult",
    "COMMON_SYSTEM_PROMPT",
    "build_answer_instruction",
    "build_common_system_prompt",
]
