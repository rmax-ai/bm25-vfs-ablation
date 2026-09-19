"""Shared package metadata and enums for the BM25/VFS ablation."""

from enum import StrEnum

__version__ = "0.1.0"


class Condition(StrEnum):
    """Primary experimental conditions."""

    SNIPPETS = "snippets"
    VFS = "vfs"


class ToolName(StrEnum):
    """Read-only tools exposed by the VFS condition."""

    GREP = "grep"
    READ = "read"
    CAT = "cat"
    LIST = "list"


class TerminationReason(StrEnum):
    """Reasons a condition stops processing a task."""

    ANSWERED = "answered"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_TOOL_CALLS = "max_tool_calls"
    INVALID_CALL_LIMIT = "invalid_call_limit"
    MODEL_ERROR = "model_error"
    MALFORMED_RESPONSE = "malformed_response"
    SKIPPED_RESUME = "skipped_resume"


class FailureClass(StrEnum):
    """Deterministic failure classifications."""

    INITIAL_RETRIEVAL_MISS = "initial_retrieval_miss"
    FAILED_EXPLORATION = "failed_exploration"
    INCOMPLETE_EVIDENCE_COVERAGE = "incomplete_evidence_coverage"
    INCORRECT_EVIDENCE_COMPOSITION = "incorrect_evidence_composition"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    CORRECT_EVIDENCE_WRONG_CONCLUSION = "correct_evidence_wrong_conclusion"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    EXCESSIVE_REPEATED_TOOL_USE = "excessive_repeated_tool_use"
    INVALID_TOOL_CALL = "invalid_tool_call"
    MALFORMED_FINAL_ANSWER = "malformed_final_answer"


__all__ = [
    "Condition",
    "FailureClass",
    "TerminationReason",
    "ToolName",
    "__version__",
]

