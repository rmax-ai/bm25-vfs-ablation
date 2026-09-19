import runpy
import sys
import types
from enum import Enum

from bm25_vfs_ablation import (
    Condition,
    FailureClass,
    TerminationReason,
    ToolName,
    __version__,
)


def test_version_is_frozen() -> None:
    assert __version__ == "0.1.0"


def test_enum_literals_are_exact() -> None:
    expected_values = {
        Condition: ("snippets", "vfs"),
        ToolName: ("grep", "read", "cat", "list"),
        TerminationReason: (
            "answered",
            "budget_exhausted",
            "max_tool_calls",
            "invalid_call_limit",
            "model_error",
            "malformed_response",
            "skipped_resume",
        ),
        FailureClass: (
            "initial_retrieval_miss",
            "failed_exploration",
            "incomplete_evidence_coverage",
            "incorrect_evidence_composition",
            "unsupported_answer",
            "correct_evidence_wrong_conclusion",
            "budget_exhaustion",
            "excessive_repeated_tool_use",
            "invalid_tool_call",
            "malformed_final_answer",
        ),
    }

    for enum_type, values in expected_values.items():
        assert issubclass(enum_type, str)
        assert issubclass(enum_type, Enum)
        assert [member.value for member in enum_type] == list(values)
        assert len(values) == len(set(values))


def test_module_entrypoint_invokes_app(monkeypatch) -> None:
    calls: list[str] = []
    cli_module = types.ModuleType("bm25_vfs_ablation.cli")

    def fake_app() -> None:
        calls.append("called")

    cli_module.app = fake_app
    monkeypatch.setitem(sys.modules, "bm25_vfs_ablation.cli", cli_module)

    runpy.run_module("bm25_vfs_ablation", run_name="__main__")

    assert calls == ["called"]
