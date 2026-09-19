"""Bounded, observable read-only tools over the virtual filesystem."""

from __future__ import annotations

import copy
import time
from typing import Any

from bm25_vfs_ablation import ToolName
from bm25_vfs_ablation.experiment.schemas import LineRange, ToolRequest, ToolResult
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.retrieval.bm25 import BM25Index, SearchHit
from bm25_vfs_ablation.retrieval.chunking import normalize_terms
from bm25_vfs_ablation.vfs.filesystem import ContentSlice, VirtualFilesystem
from bm25_vfs_ablation.vfs.security import VfsError, normalize_vfs_path

_TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search indexed corpus content and return bounded line previews.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nonblank search query.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional file or directory path filter.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of matching previews.",
                    },
                },
                "required": ["query", "max_results"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read an inclusive bounded line range from one file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root-relative POSIX file path.",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "One-based first line.",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "One-based last line.",
                    },
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cat",
            "description": "Read an entire file when it fits the result limits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root-relative POSIX file path.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list",
            "description": "List the shallow, sorted contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": "/",
                        "description": "Root-relative POSIX directory path.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
]


def tool_schema() -> list[dict[str, Any]]:
    """Return the canonical provider-neutral schema for all VFS tools."""

    return copy.deepcopy(_TOOL_SCHEMA)


def _canonical_path(raw: str) -> str:
    normalized = normalize_vfs_path(raw)
    if normalized.is_absolute():
        return normalized.as_posix()
    return f"/{normalized.as_posix()}"


def _require_keys(
    arguments: dict[str, Any],
    required: tuple[str, ...],
    allowed: frozenset[str],
    *,
    code: str,
    tool: str,
) -> None:
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise VfsError(code, f"{tool} received unsupported arguments")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise VfsError(code, f"{tool} is missing required arguments")


def _require_integer(value: Any, name: str, *, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VfsError(code, f"{name} must be an integer")
    return value


def _line_range(content: ContentSlice) -> LineRange:
    return LineRange(
        path=content.path,
        start_line=content.start_line,
        end_line=content.end_line,
    )


class VfsToolExecutor:
    """Execute bounded read-only VFS tools against one shared BM25 index."""

    __slots__ = ("_budget", "_index", "_per_call_limit", "_vfs")

    def __init__(
        self,
        vfs: VirtualFilesystem,
        index: BM25Index,
        budget: BudgetController,
        per_call_limit: int,
    ) -> None:
        if not isinstance(per_call_limit, int) or isinstance(per_call_limit, bool):
            raise TypeError("per_call_limit must be an integer")
        if per_call_limit < 1:
            raise ValueError("per_call_limit must be at least one")
        self._vfs = vfs
        self._index = index
        self._budget = budget
        self._per_call_limit = per_call_limit

    @property
    def per_call_limit(self) -> int:
        """Return the maximum content-token count for one successful result."""

        return self._per_call_limit

    def execute(self, request: ToolRequest) -> ToolResult:
        """Execute one request and return a timed result envelope."""

        if not isinstance(request, ToolRequest):
            raise TypeError("request must be a ToolRequest")

        started = time.monotonic()
        tool = request.tool
        try:
            if tool is ToolName.GREP:
                result = self._grep(request.arguments)
            elif tool is ToolName.READ:
                result = self._read(request.arguments)
            elif tool is ToolName.CAT:
                result = self._cat(request.arguments)
            elif tool is ToolName.LIST:
                result = self._list(request.arguments)
            else:
                raise VfsError("INVALID_PATH", "unknown VFS tool")
        except VfsError as error:
            result = ToolResult(
                ok=False,
                tool=tool,
                error_code=error.code,
                error_message=error.message,
            )

        elapsed_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        return result.model_copy(update={"latency_ms": elapsed_ms})

    def _count(self, text: str) -> int:
        return self._budget.tokenizer.count_text(text)

    def _fit_content(self, text: str) -> tuple[str, bool]:
        """Apply the per-call cap and then the remaining-budget fit."""

        truncated = False
        candidate = text
        if self._count(candidate) > self._per_call_limit:
            candidate = self._budget.tokenizer.truncate_text(
                candidate,
                self._per_call_limit,
            )
            truncated = True

        fitted = self._budget.fit_tool_result(candidate, reserved_output_tokens=1)
        if fitted != candidate:
            truncated = True
        return fitted, truncated

    def _success(
        self,
        tool: ToolName,
        content: str,
        *,
        paths: list[str] | None = None,
        line_ranges: list[LineRange] | None = None,
        truncated: bool = False,
    ) -> ToolResult:
        return ToolResult(
            ok=True,
            tool=tool,
            content=content,
            paths=sorted(set(paths or [])),
            line_ranges=line_ranges or [],
            token_count=self._count(content),
            truncated=truncated,
        )

    def _grep(self, arguments: dict[str, Any]) -> ToolResult:
        _require_keys(
            arguments,
            ("query", "max_results"),
            frozenset({"query", "path", "max_results"}),
            code="RESULT_LIMIT_INVALID",
            tool="grep",
        )
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise VfsError("RESULT_LIMIT_INVALID", "grep query must not be blank")

        max_results = _require_integer(
            arguments["max_results"],
            "max_results",
            code="RESULT_LIMIT_INVALID",
        )
        if not 1 <= max_results <= 50:
            raise VfsError("RESULT_LIMIT_INVALID", "max_results must be between 1 and 50")

        path_filter = arguments.get("path")
        allowed_paths: str | None = None
        if path_filter is not None:
            if not isinstance(path_filter, str):
                raise VfsError("INVALID_PATH", "grep path must be a string")
            canonical = _canonical_path(path_filter)
            indexed_paths = {str(chunk.path) for chunk in self._index.chunks}
            indexed_directories = {"/"}
            for indexed_path in indexed_paths:
                parts = indexed_path.strip("/").split("/")[:-1]
                for part_index in range(1, len(parts) + 1):
                    indexed_directories.add("/" + "/".join(parts[:part_index]))

            if canonical in indexed_paths and not path_filter.endswith("/"):
                allowed_paths = canonical
            elif canonical in indexed_directories:
                allowed_paths = "/" if canonical == "/" else f"{canonical}/"
            elif canonical in indexed_paths and path_filter.endswith("/"):
                raise VfsError("NOT_A_DIRECTORY", "grep path is a file")
            else:
                raise VfsError("NOT_FOUND", "grep path does not exist")

        query_terms = set(normalize_terms(query))
        hits = self._index.search(
            query,
            top_k=len(self._index.chunks),
            allowed_paths=allowed_paths,
        )
        previews: list[tuple[ContentSlice, SearchHit]] = []
        seen_ranges: set[tuple[str, int, int]] = set()

        for hit in hits:
            for offset, line in enumerate(hit.text.splitlines()):
                if not query_terms.intersection(normalize_terms(line)):
                    continue
                matching_line = hit.start_line + offset
                start_line = max(1, matching_line - 1)
                end_line = matching_line + 1
                preview = self._vfs.read(hit.path, start_line, end_line)
                range_key = (preview.path, preview.start_line, preview.end_line)
                if range_key in seen_ranges:
                    continue
                seen_ranges.add(range_key)
                previews.append((preview, hit))
                if len(previews) >= max_results:
                    break
            if len(previews) >= max_results:
                break

        preview_blocks = [
            (
                f"{preview.path}:{preview.start_line}-{preview.end_line}\n"
                f"{preview.text.rstrip(chr(10) + chr(13))}"
            )
            for preview, _ in previews
        ]
        content, truncated = self._fit_content("\n".join(preview_blocks))
        return self._success(
            ToolName.GREP,
            content,
            paths=[preview.path for preview, _ in previews],
            line_ranges=[_line_range(preview) for preview, _ in previews],
            truncated=truncated,
        )

    def _read(self, arguments: dict[str, Any]) -> ToolResult:
        _require_keys(
            arguments,
            ("path", "start_line", "end_line"),
            frozenset({"path", "start_line", "end_line"}),
            code="RANGE_INVALID",
            tool="read",
        )
        path = arguments["path"]
        if not isinstance(path, str):
            raise VfsError("INVALID_PATH", "read path must be a string")
        start_line = _require_integer(
            arguments["start_line"],
            "start_line",
            code="RANGE_INVALID",
        )
        end_line = _require_integer(
            arguments["end_line"],
            "end_line",
            code="RANGE_INVALID",
        )
        content_slice = self._vfs.read(path, start_line, end_line)
        content, truncated = self._fit_content(content_slice.text)
        return self._success(
            ToolName.READ,
            content,
            paths=[content_slice.path],
            line_ranges=[_line_range(content_slice)],
            truncated=truncated,
        )

    def _cat(self, arguments: dict[str, Any]) -> ToolResult:
        _require_keys(
            arguments,
            ("path",),
            frozenset({"path"}),
            code="INVALID_PATH",
            tool="cat",
        )
        path = arguments["path"]
        if not isinstance(path, str):
            raise VfsError("INVALID_PATH", "cat path must be a string")
        content_slice = self._vfs.cat(path)
        if self._count(content_slice.text) > self._per_call_limit:
            raise VfsError(
                "PER_CALL_LIMIT",
                f"cat result for {content_slice.path} exceeds the per-call limit",
            )
        if self._budget.fit_tool_result(
            content_slice.text,
            reserved_output_tokens=1,
        ) != content_slice.text:
            raise VfsError(
                "BUDGET_INSUFFICIENT",
                f"cat result for {content_slice.path} exceeds the remaining budget",
            )
        return self._success(
            ToolName.CAT,
            content_slice.text,
            paths=[content_slice.path],
            line_ranges=[_line_range(content_slice)],
        )

    def _list(self, arguments: dict[str, Any]) -> ToolResult:
        _require_keys(
            arguments,
            (),
            frozenset({"path"}),
            code="INVALID_PATH",
            tool="list",
        )
        path = arguments.get("path", "/")
        if not isinstance(path, str):
            raise VfsError("INVALID_PATH", "list path must be a string")
        paths = self._vfs.list(path)
        content, truncated = self._fit_content("\n".join(paths))
        return self._success(
            ToolName.LIST,
            content,
            paths=[item for item in paths if not item.endswith("/")],
            truncated=truncated,
        )


__all__ = ["ToolRequest", "ToolResult", "VfsToolExecutor", "tool_schema"]
