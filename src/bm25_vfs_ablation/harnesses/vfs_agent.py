"""Iterative, bounded model/tool harness for the BM25 VFS condition."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from bm25_vfs_ablation import ToolName
from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.experiment.artifacts import canonical_json
from bm25_vfs_ablation.experiment.schemas import (
    ErrorRecord,
    LatencyBreakdown,
    ParsedAnswer,
    TerminationReason,
    ToolCallCounts,
    ToolRequest,
    ToolResult,
    ToolTraceEntry,
)
from bm25_vfs_ablation.harnesses.base import (
    Harness,
    HarnessContext,
    HarnessResult,
    build_answer_instruction,
    build_common_system_prompt,
)
from bm25_vfs_ablation.harnesses.budget import (
    BudgetController,
    BudgetInvariantError,
)
from bm25_vfs_ablation.models.cache import CacheInputs, build_cache_key
from bm25_vfs_ablation.models.client import ModelRequest, ModelResponse
from bm25_vfs_ablation.retrieval.bm25 import SearchHit
from bm25_vfs_ablation.vfs.filesystem import VirtualFilesystem, materialize_corpus
from bm25_vfs_ablation.vfs.tools import VfsToolExecutor, tool_schema


class ActionParseError(ValueError):
    """Raised when a model response is neither one tool call nor one final JSON object."""


@dataclass(frozen=True, slots=True)
class FinalAction:
    """A parsed public answer returned by the model."""

    content: str
    parsed_answer: ParsedAnswer

    @property
    def answer(self) -> str:
        """Return the public answer text."""

        return self.parsed_answer.answer

    @property
    def citations(self) -> list[str]:
        """Return public citations in first-seen order."""

        return list(self.parsed_answer.citations)

    @property
    def justification(self) -> str:
        """Return the short public justification."""

        return self.parsed_answer.justification

    @property
    def raw_answer(self) -> str:
        """Return the exact public JSON response."""

        return self.content


@dataclass(frozen=True, slots=True)
class VfsHarnessResult(HarnessResult):
    """Harness result with VFS-specific observable call accounting."""

    tool_call_count: int = 0
    max_tool_calls_permitted: int = 0
    successful_tool_calls: int = 0
    invalid_tool_calls: int = 0
    repeated_tool_calls: int = 0
    tool_calls_by_type: ToolCallCounts = field(
        default_factory=lambda: ToolCallCounts(grep=0, read=0, cat=0, list=0)
    )

    @property
    def invalid_call_count(self) -> int:
        """Compatibility alias for the invalid-request counter."""

        return self.invalid_tool_calls

    @property
    def repeated_call_count(self) -> int:
        """Compatibility alias for the repeated-request counter."""

        return self.repeated_tool_calls


def _question(task: TaskRecord | str) -> str:
    if isinstance(task, str):
        return task
    return task.question


def _mapping_value(value: object, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _canonical_path(path: object) -> str:
    value = str(path)
    if value.startswith("/"):
        return value
    return f"/{value}"


def _score_text(score: object) -> str:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "0"
    return str(round(float(score), 12))


def _candidate_rows(
    candidates: Sequence[object],
    *,
    oracle_paths: Sequence[str] | None = None,
) -> list[str]:
    if oracle_paths is not None:
        return [f"- path={_canonical_path(path)}" for path in oracle_paths]

    rows: list[str] = []
    for candidate in candidates:
        path = _mapping_value(candidate, "path")
        if path is None:
            continue
        doc_id = _mapping_value(candidate, "doc_id")
        score = _mapping_value(candidate, "best_score", _mapping_value(candidate, "score", 0.0))
        pieces = [f"path={_canonical_path(path)}"]
        if doc_id is not None:
            pieces.append(f"doc_id={doc_id}")
        pieces.append(f"score={_score_text(score)}")
        rows.append("- " + " ".join(pieces))
    return rows


def build_vfs_messages(
    task: TaskRecord | str,
    candidates: Sequence[object] = (),
    remaining_budget: int = 0,
    *,
    oracle_paths: Sequence[str] | None = None,
    oracle_mode: bool = False,
) -> list[dict[str, Any]]:
    """Build the initial VFS prompt without exposing corpus content."""

    if isinstance(remaining_budget, bool) or not isinstance(remaining_budget, int):
        raise TypeError("remaining_budget must be an integer")
    if remaining_budget < 0:
        raise ValueError("remaining_budget must be nonnegative")

    candidate_values = candidates[:5] if oracle_paths is None else candidates
    rows = _candidate_rows(candidate_values, oracle_paths=oracle_paths)
    candidate_text = "\n".join(rows) if rows else "(No candidate document paths.)"
    candidate_label = (
        "Oracle gold document paths:"
        if oracle_mode or oracle_paths is not None
        else "Initial BM25 candidate document paths and scores:"
    )
    user_content = "\n\n".join(
        (
            f"Question:\n{_question(task)}",
            "Available tools: grep, read, cat, list. Use exactly one tool call "
            "or return the final JSON answer on each turn.",
            f"{candidate_label}\n{candidate_text}",
            f"Remaining estimated token budget: {remaining_budget}",
            build_answer_instruction(),
        )
    )
    return [
        {"role": "system", "content": build_common_system_prompt()},
        {"role": "user", "content": user_content},
    ]


def _parse_arguments(raw_arguments: object) -> dict[str, Any]:
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except (TypeError, ValueError) as error:
            raise ActionParseError("tool arguments were not valid JSON") from error
    if not isinstance(raw_arguments, Mapping):
        raise ActionParseError("tool arguments must be a JSON object")
    return dict(raw_arguments)


def _function_payload(call: object) -> Mapping[str, Any]:
    if not isinstance(call, Mapping):
        raise ActionParseError("tool call must be an object")
    function = call.get("function", call)
    if not isinstance(function, Mapping):
        raise ActionParseError("tool call function must be an object")
    return function


def parse_model_action(response: ModelResponse) -> FinalAction | ToolRequest:
    """Parse exactly one supported tool call or one public final JSON answer."""

    if not isinstance(response, ModelResponse):
        raise TypeError("response must be a ModelResponse")

    calls = response.tool_calls or []
    if calls:
        if len(calls) != 1 or response.content not in (None, ""):
            raise ActionParseError("response must contain exactly one tool call")
        function = _function_payload(calls[0])
        name = function.get("name", function.get("tool"))
        if not isinstance(name, str) or not name.strip():
            raise ActionParseError("tool call is missing a tool name")
        arguments = _parse_arguments(function.get("arguments", {}))
        try:
            return ToolRequest(tool=name, arguments=arguments)
        except (TypeError, ValueError) as error:
            raise ActionParseError("tool call did not match the VFS tool contract") from error

    if not isinstance(response.content, str):
        raise ActionParseError("response did not contain a final JSON object")
    try:
        payload = json.loads(response.content)
        parsed = ParsedAnswer.model_validate(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ActionParseError("response did not contain a valid final JSON object") from error
    return FinalAction(content=response.content, parsed_answer=parsed)


def _invalid_request(response: ModelResponse) -> ToolRequest:
    """Create a safe observable placeholder for an unparseable tool response."""

    calls = response.tool_calls or []
    if len(calls) == 1:
        with suppress(ActionParseError):
            function = _function_payload(calls[0])
            name = function.get("name", function.get("tool"))
            if isinstance(name, str) and name in {item.value for item in ToolName}:
                return ToolRequest(
                    tool=name,
                    arguments={"_invalid_request": "malformed_model_action"},
                )
    return ToolRequest(
        tool=ToolName.GREP,
        arguments={"_invalid_request": "exactly_one_supported_tool_call_required"},
    )


def _invalid_requests(response: ModelResponse) -> list[ToolRequest]:
    """Return one safe observable request for every malformed tool call."""

    calls = response.tool_calls or []
    return [
        _invalid_request(response.model_copy(update={"content": None, "tool_calls": [call]}))
        for call in calls
    ]


def _request_key(request: ToolRequest) -> str:
    return canonical_json(request.model_dump(mode="json"))


def _safe_error(
    stage: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> ErrorRecord:
    return ErrorRecord(
        stage=stage,
        code=code,
        message=message,
        retryable=retryable,
    )


def _provider_usage(response: ModelResponse) -> tuple[int | None, int | None, int | None]:
    usage = response.provider_usage
    if not isinstance(usage, Mapping):
        return None, None, None

    def integer(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    provider_input = integer("prompt_tokens", "input_tokens")
    provider_output = integer("completion_tokens", "output_tokens")
    provider_total = integer("total_tokens")
    if (
        provider_total is not None
        and provider_input is not None
        and provider_output is not None
        and provider_total != provider_input + provider_output
    ):
        provider_total = None
    return provider_input, provider_output, provider_total


def _response_output_tokens(response: ModelResponse, budget: BudgetController) -> int:
    total = budget.tokenizer.count_text(response.content or "")
    for call in response.tool_calls or []:
        with suppress(ActionParseError):
            function = _function_payload(call)
            arguments = function.get("arguments", "")
            if isinstance(arguments, Mapping):
                total += budget.count_tool_arguments(arguments)
            elif isinstance(arguments, str):
                total += budget.tokenizer.count_text(arguments)
            else:
                total += budget.tokenizer.count_text(str(arguments))
    return total


def _response_argument_tokens(response: ModelResponse, budget: BudgetController) -> int:
    total = 0
    for call in response.tool_calls or []:
        with suppress(ActionParseError):
            function = _function_payload(call)
            arguments = function.get("arguments", "")
            if isinstance(arguments, Mapping):
                total += budget.count_tool_arguments(arguments)
            elif isinstance(arguments, str):
                total += budget.tokenizer.count_text(arguments)
            else:
                total += budget.tokenizer.count_text(str(arguments))
    return total


def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    if response.tool_calls:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": response.tool_calls,
        }
    return {"role": "assistant", "content": response.content}


def _tool_call_id(response: ModelResponse, request_id: str, call_index: int = 0) -> str:
    calls = response.tool_calls or []
    if 0 <= call_index < len(calls) and isinstance(calls[call_index], Mapping):
        value = calls[call_index].get("id")
        if isinstance(value, str) and value:
            return value
    return request_id


def _tool_message(result: ToolResult, tool_call_id: str) -> dict[str, Any]:
    if result.ok:
        content = result.content or ""
    else:
        content = canonical_json(
            {
                "error_code": result.error_code,
                "error_message": result.error_message,
            }
        )
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
    }


def _document_candidates(index: object, hits: Sequence[SearchHit]) -> list[object]:
    document_hits = getattr(index, "document_hits", None)
    if callable(document_hits):
        return list(document_hits(hits))

    grouped: dict[str, SearchHit] = {}
    for hit in hits:
        current = grouped.get(hit.doc_id)
        if current is None or (hit.score, hit.chunk_id) > (current.score, current.chunk_id):
            grouped[hit.doc_id] = hit
    return sorted(grouped.values(), key=lambda hit: (-hit.score, hit.doc_id))


def _document_path(bundle: object, doc_id: str) -> str | None:
    documents_by_id = getattr(bundle, "documents_by_id", None)
    if documents_by_id is not None:
        document = documents_by_id.get(doc_id)
        if document is not None:
            return _canonical_path(getattr(document, "path", ""))
    for document in getattr(bundle, "documents", ()):
        if getattr(document, "doc_id", None) == doc_id:
            return _canonical_path(getattr(document, "path", ""))
    return None


def _config_hash(
    context: HarnessContext,
    *,
    max_tool_calls: int,
    oracle_mode: bool,
) -> str:
    config = context.config
    if hasattr(config, "canonical_dict"):
        payload = config.canonical_dict()
    else:  # pragma: no cover - production contexts use AppConfig
        payload = config.model_dump(mode="json")
    payload = {
        "config": payload,
        "assigned_max_tool_calls": max_tool_calls,
        "oracle_mode": oracle_mode,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _cache_key(
    context: HarnessContext,
    messages: list[dict[str, Any]],
    *,
    max_tool_calls: int,
    oracle_mode: bool,
) -> str:
    inputs = CacheInputs(
        condition="vfs",
        model_config=context.config.model.model_dump(mode="json"),
        messages=messages,
        tool_schema=tool_schema(),
        corpus_sha256=context.bundle.corpus_sha256,
        experiment_config_sha256=_config_hash(
            context,
            max_tool_calls=max_tool_calls,
            oracle_mode=oracle_mode,
        ),
    )
    return build_cache_key(inputs)


def _default_tool_counts() -> dict[ToolName, int]:
    return {tool: 0 for tool in ToolName}


class VfsHarness(Harness):
    """Run Condition B with a bounded iterative model/tool loop."""

    def __init__(
        self,
        vfs: VirtualFilesystem | None = None,
        *,
        executor_factory: Callable[[BudgetController], VfsToolExecutor] | None = None,
        max_tool_calls: int | None = None,
        invalid_call_limit: int | None = None,
        oracle_mode: bool | None = None,
    ) -> None:
        if max_tool_calls is not None and (
            isinstance(max_tool_calls, bool)
            or not isinstance(max_tool_calls, int)
            or max_tool_calls < 0
        ):
            raise ValueError("max_tool_calls must be a nonnegative integer")
        if invalid_call_limit is not None and (
            isinstance(invalid_call_limit, bool)
            or not isinstance(invalid_call_limit, int)
            or invalid_call_limit < 0
        ):
            raise ValueError("invalid_call_limit must be a nonnegative integer")
        if oracle_mode is not None and not isinstance(oracle_mode, bool):
            raise TypeError("oracle_mode must be a boolean")
        self._vfs = vfs
        self._executor_factory = executor_factory
        self._max_tool_calls = max_tool_calls
        self._invalid_call_limit = invalid_call_limit
        self._oracle_mode = oracle_mode

    def _effective_limits(self, context: HarnessContext) -> tuple[int, int, bool]:
        max_tool_calls = (
            context.config.harness.max_tool_calls
            if self._max_tool_calls is None
            else self._max_tool_calls
        )
        invalid_call_limit = (
            context.config.harness.invalid_call_limit
            if self._invalid_call_limit is None
            else self._invalid_call_limit
        )
        oracle_mode = (
            context.config.experiment.oracle_mode
            if self._oracle_mode is None
            else self._oracle_mode
        )
        return max_tool_calls, invalid_call_limit, oracle_mode

    def _executor(
        self,
        context: HarnessContext,
        budget: BudgetController,
    ) -> tuple[VfsToolExecutor, TemporaryDirectory[str] | None]:
        if self._executor_factory is not None:
            return self._executor_factory(budget), None
        if self._vfs is not None:
            return (
                VfsToolExecutor(
                    self._vfs,
                    context.index,
                    budget,
                    context.config.harness.tool_result_tokens_per_call,
                ),
                None,
            )

        temporary_root: TemporaryDirectory[str] = TemporaryDirectory(prefix="bm25-vfs-harness-")
        try:
            documents = tuple(context.bundle.documents)
            root = Path(temporary_root.name)
            materialize_corpus(root, documents)
            filesystem = VirtualFilesystem(root, documents)
            return (
                VfsToolExecutor(
                    filesystem,
                    context.index,
                    budget,
                    context.config.harness.tool_result_tokens_per_call,
                ),
                temporary_root,
            )
        except BaseException:
            temporary_root.cleanup()
            raise

    @staticmethod
    def _result_with_request_id(result: ToolResult, request_id: str) -> ToolResult:
        return result.model_copy(update={"request_id": request_id})

    def run(self, task: TaskRecord, context: HarnessContext) -> HarnessResult:
        started = time.perf_counter()
        retrieval_started = started
        try:
            hits = tuple(
                context.index.search(
                    task.question,
                    context.config.retrieval.top_k,
                )
            )
        except Exception:
            return VfsHarnessResult(
                termination=TerminationReason.MODEL_ERROR,
                errors=(
                    _safe_error(
                        "retrieval",
                        "RETRIEVAL_ERROR",
                        "initial BM25 retrieval failed",
                    ),
                ),
                latencies=LatencyBreakdown(
                    wall_ms=max(0.0, (time.perf_counter() - started) * 1000),
                    model_ms=0.0,
                    retrieval_ms=max(0.0, (time.perf_counter() - retrieval_started) * 1000),
                    tool_ms=0.0,
                ),
            )
        retrieval_ms = max(0.0, (time.perf_counter() - retrieval_started) * 1000)

        budget = context.new_budget()
        max_tool_calls, invalid_call_limit, oracle_mode = self._effective_limits(context)
        candidates = _document_candidates(context.index, hits)
        oracle_paths: list[str] | None = None
        if oracle_mode:
            oracle_paths = [
                path
                for doc_id in task.gold_document_ids
                if (path := _document_path(context.bundle, doc_id)) is not None
            ]
        messages = build_vfs_messages(
            task,
            candidates[: context.config.harness.initial_candidate_documents],
            budget.remaining,
            oracle_paths=oracle_paths,
            oracle_mode=oracle_mode,
        )
        tools = tool_schema()
        schema_tokens = budget.tokenizer.count_text(canonical_json(tools))
        trace: list[ToolTraceEntry] = []
        errors: list[ErrorRecord] = []
        seen_requests: set[str] = set()
        counts = _default_tool_counts()
        tool_call_count = 0
        successful_tool_calls = 0
        invalid_tool_calls = 0
        repeated_tool_calls = 0
        raw_answer: str | None = None
        termination: TerminationReason | None = None
        fatal_tool_error = False
        model_ms = 0.0
        tool_ms = 0.0
        cache_hit = False
        temporary_root: TemporaryDirectory[str] | None = None

        executor: VfsToolExecutor | None = None

        try:
            while termination is None:
                admission = budget.plan_call(
                    messages,
                    context.config.harness.max_answer_tokens,
                    extra_input_tokens=schema_tokens,
                )
                if not admission.admitted:
                    termination = TerminationReason.BUDGET_EXHAUSTED
                    break
                # A zero-call intervention admits the initial model turn so
                # that answer-at-zero remains measurable.  A requested tool
                # is counted below and stopped before execution.
                if tool_call_count > 0 and tool_call_count >= max_tool_calls:
                    termination = TerminationReason.MAX_TOOL_CALLS
                    break
                if invalid_tool_calls and invalid_tool_calls >= invalid_call_limit:
                    termination = TerminationReason.INVALID_CALL_LIMIT
                    break
                if fatal_tool_error:
                    termination = TerminationReason.MODEL_ERROR
                    break

                request = ModelRequest(
                    messages=list(messages),
                    tools=tools,
                    max_output_tokens=admission.output_cap,
                    temperature=context.config.model.temperature,
                    top_p=context.config.model.top_p,
                    seed=context.config.model.seed,
                )
                call_started = time.perf_counter()
                current_cache_hit = False
                current_cache_key: str | None = None
                try:
                    if context.cache is not None:
                        current_cache_key = _cache_key(
                            context,
                            messages,
                            max_tool_calls=max_tool_calls,
                            oracle_mode=oracle_mode,
                        )
                        cached = context.cache.get(current_cache_key)
                    else:
                        cached = None

                    if cached is not None:
                        response = cached
                        current_cache_hit = True
                    else:
                        response = context.model.complete(request)
                        if context.cache is not None and current_cache_key is not None:
                            context.cache.put(current_cache_key, response)
                except Exception:
                    with suppress(BudgetInvariantError):
                        budget.record_call(
                            admission,
                            response_output_tokens_estimated=0,
                        )
                    model_ms += max(0.0, (time.perf_counter() - call_started) * 1000)
                    errors.append(_safe_error("model", "MODEL_ERROR", "model completion failed"))
                    termination = TerminationReason.MODEL_ERROR
                    break

                cache_hit = cache_hit or current_cache_hit
                if not current_cache_hit:
                    model_ms += max(0.0, response.latency_ms)
                output_tokens = _response_output_tokens(response, budget)
                argument_tokens = _response_argument_tokens(response, budget)
                provider_input, provider_output, provider_total = _provider_usage(response)
                tool_result_tokens = sum(
                    budget.tokenizer.count_messages([message])
                    for message in messages
                    if message.get("role") == "tool"
                )
                try:
                    budget.record_call(
                        admission,
                        response_output_tokens_estimated=output_tokens,
                        tool_argument_output_tokens=argument_tokens,
                        tool_result_input_tokens=tool_result_tokens,
                        provider_input_tokens=provider_input,
                        provider_output_tokens=provider_output,
                        provider_total_tokens=provider_total,
                    )
                except BudgetInvariantError:
                    errors.append(
                        _safe_error(
                            "accounting",
                            "MODEL_ERROR",
                            "model response exceeded the admitted budget",
                        )
                    )
                    termination = TerminationReason.MODEL_ERROR
                    break

                messages.append(_assistant_message(response))
                try:
                    parsed_action = parse_model_action(response)
                except (ActionParseError, TypeError):
                    if response.tool_calls:
                        # Preserve a malformed request as an observable
                        # invalid call instead of dropping it from counts.
                        actions = _invalid_requests(response)
                        action_invalid = [True] * len(actions)
                    else:
                        raw_answer = response.content
                        errors.append(
                            _safe_error(
                                "response",
                                "MALFORMED_RESPONSE",
                                "model response was not one tool call or final JSON object",
                            )
                        )
                        termination = TerminationReason.MALFORMED_RESPONSE
                        break
                else:
                    if isinstance(parsed_action, FinalAction):
                        raw_answer = parsed_action.content
                        termination = TerminationReason.ANSWERED
                        break
                    actions = [parsed_action]
                    action_invalid = [False]

                for action_index, (action, was_invalid) in enumerate(
                    zip(actions, action_invalid, strict=True)
                ):
                    request_id = f"tool-{tool_call_count + 1:03d}"
                    tool_call_count += 1
                    counts[action.tool] += 1
                    request_key = _request_key(action)
                    repeated = request_key in seen_requests
                    if repeated:
                        repeated_tool_calls += 1
                    seen_requests.add(request_key)

                    requested_at = datetime.now(UTC)
                    if tool_call_count > max_tool_calls:
                        result = ToolResult(
                            ok=False,
                            tool=action.tool,
                            request_id=request_id,
                            error_code="MAX_TOOL_CALLS",
                            error_message=("tool execution was not permitted after the call limit"),
                        )
                        completed_at = datetime.now(UTC)
                        trace.append(
                            ToolTraceEntry(
                                request_id=request_id,
                                call_ordinal=tool_call_count,
                                requested_at=requested_at,
                                completed_at=completed_at,
                                request=action,
                                result=result,
                                repeated=repeated,
                                latency_ms=0.0,
                            )
                        )
                        if was_invalid:
                            invalid_tool_calls += 1
                        termination = TerminationReason.MAX_TOOL_CALLS
                        break

                    call_failed = False
                    if executor is None:
                        try:
                            executor, temporary_root = self._executor(context, budget)
                        except Exception:
                            result = ToolResult(
                                ok=False,
                                tool=action.tool,
                                request_id=request_id,
                                error_code="VFS_ERROR",
                                error_message="read-only VFS initialization failed",
                            )
                            call_failed = True
                            fatal_tool_error = True
                    if not call_failed:
                        try:
                            if executor is None:  # pragma: no cover - guarded above
                                raise RuntimeError("VFS executor was not initialized")
                            result = executor.execute(action)
                        except Exception:
                            result = ToolResult(
                                ok=False,
                                tool=action.tool,
                                request_id=request_id,
                                error_code="TOOL_ERROR",
                                error_message="VFS tool execution failed",
                            )
                            fatal_tool_error = True
                    completed_at = datetime.now(UTC)
                    result = self._result_with_request_id(result, request_id)
                    tool_ms += max(0.0, result.latency_ms)
                    if result.ok:
                        successful_tool_calls += 1
                    else:
                        invalid_tool_calls += 1
                        if result.error_code:
                            errors.append(
                                _safe_error(
                                    "tool",
                                    result.error_code,
                                    result.error_message or "VFS tool request failed",
                                )
                            )

                    trace.append(
                        ToolTraceEntry(
                            request_id=request_id,
                            call_ordinal=tool_call_count,
                            requested_at=requested_at,
                            completed_at=completed_at,
                            request=action,
                            result=result,
                            repeated=repeated,
                            latency_ms=max(0.0, result.latency_ms),
                        )
                    )
                    messages.append(
                        _tool_message(
                            result,
                            _tool_call_id(response, request_id, action_index),
                        )
                    )

                    if tool_call_count >= max_tool_calls:
                        termination = TerminationReason.MAX_TOOL_CALLS
                        break
                    if invalid_tool_calls and invalid_tool_calls >= invalid_call_limit:
                        termination = TerminationReason.INVALID_CALL_LIMIT
                        break
                    if fatal_tool_error:
                        break

                if termination is not None or fatal_tool_error:
                    break

        finally:
            if temporary_root is not None:
                temporary_root.cleanup()

        if termination is None:  # pragma: no cover - loop always chooses a stop
            termination = TerminationReason.MODEL_ERROR
        tool_counts = ToolCallCounts(
            grep=counts[ToolName.GREP],
            read=counts[ToolName.READ],
            cat=counts[ToolName.CAT],
            list=counts[ToolName.LIST],
        )
        return VfsHarnessResult(
            raw_answer=raw_answer,
            initial_hits=hits,
            trace=tuple(trace),
            accounting=budget.accounting(),
            latencies=LatencyBreakdown(
                wall_ms=max(0.0, (time.perf_counter() - started) * 1000),
                model_ms=model_ms,
                retrieval_ms=retrieval_ms,
                tool_ms=tool_ms,
            ),
            termination=termination,
            errors=tuple(errors),
            tool_call_count=tool_call_count,
            max_tool_calls_permitted=max_tool_calls,
            successful_tool_calls=successful_tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            repeated_tool_calls=repeated_tool_calls,
            tool_calls_by_type=tool_counts,
        )


__all__ = [
    "ActionParseError",
    "FinalAction",
    "VfsHarness",
    "VfsHarnessResult",
    "build_vfs_messages",
    "parse_model_action",
]
