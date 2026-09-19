"""Single-call BM25 snippet harness."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.experiment.artifacts import canonical_json
from bm25_vfs_ablation.experiment.schemas import (
    ErrorRecord,
    LatencyBreakdown,
    ParsedAnswer,
    TerminationReason,
)
from bm25_vfs_ablation.harnesses.base import (
    Harness,
    HarnessContext,
    HarnessResult,
    build_answer_instruction,
    build_common_system_prompt,
)
from bm25_vfs_ablation.harnesses.budget import BudgetController, CallAdmission
from bm25_vfs_ablation.models.cache import CacheInputs, build_cache_key
from bm25_vfs_ablation.models.client import ModelRequest, ModelResponse
from bm25_vfs_ablation.retrieval.bm25 import SearchHit


def _question(task: TaskRecord | str) -> str:
    if isinstance(task, str):
        return task
    return task.question


def _render_hit(hit: SearchHit) -> str:
    return (
        f"[doc_id={hit.doc_id} chunk_id={hit.chunk_id} path={hit.path} "
        f"lines={hit.start_line}-{hit.end_line}]\n{hit.text}"
    )


def build_snippet_messages(
    task: TaskRecord | str,
    hits: Sequence[SearchHit] = (),
) -> list[dict[str, str]]:
    """Build the deterministic system and user messages for Condition A."""

    rendered_hits = "\n\n".join(_render_hit(hit) for hit in hits)
    if not rendered_hits:
        rendered_hits = "(No retrieved snippets fit the configured bounds.)"

    user_content = "\n\n".join(
        (
            f"Question:\n{_question(task)}",
            "Retrieved snippets:\n" + rendered_hits,
            build_answer_instruction(),
        )
    )
    return [
        {"role": "system", "content": build_common_system_prompt()},
        {"role": "user", "content": user_content},
    ]


def _integer_usage(usage: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _provider_usage(response: ModelResponse) -> tuple[int | None, int | None, int | None]:
    usage = response.provider_usage
    if not isinstance(usage, Mapping):
        return None, None, None

    provider_input = _integer_usage(usage, "prompt_tokens", "input_tokens")
    provider_output = _integer_usage(
        usage,
        "completion_tokens",
        "output_tokens",
    )
    provider_total = _integer_usage(usage, "total_tokens")
    if (
        provider_total is not None
        and provider_input is not None
        and provider_output is not None
        and provider_total != provider_input + provider_output
    ):
        provider_total = None
    return provider_input, provider_output, provider_total


def _config_hash(context: HarnessContext) -> str:
    config = context.config
    if hasattr(config, "canonical_dict"):
        payload = config.canonical_dict()
    else:  # pragma: no cover - the production context always has AppConfig
        payload = config.model_dump(mode="json")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _model_settings(context: HarnessContext) -> dict[str, Any]:
    model_config = context.config.model
    return model_config.model_dump(mode="json")


def _cache_key(
    context: HarnessContext,
    messages: list[dict[str, str]],
) -> str:
    inputs = CacheInputs(
        condition="snippets",
        model_config=_model_settings(context),
        messages=messages,
        tool_schema=[],
        corpus_sha256=context.bundle.corpus_sha256,
        experiment_config_sha256=_config_hash(context),
    )
    return build_cache_key(inputs)


def _error(stage: str, code: str, message: str) -> ErrorRecord:
    return ErrorRecord(
        stage=stage,
        code=code,
        message=message,
        retryable=False,
    )


def _response_is_valid(response: ModelResponse) -> bool:
    if response.content is None or response.tool_calls:
        return False
    try:
        ParsedAnswer.model_validate(json.loads(response.content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _select_chunks(
    task: TaskRecord,
    hits: Sequence[SearchHit],
    context: HarnessContext,
    budget: BudgetController,
) -> tuple[tuple[SearchHit, ...], list[dict[str, str]], CallAdmission]:
    """Greedily admit ranked, complete chunks without mutating the budget."""

    selected: list[SearchHit] = []
    selected_tokens = 0
    context_limit = context.config.harness.retrieval_context_tokens
    output_cap = context.config.harness.max_answer_tokens

    for hit in hits:
        hit_tokens = budget.tokenizer.count_text(hit.text)
        if selected_tokens + hit_tokens > context_limit:
            continue

        candidate = (*selected, hit)
        candidate_messages = build_snippet_messages(task, candidate)
        candidate_admission = budget.plan_call(candidate_messages, output_cap)
        if not candidate_admission.admitted:
            continue

        selected.append(hit)
        selected_tokens += hit_tokens

    messages = build_snippet_messages(task, selected)
    admission = budget.plan_call(messages, output_cap)
    return tuple(selected), messages, admission


class SnippetHarness(Harness):
    """Run Condition A with one bounded prompt and one completion at most."""

    def run(self, task: TaskRecord, context: HarnessContext) -> HarnessResult:
        started = time.perf_counter()
        retrieval_started = started
        hits = tuple(context.index.search(task.question, context.config.retrieval.top_k))
        retrieval_ms = max(0.0, (time.perf_counter() - retrieval_started) * 1000)

        budget = context.new_budget()
        selected, messages, admission = _select_chunks(task, hits, context, budget)

        if not admission.admitted:
            return HarnessResult(
                initial_hits=hits,
                accounting=budget.accounting(),
                latencies=LatencyBreakdown(
                    wall_ms=max(0.0, (time.perf_counter() - started) * 1000),
                    model_ms=0.0,
                    retrieval_ms=retrieval_ms,
                    tool_ms=0.0,
                ),
                termination=TerminationReason.BUDGET_EXHAUSTED,
            )

        request = ModelRequest(
            messages=messages,
            tools=[],
            max_output_tokens=admission.output_cap,
            temperature=context.config.model.temperature,
            top_p=context.config.model.top_p,
            seed=context.config.model.seed,
        )

        response: ModelResponse
        cache_hit = False
        cache_key: str | None = None
        model_started = time.perf_counter()
        try:
            if context.cache is not None:
                cache_key = _cache_key(context, messages)
                cached = context.cache.get(cache_key)
            else:
                cached = None

            if cached is not None:
                response = cached
                cache_hit = True
            else:
                response = context.model.complete(request)
                if context.cache is not None and cache_key is not None:
                    context.cache.put(cache_key, response)
        except Exception:
            with suppress(Exception):
                budget.record_call(
                    admission,
                    response_output_tokens_estimated=0,
                )
            return HarnessResult(
                initial_hits=hits,
                accounting=budget.accounting(),
                latencies=LatencyBreakdown(
                    wall_ms=max(0.0, (time.perf_counter() - started) * 1000),
                    model_ms=max(0.0, (time.perf_counter() - model_started) * 1000),
                    retrieval_ms=retrieval_ms,
                    tool_ms=0.0,
                ),
                termination=TerminationReason.MODEL_ERROR,
                errors=(
                    _error(
                        "model",
                        "MODEL_ERROR",
                        "model completion failed",
                    ),
                ),
            )

        content = response.content if isinstance(response.content, str) else ""
        output_tokens = budget.tokenizer.count_text(content)
        snippet_tokens = sum(budget.tokenizer.count_text(hit.text) for hit in selected)
        provider_input, provider_output, provider_total = _provider_usage(response)
        try:
            budget.record_call(
                admission,
                response_output_tokens_estimated=output_tokens,
                snippet_input_tokens=snippet_tokens,
                provider_input_tokens=provider_input,
                provider_output_tokens=provider_output,
                provider_total_tokens=provider_total,
            )
        except Exception:
            return HarnessResult(
                raw_answer=response.content,
                initial_hits=hits,
                accounting=budget.accounting(),
                latencies=LatencyBreakdown(
                    wall_ms=max(0.0, (time.perf_counter() - started) * 1000),
                    model_ms=0.0 if cache_hit else response.latency_ms,
                    retrieval_ms=retrieval_ms,
                    tool_ms=0.0,
                ),
                termination=TerminationReason.MODEL_ERROR,
                errors=(
                    _error(
                        "accounting",
                        "MODEL_ERROR",
                        "model response exceeded the admitted budget",
                    ),
                ),
            )

        valid = _response_is_valid(response)
        termination = (
            TerminationReason.ANSWERED
            if valid
            else TerminationReason.MALFORMED_RESPONSE
        )
        errors = (
            ()
            if valid
            else (
                _error(
                    "response",
                    "MALFORMED_RESPONSE",
                    "model response was not a valid answer object",
                ),
            )
        )
        return HarnessResult(
            raw_answer=response.content,
            initial_hits=hits,
            accounting=budget.accounting(),
            latencies=LatencyBreakdown(
                wall_ms=max(0.0, (time.perf_counter() - started) * 1000),
                model_ms=0.0 if cache_hit else response.latency_ms,
                retrieval_ms=retrieval_ms,
                tool_ms=0.0,
            ),
            termination=termination,
            errors=errors,
        )


__all__ = ["SnippetHarness", "build_snippet_messages"]
