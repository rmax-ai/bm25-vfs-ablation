"""Exact estimated-token accounting shared by both harness conditions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from bm25_vfs_ablation.experiment.schemas import TokenAccounting, TurnAccounting
from bm25_vfs_ablation.models.tokenization import RegexTokenizer


class Tokenizer(Protocol):
    """The tokenizer operations required by the budget controller."""

    name: str

    def count_messages(self, messages: Sequence[Any]) -> int:
        """Count a serialized model request."""

    def count_text(self, text: str) -> int:
        """Count text tokens."""

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Return a prefix ending on a tokenizer boundary."""


@dataclass(frozen=True, slots=True)
class CallAdmission:
    """A non-mutating decision about whether a model call can be made."""

    admitted: bool
    input_tokens: int
    output_cap: int
    remaining_before: int
    reason: str | None = None

    @property
    def effective_output_cap(self) -> int:
        """Return the output cap after applying the remaining budget."""

        return self.output_cap


class BudgetInvariantError(ValueError):
    """Raised when a finalized call would violate the budget contract."""


def _require_integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _coalesce_counts(
    values: tuple[tuple[str, int | None], ...],
    *,
    field_name: str,
) -> int:
    present = [(name, value) for name, value in values if value is not None]
    if not present:
        raise TypeError(f"{field_name} is required")
    first = present[0][1]
    if any(value != first for _, value in present[1:]):
        names = ", ".join(name for name, _ in present)
        raise TypeError(f"conflicting {field_name} values: {names}")
    assert first is not None
    return first


def _sum_if_complete(values: Sequence[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


class BudgetController:
    """Enforce one hard estimated-token ceiling across model turns.

    Planning is deliberately non-mutating.  A call is charged only when its
    response is finalized through :meth:`record_call`; provider usage never
    participates in admission.
    """

    __slots__ = (
        "_limit",
        "_provider_turn_totals",
        "_tokenizer",
        "_total_input",
        "_total_output",
        "_turns",
    )

    def __init__(self, limit: int, tokenizer: Tokenizer | None = None) -> None:
        self._limit = _require_integer(limit, "limit", minimum=1)
        self._tokenizer = tokenizer if tokenizer is not None else RegexTokenizer()
        self._turns: tuple[TurnAccounting, ...] = ()
        self._provider_turn_totals: tuple[int | None, ...] = ()
        self._total_input = 0
        self._total_output = 0

    @property
    def limit(self) -> int:
        """Return the immutable estimated-token ceiling."""

        return self._limit

    @property
    def tokenizer(self) -> Tokenizer:
        """Return the tokenizer used for all estimates."""

        return self._tokenizer

    @property
    def remaining(self) -> int:
        """Return ``limit - total_tokens_estimated``."""

        return self._limit - self.total_tokens_estimated

    @property
    def input_tokens_estimated(self) -> int:
        """Return the finalized estimated input total."""

        return self._total_input

    @property
    def output_tokens_estimated(self) -> int:
        """Return the finalized estimated output total."""

        return self._total_output

    @property
    def total_tokens_estimated(self) -> int:
        """Return finalized estimated input plus output tokens."""

        return self._total_input + self._total_output

    @property
    def total_estimated(self) -> int:
        """Compatibility alias for the finalized estimated total."""

        return self.total_tokens_estimated

    @property
    def turns(self) -> tuple[TurnAccounting, ...]:
        """Return defensive copies of the immutable finalized turns."""

        return tuple(turn.model_copy(deep=True) for turn in self._turns)

    def plan_call(
        self,
        messages: Sequence[Any],
        output_cap: int,
        *,
        extra_input_tokens: int = 0,
    ) -> CallAdmission:
        """Estimate and admit a call without changing controller state.

        ``extra_input_tokens`` charges request surface that is not part of the
        message list itself — most importantly the tool schema, which is
        resent on every VFS turn and must reduce the admitted budget (AIR-1).
        """

        requested_cap = _require_integer(output_cap, "output_cap")
        extras = _require_integer(extra_input_tokens, "extra_input_tokens")
        input_tokens = _require_integer(
            self._tokenizer.count_messages(messages),
            "tokenizer input count",
        ) + extras
        remaining_before = self.remaining
        effective_cap = min(requested_cap, max(0, remaining_before - input_tokens))
        if effective_cap < 1:
            return CallAdmission(
                admitted=False,
                input_tokens=input_tokens,
                output_cap=0,
                remaining_before=remaining_before,
                reason="budget_exhausted",
            )
        return CallAdmission(
            admitted=True,
            input_tokens=input_tokens,
            output_cap=effective_cap,
            remaining_before=remaining_before,
        )

    def record_call(
        self,
        admission: CallAdmission,
        response_output_tokens_estimated: int | None = None,
        *,
        response_output_tokens: int | None = None,
        output_tokens_estimated: int | None = None,
        output_tokens: int | None = None,
        snippet_input_tokens: int = 0,
        tool_argument_output_tokens: int = 0,
        tool_result_input_tokens: int = 0,
        provider_input_tokens: int | None = None,
        provider_output_tokens: int | None = None,
        provider_total_tokens: int | None = None,
    ) -> TurnAccounting:
        """Finalize one admitted call and append one accounting turn.

        All validation happens before the controller state is replaced.  This
        keeps a failed finalization from partially charging the task.
        """

        if not isinstance(admission, CallAdmission):
            raise TypeError("admission must be a CallAdmission")
        if not admission.admitted:
            raise BudgetInvariantError("cannot record a rejected admission")
        if admission.remaining_before != self.remaining:
            raise BudgetInvariantError("admission is stale")
        if admission.output_cap < 1:
            raise BudgetInvariantError("admitted output cap must be at least one")
        if admission.input_tokens < 0:
            raise BudgetInvariantError("admission input tokens must be nonnegative")

        response_tokens = _coalesce_counts(
            (
                ("response_output_tokens_estimated", response_output_tokens_estimated),
                ("response_output_tokens", response_output_tokens),
                ("output_tokens_estimated", output_tokens_estimated),
                ("output_tokens", output_tokens),
            ),
            field_name="response output tokens",
        )
        response_tokens = _require_integer(
            response_tokens,
            "response_output_tokens_estimated",
        )
        if response_tokens > admission.output_cap:
            raise BudgetInvariantError("response output exceeds admitted output cap")

        snippet_tokens = _require_integer(snippet_input_tokens, "snippet_input_tokens")
        argument_tokens = _require_integer(
            tool_argument_output_tokens,
            "tool_argument_output_tokens",
        )
        result_tokens = _require_integer(
            tool_result_input_tokens,
            "tool_result_input_tokens",
        )
        if snippet_tokens > admission.input_tokens:
            raise BudgetInvariantError("snippet input must be a request-input subset")
        if result_tokens > admission.input_tokens:
            raise BudgetInvariantError("tool-result input must be a request-input subset")
        if argument_tokens > response_tokens:
            raise BudgetInvariantError("tool arguments must be an output subset")

        provider_input = self._validate_provider_count(
            provider_input_tokens,
            "provider_input_tokens",
        )
        provider_output = self._validate_provider_count(
            provider_output_tokens,
            "provider_output_tokens",
        )
        provider_total = self._validate_provider_count(
            provider_total_tokens,
            "provider_total_tokens",
        )
        if (
            provider_total is not None
            and provider_input is not None
            and provider_output is not None
            and provider_total != provider_input + provider_output
        ):
            raise BudgetInvariantError(
                "provider_total_tokens must equal provider input plus output"
            )

        next_input = self._total_input + admission.input_tokens
        next_output = self._total_output + response_tokens
        next_total = next_input + next_output
        if next_total > self._limit:
            raise BudgetInvariantError("estimated token total exceeds limit")

        turn = TurnAccounting(
            turn_index=len(self._turns),
            request_input_tokens_estimated=admission.input_tokens,
            request_output_cap=admission.output_cap,
            response_output_tokens_estimated=response_tokens,
            snippet_input_tokens=snippet_tokens,
            tool_argument_output_tokens=argument_tokens,
            tool_result_input_tokens=result_tokens,
            provider_input_tokens=provider_input,
            provider_output_tokens=provider_output,
            remaining_after_turn=self._limit - next_total,
        )

        self._turns = (*self._turns, turn)
        self._provider_turn_totals = (*self._provider_turn_totals, provider_total)
        self._total_input = next_input
        self._total_output = next_output
        return turn.model_copy(deep=True)

    @staticmethod
    def _validate_provider_count(value: int | None, field_name: str) -> int | None:
        if value is None:
            return None
        return _require_integer(value, field_name)

    def count_tool_arguments(self, arguments: Mapping[str, Any] | Any) -> int:
        """Count canonical compact JSON tool arguments as output tokens."""

        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return self._tokenizer.count_text(serialized)

    @staticmethod
    def _tool_result_message(text: str) -> dict[str, str]:
        return {"role": "tool", "content": text}

    def fit_tool_result(self, text: str, reserved_output_tokens: int = 1) -> str:
        """Fit the largest tokenizer-boundary result while reserving output.

        The candidate is counted as a complete tool message, including the
        canonical request and message framing used by ``count_messages``.
        """

        if not isinstance(text, str):
            raise TypeError("tool result text must be a string")
        reserve = _require_integer(reserved_output_tokens, "reserved_output_tokens")
        available = self.remaining - reserve
        if available < 0:
            return ""

        text_token_count = _require_integer(
            self._tokenizer.count_text(text),
            "tool result token count",
        )

        def candidate(token_limit: int) -> str:
            if token_limit >= text_token_count:
                return text
            return self._tokenizer.truncate_text(text, token_limit)

        def cost(value: str) -> int:
            return self._tokenizer.count_messages([self._tool_result_message(value)])

        low = 0
        high = text_token_count + 1
        best = ""
        while low < high:
            midpoint = (low + high) // 2
            value = candidate(midpoint)
            if cost(value) <= available:
                best = value
                low = midpoint + 1
            else:
                high = midpoint
        return best

    def _provider_totals(self) -> tuple[int | None, int | None, int | None]:
        provider_input = _sum_if_complete(
            [turn.provider_input_tokens for turn in self._turns]
        )
        provider_output = _sum_if_complete(
            [turn.provider_output_tokens for turn in self._turns]
        )
        provider_total = _sum_if_complete(self._provider_turn_totals)
        if provider_total is None and provider_input is not None and provider_output is not None:
            provider_total = provider_input + provider_output
        return provider_input, provider_output, provider_total

    @property
    def token_accounting(self) -> TokenAccounting:
        """Build the persisted accounting model from finalized turns."""

        provider_input, provider_output, provider_total = self._provider_totals()
        estimated_total = self.total_tokens_estimated
        return TokenAccounting(
            tokenizer=getattr(self._tokenizer, "name", type(self._tokenizer).__name__),
            limit=self._limit,
            turns=[turn.model_copy(deep=True) for turn in self._turns],
            input_tokens_estimated=self._total_input,
            output_tokens_estimated=self._total_output,
            total_tokens_estimated=estimated_total,
            provider_input_tokens=provider_input,
            provider_output_tokens=provider_output,
            provider_total_tokens=provider_total,
            provider_minus_estimated={
                "input": 0
                if provider_input is None
                else provider_input - self._total_input,
                "output": 0
                if provider_output is None
                else provider_output - self._total_output,
                "total": 0
                if provider_total is None
                else provider_total - estimated_total,
            },
        )

    def accounting(self) -> TokenAccounting:
        """Return the current persisted accounting model."""

        return self.token_accounting


__all__ = [
    "BudgetController",
    "BudgetInvariantError",
    "CallAdmission",
    "Tokenizer",
]
