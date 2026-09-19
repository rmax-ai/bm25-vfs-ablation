"""Acceptance tests for the exact estimated-token budget controller."""

from __future__ import annotations

import json

from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.models.tokenization import RegexTokenizer


def _user_messages(content: str = "alpha beta") -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def test_admit_exact_boundary() -> None:
    tokenizer = RegexTokenizer()
    messages = _user_messages()
    input_tokens = tokenizer.count_messages(messages)
    controller = BudgetController(input_tokens + 4, tokenizer)

    admission = controller.plan_call(messages, output_cap=4)

    assert admission.admitted is True
    assert admission.input_tokens == input_tokens
    assert admission.output_cap == 4
    assert admission.remaining_before == input_tokens + 4
    turn = controller.record_call(admission, response_output_tokens_estimated=4)
    assert turn.remaining_after_turn == 0
    assert controller.remaining == 0


def test_reject_one_token_over() -> None:
    tokenizer = RegexTokenizer()
    messages = _user_messages()
    input_tokens = tokenizer.count_messages(messages)
    controller = BudgetController(input_tokens, tokenizer)

    admission = controller.plan_call(messages, output_cap=1)

    assert admission.admitted is False
    assert admission.input_tokens == input_tokens
    assert admission.output_cap == 0
    assert admission.reason == "budget_exhausted"


def test_rejected_call_does_not_mutate() -> None:
    tokenizer = RegexTokenizer()
    messages = _user_messages()
    input_tokens = tokenizer.count_messages(messages)
    controller = BudgetController(input_tokens, tokenizer)
    before = (controller.remaining, controller.turns, controller.total_tokens_estimated)

    admission = controller.plan_call(messages, output_cap=1)

    assert admission.admitted is False
    assert (controller.remaining, controller.turns, controller.total_tokens_estimated) == before


def test_multiturn_accounting_no_double_count() -> None:
    tokenizer = RegexTokenizer()
    first_messages = _user_messages("alpha")
    second_messages = [*first_messages, {"role": "assistant", "content": "beta"}]
    first_input = tokenizer.count_messages(first_messages)
    second_input = tokenizer.count_messages(second_messages)
    controller = BudgetController(first_input + 1 + second_input + 2, tokenizer)

    first = controller.plan_call(first_messages, output_cap=1)
    first_turn = controller.record_call(
        first,
        response_output_tokens_estimated=1,
        snippet_input_tokens=1,
    )
    second = controller.plan_call(second_messages, output_cap=3)
    second_turn = controller.record_call(
        second,
        response_output_tokens_estimated=2,
        tool_result_input_tokens=2,
    )

    accounting = controller.token_accounting
    assert second.output_cap == 2
    assert accounting.input_tokens_estimated == first_input + second_input
    assert accounting.output_tokens_estimated == 3
    assert accounting.total_tokens_estimated == first_input + second_input + 3
    assert accounting.total_tokens_estimated == sum(
        turn.request_input_tokens_estimated + turn.response_output_tokens_estimated
        for turn in accounting.turns
    )
    assert first_turn.snippet_input_tokens == 1
    assert second_turn.tool_result_input_tokens == 2


def test_tool_arguments_count_output_and_replayed_input() -> None:
    tokenizer = RegexTokenizer()
    first_messages = _user_messages("inspect the record")
    arguments = {"path": "/placeholder/item.txt", "start_line": 1, "end_line": 2}
    argument_json = json.dumps(
        arguments,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    argument_tokens = tokenizer.count_text(argument_json)
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-001",
                "type": "function",
                "function": {"name": "read", "arguments": argument_json},
            }
        ],
    }
    tool_message = {
        "role": "tool",
        "content": "record value",
        "tool_call_id": "call-001",
    }
    replayed_messages = [*first_messages, assistant_message, tool_message]
    first_input = tokenizer.count_messages(first_messages)
    replayed_input = tokenizer.count_messages(replayed_messages)
    controller = BudgetController(first_input + argument_tokens + replayed_input + 2, tokenizer)

    first = controller.plan_call(first_messages, output_cap=argument_tokens)
    first_turn = controller.record_call(
        first,
        response_output_tokens_estimated=argument_tokens,
        tool_argument_output_tokens=argument_tokens,
    )
    second = controller.plan_call(replayed_messages, output_cap=2)
    second_turn = controller.record_call(
        second,
        response_output_tokens_estimated=2,
        tool_result_input_tokens=replayed_input,
    )

    assert first_turn.tool_argument_output_tokens == argument_tokens
    assert first_turn.response_output_tokens_estimated == argument_tokens
    assert second_turn.tool_result_input_tokens == replayed_input
    assert controller.total_tokens_estimated == (
        first_input + argument_tokens + replayed_input + 2
    )


def test_fit_tool_result_reserves_final_token() -> None:
    tokenizer = RegexTokenizer()
    messages = _user_messages("seed")
    input_tokens = tokenizer.count_messages(messages)
    empty_result_tokens = tokenizer.count_messages([{"role": "tool", "content": ""}])
    one_result_tokens = tokenizer.count_messages([{"role": "tool", "content": "alpha"}])
    two_result_tokens = tokenizer.count_messages([{"role": "tool", "content": "alpha beta"}])
    assert empty_result_tokens < one_result_tokens < two_result_tokens

    controller = BudgetController(input_tokens + 1 + one_result_tokens + 1, tokenizer)
    admission = controller.plan_call(messages, output_cap=1)
    controller.record_call(admission, response_output_tokens_estimated=1)

    fitted = controller.fit_tool_result("alpha beta", reserved_output_tokens=1)

    assert fitted == "alpha"
    assert one_result_tokens + 1 <= controller.remaining
    assert two_result_tokens + 1 > controller.remaining
    next_call = controller.plan_call([{"role": "tool", "content": fitted}], output_cap=1)
    assert next_call.admitted is True
    assert next_call.output_cap == 1


def test_provider_reconciliation_does_not_change_limit() -> None:
    tokenizer = RegexTokenizer()
    messages = _user_messages("provider check")
    input_tokens = tokenizer.count_messages(messages)
    controller = BudgetController(input_tokens + 3, tokenizer)
    admission = controller.plan_call(messages, output_cap=3)

    controller.record_call(
        admission,
        response_output_tokens_estimated=3,
        provider_input_tokens=input_tokens + 2,
        provider_output_tokens=3,
        provider_total_tokens=input_tokens + 5,
    )
    accounting = controller.token_accounting

    assert accounting.limit == input_tokens + 3
    assert controller.remaining == 0
    assert accounting.provider_input_tokens == input_tokens + 2
    assert accounting.provider_output_tokens == 3
    assert accounting.provider_total_tokens == input_tokens + 5
    assert accounting.provider_minus_estimated == {"input": 2, "output": 0, "total": 2}
    assert type(accounting).model_validate_json(accounting.model_dump_json()) == accounting
