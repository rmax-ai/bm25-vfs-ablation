"""Deterministic regex-based token counting used by the experiment."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:['-][^\W_]+)*|[^\w\s]",
    flags=re.UNICODE,
)


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _tokens(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(_normalized(value))


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(key)
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped.get(key)
    return getattr(message, key, None)


class RegexTokenizer:
    """The frozen ``regex_v1`` tokenizer approximation."""

    name = "regex_v1"

    def count_text(self, text: str) -> int:
        """Count word and standalone-punctuation tokens in *text*."""

        return len(_tokens(text))

    def count_messages(self, messages: Sequence[Any]) -> int:
        """Count a request using the canonical message framing contract."""

        total = 2
        for message in messages:
            canonical = {
                "role": _message_value(message, "role"),
                "content": _message_value(message, "content"),
                "tool_calls": _message_value(message, "tool_calls"),
                "tool_call_id": _message_value(message, "tool_call_id"),
            }
            serialized = json.dumps(
                canonical,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            total += 4 + self.count_text(serialized)
        return total

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Return the longest source prefix ending on a token boundary."""

        if max_tokens <= 0:
            return ""

        if self.count_text(text) <= max_tokens:
            return text

        matches = list(_TOKEN_PATTERN.finditer(text))
        if not matches:
            return ""

        end = 0
        for match in matches[:max_tokens]:
            end = match.end()
        candidate = text[:end]

        # A normalization expansion can make a source-token prefix differ from
        # the count of the source text.  Back off until the public invariant is
        # true rather than returning a prefix over the requested limit.
        while end and self.count_text(candidate) > max_tokens:
            matches = matches[:-1]
            end = matches[-1].end() if matches else 0
            candidate = text[:end]
        return candidate


__all__ = ["RegexTokenizer"]
