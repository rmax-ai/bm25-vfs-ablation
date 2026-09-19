"""Deterministic JSONL artifact and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any


def _expanded_path(path: Path) -> Path:
    """Return a path with a user-home prefix expanded."""

    return Path(os.path.expanduser(os.fspath(path)))


def canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value with the repository's stable settings."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _jsonl_row(row: object) -> str:
    if not isinstance(row, dict):
        raise TypeError("JSONL rows must be JSON objects")
    return canonical_json(row)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""

    target = _expanded_path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> str:
    """Atomically write JSON objects as UTF-8 JSONL and return the file digest."""

    target = _expanded_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            for row in rows:
                stream.write(_jsonl_row(row))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                temporary_path.unlink()
        raise

    return sha256_file(target)


def append_jsonl(path: Path, row: dict) -> None:
    """Append exactly one flushed JSON object line to a JSONL file."""

    line = _jsonl_row(row)
    target = _expanded_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="") as stream:
        stream.write(line)
        stream.write("\n")
        stream.flush()


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects, reporting malformed input with its path and line."""

    target = _expanded_path(path)
    with target.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"invalid UTF-8 in {target} at line {line_number}: {error}"
                ) from error

            try:
                value: Any = json.loads(
                    line,
                    parse_constant=_reject_non_finite,
                )
            except ValueError as error:
                raise ValueError(
                    f"invalid JSON in {target} at line {line_number}: {error}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(
                    f"invalid JSON object in {target} at line {line_number}"
                )
            yield value


__all__ = [
    "append_jsonl",
    "canonical_json",
    "read_jsonl",
    "sha256_file",
    "write_jsonl_atomic",
]
