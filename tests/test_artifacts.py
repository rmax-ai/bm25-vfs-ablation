import hashlib

import pytest

from bm25_vfs_ablation.experiment.artifacts import (
    append_jsonl,
    canonical_json,
    read_jsonl,
    sha256_file,
    write_jsonl_atomic,
)


def test_canonical_json_is_byte_stable() -> None:
    first = {"z": ["café", 1], "a": {"second": False, "first": None}}
    second = {"a": {"first": None, "second": False}, "z": ["café", 1]}

    expected = '{"a":{"first":null,"second":false},"z":["café",1]}'
    assert canonical_json(first) == expected
    assert canonical_json(second) == expected

    with pytest.raises(ValueError):
        canonical_json({"not_finite": float("nan")})


def test_atomic_jsonl_has_trailing_newline(tmp_path) -> None:
    target = tmp_path / "nested" / "records.jsonl"
    rows = [{"b": 2, "a": 1}, {"message": "café"}]

    digest = write_jsonl_atomic(target, rows)

    assert target.read_bytes() == (
        b'{"a":1,"b":2}\n{"message":"caf\xc3\xa9"}\n'
    )
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()

    empty_target = tmp_path / "nested" / "empty.jsonl"
    write_jsonl_atomic(empty_target, [])
    assert empty_target.read_bytes() == b""


def test_append_jsonl_one_record(tmp_path) -> None:
    target = tmp_path / "append" / "records.jsonl"

    append_jsonl(target, {"id": "record-1", "value": 7})

    assert target.read_bytes() == b'{"id":"record-1","value":7}\n'
    assert list(read_jsonl(target)) == [{"id": "record-1", "value": 7}]


def test_read_jsonl_reports_line(tmp_path) -> None:
    target = tmp_path / "invalid.jsonl"
    target.write_text('{"ok":true}\n{"broken":\n', encoding="utf-8")

    with pytest.raises(ValueError) as error:
        list(read_jsonl(target))

    message = str(error.value)
    assert str(target) in message
    assert "line 2" in message


def test_sha256_file_known_vector(tmp_path) -> None:
    target = tmp_path / "hello.txt"
    target.write_bytes(b"hello")

    assert sha256_file(target) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
