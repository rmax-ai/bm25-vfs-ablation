"""Acceptance tests for the VFS path security boundary."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from bm25_vfs_ablation.vfs.security import (
    VfsError,
    normalize_vfs_path,
    resolve_sandboxed,
)


def test_accepts_root_relative_posix_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "services" / "placeholder.md"
    target.parent.mkdir(parents=True)
    target.write_text("content\n", encoding="utf-8")

    assert normalize_vfs_path("/services/placeholder.md") == PurePosixPath(
        "/services/placeholder.md"
    )
    assert resolve_sandboxed(root, "/services/placeholder.md", "file") == target
    assert resolve_sandboxed(root, "services/placeholder.md", "file") == target


def test_rejects_dotdot_traversal_exact_code(tmp_path: Path) -> None:
    raw = "/services/../placeholder.md"

    with pytest.raises(VfsError) as caught:
        resolve_sandboxed(tmp_path, raw, "file")

    assert caught.value.code == "PATH_TRAVERSAL"
    assert raw in caught.value.message


def test_rejects_encoded_or_backslash_tricks(tmp_path: Path) -> None:
    rejected_paths = (
        "/services/%2e%2e/placeholder.md",
        r"/services\placeholder.md",
        "/services//placeholder.md",
        "https://placeholder.invalid/file",
        "C:/placeholder.md",
        "/services/\x00placeholder.md",
    )

    for raw in rejected_paths:
        with pytest.raises(VfsError) as caught:
            resolve_sandboxed(tmp_path, raw, "file")
        assert caught.value.code == "INVALID_PATH"


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "placeholder.md").write_text("outside\n", encoding="utf-8")
    (root / "link").symlink_to(outside, target_is_directory=True)
    raw = "/link/placeholder.md"

    with pytest.raises(VfsError) as caught:
        resolve_sandboxed(root, raw, "file")

    assert caught.value.code == "PATH_TRAVERSAL"
    assert raw in caught.value.message


def test_error_does_not_leak_host_root(tmp_path: Path) -> None:
    raw = "/missing/placeholder.md"

    with pytest.raises(VfsError) as caught:
        resolve_sandboxed(tmp_path, raw, "file")

    assert raw in caught.value.message
    assert str(tmp_path) not in caught.value.message


def test_type_expectation_codes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "placeholder.md").write_text("content\n", encoding="utf-8")
    (root / "directory").mkdir()

    with pytest.raises(VfsError) as file_as_directory:
        resolve_sandboxed(root, "/placeholder.md", "directory")
    assert file_as_directory.value.code == "NOT_A_DIRECTORY"

    with pytest.raises(VfsError) as directory_as_file:
        resolve_sandboxed(root, "/directory", "file")
    assert directory_as_file.value.code == "NOT_A_FILE"

    with pytest.raises(VfsError) as missing:
        resolve_sandboxed(root, "/missing", "either")
    assert missing.value.code == "NOT_FOUND"

    assert resolve_sandboxed(root, "/", "directory") == root
    with pytest.raises(VfsError) as root_as_file:
        resolve_sandboxed(root, "/", "file")
    assert root_as_file.value.code == "NOT_A_FILE"
