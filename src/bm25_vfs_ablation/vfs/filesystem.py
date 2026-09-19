"""Immutable, line-bounded access to a materialized corpus."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from bm25_vfs_ablation.corpus.schema import DocumentRecord
from bm25_vfs_ablation.retrieval.chunking import estimate_tokens
from bm25_vfs_ablation.vfs.security import VfsError, normalize_vfs_path, resolve_sandboxed


@dataclass(frozen=True, slots=True)
class ContentSlice:
    """A bounded, one-based inclusive slice of one materialized file."""

    path: str
    start_line: int
    end_line: int
    text: str
    token_count: int


def _expanded_path(root: Path) -> Path:
    return Path(os.path.expanduser(os.fspath(root)))


def _canonical_path(raw: str) -> str:
    normalized = normalize_vfs_path(raw)
    if normalized.is_absolute():
        return normalized.as_posix()
    return f"/{normalized.as_posix()}"


def _path_parts(canonical_path: str) -> tuple[str, ...]:
    if canonical_path == "/":
        return ()
    return tuple(canonical_path.removeprefix("/").split("/"))


def _document_paths(
    documents: Iterable[DocumentRecord],
) -> list[tuple[DocumentRecord, str, bytes]]:
    prepared: list[tuple[DocumentRecord, str, bytes]] = []
    seen_paths: set[str] = set()
    for document in documents:
        if not isinstance(document, DocumentRecord):
            raise TypeError("documents must contain DocumentRecord values")
        canonical_path = _canonical_path(document.path)
        if canonical_path == "/":
            raise ValueError("document paths must identify files")
        if canonical_path in seen_paths:
            raise ValueError(f"duplicate document path: {document.path}")
        seen_paths.add(canonical_path)
        prepared.append((document, canonical_path, document.content.encode("utf-8")))
    return prepared


def _ensure_empty_root(root: Path) -> None:
    try:
        root.lstat()
    except FileNotFoundError:
        root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise VfsError("NOT_FOUND", "sandbox root is unavailable") from exc

    if root.is_symlink():
        raise VfsError("PATH_TRAVERSAL", "sandbox root cannot be a symlink")
    if not root.is_dir():
        raise VfsError("NOT_A_DIRECTORY", "sandbox root is not a directory")

    try:
        next(root.iterdir())
    except StopIteration:
        return
    except OSError as exc:
        raise VfsError("NOT_FOUND", "sandbox root is unavailable") from exc
    raise ValueError("materialization root must be empty")


def materialize_corpus(root: Path, documents: Iterable[DocumentRecord]) -> None:
    """Write validated document content below a new, empty root directory."""

    prepared = _document_paths(documents)
    target_root = _expanded_path(root)
    _ensure_empty_root(target_root)

    for _, canonical_path, content_bytes in prepared:
        target = target_root.joinpath(*_path_parts(canonical_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(content_bytes)
        except FileExistsError as exc:
            raise ValueError(f"materialization target already exists: {canonical_path}") from exc


class VirtualFilesystem:
    """A read-only snapshot of validated files below a sandbox root."""

    __slots__ = ("_children", "_contents", "_directories", "_root")

    def __init__(self, root: Path, documents: Iterable[DocumentRecord]) -> None:
        self._root = _expanded_path(root)
        prepared = _document_paths(documents)
        resolve_sandboxed(self._root, "/", "directory")

        contents: dict[str, str] = {}
        directories: set[str] = {"/"}
        children: dict[str, set[str]] = {"/": set()}

        for document, canonical_path, expected_bytes in prepared:
            materialized_path = resolve_sandboxed(self._root, document.path, "file")
            try:
                actual_bytes = materialized_path.read_bytes()
            except OSError as exc:
                raise ValueError(f"could not read materialized file: {canonical_path}") from exc
            if actual_bytes != expected_bytes:
                raise ValueError(f"materialized content mismatch: {canonical_path}")

            contents[canonical_path] = document.content
            parent = "/"
            parts = _path_parts(canonical_path)
            for component_index, component in enumerate(parts):
                child = f"{parent.rstrip('/')}/{component}"
                is_file = component_index == len(parts) - 1
                if is_file:
                    children.setdefault(parent, set()).add(child)
                    continue
                directories.add(child)
                children.setdefault(parent, set()).add(f"{child}/")
                children.setdefault(child, set())
                parent = child

        self._contents: Mapping[str, str] = MappingProxyType(contents)
        self._directories = frozenset(directories)
        self._children: Mapping[str, tuple[str, ...]] = MappingProxyType(
            {path: tuple(sorted(entries)) for path, entries in children.items()}
        )

    def _resolve_content(self, path: str) -> tuple[str, str]:
        canonical_path = _canonical_path(path)
        resolve_sandboxed(self._root, path, "file")
        try:
            return canonical_path, self._contents[canonical_path]
        except KeyError as exc:
            raise VfsError("NOT_FOUND", f"VFS path {path} is not materialized") from exc

    @staticmethod
    def _validate_range(path: str, start_line: int, end_line: int, line_count: int) -> int:
        if (
            isinstance(start_line, bool)
            or isinstance(end_line, bool)
            or not isinstance(start_line, int)
            or not isinstance(end_line, int)
            or start_line <= 0
            or end_line <= 0
            or start_line > end_line
            or start_line > line_count
        ):
            raise VfsError(
                "RANGE_INVALID",
                f"VFS line range {start_line}:{end_line} rejected for {path}",
            )
        return min(end_line, line_count)

    def list(self, path: str = "/") -> list[str]:
        """Return the shallow, lexicographically sorted contents of a directory."""

        canonical_path = _canonical_path(path)
        resolve_sandboxed(self._root, path, "directory")
        if canonical_path not in self._directories:
            raise VfsError("NOT_FOUND", f"VFS path {path} is not materialized")
        return list(self._children.get(canonical_path, ()))

    def read(self, path: str, start_line: int, end_line: int) -> ContentSlice:
        """Read an inclusive positive line range, clamping only its end to EOF."""

        canonical_path, content = self._resolve_content(path)
        lines = content.splitlines(keepends=True)
        bounded_end = self._validate_range(path, start_line, end_line, len(lines))
        text = "".join(lines[start_line - 1 : bounded_end])
        return ContentSlice(
            path=canonical_path,
            start_line=start_line,
            end_line=bounded_end,
            text=text,
            token_count=estimate_tokens(text),
        )

    def cat(self, path: str) -> ContentSlice:
        """Return the complete content of one materialized file."""

        canonical_path, content = self._resolve_content(path)
        line_count = len(content.splitlines(keepends=True))
        return ContentSlice(
            path=canonical_path,
            start_line=1,
            end_line=line_count,
            text=content,
            token_count=estimate_tokens(content),
        )


__all__ = ["ContentSlice", "VirtualFilesystem", "materialize_corpus"]
