"""Security checks for paths entering the read-only virtual filesystem."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Literal

_EXPECTED_VALUES = frozenset({"file", "directory", "either"})
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")

ExpectedPath = Literal["file", "directory", "either"]


class VfsError(Exception):
    """A safe, machine-readable error raised at the VFS security boundary."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _raise_path_error(code: str, raw: object, reason: str) -> None:
    """Raise an error whose diagnostic contains only the logical path."""

    raise VfsError(code, f"VFS path {raw} rejected: {reason}")


def normalize_vfs_path(raw: str) -> PurePosixPath:
    """Validate and normalize a logical POSIX path without touching the host FS."""

    if not isinstance(raw, str):
        _raise_path_error("INVALID_PATH", raw, "path must be a string")
    if not raw:
        _raise_path_error("INVALID_PATH", raw, "path must not be empty")
    if "\x00" in raw:
        _raise_path_error("INVALID_PATH", raw, "path contains NUL")
    if "\\" in raw:
        _raise_path_error("INVALID_PATH", raw, "backslashes are not allowed")
    if "//" in raw:
        _raise_path_error("INVALID_PATH", raw, "repeated slashes are not allowed")
    if _DRIVE_PREFIX.match(raw) or _URI_SCHEME.match(raw):
        _raise_path_error("INVALID_PATH", raw, "drive prefixes and URI schemes are not allowed")
    if _PERCENT_ESCAPE.search(raw) or "%" in raw:
        _raise_path_error("INVALID_PATH", raw, "percent-encoded paths are not allowed")

    segments = raw.split("/")
    if any(segment in {".", ".."} for segment in segments):
        _raise_path_error("PATH_TRAVERSAL", raw, "dot segments are not allowed")

    return PurePosixPath(raw)


def _logical_parts(path: PurePosixPath) -> tuple[str, ...]:
    """Return path components without the optional VFS root marker."""

    if path == PurePosixPath("/"):
        return ()
    if path.is_absolute():
        return path.parts[1:]
    return path.parts


def _check_target_type(candidate: Path, raw: str, expected: str) -> None:
    """Enforce the caller's file/directory expectation without leaking host paths."""

    is_file = candidate.is_file()
    is_directory = candidate.is_dir()
    if expected == "file" and not is_file:
        _raise_path_error("NOT_A_FILE", raw, "expected a file")
    if expected == "directory" and not is_directory:
        _raise_path_error("NOT_A_DIRECTORY", raw, "expected a directory")
    if expected == "either" and not (is_file or is_directory):
        _raise_path_error("NOT_FOUND", raw, "path is not a file or directory")


def resolve_sandboxed(
    root: Path,
    raw: str,
    expected: ExpectedPath | str,
) -> Path:
    """Resolve a logical path below ``root`` while refusing symlinks and escapes."""

    normalized = normalize_vfs_path(raw)
    if expected not in _EXPECTED_VALUES:
        _raise_path_error("INVALID_PATH", raw, "expected must be file, directory, or either")

    sandbox_root = Path(root).expanduser()
    try:
        sandbox_root.lstat()
    except FileNotFoundError:
        _raise_path_error("NOT_FOUND", raw, "sandbox root does not exist")
    except OSError:
        _raise_path_error("NOT_FOUND", raw, "sandbox root is unavailable")

    if sandbox_root.is_symlink():
        _raise_path_error("PATH_TRAVERSAL", raw, "sandbox root cannot be a symlink")
    if not sandbox_root.is_dir():
        _raise_path_error("NOT_A_DIRECTORY", raw, "sandbox root is not a directory")

    # Keep the result rooted even when a caller supplies an absolute VFS path.
    candidate = sandbox_root.joinpath(*_logical_parts(normalized))
    components = _logical_parts(normalized)
    current = sandbox_root

    for index, component in enumerate(components):
        current /= component
        try:
            current.lstat()
        except FileNotFoundError:
            if index != len(components) - 1:
                _raise_path_error("NOT_FOUND", raw, "a path component does not exist")
            _raise_path_error("NOT_FOUND", raw, "path does not exist")
        except NotADirectoryError:
            _raise_path_error("NOT_A_DIRECTORY", raw, "a path component is not a directory")
        except OSError:
            _raise_path_error("NOT_FOUND", raw, "path is unavailable")

        if current.is_symlink():
            _raise_path_error("PATH_TRAVERSAL", raw, "symlink components are not allowed")
        if index != len(components) - 1 and not current.is_dir():
            _raise_path_error("NOT_A_DIRECTORY", raw, "a path component is not a directory")

    try:
        resolved_root = sandbox_root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        _raise_path_error("PATH_TRAVERSAL", raw, "path could not be safely resolved")

    if not resolved_candidate.is_relative_to(resolved_root):
        _raise_path_error("PATH_TRAVERSAL", raw, "resolved path escapes the sandbox")

    _check_target_type(resolved_candidate, raw, expected)
    return resolved_candidate
