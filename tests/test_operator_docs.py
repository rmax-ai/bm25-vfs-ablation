from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ENV_NAMES = frozenset(
    {
        "BM25_VFS_BASE_URL",
        "BM25_VFS_API_KEY",
        "BM25_VFS_MODEL",
    }
)
MAKE_TARGET_PATTERN = re.compile(r"(?m)^([a-z][a-z0-9_-]*):(?:[^\n]*)$")
ENV_ASSIGNMENT_PATTERN = re.compile(
    r"^(?:export\s+)?(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>.*)$"
)


def test_readme_required_topics() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8").casefold()

    required_fragments = (
        "bm25 vfs workspace vs bm25 snippet ablation",
        "## setup",
        "make generate",
        "make experiment",
        "make evaluate",
        "make report",
        "make smoke",
        "experiment design and fairness",
        "limitations and threats to validity",
        "associative warning",
        "live model runs",
        "docker is untested",
        "bm25_vfs_base_url",
        "bm25_vfs_api_key",
        "bm25_vfs_model",
    )

    for fragment in required_fragments:
        assert fragment in readme, f"README.md is missing required topic: {fragment}"


def test_makefile_required_targets() -> None:
    makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(MAKE_TARGET_PATTERN.findall(makefile))
    required_targets = {
        "setup",
        "lint",
        "test",
        "generate",
        "experiment",
        "intervention",
        "evaluate",
        "report",
        "smoke",
        "accept",
    }

    assert required_targets <= targets


def test_env_example_contract() -> None:
    env_example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    assignments: dict[str, str] = {}

    for line_number, line in enumerate(env_example.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_ASSIGNMENT_PATTERN.fullmatch(stripped)
        assert match is not None, f"unsupported .env.example line {line_number}"
        name = match.group("name")
        assert name in ALLOWED_ENV_NAMES, f"unexpected .env.example variable: {name}"
        assert name not in assignments, f"duplicate .env.example variable: {name}"
        assignments[name] = match.group("value").strip()

    assert set(assignments) == ALLOWED_ENV_NAMES
    assert all(not value for value in assignments.values())


def test_spec_present() -> None:
    spec = REPOSITORY_ROOT / "SPEC.md"
    assert spec.is_file()
