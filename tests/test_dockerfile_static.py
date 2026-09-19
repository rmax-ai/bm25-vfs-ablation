from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_declares_untested_host_constraint() -> None:
    assert "UNTESTED: build host has no Docker daemon" in _dockerfile_text()


def test_dockerfile_uses_frozen_lock() -> None:
    text = _dockerfile_text()
    metadata_copy = "COPY pyproject.toml uv.lock ./"
    sync_command = "RUN uv sync --frozen --no-dev"
    source_copy = "COPY src ./src"
    config_copy = "COPY configs ./configs"

    assert "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim" in text
    assert metadata_copy in text
    assert sync_command in text
    assert text.index(metadata_copy) < text.index(sync_command)
    assert text.index(sync_command) < text.index(source_copy)
    assert text.index(source_copy) < text.index(config_copy)

    lowered = text.lower()
    assert "copy . " not in lowered
    assert ".env" not in lowered
    assert "api_key=" not in lowered
    assert "password=" not in lowered
    assert "secret=" not in lowered


def test_dockerfile_runs_nonroot() -> None:
    user_lines = [
        line.strip()
        for line in _dockerfile_text().splitlines()
        if line.strip().startswith("USER ")
    ]

    assert user_lines == ["USER 10001"]


def test_dockerfile_entrypoint() -> None:
    assert 'ENTRYPOINT ["python", "-m", "bm25_vfs_ablation"]' in _dockerfile_text()
