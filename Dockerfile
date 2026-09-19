# UNTESTED: build host has no Docker daemon
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/src"
RUN chown -R 10001:10001 /app
USER 10001

ENTRYPOINT ["python", "-m", "bm25_vfs_ablation"]
