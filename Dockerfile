# UNTESTED: build host has no Docker daemon
# Base digest resolved from the GHCR registry API 2026-09-19 (multi-arch OCI index;
# linux/arm64 + linux/amd64). Pinning freezes the base; bump intentionally.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

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
