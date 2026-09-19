.PHONY: setup lint test generate experiment intervention evaluate report smoke accept

setup:
	uv sync --frozen

lint:
	uv run ruff check .

test:
	uv run pytest

generate:
	uv run python -m bm25_vfs_ablation generate --tasks 200 --seed 42

experiment:
	uv run python -m bm25_vfs_ablation run --config configs/default.yaml

intervention:
	uv run python -m bm25_vfs_ablation run --config configs/tool_call_intervention.yaml

evaluate:
	uv run python -m bm25_vfs_ablation evaluate --runs results/runs.jsonl

report:
	uv run python -m bm25_vfs_ablation report --runs results/runs.jsonl --output reports/experiment.md

smoke:
	uv run python -m bm25_vfs_ablation smoke --workdir .smoke --seed 42

accept:
	uv run ruff check . && uv run pytest && uv run python -m bm25_vfs_ablation smoke --workdir .smoke --seed 42
