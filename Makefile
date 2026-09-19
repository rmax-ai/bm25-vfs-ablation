.PHONY: setup test lint generate experiment experiment-intervention evaluate report

setup:
	uv sync --dev

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

generate:
	uv run python -m bm25_vfs_ablation generate --tasks 200 --seed 42

experiment:
	uv run python -m bm25_vfs_ablation run --config configs/default.yaml

experiment-intervention:
	uv run python -m bm25_vfs_ablation run --config configs/tool_call_intervention.yaml

evaluate:
	uv run python -m bm25_vfs_ablation evaluate --runs results/runs.jsonl

report:
	uv run python -m bm25_vfs_ablation report --runs results/runs.jsonl --output reports/experiment.md
