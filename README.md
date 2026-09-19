# BM25 VFS Workspace vs BM25 Snippet Ablation

Reproducible proof of concept: does giving an LLM a navigable, read-only virtual filesystem (VFS)
with BM25-backed search/read tools improve multi-hop question answering over returning BM25 top-k
snippets directly — when both conditions share the same corpus, BM25 index, model, task set, and
total token budget?

- `SPEC.md` — authoritative operator brief (the contract).
- `PLAN.md` — execution plan and build order (added by the planning pass).
- `reports/experiment.md` — research report, generated from experiment outputs.

Status: scaffold. Setup, experiment design, limitations, and interpretation sections are filled in
by the implementation (see SPEC §8–§15).

## Quickstart

```bash
uv sync --dev          # host-side dependency install
uv run pytest -q       # test suite (hermetic; mocked model, no network)
```

Full reproduction commands land with the implementation.
