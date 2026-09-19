# AGENTS.md — bm25-vfs-ablation conventions for coding agents

**SPEC.md is the sole source of truth.** When this file and SPEC.md disagree, SPEC.md wins.
`PLAN.md` is the authoritative build order; your task card in its §3 defines exactly what your
batch changes. Cite SPEC section numbers in commits and issues.

## Project DNA

This is a research instrument, not a product. It compares two LLM harnesses (BM25 snippets vs
BM25-backed VFS tools) on deterministically generated multi-hop QA tasks, under an identical
per-task token ceiling. Ground truth, evidence provenance, and every tool interaction must be
exactly measurable. A negative result is first-class; over-claiming is the failure mode.

**What must never happen:** weakening determinism of dataset generation; comparing conditions
under different token budgets; counting tool-call effects as causal without the call-limit
intervention; real network calls in tests (the mock model is the acceptance path).

## Batch discipline (hard rules)

- One batch changes at most 2–3 files (production and test counted together).
- Do not touch files outside your batch card. "Do not touch" lists are absolute.
- Do not run git commands beyond `git status`, `git diff`, `git diff --check`. The operator owns
  commits, pushes, and releases.
- Stop on any contract delta and report it as `blocked:design-review` — never implement it implicitly.
- No network access from code, tests, or tooling inside a batch.

## Commands

```bash
uv sync --dev                             # host-side install (sandbox has no network)
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                          # hermetic; mocked model
```

## Conventions

- Python 3.12+, uv-managed. Never `pip install` directly.
- Pydantic v2 idioms: `StrEnum` for taxonomies; `object | None` unions; `Field(default_factory=...)`
  instead of mutable defaults; timezone-aware `datetime.now(UTC)` (**never** `datetime.utcnow()`).
- All persisted contracts are Pydantic models. Every model gets a JSON round-trip test.
- Determinism: seeded `random.Random(seed)` only; no global RNG; no wall-clock or `time`-derived
  values in generated artifacts. Canonical JSON: `json.dumps(..., sort_keys=True)` + trailing
  newline; regeneration must be byte-identical (sha256 gate).
- Answer scoring is deterministic first (canonical answer + acceptable variants); an LLM judge, if
  ever enabled, is a secondary scorer only.
- Never hardcode secrets. Committed config uses `.env.example` only; real keys come from the
  environment in operator-run live sessions.
- Keep logs free of private chain-of-thought: store observable messages, tool calls, evidence
  accessed, answers, and concise justifications only.

## Pitfalls (learned from sibling projects)

- Do NOT put `__init__.py` under `tests/` — it shadows top-level packages in pytest's prepend
  import mode. `pythonpath = ["."]` is already set in pyproject.
- `ruff format` reformats Python code fences inside markdown — `extend-exclude = ["*.md"]` is set;
  don't remove it.
- Agent runs leave a `.serena/` artifact dir — gitignored; remove it before commits.
- Never `git add -A` while a background agent session is writing in the same tree.

## Git / board conventions

- One coherent commit per task: `feat: implements #N` (or `test:`/`fix:` as appropriate).
- State lives in GitHub issue labels: `status:*`, `type:{epic,story}`; the board is the resume
  mechanism.
- Operators own git operations; delegated coding agents write files only.
