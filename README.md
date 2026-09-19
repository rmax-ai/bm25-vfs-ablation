# BM25 VFS Workspace vs BM25 Snippet Ablation

A small, inspectable Python 3.12 proof of concept answering one question:

> Does giving an LLM a navigable, read-only virtual filesystem (VFS) with BM25-backed
> search/read tools improve multi-hop question answering over returning BM25 top-k snippets
> directly — when both conditions share the same corpus, BM25 index, model, question set,
> answer scorer, and total per-task token ceiling?

This repository is a research instrument, not a product. A negative result is first-class; the
failure mode to avoid is over-claiming. Deterministic dataset generation, exact evidence
provenance, and observable tool interactions are the core design constraints.

- `SPEC.md` — authoritative operator brief (the contract).
- `PLAN.md` — execution plan, build order, and full design tables.
- `reports/experiment.md` — research report generated from recorded runs (produced by the pipeline, not hand-written).

## Setup

```bash
uv sync --frozen     # or: make setup
uv run pytest        # or: make test — hermetic; mocked model, no network
```

Python 3.12+, `uv`-managed. Do not `pip install` directly; the lockfile is committed.

## One-command offline smoke

```bash
make smoke
# or: uv run python -m bm25_vfs_ablation smoke --workdir .smoke --seed 42
```

`smoke` generates a small dev+eval corpus, runs **both conditions** with the offline mock
model, evaluates, and reports — in one command, with no network. It prints the corpus and
task SHA-256 values, then writes a self-contained workspace under `.smoke/run-{seed}/`
(config, generated data, run records, 10 aggregate tables, 10 plots, report).

The mocked model is gold-aware by design (it demonstrates the machinery, reaching ceiling
success); every record it produces is stamped `scoring_details.mock_oracle_policy=true` so
mocked results can never be mistaken for performance evidence.

## Full pipeline

| step | direct CLI | make target |
|---|---|---|
| generate corpus + tasks | `uv run python -m bm25_vfs_ablation generate --tasks 200 --seed 42` | `make generate` |
| run primary experiment | `uv run python -m bm25_vfs_ablation run --config configs/default.yaml` | `make experiment` |
| run call-limit intervention | `uv run python -m bm25_vfs_ablation run --config configs/tool_call_intervention.yaml` | `make intervention` |
| evaluate runs | `uv run python -m bm25_vfs_ablation evaluate --runs results/runs.jsonl` | `make evaluate` |
| generate report | `uv run python -m bm25_vfs_ablation report --runs results/runs.jsonl --output reports/experiment.md` | `make report` |
| offline smoke | `uv run python -m bm25_vfs_ablation smoke --workdir .smoke --seed 42` | `make smoke` |
| full local gate | `uv run ruff check . && uv run pytest && uv run python -m bm25_vfs_ablation smoke --workdir .smoke --seed 42` | `make accept` |

All commands return 0 on success, 2 for user/config/schema errors, and 1 for runtime/model errors.
Resume is safe: `run` validates every existing record, skips completed run keys, and refuses
to continue past a truncated or duplicated file.

## Experiment design and fairness

Two primary conditions execute over one shared corpus and one shared BM25 index:

- **Condition A (snippets):** single model call; the prompt carries the top-k BM25 chunks with
  stable document/chunk identifiers and answer+citation instructions.
- **Condition B (VFS):** the model iterates with sandboxed read-only tools — `grep` (BM25-ranked,
  previews with line ranges), `read` (bounded line ranges), `cat` (budget-fitted whole file),
  `list` — under explicit stop rules (answer, budget exhausted, max calls, invalid-call threshold).

Held constant across conditions: model and sampling settings, question set, corpus, BM25
implementation/chunking, answer format, scoring logic, the **total per-task token ceiling**
(default 4096 estimated tokens; the same ceiling is enforced for both conditions), retry policy,
and concurrency policy. Condition order per task is randomized from a seeded SHA-256 so time/
provider bias cannot systematically favor one arm. Model responses are cached under a SHA-256
key covering condition, model config, messages, tool schema, corpus hash, and experiment hash.

Token accounting deliberately charges the **full request surface on every turn**: all messages
(repeated history), tool-call arguments, tool results, and — on every VFS turn — the complete
tool schema (≈558 estimated tokens for the four-tool schema). The VFS arm therefore spends part
of its budget on protocol rather than content; this fixed overhead asymmetry is reported as a
validity threat rather than waived, and it means low ceilings starve the VFS arm first.

The optional intervention experiment assigns **permitted** maximum tool calls from
`[0, 1, 2, 4, 8, 12]`; the dose axis is the assigned limit, never the observed call count.

## Importing a custom corpus

Point the config's `corpus.corpus_path` and `corpus.tasks_path` at your own JSONL files
(same schemas as `generate` writes) and run `run --config` as usual. Imports must be
**schema-complete**: exact fact spans, gold document/chunk ids consistent with the active
chunking configuration (the runner rejects gold chunk ids that are absent from the current
index at startup), and — for deterministic scoring — exhaustive acceptable answer variants.
There is no inference path from an unlabeled folder: arbitrary documents cannot be
auto-annotated into gold evidence, by design.

## Live model runs (operator-gated)

Tests and the smoke workflow are fully offline. Live runs are explicit and operator-controlled;
copy `.env.example` to `.env` and set exactly these variables (they substitute into the YAML only
when present):

| variable | meaning |
|---|---|
| `BM25_VFS_BASE_URL` | OpenAI-compatible endpoint base URL |
| `BM25_VFS_API_KEY` | API key (never commit; never logged) |
| `BM25_VFS_MODEL` | served model identifier |

Then run with `--model-provider openai_compatible` (make targets use the config default). The
provider's returned model identifier is recorded on every record, alongside timestamps and
cache-hit flags, so provider drift is visible in the artifacts. Live spend is intentionally
kept behind the operator gate.

## Docker

`Dockerfile` is a best-effort, ARM64-capable recipe (uv base image pinned to a resolved
multi-arch digest, lockfile-first `uv sync --frozen`, nonroot user, module entrypoint).
**Docker is untested on this build host — no Docker daemon is available**, and the repository
makes no claim that the image was built or run. Static assertions only
(`tests/test_dockerfile_static.py`); a full build/run check remains an operator follow-up.

## Limitations and threats to validity

- **Synthetic tasks.** Generated enterprise-flavored documents and multi-hop questions are not
  production traffic; realism, noise, and stakes are limited by construction.
- **Exact-answer scoring.** A single canonical answer plus variants can still penalize correct
  paraphrases (e.g., quoted or re-worded answers); imported tasks must supply exhaustive variants
  or accept conservative false negatives. The optional judge never replaces primary scoring.
- **Fixed protocol overhead.** The VFS arm pays tool-schema and history costs from the same
  ceiling; low ceilings starve it first. Prompt/accounting components are reported per turn.
- **Prompt sensitivity.** Small changes to instructions, context formatting, or tool descriptions
  may change outcomes; the frozen prompts here are one defensible point in that space.
- **Provider drift.** A different served model or later version may not reproduce observations;
  records include the returned model ID so drift is inspectable.
- **Tokenizer/accounting error.** Budgets are enforced with a documented local approximation and
  reconciled against provider usage metadata; approximation error is possible.
- **VFS interface quality.** Tool names, ranking, previews, and limits could favor or hinder the
  VFS arm independently of the mechanism.
- **Unequal prompt overhead.** Condition-specific instructions and tool schemas create
  different fixed overheads even under a common total ceiling.
- **Shared templates.** Tasks generated from shared templates reduce effective independence;
  paired analyses are the primary lens and template structure is disclosed.
- **Judge-model bias.** Not applicable by default: the secondary judge is disabled and
  deterministic scoring is primary.

**Associative warning (tool-call analysis).** Within the VFS condition, tool-call counts are
post-treatment variables: zero calls can mean premature answering, moderate calls adaptive
search, and excessive calls search failure. Regression and stratified analyses over call counts
are **associative only** and are labeled as such in every artifact. The only causal-flavored
evidence in this repository comes from the explicit **call-limit intervention**, where permitted
calls are assigned rather than observed.

## Repository layout

```
src/bm25_vfs_ablation/   corpus, retrieval, vfs, harnesses, models, evaluation, experiment, cli
tests/                   unit, integration, scientific-contract, and acceptance suites (all hermetic)
configs/                 default.yaml, tool_call_intervention.yaml
data/ results/ reports/  pipeline outputs (regenerable; gitignored except .gitkeep)
```

## Reproducibility anchors

- Seeded generation is byte-reproducible: regenerating seed 42 twice yields identical bytes
  (the acceptance suite enforces this without hardcoding golden hashes).
- Run records are append-only JSONL with a validated schema (49 fields), one record per
  task-condition run, including gold evidence, accessed evidence, tool traces, token accounting,
  latency, termination reason, and prompt/config hashes.
- The report generator is evidence-disciplined: every numeric paragraph is tagged
  `[OBSERVED]`, `[INFERENCE]`, `[CAUSAL-INTERVENTION]`, or `[SPECULATION]`.
