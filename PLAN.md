# Implementation plan: BM25 VFS Workspace vs BM25 Snippet Ablation

This plan is normative for implementation.
`SPEC.md` remains normative for scientific intent; where it leaves an implementation choice open, this plan freezes the choice.
Keywords MUST, MUST NOT, SHOULD, and MAY are contractual.

## §0 Settled decisions & overrides

| decision | choice | rationale | alternative rejected |
|---|---|---|---|
| Python/platform | Python `>=3.12,<3.13`; Linux aarch64 is the acceptance host | Matches operator-confirmed runtime and makes lock resolution bounded | Supporting older Python or gating other platforms |
| environment | `uv sync --frozen`; operator creates and commits `uv.lock` before dispatch | Agents have no network and must consume one immutable environment | Agent-installed dependencies |
| package layout | `src/bm25_vfs_ablation`; entry point `python -m bm25_vfs_ablation` | Required src layout and one stable CLI surface | Flat package or console-script-only entry |
| BM25 library | `rank-bm25`, class `BM25Okapi` | Pure Python/NumPy, small, inspectable, aarch64/Python 3.12 compatible; corpus is PoC-sized | `bm25s`, whose speed is unnecessary and whose extra serialization/tokenization choices add variability |
| BM25 parameters | Okapi `k1=1.5`, `b=0.75`, `epsilon=0.25`; config keys `retrieval.k1/b/epsilon` | Library defaults made explicit and shared by both conditions | Hidden library defaults or condition-specific tuning |
| ranking ties | Sort by `(-score, chunk_id)` after scoring every chunk | Stable across insertion order and NumPy sorting details | Relying on `get_top_n` tie behavior |
| retrieval query tokenization | Unicode NFKC, `casefold()`, regex `[^\W_]+(?:['-][^\W_]+)*`, no stemming, no stopword removal | Deterministic, language-neutral-enough, dependency-free, preserves negation and identifiers | NLTK/spaCy, locale rules, stopword lists |
| token approximation | Same NFKC regex plus standalone punctuation; count every regex token; tokenizer name `regex_v1` | Deterministic offline enforcement for any provider | `tiktoken` model coupling and another binary/data dependency |
| provider usage | Enforce using estimates before calls; record estimates and provider counts separately; never retroactively permit an over-budget call | Fair ceiling remains deterministic while reconciliation remains observable | Trusting possibly absent/inconsistent provider metadata |
| chunking | `chunk_size_tokens=180`, `chunk_overlap_tokens=30`; line-aware greedy chunks, long lines split at token boundaries | Supports evidence localization while keeping multi-hop facts separate | Character chunks or adaptive condition-specific chunks |
| chunk identity | `chunk_id = "{doc_id}::c{zero_based_index:04d}"` | Stable, readable provenance | Content-only opaque IDs |
| line numbering | One-based inclusive `start_line/end_line`; root-relative POSIX paths | Human-readable traces and unambiguous overlap | Zero-based or half-open line APIs |
| synthetic corpus defaults | corpus version `synthetic-v1`; 50 dev + 200 eval tasks; seed 42; corpus entities 80; distractors/task 6; hops 2–4 | Meets minimums with a modest inspectable corpus | Huge benchmark or random split only |
| split rule | Dev uses template IDs ending in even template ordinal; eval uses odd ordinal plus disjoint entity tuples; generator validates no tuple crosses splits | Structural generalization rather than random-row leakage | Random shuffle/slice |
| generator RNG | Every public generation call creates only `random.Random(seed)`; child streams use SHA-256-derived integer seeds; never global `random` or NumPy RNG | Reproducible across process order; isolates stages | Global seeding or time-derived seeds |
| generated JSON | UTF-8, one compact JSON object/line, `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, final newline | Byte reproducibility and SHA-256 gates | Pretty JSON or unstable key order |
| source leakage | Questions never occur verbatim in documents; aliases required; no fact IDs rendered in document text; generator rejects any one chunk covering all required facts | Preserves multi-hop character | Embedding provenance markers in visible text |
| provenance | Corpus JSONL stores hidden fact spans as metadata beside document content; harness prompts receive content/path only | Exact evidence scoring without exposing gold labels | Parsing fact markers from model-visible text |
| imported data | Corpus and tasks use the same validated JSONL schemas; imported tasks must provide all gold fields; no automatic gold inference | Exact provenance cannot be guessed safely | Heuristic labeling of arbitrary text |
| conditions | Enum values exactly `snippets`, `vfs`; optional design flags do not create a third primary condition | Exactly two primary conditions | Naming oracle/intervention as harnesses |
| answer format | Model must return one JSON object: `answer` string, `citations` array of stable IDs, `justification` string ≤240 chars | Deterministic parsing while avoiding chain-of-thought storage | Free-form primary answers |
| primary answer scoring | Parse JSON; normalize candidate by NFKC, casefold, trim, collapse whitespace, strip only terminal `.?!`; success iff equal to normalized canonical answer or variant | Transparent exact/structured scoring | Substring, fuzzy, embedding, or LLM judge as primary |
| optional judge | Interface reserved, disabled by default, mock-only tests; no live judge implementation in this build | Brief permits only secondary judge and acceptance must be offline | Judge-dependent correctness |
| citation correctness | Citation is correct iff it names a gold `doc_id` or `chunk_id`; precision is correct/unique cited IDs; recall is covered gold docs/required gold docs | Deterministic and explainable | Semantic citation judging |
| evidence accessed | A fact is accessed when model-visible returned content overlaps any one-based fact line span: snippet content, successful `grep` preview, `read`, or `cat`; manifests/path lists do not count | Measures content exposure, independent of citation or inference | Counting retrieval rank or filenames as evidence |
| grep semantics | BM25-rank chunks matching normalized query terms, optional path prefix filter, return at most `max_results`, each preview bounded to matching line ±1 line | Uses shared index while producing compact navigable evidence | Filesystem regex grep with unrelated ranking |
| list tool | Included; name literal `list`; shallow sorted directory listing only | Corpus navigation is useful and low-risk | Omitting a requested navigation primitive |
| VFS sandbox | Accept absolute VFS paths like `/services/a.md` or relative paths; convert `\` to rejection, POSIX normalize, reject empty/NUL, `.`/`..` segments, symlinks, non-files, and resolved paths outside root | Concrete read-only boundary | `Path.resolve()` alone or host absolute paths |
| VFS errors | Exact codes: `INVALID_PATH`, `PATH_TRAVERSAL`, `NOT_FOUND`, `NOT_A_FILE`, `NOT_A_DIRECTORY`, `RANGE_INVALID`, `RESULT_LIMIT_INVALID`, `PER_CALL_LIMIT`, `BUDGET_INSUFFICIENT` | Stable tests/traces | Exception-message matching |
| tool result envelope | `{ok, tool, request_id, content, paths, line_ranges, token_count, latency_ms, error_code, error_message}` | One observable schema across tools | Ad hoc strings |
| call limits | Default VFS `max_tool_calls=8`, `invalid_call_limit=3`; intervention grid `[0,1,2,4,8,12]` | Covers specified dose levels and finite primary behavior | Unlimited agent loop |
| total token ceiling | Default `4096` estimated LLM tokens/task in both conditions; `max_answer_tokens=256`; tool result per call `768`; snippet retrieval context `1800` | Leaves prompt/reasoning room and applies identical ceiling | Equal context only or unconstrained VFS |
| accounting boundary | `total_tokens = sum(turn.input_tokens_estimated + turn.output_tokens_estimated)`; arguments are generated output and, when replayed, input; snippets/tool results are classified subsets of request input, never added twice | Mirrors provider charging and avoids hidden/double costs | Counting unique text once across turns |
| pre-call admission | Set effective output cap to `min(requested_cap, remaining-input_estimate)` and admit only when it is at least 1; otherwise terminate `budget_exhausted`; tool results preserve that 1-token minimum | Ceiling cannot be exceeded while usable residual budget is not discarded | Call then truncate/provider reconcile |
| post-call reconciliation | Persist provider input/output/total if returned and deltas from estimates; enforcement and `total_tokens` remain estimated fields | Comparable offline/online behavior | Replacing enforced totals after execution |
| snippet fairness | Retrieval context is greedily truncated by ranked whole chunks to context cap and total pre-call admission; one call only | Stable IDs and no partial evidence ambiguity | Character truncation or second call |
| VFS initial candidates | Sorted top 5 document IDs from shared BM25 results, scores included, no content; oracle replaces candidates with all gold document IDs | Minimal navigation hint without free evidence | Initial snippet text in VFS |
| execution order | Per task, seeded SHA-256 of `(experiment_seed, task_id)` chooses `snippets,vfs` or reverse; persist `condition_order` | Reduces provider/time bias reproducibly | Global block order |
| concurrency/retry | `concurrency=1`; `max_attempts=2`; retry only timeout/HTTP 429/5xx; deterministic exponential delays 0.25 then 0.5 seconds, no jitter; mock never sleeps | Comparable and simple | Parallel default or random jitter |
| cache | SHA-256 of canonical JSON containing condition, full model config excluding API key, messages, tool schema, corpus SHA-256, experiment config SHA-256 | Matches brief, avoids secret material, stable reuse | URL/key in hash or prompt-only cache |
| cache storage | One JSON file per key under `results/cache/{key}.json`, atomic temp-file replace; cache hit still creates run record with `cache_hit=true` | Resume-safe and inspectable | SQLite or pickle |
| model client | Minimal `httpx` OpenAI-compatible `/chat/completions` client plus dependency-free scripted mock | Avoids SDK churn/frameworks; endpoint is common | `openai` SDK, LangChain |
| returned model ID | Persist response `model`; mock returns `mock-scripted-v1`; requested and returned identifiers are separate | Provider drift is measurable | Assuming requested model equals served model |
| mock model | `ScriptedMockClient` consumes test-owned `MockPolicy` actions (`tool_call` or `final`) keyed by task/condition; smoke derives a policy from task gold only in explicit mock mode | Exact single/multi-turn offline assertions without pretending to model intelligence | Keyword heuristic masquerading as inference |
| run identity | Experiment ID supplied or deterministic `exp-{config_hash[:12]}-{seed}`; run key `{experiment_id}:{task_id}:{condition}:{design_cell}` | Resume and paired joins are stable | Timestamps or UUIDs |
| timestamps | Runtime traces use UTC RFC3339 wall timestamps; generated corpus/tasks never do; hashes exclude runtime timestamps and latencies | Observability without breaking artifact determinism | Time in generated data or cache keys |
| resume | Append-only runs JSONL; validate every existing row; skip only complete unique run keys; conflicting duplicate keys fail | Interruption safety without silent overwrite | Rewrite-in-place |
| paired bootstrap | 10,000 paired task resamples with `numpy.random.default_rng(20250308)` used only in evaluation; percentile 2.5/97.5%; point estimate VFS minus snippets | Standard paired uncertainty, fixed seed | Unpaired bootstrap or BCa complexity |
| McNemar | Exact two-sided binomial test on discordant pairs via `scipy.stats.binomtest(min(b,c), b+c, 0.5)`; p=1 when no discordance | Correct for small samples and explicit variant | Asymptotic chi-square |
| regression | `statsmodels` Logit/GLM binomial with HC3 robust covariance; predictors calls, calls², initial recall, hops, category dummies, tokens, distractors; report failure/convergence | Meets associative nonlinear requirement | Calling it mediation or silently dropping failures |
| stratification | Token quartiles computed on pooled paired observations; recall bands `0`, `(0,1)`, `1`; hop/category exact; report paired difference where both observations occupy same band | Prespecified comparable bands | Post-hoc cut points |
| intervention | Factor is permitted max calls `[0,1,2,4,8,12]`; same task/token ceiling/order seed; actual calls reported separately | Stronger causal test is assignment to limit | Interpreting observed calls causally |
| oracle | `oracle_mode` boolean, off by default; stored as design cell, reported separately, never pooled with primary | Partial retrieval/reasoning separation | Gold content injection or mixed headline result |
| aggregate format | CSV only, no Parquet | Avoids `pyarrow`; inspectable | Parquet dependency |
| plots | Matplotlib noninteractive `Agg`, fixed style/size/DPI, sorted categories; PNG outputs listed in §2 | Deterministic-enough visual artifacts on headless ARM | Seaborn and GUI backends |
| report | Generated Markdown, observed/inference/causal/speculation badges; no fabricated result when cell absent | Enforces epistemic framing | Handwritten headline conclusions |
| Docker | Deliver `Dockerfile`; static-content test only; explicitly untestable on host and never gates acceptance | No daemon exists | Docker build/run acceptance |
| CI | No workflow authored | Explicitly not required; avoids another maintenance surface | Non-gating workflow that tasks cannot exercise |
| dependencies | Runtime: `pydantic`, `typer`, `PyYAML`, `rank-bm25`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `httpx`; dev: `pytest`, `ruff` | Exact minimal package names; all have Python 3.12/aarch64 support or pure-Python path | `bm25s`, `openai`, `tiktoken`, `seaborn`, `pyarrow`, agent frameworks |
| dependency versions | Operator resolves latest compatible versions once and commits exact `uv.lock`; task code targets only public stable APIs named here | Requirement says assume latest resolvable PyPI versions, while offline workers need exact installed bytes | Inventing unverified version numbers without network |
| operator-owned files | Tasks never edit `pyproject.toml`, `Makefile`, `.gitignore`, `SPEC.md`, `README.md`, `AGENTS.md`, `.env.example`, or `uv.lock` | Ownership boundary is explicit | Agents opportunistically fixing scaffolding |
| build exclusions | No live headline experiment, live judge validation, website, CI, Docker validation, or publication claim | Credentials/daemon/external validity are operator-gated | Making offline acceptance depend on external systems |

### §0.1 Canonical configuration schema and defaults

Unknown keys are errors at every level; all models use Pydantic `extra="forbid"`.
Paths are repository-relative POSIX strings and are resolved from repository root, never current shell directory.
Environment substitution is allowed only for `BM25_VFS_BASE_URL`, `BM25_VFS_API_KEY`, and `BM25_VFS_MODEL`.
An absent environment variable leaves the YAML value unchanged; secrets are redacted from records.

`configs/default.yaml` MUST encode exactly these values:

```yaml
schema_version: 1
experiment:
  experiment_id: null
  seed: 42
  conditions: [snippets, vfs]
  token_ceiling: 4096
  concurrency: 1
  oracle_mode: false
  resume: true
  output_runs: results/runs.jsonl
corpus:
  corpus_path: data/generated/corpus.jsonl
  tasks_path: data/generated/tasks.jsonl
  corpus_version: synthetic-v1
chunking:
  chunk_size_tokens: 180
  chunk_overlap_tokens: 30
retrieval:
  implementation: rank_bm25_okapi
  top_k: 8
  k1: 1.5
  b: 0.75
  epsilon: 0.25
  query_tokenizer: unicode_word_v1
harness:
  retrieval_context_tokens: 1800
  max_answer_tokens: 256
  max_tool_calls: 8
  invalid_call_limit: 3
  tool_result_tokens_per_call: 768
  initial_candidate_documents: 5
model:
  provider: mock
  base_url: http://127.0.0.1:8000/v1
  api_key: null
  model: mock-scripted-v1
  temperature: 0.0
  top_p: 1.0
  seed: 42
  timeout_seconds: 60.0
  max_attempts: 2
  tokenizer: regex_v1
evaluation:
  bootstrap_resamples: 10000
  bootstrap_seed: 20250308
  confidence_level: 0.95
  judge_enabled: false
  estimated_input_usd_per_million: 0.0
  estimated_output_usd_per_million: 0.0
```

`configs/tool_call_intervention.yaml` MUST equal the default except for this additive top-level block and output path:

```yaml
experiment:
  output_runs: results/intervention_runs.jsonl
intervention:
  enabled: true
  max_tool_calls: [0, 1, 2, 4, 8, 12]
```

The default config logically supplies `intervention.enabled=false` and `intervention.max_tool_calls=[]` when the block is absent.
CLI overrides are applied after YAML validation, revalidated, canonicalized, and included in `experiment_config_sha256`.

### §0.2 Run-record top-level schema

Every run line contains every key below; unavailable scalar values are `null`, unavailable collections are empty, and no key is omitted.

| field | type and meaning |
|---|---|
| `schema_version` | integer `1` |
| `run_key` | stable grammar defined in §2 |
| `experiment_id` | stable experiment ID |
| `design_cell` | `primary`, `oracle`, or `max_calls_{n}` |
| `task_id` | source task ID |
| `condition` | `snippets` or `vfs` |
| `condition_order` | integer 0 or 1 within task/cell |
| `model_config` | redacted object containing provider, base_url, requested model, sampling, seed, tokenizer |
| `requested_model_id` | configured model string |
| `returned_model_id` | provider-returned model string |
| `seed` | experiment integer seed |
| `corpus_version` | corpus version string |
| `question` | exact task question |
| `final_answer` | raw observable model response text |
| `parsed_answer` | object `{answer,citations,justification}` or `null` |
| `correctness` | primary boolean |
| `secondary_judge` | object `{enabled,score,rationale,model_id}`; null values when disabled |
| `scoring_details` | normalization, matched variant, parse status, citation metrics |
| `citations` | deduplicated model citation strings in first-seen order |
| `initial_retrieved_documents` | ordered `{doc_id,best_score,rank}` objects |
| `initial_retrieved_chunks` | ordered `{chunk_id,doc_id,score,rank,start_line,end_line}` objects |
| `gold_evidence` | `{fact_ids,doc_ids,chunk_ids}` sorted arrays |
| `accessed_evidence` | `{fact_ids,doc_ids,chunk_ids,line_ranges}` sorted arrays/objects |
| `retrieval_metrics` | initial chunk/doc recall-at-k floats and k |
| `evidence_metrics` | final recall, precision, distinct gold facts, all_required_accessed, citation precision/recall |
| `tool_trace` | ordered tool request/result records; empty for snippets |
| `tool_call_count` | number of requested tool calls, including invalid calls |
| `max_tool_calls_permitted` | effective assigned integer for VFS; `null` for snippets |
| `tool_calls_by_type` | all four tool keys with integer counts |
| `successful_tool_calls` | integer |
| `invalid_tool_calls` | integer |
| `repeated_tool_calls` | integer exact duplicate canonical request count after first |
| `unique_files_read` | sorted paths whose content was returned |
| `unique_line_ranges_accessed` | canonical merged path/range objects |
| `time_to_first_gold_fact_ms` | elapsed monotonic milliseconds or null |
| `calls_to_complete_gold_coverage` | call ordinal or null |
| `token_accounting` | object defined in §2 with ordered turns |
| `total_tokens` | estimated enforced input plus output total |
| `latency` | `{wall_ms,model_ms,retrieval_ms,tool_ms}` nonnegative floats |
| `estimated_cost_usd` | config-price calculation from estimated input/output |
| `termination_reason` | frozen enum |
| `errors` | ordered `{stage,code,message,retryable}` objects |
| `prompt_sha256` | SHA-256 of canonical initial messages |
| `config_sha256` | canonical complete config hash |
| `corpus_sha256` | exact corpus file bytes hash |
| `cache_key` | cache digest or null before a call |
| `cache_hit` | boolean |
| `started_at` | UTC RFC3339 runtime timestamp |
| `finished_at` | UTC RFC3339 runtime timestamp |

## §1 System restatement

The repository is an offline-testable research instrument comparing two harnesses on identical synthetic or imported corpora and tasks.
The snippet condition performs one shared-index BM25 search, places bounded top-ranked chunks in one prompt, and obtains one structured answer without tools.
The VFS condition receives only the question, tool contract, remaining budget, and ranked candidate paths, then iterates deterministic `grep`, `read`, `cat`, and `list` calls until it answers or reaches a frozen stop rule.
Both conditions use the same `Corpus`, `Chunker`, and `BM25Index` objects, the same model/sampling/retry settings, answer schema, scoring code, task token ceiling, and randomized within-task order.

Synthetic generation creates versioned enterprise documents, hidden fact-span provenance, keyword-sharing distractors, and tasks requiring two to four facts across documents.
Generation validates structural dev/eval separation, question non-leakage, exact IDs, and that no one chunk contains every required fact.
Imported JSONL must satisfy the identical schemas and validation rules.

Every run becomes one schema-validated append-only JSONL record.
Evaluation joins the paired conditions, separately scores answer correctness, initial retrieval, accessed evidence, citations, tool behavior, tokens, latency, cost, and reproducible failure classes.
It produces CSV tables, ten required PNGs, and a Markdown report that distinguishes observations, inference, causal intervention evidence, and speculation.
The optional assigned maximum-tool-call grid and optional oracle candidate mode are separate design cells and never contaminate primary estimates.

Acceptance uses only the scripted mock and local files: lint, tests, a small CLI smoke run, and byte-identical regeneration.
The HTTP model client exists for later operator runs, but live inference, live judging, Docker execution, and scientific headline claims are not build gates.

## §2 Contract freeze table

### §2.1 Modules and public interfaces

| module | frozen public interface |
|---|---|
| `bm25_vfs_ablation.config` | `load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> AppConfig`; Pydantic models `AppConfig`, `ExperimentConfig`, `CorpusConfig`, `ChunkingConfig`, `RetrievalConfig`, `HarnessConfig`, `ModelConfig`, `EvaluationConfig`, `InterventionConfig` |
| `corpus.schema` | `FactSpan`, `DocumentRecord`, `TaskRecord`; `validate_dataset(documents, tasks, chunker) -> None` |
| `corpus.generator` | `GenerationConfig`; `generate_dataset(config: GenerationConfig) -> tuple[list[DocumentRecord], list[TaskRecord]]`; `write_dataset(output_dir, documents, tasks) -> ArtifactHashes` |
| `corpus.loader` | `CorpusBundle`; `load_corpus(path: Path) -> list[DocumentRecord]`; `load_tasks(path: Path) -> list[TaskRecord]`; `load_bundle(corpus_path, tasks_path) -> CorpusBundle` |
| `retrieval.chunking` | `normalize_terms(text: str) -> list[str]`; `estimate_tokens(text: str) -> int`; `ChunkRecord`; `Chunker(size_tokens=180, overlap_tokens=30).chunk(documents) -> list[ChunkRecord]` |
| `retrieval.bm25` | `SearchHit`; `BM25Index(chunks, k1, b, epsilon)`; `.search(query, top_k, allowed_paths=None) -> list[SearchHit]`; `.document_hits(hits) -> list[DocumentHit]`; `.corpus_fingerprint` |
| `experiment.artifacts` | `canonical_json(value) -> str`; `sha256_file(path) -> str`; `write_jsonl_atomic(path, rows) -> str`; `append_jsonl(path, row) -> None`; `read_jsonl(path) -> Iterator[dict]` |
| `experiment.schemas` | all enums and run Pydantic models; `RunRecord`, `TokenAccounting`, `TurnAccounting`, `ToolTraceEntry`, `ParsedAnswer` |
| `models.tokenization` | `RegexTokenizer.name == "regex_v1"`; `.count_text`; `.count_messages`; `.truncate_text(text, max_tokens) -> str` |
| `harnesses.budget` | `BudgetController(limit, tokenizer)`; `.plan_call(messages, output_cap) -> CallAdmission`; `.record_call(...) -> TurnAccounting`; `.remaining`; `.fit_tool_result(text, reserved_output_tokens=1) -> str` |
| `models.client` | `ModelClient` protocol `.complete(request: ModelRequest) -> ModelResponse`; `OpenAICompatibleClient`; `ScriptedMockClient`; `MockPolicy`; `MockAction` |
| `models.cache` | `build_cache_key(CacheInputs) -> str`; `FileResponseCache(root).get(key)`; `.put(key,response)` |
| `harnesses.base` | `HarnessContext`; `HarnessResult`; abstract `Harness.run(task, context) -> HarnessResult`; `build_answer_instruction() -> str` |
| `harnesses.snippets` | `SnippetHarness(Harness).run(...)`; `build_snippet_messages(...)` |
| `vfs.security` | `VfsError(code,message)`; `normalize_vfs_path(raw) -> PurePosixPath`; `resolve_sandboxed(root, raw, expected) -> Path` |
| `vfs.filesystem` | `materialize_corpus(root, documents) -> None`; `VirtualFilesystem(root, documents)`; `.list(path="/")`; `.read(path,start_line,end_line)`; `.cat(path)`; immutable `ContentSlice` |
| `vfs.tools` | `ToolRequest`; `ToolResult`; `VfsToolExecutor(vfs,index,budget,per_call_limit).execute(request) -> ToolResult`; `tool_schema() -> list[dict]` |
| `harnesses.vfs_agent` | `VfsHarness(Harness).run(...)`; `build_vfs_messages(...)`; `parse_model_action(response) -> FinalAction | ToolRequest` |
| `evaluation.answer_scoring` | `normalize_answer`; `parse_answer`; `score_answer(task, raw) -> AnswerScore`; `score_citations` |
| `evaluation.evidence_scoring` | `derive_accessed_evidence(task, chunks, trace) -> AccessedEvidence`; `score_retrieval`; `score_evidence` |
| `experiment.runner` | `ExperimentRunner(config, bundle, index, model, cache)`; `.run() -> RunSummary`; `condition_order(seed,task_id) -> tuple[Condition,...]`; `load_completed_run_keys(path)` |
| `experiment.designs` | `DesignCell`; `expand_design(config,tasks) -> list[DesignAssignment]` |
| `evaluation.statistics` | `paired_summary(frame, resamples, seed) -> PairedStats`; `exact_mcnemar`; `within_vfs_regression`; `stratified_summaries` |
| `evaluation.failures` | `FailureClass`; `classify_failure(record) -> list[FailureClass]`; `select_trace_examples(records) -> dict[str,RunRecord]` |
| `evaluation.aggregates` | `build_run_frame(records) -> DataFrame`; `write_aggregate_tables(records,out_dir) -> list[Path]` |
| `evaluation.plots` | ten producer functions below; `produce_all_plots(frame,out_dir) -> list[Path]` |
| `experiment.reporting` | `generate_report(records, aggregates_dir, plots_dir) -> str`; `write_report(..., output) -> Path` |
| `cli` | Typer `app`; commands `generate`, `run`, `evaluate`, `report`, `smoke` |

### §2.2 Enums, IDs, formats, and literals

| item | frozen value |
|---|---|
| condition | `snippets`, `vfs` |
| tool name | `grep`, `read`, `cat`, `list` |
| termination reason | `answered`, `budget_exhausted`, `max_tool_calls`, `invalid_call_limit`, `model_error`, `malformed_response`, `skipped_resume` |
| failure class | `initial_retrieval_miss`, `failed_exploration`, `incomplete_evidence_coverage`, `incorrect_evidence_composition`, `unsupported_answer`, `correct_evidence_wrong_conclusion`, `budget_exhaustion`, `excessive_repeated_tool_use`, `invalid_tool_call`, `malformed_final_answer` |
| split | `dev`, `eval` |
| task ID | `task-{split}-{six_digit_sequence}`, e.g. `task-eval-000001` |
| fact ID | `fact-{six_digit_sequence}` unique in corpus version |
| doc ID | `doc-{category}-{four_digit_sequence}` where category is lowercase ASCII slug |
| path | `/{category}/{doc_id}.md`; no spaces |
| chunk ID | `{doc_id}::c{four_digit_zero_based_index}` |
| experiment ID | `exp-{12_lower_hex}-{decimal_seed}` unless explicit value matches `[a-z0-9][a-z0-9-]{2,63}` |
| design cell | `primary`, `oracle`, `max_calls_0`, `max_calls_1`, `max_calls_2`, `max_calls_4`, `max_calls_8`, `max_calls_12` |
| run key | `{experiment_id}:{task_id}:{condition}:{design_cell}` |
| request ID | `tool-{one_based_call_ordinal:03d}` |
| corpus JSONL | one `DocumentRecord`/line sorted by `doc_id` |
| tasks JSONL | one `TaskRecord`/line sorted by `(split,task_id)` with `dev` before `eval` |
| runs JSONL | append-only `RunRecord`/line in assignment order; runtime values allowed |
| aggregates CSV | UTF-8, header, `lineterminator="\n"`, stable documented column order, rows sorted by grouping keys |

`FactSpan` fields are `fact_id`, `subject`, `predicate`, `object`, `line_start`, `line_end`.
`DocumentRecord` fields are `schema_version`, `corpus_version`, `doc_id`, `path`, `category`, `title`, `content`, `facts`, `generator_seed`.
`TaskRecord` fields are `schema_version`, `task_id`, `split`, `question`, `canonical_answer`, `acceptable_answer_variants`, `required_fact_ids`, `gold_document_ids`, `gold_chunk_ids`, `hop_count`, `task_template`, `category`, `distractor_document_ids`, `distractor_count`, `generator_seed`, `corpus_version`.
`ChunkRecord` fields are `chunk_id`, `doc_id`, `path`, `chunk_index`, `start_line`, `end_line`, `text`, `token_count`, `fact_ids`.
Gold chunk IDs are populated only after fixed chunking and validated on load against the config's chunk fingerprint.

### §2.3 Token accounting contract

`TokenAccounting` has `tokenizer`, `limit`, `turns`, `input_tokens_estimated`, `output_tokens_estimated`, `total_tokens_estimated`, `provider_input_tokens`, `provider_output_tokens`, `provider_total_tokens`, and `provider_minus_estimated`.
`TurnAccounting` has `turn_index`, `request_input_tokens_estimated`, `request_output_cap`, `response_output_tokens_estimated`, `snippet_input_tokens`, `tool_argument_output_tokens`, `tool_result_input_tokens`, `provider_input_tokens`, `provider_output_tokens`, `remaining_after_turn`.
`ToolTraceEntry` has `request_id`, `call_ordinal`, `requested_at`, `completed_at`, `request`, `result`, `repeated`, and `latency_ms`; request/result are the strict tool models, not free-form logs.
Component token fields are diagnostic subsets and MUST NOT be summed on top of request input/output totals.
Message serialization count is the token count of canonical compact JSON for `{role,content,tool_calls,tool_call_id}` plus four framing tokens/message and two request framing tokens.
Tool arguments count in response output using canonical compact JSON.
The same arguments count again inside a later request if conversation history is replayed.
Tool results count only when present in a model request; generating a local result alone consumes no LLM token budget.
Snippet text counts within the single snippet request input.
`plan_call` returns `{admitted,input_tokens,output_cap,remaining_before,reason}`, clamps output cap to available budget, and MUST reject when the effective cap is below one.
Each harness passes configured `max_answer_tokens` as requested cap on every turn; unused output allowance is not charged.
`record_call` MUST raise `BudgetInvariantError` if estimated cumulative total would exceed `limit`.
Provider counts are nullable sums and never determine admission.
Exact exhaustion termination literal is `budget_exhausted`.

### §2.4 Determinism and hash rules

Generated artifacts use no timestamps, UUIDs, directory enumeration order, global RNG, provider response, or Python `hash()`.
Generator stage seeds are `int.from_bytes(sha256(f"{seed}:{stage}:{ordinal}".encode()).digest()[:8], "big")`.
All sets become lexicographically sorted arrays before serialization.
Floating BM25 scores are stored as JSON numbers rounded to 12 decimal places, but sorting uses unrounded `float64` values then `chunk_id`.
Artifact regeneration deletes nothing: generation writes temp files, byte-compares/hashes, then atomically replaces the two explicit targets.
The determinism gate generates twice into two temporary directories and requires equal SHA-256 for `corpus.jsonl` and `tasks.jsonl` separately.
Canonical config hash uses fully defaulted, post-override, secret-redacted `model_dump(mode="json")`.
Prompt hash uses canonical JSON of initial messages before any model response.
Cache hash uses canonical JSON and lower-case 64-character SHA-256; the experiment-config input is the effective design-cell config, including assigned maximum calls and oracle flag.

### §2.5 Plot and table producers

| required output | producer | filename |
|---|---|---|
| success by harness with paired CI | `plot_success_by_condition` | `01_success_by_condition.png` |
| success by hop count | `plot_success_by_hop` | `02_success_by_hop.png` |
| success vs actual VFS calls | `plot_success_by_actual_calls` | `03_success_by_actual_tool_calls.png` |
| success vs permitted calls | `plot_intervention_dose_response` | `04_success_by_max_tool_calls.png` |
| initial recall vs success | `plot_initial_recall_vs_success` | `05_initial_recall_vs_success.png` |
| final evidence recall vs success | `plot_evidence_recall_vs_success` | `06_evidence_recall_vs_success.png` |
| tokens vs success | `plot_tokens_vs_success` | `07_tokens_vs_success.png` |
| latency and cost | `plot_efficiency` | `08_latency_cost_by_condition.png` |
| failure modes | `plot_failure_modes` | `09_failure_modes.png` |
| paired outcomes | `plot_paired_outcomes` | `10_paired_outcomes.png` |
| paired result table | `write_aggregate_tables` | `paired_outcomes.csv` |
| condition summary | `write_aggregate_tables` | `condition_summary.csv` |
| hop/category results | `write_aggregate_tables` | `subgroup_summary.csv` |
| intervention results | `write_aggregate_tables` | `intervention_summary.csv` |
| regression coefficients | `write_aggregate_tables` | `within_vfs_regression.csv` |
| latency, tokens, cost, success/1k tokens | `write_aggregate_tables` | `efficiency_summary.csv` |
| failure examples | `write_aggregate_tables` | `trace_examples.json` |

### §2.6 CLI and Make contracts

`generate` flags: `--output-dir PATH` default `data/generated`; `--dev-tasks INT` 50; `--tasks INT` 200 eval tasks; `--seed INT` 42; `--corpus-size INT` 80; `--distractors INT` 6; `--min-hops INT` 2; `--max-hops INT` 4; `--force/--no-force` default false.
`run` flags: required `--config PATH`; optional `--runs PATH`; `--task-limit INT`; `--split [dev|eval|all]` default eval; `--model-provider [mock|openai_compatible]`; `--resume/--no-resume` default from config.
`evaluate` flags: required `--runs PATH`; `--output-dir PATH` default `results/aggregates`; `--plots-dir PATH` default `results/plots`; `--include-design [primary|all]` default primary.
`report` flags: required `--runs PATH`; `--aggregates-dir PATH` default `results/aggregates`; `--plots-dir PATH` default `results/plots`; `--output PATH` default `reports/experiment.md`.
`smoke` flags: `--workdir PATH` default `.smoke`; `--seed INT` default 42; it generates 4 dev and 8 eval tasks, runs both conditions with mock, evaluates, reports, and prints artifact hashes.
All commands return 0 on success, 2 for user/config/schema errors, and 1 for runtime/model errors.

Operator-owned `Makefile` MUST expose exact targets:
`setup: uv sync --frozen`;
`lint: uv run ruff check .`;
`test: uv run pytest`;
`generate: uv run python -m bm25_vfs_ablation generate --tasks 200 --seed 42`;
`experiment: uv run python -m bm25_vfs_ablation run --config configs/default.yaml`;
`intervention: uv run python -m bm25_vfs_ablation run --config configs/tool_call_intervention.yaml`;
`evaluate: uv run python -m bm25_vfs_ablation evaluate --runs results/runs.jsonl`;
`report: uv run python -m bm25_vfs_ablation report --runs results/runs.jsonl --output reports/experiment.md`;
`smoke: uv run python -m bm25_vfs_ablation smoke --workdir .smoke --seed 42`;
`accept: uv run ruff check . && uv run pytest && uv run python -m bm25_vfs_ablation smoke --workdir .smoke --seed 42`.

## §3 Epics and task breakdown

Each card is one isolated agent run.
The FILE allowlist is exhaustive: reading is unrestricted, but writing, formatting, renaming, or deleting any other path is forbidden.
Every agent runs its acceptance commands after its edits and leaves all pre-existing tests green.
Evidence bundles are pasted into the issue comment; they are not committed as extra files.

### Epic E1 — Package and configuration foundation

#### B01 — Package bootstrap and frozen enums

- **Epic:** E1.
- **Goal:** Create the importable package, module entry point, and shared string enums used by every later schema.
- **FILE allowlist (3):** `src/bm25_vfs_ablation/__init__.py`; `src/bm25_vfs_ablation/__main__.py`; `tests/test_package.py`.
- **`__init__.py`:** expose `__version__ = "0.1.0"`, `Condition`, `ToolName`, `TerminationReason`, and `FailureClass` as `str, Enum`; literals exactly match §2.2.
- **`__main__.py`:** import `app` lazily inside `main()`, invoke it, and contain only the standard `if __name__ == "__main__"` guard; importing the module MUST have no side effects.
- **Test spec:** assert exact enum member values, unique values, version, and that `runpy.run_module` can reach a monkeypatched CLI without network.
- **Named tests:** `test_version_is_frozen`; `test_enum_literals_are_exact`; `test_module_entrypoint_invokes_app`.
- **Acceptance:** `uv run pytest tests/test_package.py`; `uv run ruff check src/bm25_vfs_ablation/__init__.py src/bm25_vfs_ablation/__main__.py tests/test_package.py`.
- **Dependencies:** none after operator prep P0.
- **Do not touch:** all operator-owned files; any config, corpus, harness, or CLI implementation.
- **Evidence bundle:** three changed paths; `3 passed`; full two command outputs.

#### B02 — Typed config loader and default config

- **Epic:** E1.
- **Goal:** Implement the strict nested config contract and ship the primary defaults.
- **FILE allowlist (3):** `src/bm25_vfs_ablation/config.py`; `configs/default.yaml`; `tests/test_config.py`.
- **`config.py`:** define every Pydantic model/signature in §2.1; `extra="forbid"`; numeric bounds (`token_ceiling>0`, overlap `<` size, top_k>0, limits nonnegative); resolve exactly three environment variables; redact API key in `canonical_dict()`.
- **`default.yaml`:** byte-for-byte semantic values from §0.1; do not add comments or aliases that alter loaded shape.
- **Test spec:** load defaults, reject unknown/nonsensical keys, test allowed env override and secret redaction, and prove paths remain repo-relative strings.
- **Named tests:** `test_default_config_exact_values`; `test_config_rejects_unknown_key`; `test_config_cross_field_bounds`; `test_env_override_allowlist`; `test_api_key_redacted_from_canonical_dict`.
- **Acceptance:** `uv run pytest tests/test_config.py`; `uv run ruff check src/bm25_vfs_ablation/config.py tests/test_config.py`.
- **Dependencies:** B01.
- **Do not touch:** `configs/tool_call_intervention.yaml`; CLI; pyproject/lock/Makefile.
- **Evidence bundle:** three changed paths; `5 passed`; command outputs; redacted canonical-config sample.

#### B03 — Intervention config overlay

- **Epic:** E1.
- **Goal:** Add and validate the prescribed maximum-tool-call intervention configuration.
- **FILE allowlist (3):** `src/bm25_vfs_ablation/config.py`; `configs/tool_call_intervention.yaml`; `tests/test_intervention_config.py`.
- **`config.py`:** default missing intervention to disabled/empty; validate enabled grid is exactly increasing unique nonnegative ints; shipped intervention grid must be `[0,1,2,4,8,12]`.
- **YAML:** repeat the complete default config, changing only output path and adding the block in §0.1; it MUST be independently loadable, not include/merge another YAML.
- **Test spec:** compare model dumps after normalizing the two permitted differences; reject duplicate/negative grids; verify absent block defaults.
- **Named tests:** `test_shipped_intervention_diff_is_exact`; `test_intervention_grid_validation`; `test_absent_intervention_defaults_disabled`.
- **Acceptance:** `uv run pytest tests/test_config.py tests/test_intervention_config.py`; `uv run ruff check src/bm25_vfs_ablation/config.py tests/test_intervention_config.py`.
- **Dependencies:** B02.
- **Do not touch:** primary YAML; package entrypoint; operator-owned files.
- **Evidence bundle:** three changed paths; combined test count/output; normalized config diff.

### Epic E2 — Schemas and deterministic artifacts

#### B04 — Corpus, task, and run schemas

- **Epic:** E2.
- **Goal:** Freeze validated data structures before any producer consumes them.
- **FILE allowlist (3):** `src/bm25_vfs_ablation/corpus/schema.py`; `src/bm25_vfs_ablation/experiment/schemas.py`; `tests/test_schemas.py`.
- **`corpus/schema.py`:** implement `FactSpan`, `DocumentRecord`, `TaskRecord` fields from §2.2; forbid extras; validate ID grammars, one-based spans, sorted unique gold/distractor lists, hop count 2–4, nonempty canonical/variants.
- **`experiment/schemas.py`:** implement the complete §0.2 run shape and §2.3 accounting models; re-export enums from package; enforce condition-specific invariants (snippet trace empty, nonnegative metrics, totals equal sums).
- **Test spec:** valid round trips plus one focused rejection for every ID family, omitted required run key, bad totals, and invalid enum.
- **Named tests:** `test_document_and_task_round_trip`; `test_id_grammars`; `test_fact_span_is_one_based`; `test_run_record_requires_complete_shape`; `test_token_totals_are_consistent`; `test_condition_specific_trace_invariant`.
- **Acceptance:** `uv run pytest tests/test_schemas.py`; `uv run ruff check src/bm25_vfs_ablation/corpus/schema.py src/bm25_vfs_ablation/experiment/schemas.py tests/test_schemas.py`.
- **Dependencies:** B01.
- **Do not touch:** generator, config, tests outside allowlist.
- **Evidence bundle:** three changed paths; `6 passed` minimum; validation-error excerpts; command outputs.

#### B05 — Canonical JSONL and hashing utilities

- **Epic:** E2.
- **Goal:** Supply byte-stable serialization, atomic writes, strict reads, and file hashes.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/experiment/artifacts.py`; `tests/test_artifacts.py`.
- **Source spec:** implement all §2.1 signatures; canonical compact JSON rules from §0; atomic write uses sibling temporary file, flush, `os.fsync`, `os.replace`; append writes exactly one flushed line; reader reports path and one-based line on invalid JSON.
- **Invariants:** empty sequence writes empty file; nonempty JSONL ends once with newline; no NaN/Infinity; parent directories may be created only for explicit target.
- **Named tests:** `test_canonical_json_is_byte_stable`; `test_atomic_jsonl_has_trailing_newline`; `test_append_jsonl_one_record`; `test_read_jsonl_reports_line`; `test_sha256_file_known_vector`.
- **Acceptance:** `uv run pytest tests/test_artifacts.py`; `uv run ruff check src/bm25_vfs_ablation/experiment/artifacts.py tests/test_artifacts.py`.
- **Dependencies:** B04.
- **Do not touch:** schemas, generated data, operator-owned files.
- **Evidence bundle:** two changed paths; `5 passed`; known digest; command outputs.

### Epic E3 — Synthetic dataset and imports

#### B06 — Deterministic enterprise corpus generator

- **Epic:** E3.
- **Goal:** Generate versioned enterprise documents and exact hidden fact spans without leakage.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/corpus/generator.py`; `tests/test_generator.py`.
- **Source spec:** implement `GenerationConfig`, child-seed formula, entity/alias tables, seven document categories from SPEC, four template families, and deterministic fact-span line calculation; use only local `random.Random`.
- **Task spec:** generate requested dev/eval counts, 2–4 required facts across at least two docs, aliases/paraphrases, configured distractors, disjoint split entity tuples, canonical answer plus at least one acceptable variant.
- **Validation:** reject exact normalized question substring in any document, one chunk containing all facts, missing referenced IDs, duplicate tuples/IDs, or unmet counts; `write_dataset` uses B05.
- **Named tests:** `test_generation_same_seed_same_models`; `test_generation_different_seed_changes_content`; `test_no_question_text_leakage`; `test_tasks_require_multiple_documents`; `test_split_templates_or_entities_are_disjoint`; `test_generator_uses_no_global_random`.
- **Acceptance:** `uv run pytest tests/test_generator.py`; `uv run ruff check src/bm25_vfs_ablation/corpus/generator.py tests/test_generator.py`.
- **Dependencies:** B04, B05. The generator implements the frozen 180/30 prospective-chunk check as a private pure validator; B08 later proves its production chunker yields identical boundaries.
- **Do not touch:** committed `data/`; loader; retrieval implementation.
- **Evidence bundle:** two paths; `6 passed` minimum; seed-42 model hash pair; outputs.

#### B07 — Validated corpus/task loader and import path

- **Epic:** E3.
- **Goal:** Load generated or user-provided JSONL into one validated bundle.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/corpus/loader.py`; `tests/test_loader.py`.
- **Source spec:** implement §2.1 loaders; reject duplicate IDs/paths, cross-version records, dangling gold/distractor IDs, unsorted input, absent files, blank lines, and non-UTF-8; compute source byte hashes.
- **`CorpusBundle`:** frozen dataclass fields `documents`, `tasks`, `corpus_sha256`, `tasks_sha256`, `corpus_version`; expose `documents_by_id`, `tasks_by_id` read-only mappings.
- **Named tests:** `test_load_valid_bundle`; `test_loader_rejects_duplicate_ids`; `test_loader_rejects_dangling_gold`; `test_loader_rejects_mixed_versions`; `test_user_import_uses_same_schema`.
- **Acceptance:** `uv run pytest tests/test_loader.py`; `uv run ruff check src/bm25_vfs_ablation/corpus/loader.py tests/test_loader.py`.
- **Dependencies:** B04, B05.
- **Do not touch:** generator, configs, data files.
- **Evidence bundle:** two paths; `5 passed`; invalid-line diagnostic example; outputs.

### Epic E4 — Shared retrieval layer

#### B08 — Normalization, approximation, and fixed chunking

- **Epic:** E4.
- **Goal:** Implement the single text normalization/token count and line-aware chunker used everywhere.
- **FILE allowlist (3):** `src/bm25_vfs_ablation/retrieval/chunking.py`; `src/bm25_vfs_ablation/models/tokenization.py`; `tests/test_chunking.py`.
- **`chunking.py`:** exact regex/case/NFKC policy from §0; `Chunker` greedily adds complete lines up to 180 tokens, splits an overlong line by tokens, and starts next chunk with last 30 tokens while preserving source line overlap metadata.
- **`tokenization.py`:** `RegexTokenizer` counts words and standalone punctuation, canonical messages with fixed framing, and truncates on token boundary without exceeding limit.
- **Invariants:** no empty chunks, full document coverage, stable IDs, stable input order independent output sorted by `(doc_id,chunk_index)`, fact IDs assigned by line overlap.
- **Named tests:** `test_unicode_normalization_and_casefold`; `test_stopwords_are_retained`; `test_chunk_boundaries_and_overlap`; `test_long_line_split_is_bounded`; `test_chunk_ids_stable`; `test_regex_tokenizer_never_exceeds_limit`; `test_generated_tasks_have_no_single_complete_chunk`.
- **Acceptance:** `uv run pytest tests/test_chunking.py`; `uv run ruff check src/bm25_vfs_ablation/retrieval/chunking.py src/bm25_vfs_ablation/models/tokenization.py tests/test_chunking.py`.
- **Dependencies:** B04.
- **Do not touch:** BM25, generator, config.
- **Evidence bundle:** three paths; `6 passed`; representative chunk table; outputs.

#### B09 — Stable shared BM25 index

- **Epic:** E4.
- **Goal:** Wrap `rank-bm25` with frozen scoring, filtering, tie-breaking, and document aggregation.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/retrieval/bm25.py`; `tests/test_bm25.py`.
- **Source spec:** construct `BM25Okapi(tokenized_corpus,k1,b,epsilon)` once; score all chunks; allowed paths are exact files or slash-terminated prefixes; return top positive or zero-score hits up to k sorted `(-raw_score,chunk_id)`; round only serialized score.
- **Fingerprint:** SHA-256 canonical sequence of chunk IDs/text plus BM25/tokenizer parameters; document score is max chunk score and rank uses `(-best_score,doc_id)`.
- **Named tests:** `test_bm25_known_ranking`; `test_bm25_tie_breaks_by_chunk_id`; `test_allowed_path_filter`; `test_document_hit_aggregation`; `test_index_fingerprint_stable`; `test_same_index_object_can_serve_both_conditions`.
- **Acceptance:** `uv run pytest tests/test_bm25.py`; `uv run ruff check src/bm25_vfs_ablation/retrieval/bm25.py tests/test_bm25.py`.
- **Dependencies:** B08.
- **Do not touch:** chunker, harnesses, rank-bm25 internals.
- **Evidence bundle:** two paths; `6 passed`; known ranking/score printout; outputs.

### Epic E5 — Token fairness and model boundary

#### B10 — Exact budget controller

- **Epic:** E5.
- **Goal:** Enforce the same hard estimated-token ceiling across one-turn and iterative harnesses.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/harnesses/budget.py`; `tests/test_budget.py`.
- **Source spec:** implement contracts in §2.3; immutable finalized turns; remaining is `limit-total_estimated`; rejected admission mutates nothing; `fit_tool_result` binary-searches tokenizer boundaries for the largest insertable result preserving next-call framing and reserve.
- **Reconciliation:** provider values nullable/nonnegative, diagnostic deltas only; tool argument and result subset counters checked against respective totals.
- **Named tests:** `test_admit_exact_boundary`; `test_reject_one_token_over`; `test_rejected_call_does_not_mutate`; `test_multiturn_accounting_no_double_count`; `test_tool_arguments_count_output_and_replayed_input`; `test_fit_tool_result_reserves_final_token`; `test_provider_reconciliation_does_not_change_limit`.
- **Acceptance:** `uv run pytest tests/test_budget.py`; `uv run ruff check src/bm25_vfs_ablation/harnesses/budget.py tests/test_budget.py`.
- **Dependencies:** B04, B08.
- **Do not touch:** harness implementations, client, config.
- **Evidence bundle:** two paths; `7 passed`; boundary accounting JSON; outputs.

#### B11 — Model protocol, HTTP client, and scripted mock

- **Epic:** E5.
- **Goal:** Add one narrow provider boundary and deterministic no-network behavior for all acceptance paths.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/models/client.py`; `tests/test_model_client.py`.
- **Protocol:** `ModelRequest(messages,tools,max_output_tokens,temperature,top_p,seed)`; `ModelResponse(content,tool_calls,returned_model_id,provider_usage,latency_ms,raw_id)`; never store hidden reasoning fields.
- **HTTP client:** POST `{base_url}/chat/completions` with bearer key, parse OpenAI-compatible choices/usage/model, retry only frozen classes/delays, sanitize errors, never log key; injectable `httpx.Client` and sleeper.
- **Mock:** `MockAction(kind,tool_name,arguments,answer)`; `MockPolicy` keyed `(task_id,condition,design_cell)`; cursor advances deterministically; tool call arguments canonical; missing/exhausted policy raises `MOCK_POLICY_EXHAUSTED`; returned ID fixed.
- **Named tests:** `test_protocol_response_shape`; `test_http_request_and_usage_parse`; `test_http_retry_policy_exact`; `test_api_key_not_in_error`; `test_mock_single_shot_answer`; `test_mock_multiturn_tool_sequence`; `test_mock_policy_exhaustion`.
- **Acceptance:** `uv run pytest tests/test_model_client.py`; `uv run ruff check src/bm25_vfs_ablation/models/client.py tests/test_model_client.py`.
- **Dependencies:** B04.
- **Do not touch:** cache, harnesses, any network endpoint.
- **Evidence bundle:** two paths; `7 passed`; mocked request body with redaction; outputs.

#### B12 — Response cache

- **Epic:** E5.
- **Goal:** Implement the exact cache key and atomic file cache without secrets.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/models/cache.py`; `tests/test_cache.py`.
- **Source spec:** `CacheInputs` contains condition, redacted model config, messages, tool schema, corpus hash, experiment config hash; build canonical SHA-256; validate cached `ModelResponse`; atomic per-key JSON writes; malformed entries are errors, not misses.
- **Invariants:** changing any required input changes key; API key and runtime timestamps cannot affect or appear in key payload; disabled cache may be represented by `None` at caller, not inside class.
- **Named tests:** `test_cache_key_known_vector`; `test_every_required_input_changes_key`; `test_api_key_never_affects_key`; `test_cache_round_trip`; `test_malformed_cache_entry_fails_closed`.
- **Acceptance:** `uv run pytest tests/test_cache.py`; `uv run ruff check src/bm25_vfs_ablation/models/cache.py tests/test_cache.py`.
- **Dependencies:** B05, B11.
- **Do not touch:** runner, configs, results directory.
- **Evidence bundle:** two paths; `5 passed`; known key; outputs.

### Epic E6 — Primary harnesses

#### B13 — Base harness contracts and answer instruction

- **Epic:** E6.
- **Goal:** Centralize the truly shared prompt, context, and result contracts without framework machinery.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/harnesses/base.py`; `tests/test_harness_base.py`.
- **Source spec:** frozen dataclasses/protocol from §2.1; context holds exact shared `CorpusBundle`, `BM25Index`, model, cache, config, budget factory; result carries raw answer, initial hits, trace, accounting, latencies, termination, errors.
- **Prompt rule:** common system text is byte-identical; `build_answer_instruction()` specifies only JSON answer/citations/short justification and no chain-of-thought; condition additions are separate messages.
- **Named tests:** `test_answer_instruction_exact_schema`; `test_common_system_prompt_has_no_condition_language`; `test_harness_context_preserves_index_identity`; `test_harness_result_defaults_are_complete`.
- **Acceptance:** `uv run pytest tests/test_harness_base.py`; `uv run ruff check src/bm25_vfs_ablation/harnesses/base.py tests/test_harness_base.py`.
- **Dependencies:** B04, B09, B10, B11, B12.
- **Do not touch:** concrete harnesses, CLI, runner.
- **Evidence bundle:** two paths; `4 passed`; prompt digest; outputs.

#### B14 — One-call snippet harness

- **Epic:** E6.
- **Goal:** Implement Condition A with bounded whole chunks and exactly one model call.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/harnesses/snippets.py`; `tests/test_snippet_harness.py`.
- **Source spec:** search shared index once; preserve top-k metadata; greedily include ranked complete chunks under retrieval-context and total admission caps; message renders stable doc/chunk/line IDs; call cache/client once; no tool schema.
- **Stops:** answer on valid or malformed content with corresponding reason; pre-call rejection is `budget_exhausted`; model failure is `model_error`; never perform second completion.
- **Named tests:** `test_snippet_prompt_contains_stable_ids`; `test_snippet_uses_shared_index_once`; `test_snippet_context_whole_chunk_truncation`; `test_snippet_exactly_one_model_call`; `test_snippet_budget_exhaustion_before_call`; `test_snippet_has_no_tools`.
- **Acceptance:** `uv run pytest tests/test_snippet_harness.py`; `uv run ruff check src/bm25_vfs_ablation/harnesses/snippets.py tests/test_snippet_harness.py`.
- **Dependencies:** B13.
- **Do not touch:** VFS modules, scoring, runner.
- **Evidence bundle:** two paths; `6 passed`; captured prompt and accounting summary; outputs.

### Epic E7 — Secure read-only VFS

#### B15 — Path security boundary

- **Epic:** E7.
- **Goal:** Make traversal and host-file access impossible with exact rejection semantics.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/vfs/security.py`; `tests/test_vfs_security.py`.
- **Source spec:** reject NUL, backslash, empty non-root, repeated slash ambiguity, literal `.`/`..` before normalization, URI schemes, drive prefixes, symlinks at any component, escape after resolve; `/` maps only to root for directory expectation.
- **Errors:** raise `VfsError` with exact code from §0 and safe message containing raw logical path but never host root; expected is `file`, `directory`, or `either`.
- **Named tests:** `test_accepts_root_relative_posix_path`; `test_rejects_dotdot_traversal_exact_code`; `test_rejects_encoded_or_backslash_tricks`; `test_rejects_symlink_escape`; `test_error_does_not_leak_host_root`; `test_type_expectation_codes`.
- **Acceptance:** `uv run pytest tests/test_vfs_security.py`; `uv run ruff check src/bm25_vfs_ablation/vfs/security.py tests/test_vfs_security.py`.
- **Dependencies:** B01.
- **Do not touch:** filesystem/tool executor, real corpus files.
- **Evidence bundle:** two paths; `6 passed`; rejection matrix; outputs.

#### B16 — Bounded virtual filesystem operations

- **Epic:** E7.
- **Goal:** Expose immutable sorted list/read/cat primitives over materialized corpus files.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/vfs/filesystem.py`; `tests/test_vfs_filesystem.py`.
- **Source spec:** `materialize_corpus` writes only validated document paths beneath an explicit empty root; constructor verifies file bytes exactly equal `DocumentRecord.content`; `list` is shallow, directories end `/`, lexicographically sorted; `read` uses inclusive positive bounds and clamps end to EOF; `cat` returns all lines; writes are absent after construction.
- **ContentSlice:** path, start/end lines, text, token count; preserve source newlines; empty files forbidden by corpus schema.
- **Named tests:** `test_list_is_shallow_and_sorted`; `test_read_inclusive_bounded_lines`; `test_read_rejects_invalid_range`; `test_cat_returns_exact_file`; `test_materialized_content_must_match`; `test_filesystem_exposes_no_write_method`.
- **Acceptance:** `uv run pytest tests/test_vfs_filesystem.py tests/test_vfs_security.py`; `uv run ruff check src/bm25_vfs_ablation/vfs/filesystem.py tests/test_vfs_filesystem.py`.
- **Dependencies:** B07, B08, B15.
- **Do not touch:** tools, harness, loader.
- **Evidence bundle:** two paths; test count/output; sample content slice; lint output.

#### B17 — VFS tool executor and schemas

- **Epic:** E7.
- **Goal:** Add observable bounded `grep/read/cat/list` envelopes over the secure filesystem.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/vfs/tools.py`; `tests/test_vfs_tools.py`.
- **Source spec:** strict `ToolRequest`; canonical `tool_schema`; executor times monotonic work, returns rather than raises user errors, assigns request IDs outside executor, and truncates successful content through budget/per-call caps.
- **`grep`:** require nonblank query and `1<=max_results<=50`; optional path filter; shared BM25 index; matching term line ±1; deduplicate same path/range; rank deterministically.
- **`cat`:** if exact entire content exceeds per-call or remaining fit, return `PER_CALL_LIMIT` or `BUDGET_INSUFFICIENT`, never partial; `read/grep/list` may truncate with explicit `truncated=true` metadata.
- **Named tests:** `test_tool_schema_exact_names`; `test_grep_rank_and_preview_bounds`; `test_read_tool_envelope`; `test_cat_refuses_partial_result`; `test_list_tool_sorted`; `test_invalid_arguments_exact_codes`; `test_tool_result_token_count_exact`.
- **Acceptance:** `uv run pytest tests/test_vfs_tools.py`; `uv run ruff check src/bm25_vfs_ablation/vfs/tools.py tests/test_vfs_tools.py`.
- **Dependencies:** B09, B10, B16.
- **Do not touch:** VFS agent, runner, security semantics.
- **Evidence bundle:** two paths; `7 passed`; all four result envelopes; outputs.

#### B18 — Iterative VFS harness

- **Epic:** E7.
- **Goal:** Implement Condition B's bounded model/tool loop and fully observable trace.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/harnesses/vfs_agent.py`; `tests/test_vfs_harness.py`.
- **Source spec:** initial message has question, exact tools, top five candidate document paths/scores or oracle gold paths, and remaining budget; no content; parse exactly one tool call or final JSON; append observable assistant/tool messages only.
- **Loop:** count every requested call; invalid requests increment invalid count; canonical duplicate requests increment repeated; check max before execution; record timestamps/latencies/results; re-admit every model call; precedence `answered`, budget, max calls, invalid threshold, errors.
- **Zero-call limit:** perform one initial model call with no tool execution permitted so intervention measures answer-at-zero; a requested tool terminates `max_tool_calls`.
- **Named tests:** `test_vfs_initial_context_has_paths_not_content`; `test_vfs_scripted_multiturn_trace`; `test_vfs_budget_across_turns`; `test_vfs_invalid_call_threshold`; `test_vfs_max_call_limit_zero`; `test_vfs_repeated_call_count`; `test_vfs_oracle_candidates_are_gold_only`; `test_vfs_never_logs_hidden_reasoning`.
- **Acceptance:** `uv run pytest tests/test_vfs_harness.py`; `uv run ruff check src/bm25_vfs_ablation/harnesses/vfs_agent.py tests/test_vfs_harness.py`.
- **Dependencies:** B13, B17.
- **Do not touch:** runner, scoring, tool implementation.
- **Evidence bundle:** two paths; `8 passed`; sample trace and accounting totals; outputs.

### Epic E8 — Scoring and run construction

#### B19 — Deterministic answer and citation scoring

- **Epic:** E8.
- **Goal:** Parse the public answer envelope and compute primary correctness without an LLM.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/evaluation/answer_scoring.py`; `tests/test_answer_scoring.py`.
- **Source spec:** extract a single JSON object only (optional surrounding whitespace, no Markdown fence/trailing prose); require string answer, string-array citations, justification string ≤240 chars; normalization exactly §0; preserve raw.
- **Scoring:** equality against canonical/variants after normalization; malformed is false; deduplicate citations first-seen; calculate document/chunk citation correctness separately; secondary judge object disabled/null.
- **Named tests:** `test_answer_normalization_exact`; `test_variant_exact_match`; `test_substring_is_not_correct`; `test_malformed_answer_is_incorrect`; `test_markdown_fence_rejected`; `test_citation_precision_and_recall`; `test_justification_length_bound`.
- **Acceptance:** `uv run pytest tests/test_answer_scoring.py`; `uv run ruff check src/bm25_vfs_ablation/evaluation/answer_scoring.py tests/test_answer_scoring.py`.
- **Dependencies:** B04.
- **Do not touch:** evidence scoring, harness prompts.
- **Evidence bundle:** two paths; `7 passed`; normalization examples; outputs.

#### B20 — Retrieval and accessed-evidence scoring

- **Epic:** E8.
- **Goal:** Keep retrieval recall, model-visible evidence, and answer success as separate quantities.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/evaluation/evidence_scoring.py`; `tests/test_evidence_scoring.py`.
- **Source spec:** initial chunk recall is unique gold chunks retrieved / gold chunks; doc recall analogous; accessed spans include snippet chunks and successful VFS content results; line overlap maps hidden `FactSpan`; manifests/errors do not count.
- **Metrics:** fact precision denominator all distinct corpus facts whose spans were exposed; recall required facts exposed; merge overlapping ranges per path; all-required boolean; exact time/call coverage delegated from ordered trace.
- **Named tests:** `test_initial_chunk_and_document_recall_separate`; `test_snippet_content_counts_as_accessed`; `test_grep_preview_counts_by_line_overlap`; `test_manifest_does_not_count_as_evidence`; `test_accessed_ranges_are_merged`; `test_evidence_precision_and_recall`; `test_complete_coverage_call_ordinal`.
- **Acceptance:** `uv run pytest tests/test_evidence_scoring.py`; `uv run ruff check src/bm25_vfs_ablation/evaluation/evidence_scoring.py tests/test_evidence_scoring.py`.
- **Dependencies:** B09, B17, B19.
- **Do not touch:** answer scoring, runner.
- **Evidence bundle:** two paths; `7 passed`; hand-calculated metric fixture; outputs.

#### B21 — Design expansion and randomized order

- **Epic:** E8.
- **Goal:** Expand primary/intervention/oracle cells and freeze reproducible paired assignment order.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/experiment/designs.py`; `tests/test_designs.py`.
- **Source spec:** primary yields two conditions/task; intervention yields snippet once per limit only if configured policy explicitly requests paired controls, otherwise VFS grid plus one primary snippet baseline tagged in each comparison; implement plan as both conditions per max-calls cell to preserve paired records.
- **Clarification:** in every `max_calls_n` cell, snippet config is unchanged and VFS limit is n; this duplicates snippet calls unless cache hits, but yields self-contained pairing and identical order/cell keys.
- **Order:** SHA-256-derived low bit selects condition tuple; assignments sorted `(task_id,design_cell,condition_order)`; oracle is separate run invocation/config override, not mixed with intervention by default.
- **Named tests:** `test_primary_expands_two_conditions`; `test_intervention_expands_paired_cells`; `test_condition_order_is_seeded_and_stable`; `test_oracle_cell_is_separate`; `test_design_run_keys_unique`.
- **Acceptance:** `uv run pytest tests/test_designs.py`; `uv run ruff check src/bm25_vfs_ablation/experiment/designs.py tests/test_designs.py`.
- **Dependencies:** B03, B04.
- **Do not touch:** runner, configs, harnesses.
- **Evidence bundle:** two paths; `5 passed`; assignment table for two tasks; outputs.

#### B22 — Experiment runner, complete records, and resume

- **Epic:** E8.
- **Goal:** Orchestrate both harnesses over identical objects and append complete resumable records.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/experiment/runner.py`; `tests/test_runner.py`.
- **Source spec:** build chunks/index once per runner and pass object identity to both harnesses; iterate B21 assignments; create budgets; score B19/B20; construct every §0.2 key; atomic cache, append validated record only after completion.
- **Resume:** load/validate all lines; duplicate same run key is fatal even if identical; skip completed keys before model call; incomplete final JSON line is fatal with recovery instruction, never silently truncated.
- **Timing:** monotonic for durations, UTC for timestamps; no timestamps in prompt/cache/config hashes; caught expected errors become record errors and termination, schema/programming errors fail run.
- **Named tests:** `test_runner_shares_corpus_and_index_identity`; `test_runner_randomized_condition_order`; `test_runner_writes_complete_records`; `test_runner_resume_skips_completed_calls`; `test_runner_rejects_duplicate_run_key`; `test_runner_records_returned_model_id`; `test_runner_token_ceiling_equal_by_condition`.
- **Acceptance:** `uv run pytest tests/test_runner.py`; `uv run ruff check src/bm25_vfs_ablation/experiment/runner.py tests/test_runner.py`.
- **Dependencies:** B14, B18, B19, B20, B21.
- **Do not touch:** CLI, plots, report.
- **Evidence bundle:** two paths; `7 passed`; two-record schema sample with timestamps redacted; outputs.

### Epic E9 — Statistics, failure analysis, and aggregation

#### B23 — Paired inference and associative mechanism models

- **Epic:** E9.
- **Goal:** Implement prespecified paired statistics, strata, and clearly associative within-VFS regression.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/evaluation/statistics.py`; `tests/test_statistics.py`.
- **Source spec:** require exactly one primary record/condition/task; paired success difference VFS-snippets; 10,000 paired resamples/default seed; percentile CI; exact McNemar formula; deterministic sorted outputs.
- **Regression:** GLM binomial or Logit with HC3, calls and squared calls, recall, hops, token count, distractors, category dummies with sorted reference; return coefficients/SE/CI/p plus convergence/error row, label `associative=true`.
- **Strata:** pooled token quartiles with deterministic duplicate-edge collapse; recall bands, hop, category; only within-band pairs; include n and missing reason.
- **Named tests:** `test_paired_difference_and_percentile_ci_known_seed`; `test_exact_mcnemar_known_table`; `test_mcnemar_no_discordance`; `test_unpaired_rows_rejected`; `test_stratified_pairing`; `test_within_vfs_regression_columns_and_hc3`; `test_regression_failure_is_reported`.
- **Acceptance:** `uv run pytest tests/test_statistics.py`; `uv run ruff check src/bm25_vfs_ablation/evaluation/statistics.py tests/test_statistics.py`.
- **Dependencies:** B22.
- **Do not touch:** aggregation, plots, runner.
- **Evidence bundle:** two paths; `7 passed`; fixed-seed CI and McNemar p; outputs.

#### B24 — Reproducible failure classification and trace examples

- **Epic:** E9.
- **Goal:** Apply ordered, nonexclusive heuristics and choose representative observable traces.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/evaluation/failures.py`; `tests/test_failures.py`.
- **Heuristics:** map literal failures using precedence/data: recall zero; VFS no new gold after calls; recall<1; all evidence+wrong with unsupported citations distinction; explicit budget; repeats `>=3` or repeats/calls `>=0.5`; invalid>0; parse false; allow multiple labels sorted enum order.
- **Examples:** choose lexicographically smallest task ID for VFS win, snippet win, both fail; excessive case chooses most repeats then task ID; export only question, observable trace, evidence, answers, metrics, no hidden reasoning.
- **Named tests:** `test_initial_retrieval_miss_rule`; `test_incomplete_vs_composition_rules`; `test_budget_and_invalid_rules`; `test_excessive_repeat_threshold`; `test_failure_labels_stable_order`; `test_trace_example_categories`; `test_examples_exclude_hidden_reasoning`.
- **Acceptance:** `uv run pytest tests/test_failures.py`; `uv run ruff check src/bm25_vfs_ablation/evaluation/failures.py tests/test_failures.py`.
- **Dependencies:** B22.
- **Do not touch:** schemas, stats, report.
- **Evidence bundle:** two paths; `7 passed`; heuristic matrix; outputs.

#### B25 — Aggregate CSV and trace table writer

- **Epic:** E9.
- **Goal:** Convert validated records into all stable tables required by analysis and reporting.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/evaluation/aggregates.py`; `tests/test_aggregates.py`.
- **Source spec:** load records to typed flattened DataFrame; compute condition/hop/category/intervention summaries, paired outcomes (`both_succeed`, `snippets_only`, `vfs_only`, `both_fail`), retrieval/evidence/tool/efficiency metrics, success per 1000 tokens, budget buckets, costs, regression table, trace JSON.
- **Files:** exact seven table outputs in §2.5 plus `task_level.csv`, `stratified_summary.csv`, `failure_summary.csv`; stable column/row order; empty optional cells produce headers and a status column, not crashes.
- **Named tests:** `test_run_frame_flattens_complete_schema`; `test_paired_outcome_counts`; `test_success_per_thousand_tokens`; `test_intervention_uses_permitted_not_actual_calls`; `test_csv_order_and_newlines`; `test_empty_optional_analysis_writes_status`; `test_trace_examples_json_safe`.
- **Acceptance:** `uv run pytest tests/test_aggregates.py`; `uv run ruff check src/bm25_vfs_ablation/evaluation/aggregates.py tests/test_aggregates.py`.
- **Dependencies:** B23, B24.
- **Do not touch:** plots/report/CLI.
- **Evidence bundle:** two paths; `7 passed`; output filenames and headers; outputs.

### Epic E10 — Plots and research report

#### B26 — Ten required deterministic plots

- **Epic:** E10.
- **Goal:** Produce every SPEC §10 figure headlessly with honest missing-cell behavior.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/evaluation/plots.py`; `tests/test_plots.py`.
- **Source spec:** implement all ten named functions/files in §2.5; force `Agg`; fixed `figsize=(8,5)`, `dpi=120`, font DejaVu Sans, white background, deterministic category/color mapping; close figures.
- **Semantics:** plot 1 uses paired bootstrap interval; actual/permitted calls never conflated; binary success plots show mean and n; efficiency has latency and cost panels; missing intervention renders an annotated valid figure `No intervention data`.
- **Named tests:** `test_produce_all_ten_filenames`; `test_plot_backend_is_agg`; `test_success_plot_uses_paired_interval`; `test_actual_and_permitted_call_axes_differ`; `test_missing_intervention_plot_is_valid`; `test_all_figures_are_closed`.
- **Acceptance:** `uv run pytest tests/test_plots.py`; `uv run ruff check src/bm25_vfs_ablation/evaluation/plots.py tests/test_plots.py`.
- **Dependencies:** B25.
- **Do not touch:** aggregate calculations, report, image snapshots.
- **Evidence bundle:** two paths; `6 passed`; ten file sizes/dimensions; outputs.

#### B27 — Evidence-disciplined Markdown report

- **Epic:** E10.
- **Goal:** Generate the prescribed report from records/tables without inventing conclusions.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/experiment/reporting.py`; `tests/test_reporting.py`.
- **Source spec:** sections exactly executive summary, hypothesis/design, controls, dataset, primary, retrieval/evidence, tool use, intervention, efficiency, failures, threats, conclusions; embed relative links to ten plots and key tables.
- **Epistemic tags:** paragraphs with numeric claims start `[OBSERVED]`, `[INFERENCE]`, `[CAUSAL-INTERVENTION]`, or `[SPECULATION]`; intervention causal wording only when cell exists; calls regression always says associative/post-treatment.
- **Threats:** include all nine required threats verbatim in meaning; state judge disabled when so; missing analyses receive `Not run` with reason; report includes config/corpus hashes and exact reproduction commands.
- **Named tests:** `test_report_has_all_required_sections`; `test_report_has_all_validity_threats`; `test_report_labels_claim_types`; `test_report_never_calls_observed_calls_causal`; `test_report_handles_missing_intervention`; `test_report_links_ten_plots`; `test_report_contains_reproduction_commands`.
- **Acceptance:** `uv run pytest tests/test_reporting.py`; `uv run ruff check src/bm25_vfs_ablation/experiment/reporting.py tests/test_reporting.py`.
- **Dependencies:** B25, B26.
- **Do not touch:** committed `reports/experiment.md`; plots; README.
- **Evidence bundle:** two paths; `7 passed`; generated heading list and claim-tag grep; outputs.

### Epic E11 — CLI and offline integration

#### B28 — Generate and run CLI commands

- **Epic:** E11.
- **Goal:** Wire strict direct commands for dataset creation and experiment execution.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/cli.py`; `tests/test_cli_generate_run.py`.
- **Source spec:** Typer app/no shell calls; implement exact flags/exit codes from §2.6; resolve repo-relative config paths; `generate` refuses existing targets without force; `run` loads configured/imported bundle and selects mock or HTTP client.
- **Mock policy:** explicit mock run derives actions from task gold only when provider `mock`; snippet returns canonical answer/citations; VFS reads required gold spans in sorted fact order then answers; record `mock_oracle_policy=true` in scoring details so results cannot be mistaken for performance.
- **Named tests:** `test_generate_cli_defaults_and_hashes`; `test_generate_cli_refuses_overwrite`; `test_run_cli_mock_writes_two_records_per_task`; `test_run_cli_flag_overrides`; `test_cli_schema_error_exit_two`; `test_live_provider_requires_api_key`.
- **Acceptance:** `uv run pytest tests/test_cli_generate_run.py`; `uv run ruff check src/bm25_vfs_ablation/cli.py tests/test_cli_generate_run.py`.
- **Dependencies:** B06, B07, B22.
- **Do not touch:** entrypoint, evaluation/report commands, operator files.
- **Evidence bundle:** two paths; `6 passed`; CLI help excerpts and run-record count; outputs.

#### B29 — Evaluate and report CLI commands

- **Epic:** E11.
- **Goal:** Wire validation, aggregates, plots, and report generation with exact paths.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/cli.py`; `tests/test_cli_evaluate_report.py`.
- **Source spec:** add exact §2.6 flags; validate every run before producing anything; evaluate writes tables then all plots; primary default excludes oracle/intervention; report consumes or regenerates missing aggregate files deterministically and creates only target parents.
- **Named tests:** `test_evaluate_cli_writes_expected_outputs`; `test_evaluate_primary_excludes_design_cells`; `test_evaluate_invalid_jsonl_exit_two`; `test_report_cli_writes_markdown`; `test_report_cli_missing_aggregates_regenerates`.
- **Acceptance:** `uv run pytest tests/test_cli_evaluate_report.py`; `uv run ruff check src/bm25_vfs_ablation/cli.py tests/test_cli_evaluate_report.py`.
- **Dependencies:** B26, B27, B28.
- **Do not touch:** analysis modules, config YAML, operator files.
- **Evidence bundle:** two paths; `5 passed`; output tree listing; outputs.

#### B30 — Single-command smoke workflow

- **Epic:** E11.
- **Goal:** Implement and prove the fully offline generate→run→evaluate→report acceptance path.
- **FILE allowlist (2):** `src/bm25_vfs_ablation/cli.py`; `tests/test_smoke.py`.
- **Source spec:** `smoke` exact flags from §2.6; use workdir-contained config/data/results/reports/cache; remove no preexisting directory; fail if workdir nonempty except reuse through a newly created `run-{seed}` child; print corpus/task SHA-256 and artifact paths.
- **Test spec:** invoke CLI twice with separate temp roots; require identical corpus/tasks hashes, 16 primary records each, all schema-valid, both conditions/task, token ceilings respected, ten plots, CSVs, and report.
- **Named tests:** `test_offline_smoke_end_to_end`; `test_smoke_two_runs_have_equal_generation_hashes`; `test_smoke_records_are_paired_and_bounded`; `test_smoke_outputs_report_and_ten_plots`; `test_smoke_makes_no_network_request`.
- **Acceptance:** `uv run pytest tests/test_smoke.py`; `uv run ruff check src/bm25_vfs_ablation/cli.py tests/test_smoke.py`; `uv run python -m bm25_vfs_ablation smoke --workdir /tmp/bm25-vfs-smoke --seed 42`.
- **Dependencies:** B29.
- **Do not touch:** Makefile, committed data/results/reports, HTTP client.
- **Evidence bundle:** two paths; `5 passed`; smoke stdout, record count, both hashes; outputs.

### Epic E12 — Static deployment artifact and final contract gates

#### B31 — Best-effort Dockerfile

- **Epic:** E12.
- **Goal:** Deliver an ARM64-capable container recipe without pretending it was daemon-tested.
- **FILE allowlist (2):** `Dockerfile`; `tests/test_dockerfile_static.py`.
- **Dockerfile:** use `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`, copy `pyproject.toml`/`uv.lock` before source, run `uv sync --frozen --no-dev`, then copy source/config; set a nonroot user and module entrypoint. The operator may later replace the mutable base tag with a verified digest.
- **Static test:** assert base family, frozen sync, no secrets, nonroot `USER`, module entrypoint, and comment `UNTESTED: build host has no Docker daemon`.
- **Named tests:** `test_dockerfile_declares_untested_host_constraint`; `test_dockerfile_uses_frozen_lock`; `test_dockerfile_runs_nonroot`; `test_dockerfile_entrypoint`.
- **Acceptance:** `uv run pytest tests/test_dockerfile_static.py`; `uv run ruff check tests/test_dockerfile_static.py`; MUST NOT run Docker.
- **Dependencies:** B30 and operator prep P0 lockfile.
- **Do not touch:** `.dockerignore`, pyproject, lock, Makefile, source.
- **Evidence bundle:** two paths; `4 passed`; static assertions; explicit `Docker not run: daemon unavailable`.

#### B32 — Cross-cutting scientific contract tests

- **Epic:** E12.
- **Goal:** Add black-box assertions for the acceptance criteria most vulnerable to silent drift.
- **FILE allowlist (1):** `tests/test_scientific_contract.py`.
- **Test spec:** construct tiny fixture and assert identical object/fingerprint/corpus/config across conditions; equal ceiling; condition order stability; separate retrieval/evidence/success fields; complete VFS observability; associative labeling; intervention actual/permitted separation; no CoT-shaped keys.
- **Named tests:** `test_conditions_share_exact_index_and_corpus`; `test_conditions_receive_equal_token_ceiling`; `test_seed_and_config_reproduce_assignments`; `test_retrieval_evidence_success_are_separate`; `test_every_vfs_interaction_is_observable`; `test_tool_call_analysis_is_associative`; `test_intervention_separates_permitted_and_actual`; `test_run_schema_has_no_chain_of_thought_field`.
- **Acceptance:** `uv run pytest tests/test_scientific_contract.py`; `uv run ruff check tests/test_scientific_contract.py`.
- **Dependencies:** B30.
- **Do not touch:** production files or other tests.
- **Evidence bundle:** one path; `8 passed`; criterion-to-test matrix; outputs.

#### B33 — Final deterministic regeneration and acceptance test

- **Epic:** E12.
- **Goal:** Encode the exact byte-hash gate and one compact full-suite acceptance meta-test.
- **FILE allowlist (1):** `tests/test_acceptance.py`.
- **Test spec:** generate seed 42 twice in distinct `tmp_path` roots using public API; compare corpus and task bytes/hashes; run four-task public APIs twice excluding timestamps/latencies and compare canonical stable projections; assert expected package artifacts exist.
- **Named tests:** `test_seed42_corpus_and_tasks_sha256_equal_on_regeneration`; `test_stable_run_projection_equal_on_rerun`; `test_required_repository_artifacts_exist`.
- **Acceptance:** `uv run pytest tests/test_acceptance.py`; `uv run ruff check .`; `uv run pytest`.
- **Dependencies:** B31, B32.
- **Do not touch:** any production or operator-owned file; no golden hash hardcoded until generator is intentionally version-bumped.
- **Evidence bundle:** one path; `3 passed`; full-suite passed count; both regeneration hashes; lint output.

### Epic E13 — Operator-authored documentation handoff

#### B34 — Documentation content packet test

- **Epic:** E13.
- **Goal:** Provide executable checks for operator-authored README/Makefile/env skeleton completion without editing them.
- **FILE allowlist (1):** `tests/test_operator_docs.py`.
- **Test spec:** assert README mentions title, setup, generate/run/evaluate/report/smoke, design/fairness, limitations, associative warning, live model env names, Docker untested; Makefile includes exact targets; `.env.example` names only the three allowed vars and no secret value; SPEC exists.
- **Named tests:** `test_readme_required_topics`; `test_makefile_required_targets`; `test_env_example_contract`; `test_spec_present`.
- **Acceptance:** operator first completes P2; then `uv run pytest tests/test_operator_docs.py`; `uv run ruff check tests/test_operator_docs.py`.
- **Dependencies:** B33, operator prep P2.
- **Do not touch:** README, Makefile, env, SPEC, source.
- **Evidence bundle:** one path; `4 passed`; docs checklist; outputs.

#### B35 — Final repository acceptance run

- **Epic:** E13.
- **Goal:** Run, do not redesign, the final local gate and record the release-ready evidence.
- **FILE allowlist (0):** none; this is a verification-only agent run.
- **Verification:** confirm `git diff --name-only` against task start is empty; run exact block in §5; inspect smoke report for required sections; compute hashes from two smoke roots.
- **Failure handling:** report the first failing owning task ID from the traceability map; do not patch around it; board returns that issue to `status:ready`.
- **Named tests:** no new tests; all previously named tests must collect.
- **Acceptance:** exact §5 final command block, with Docker explicitly absent.
- **Dependencies:** B34.
- **Do not touch:** entire repository.
- **Evidence bundle:** package version; Python/uv versions; ruff output; pytest collected/passed count; smoke paths/hashes; `git status --short`; Docker-not-tested statement.

## §4 GitHub board specification

### §4.1 Labels, milestones, and workflow

Create labels exactly as follows.

| label family | labels and meaning |
|---|---|
| status | `status:blocked`, `status:ready`, `status:in-progress`, `status:review`, `status:done`; exactly one per task |
| type | `type:epic`, `type:build`, `type:test`, `type:operator`; exactly one per issue |
| epic | `epic:E1` through `epic:E13`; exactly one per issue |
| risk | `risk:science`, `risk:security`, `risk:determinism`, `risk:integration`, `risk:docs`; zero or more |
| scope | `scope:config`, `scope:data`, `scope:retrieval`, `scope:model`, `scope:harness`, `scope:vfs`, `scope:evaluation`, `scope:cli`, `scope:release` |

Use milestones `M1-foundation` (E1–E4), `M2-harnesses` (E5–E8), `M3-analysis` (E9–E10), and `M4-acceptance` (E11–E13).
Only the next dependency-satisfied issue is `status:ready`; later issues remain `status:blocked`.
Moving an issue to `status:done` requires its evidence bundle in the final comment and all acceptance commands green.
Issue titles begin with sortable IDs, so `gh issue list --json number,title,labels` exposes the next action without opening bodies.

Every epic issue body uses this exact outline:

- `Outcome:` one sentence copied from the epic purpose below.
- `Child issues:` checklist of its B IDs in strict order.
- `Exit gate:` all child issues done and milestone-wide tests green.
- `Scientific reference:` relevant SPEC sections and PLAN §0/§2 contracts.
- `Non-goals:` no implementation in the epic issue itself.

| epic issue title | labels | outcome / exit gate |
|---|---|---|
| `E1 Package and configuration foundation` | `type:epic, epic:E1, risk:determinism` | Importable strict-config package / B01–B03 done |
| `E2 Schemas and deterministic artifacts` | `type:epic, epic:E2, risk:determinism` | Frozen records and byte-stable I/O / B04–B05 done |
| `E3 Synthetic dataset and imports` | `type:epic, epic:E3, risk:science` | Valid exact-provenance data / B06–B07 done |
| `E4 Shared retrieval layer` | `type:epic, epic:E4, risk:science` | One stable chunk/index implementation / B08–B09 done |
| `E5 Token fairness and model boundary` | `type:epic, epic:E5, risk:science` | Budgeted offline/HTTP model boundary / B10–B12 done |
| `E6 Primary harnesses` | `type:epic, epic:E6, risk:science` | Shared contract plus Condition A / B13–B14 done |
| `E7 Secure read-only VFS` | `type:epic, epic:E7, risk:security` | Sandboxed tools plus Condition B / B15–B18 done |
| `E8 Scoring and run construction` | `type:epic, epic:E8, risk:science` | Complete paired resumable records / B19–B22 done |
| `E9 Statistics and failure analysis` | `type:epic, epic:E9, risk:science` | Prespecified inference and tables / B23–B25 done |
| `E10 Plots and research report` | `type:epic, epic:E10, risk:docs` | Ten figures and disciplined report / B26–B27 done |
| `E11 CLI and offline integration` | `type:epic, epic:E11, risk:integration` | Direct commands and smoke path / B28–B30 done |
| `E12 Deployment artifact and gates` | `type:epic, epic:E12, risk:integration` | Static Docker plus cross-cutting gates / B31–B33 done |
| `E13 Operator documentation handoff` | `type:epic, epic:E13, risk:docs` | Operator files verified and acceptance recorded / B34–B35 done |

Every task issue body MUST contain these headings and no substituted shorthand:

1. `Goal` — copy the card goal.
2. `Deliverables / FILE allowlist` — copy every path and per-file spec; state writes outside it are forbidden.
3. `Acceptance` — copy named tests and exact commands.
4. `Dependencies` — link prior B issues; unchecked dependencies imply `status:blocked`.
5. `Verification evidence` — copy the evidence bundle as an unchecked list.
6. `Reference` — `PLAN.md §3 Bxx`, applicable §2 contract, and SPEC section numbers.
7. `Do not touch` — copy verbatim.

Create task issues with these exact titles; the phrase after the dash is sufficient to identify next work in a list-only view.

| issue title | labels | body-specific deliverable reference |
|---|---|---|
| `B01 — Package bootstrap and frozen enums` | `type:build, epic:E1, scope:config` | §3 B01 three package files |
| `B02 — Typed config loader and default config` | `type:build, epic:E1, scope:config, risk:determinism` | §3 B02 strict config/default YAML |
| `B03 — Intervention config overlay` | `type:build, epic:E1, scope:config, risk:science` | §3 B03 intervention YAML |
| `B04 — Corpus, task, and run schemas` | `type:build, epic:E2, scope:data, risk:science` | §3 B04 complete schemas |
| `B05 — Canonical JSONL and hashing utilities` | `type:build, epic:E2, scope:data, risk:determinism` | §3 B05 artifact primitives |
| `B06 — Deterministic enterprise corpus generator` | `type:build, epic:E3, scope:data, risk:determinism` | §3 B06 generator |
| `B07 — Validated corpus/task loader and import path` | `type:build, epic:E3, scope:data` | §3 B07 loader |
| `B08 — Normalization, approximation, and fixed chunking` | `type:build, epic:E4, scope:retrieval, risk:science` | §3 B08 chunker/tokenizer |
| `B09 — Stable shared BM25 index` | `type:build, epic:E4, scope:retrieval, risk:determinism` | §3 B09 index |
| `B10 — Exact budget controller` | `type:build, epic:E5, scope:harness, risk:science` | §3 B10 budget controller |
| `B11 — Model protocol, HTTP client, and scripted mock` | `type:build, epic:E5, scope:model` | §3 B11 clients/mock |
| `B12 — Response cache` | `type:build, epic:E5, scope:model, risk:determinism` | §3 B12 cache |
| `B13 — Base harness contracts and answer instruction` | `type:build, epic:E6, scope:harness` | §3 B13 base contract |
| `B14 — One-call snippet harness` | `type:build, epic:E6, scope:harness, risk:science` | §3 B14 Condition A |
| `B15 — Path security boundary` | `type:build, epic:E7, scope:vfs, risk:security` | §3 B15 path resolver |
| `B16 — Bounded virtual filesystem operations` | `type:build, epic:E7, scope:vfs, risk:security` | §3 B16 filesystem |
| `B17 — VFS tool executor and schemas` | `type:build, epic:E7, scope:vfs, risk:security` | §3 B17 four tools |
| `B18 — Iterative VFS harness` | `type:build, epic:E7, scope:harness, risk:science` | §3 B18 Condition B |
| `B19 — Deterministic answer and citation scoring` | `type:build, epic:E8, scope:evaluation, risk:science` | §3 B19 scorer |
| `B20 — Retrieval and accessed-evidence scoring` | `type:build, epic:E8, scope:evaluation, risk:science` | §3 B20 evidence metrics |
| `B21 — Design expansion and randomized order` | `type:build, epic:E8, scope:evaluation, risk:science` | §3 B21 assignments |
| `B22 — Experiment runner, complete records, and resume` | `type:build, epic:E8, scope:harness, risk:integration` | §3 B22 runner |
| `B23 — Paired inference and associative mechanism models` | `type:build, epic:E9, scope:evaluation, risk:science` | §3 B23 statistics |
| `B24 — Reproducible failure classification and trace examples` | `type:build, epic:E9, scope:evaluation` | §3 B24 failures |
| `B25 — Aggregate CSV and trace table writer` | `type:build, epic:E9, scope:evaluation` | §3 B25 tables |
| `B26 — Ten required deterministic plots` | `type:build, epic:E10, scope:evaluation` | §3 B26 figures |
| `B27 — Evidence-disciplined Markdown report` | `type:build, epic:E10, scope:evaluation, risk:docs` | §3 B27 report generator |
| `B28 — Generate and run CLI commands` | `type:build, epic:E11, scope:cli, risk:integration` | §3 B28 CLI first half |
| `B29 — Evaluate and report CLI commands` | `type:build, epic:E11, scope:cli, risk:integration` | §3 B29 CLI second half |
| `B30 — Single-command smoke workflow` | `type:test, epic:E11, scope:cli, risk:integration` | §3 B30 smoke |
| `B31 — Best-effort Dockerfile` | `type:build, epic:E12, scope:release` | §3 B31 static-only container artifact |
| `B32 — Cross-cutting scientific contract tests` | `type:test, epic:E12, scope:evaluation, risk:science` | §3 B32 black-box contracts |
| `B33 — Final deterministic regeneration and acceptance test` | `type:test, epic:E12, scope:release, risk:determinism` | §3 B33 hash gates |
| `B34 — Documentation content packet test` | `type:operator, epic:E13, scope:release, risk:docs` | §3 B34 operator-file checks |
| `B35 — Final repository acceptance run` | `type:test, epic:E13, scope:release, risk:integration` | §3 B35 verification only |

Initial board state is B01 `status:ready`; B02–B35 `status:blocked`; all epics `status:blocked` except E1 `status:in-progress`.

## §5 Test and verification plan

### §5.1 Required-test traceability

| SPEC §12 requirement | concrete gate |
|---|---|
| identical corpus/index usage | `tests/test_bm25.py::test_same_index_object_can_serve_both_conditions`; `tests/test_runner.py::test_runner_shares_corpus_and_index_identity`; B32 black-box equivalent |
| deterministic dataset generation | B06 same/different seed tests; B33 byte-level SHA-256 regeneration test |
| BM25 ranking stability | `test_bm25_known_ranking`; `test_bm25_tie_breaks_by_chunk_id`; fingerprint test |
| VFS traversal prevention | all B15 security tests, including symlink and exact code assertions |
| bounded read/cat/grep | B16 range/cat tests; B17 preview, refusal, and token-bound tests |
| exact token-budget enforcement | all seven B10 boundary/multiturn tests; harness pre-call tests |
| accounting across model turns | `test_multiturn_accounting_no_double_count`; argument replay test; `test_vfs_budget_across_turns` |
| task scorer correctness | all B19 normalization/parser/citation tests |
| evidence recall calculation | all B20 hand-calculated span/recall tests |
| JSONL schema validation | B04 complete-shape tests; B05 bad-line test; B22 complete-record test |
| resume after interruption | `test_runner_resume_skips_completed_calls`; duplicate/incomplete-file rejection |
| paired statistical analysis | all B23 paired bootstrap/McNemar/strata tests |
| mocked end-to-end experiment | B30 five smoke tests and direct smoke command |

The planned suite contains 197 explicitly named test functions in §3; parametrized cases may make pytest's reported case count larger.
No test may require network, credentials, system time equality, Docker, a GUI, or a writable path outside pytest temp directories.
Tests use fixed hand-built records where scientific arithmetic matters and generated fixtures only where pipeline composition matters.
Floating comparisons specify `pytest.approx(abs=1e-12)` unless a statistical library result reasonably needs `1e-8`.

### §5.2 Determinism gates

Gate D1 compares exact `corpus.jsonl` bytes and exact `tasks.jsonl` bytes from two seed-42 roots.
Gate D2 asserts every generated list/order/ID is stable and no generated field contains a timestamp.
Gate D3 compares BM25 chunk order and corpus fingerprint across construction runs.
Gate D4 compares condition assignments and cache keys across processes through known vectors.
Gate D5 compares stable run projections after removing `started_at`, `finished_at`, latency fields, and cache-hit state; answers, traces, tokens, hashes, and scores remain equal.
PNG byte equality is not a gate because Matplotlib metadata may vary; dimensions, filenames, plotted data inputs, and semantic labels are tested.

### §5.3 Mocked end-to-end definition

The smoke dataset has 4 development and 8 evaluation tasks, seed 42, at least two hop counts, and six distractors/task.
The smoke run selects eight eval tasks, primary cell, both conditions, ceiling 4096, and `mock-scripted-v1`.
The scripted snippet response is one final action.
The scripted VFS response reads each required fact span in sorted fact-ID order, then emits the same answer.
This means smoke validates plumbing, observability, scoring, and budgets; its success rate MUST NOT be interpreted as evidence for either hypothesis.
The run must contain 16 unique records, 8 paired tasks, no errors, valid returned model IDs, nonempty VFS traces, empty snippet traces, and no network requests.
Evaluation must yield nine CSV plus one trace JSON outputs, ten PNGs, and one Markdown report.

### §5.4 Acceptance criteria traceability

| SPEC §15 criterion | owner task and named gate |
|---|---|
| same corpus/index | B22/B32; identity tests |
| same token ceiling | B10/B22/B32; exact-boundary and equal-ceiling tests |
| seed + committed config reproducibility | B02/B06/B33; default config and SHA gates |
| observable VFS | B17/B18/B32; envelope and interaction tests |
| paired uncertainty | B23; paired fixed-seed CI and exact McNemar |
| separate retrieval/evidence/reasoning/efficiency | B20/B25/B32; schema and aggregate tests |
| associative call analysis | B23/B27/B32; HC3 and wording tests |
| one-command local smoke | B30; `test_offline_smoke_end_to_end` |
| tests pass | B35; full pytest command |
| README setup/design/limitations | operator P2 + B34 content test |

### §5.5 Exact final acceptance block

Run from repository root, with the operator-prepared lock and no credentials:

```bash
uv sync --frozen
uv run ruff check .
uv run pytest
tmp_a="$(mktemp -d /tmp/bm25-vfs-smoke-a.XXXXXX)"
tmp_b="$(mktemp -d /tmp/bm25-vfs-smoke-b.XXXXXX)"
uv run python -m bm25_vfs_ablation smoke --workdir "$tmp_a" --seed 42
uv run python -m bm25_vfs_ablation smoke --workdir "$tmp_b" --seed 42
sha256sum "$tmp_a"/run-42/data/generated/corpus.jsonl "$tmp_b"/run-42/data/generated/corpus.jsonl
sha256sum "$tmp_a"/run-42/data/generated/tasks.jsonl "$tmp_b"/run-42/data/generated/tasks.jsonl
cmp "$tmp_a"/run-42/data/generated/corpus.jsonl "$tmp_b"/run-42/data/generated/corpus.jsonl
cmp "$tmp_a"/run-42/data/generated/tasks.jsonl "$tmp_b"/run-42/data/generated/tasks.jsonl
```

The two pairs of SHA-256 values MUST match and both `cmp` commands MUST exit zero.
Do not add `docker build`, a live endpoint call, or a judge call to this block.

## §6 Execution order and parallelization

### §6.1 Operator preparation before dispatch

- **P0 required before B01:** complete `pyproject.toml` for Python `>=3.12,<3.13`, src-layout packaging, and the exact runtime dependency names `pydantic`, `typer`, `PyYAML`, `rank-bm25`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `httpx`; dev names `pytest`, `ruff`; resolve latest compatible releases once with `uv lock`, commit `uv.lock`, then run `uv sync --frozen`.
- **P0 constraint:** no implementer may run `uv add`, `pip`, or network-backed commands; a genuinely missing dependency returns to operator rather than widening a card.
- **P1 before B01:** ensure ruff targets Python 3.12 and pytest discovers `tests`; do not create source placeholders on behalf of agents.
- **P2 after B33 and before B34:** update operator-owned `Makefile`, `README.md`, `AGENTS.md`, `.gitignore`, and `.env.example` to the exact §2.6 and B34 contracts; ignore `.smoke/`, generated data, results cache/plots/aggregates, and generated reports while retaining `.gitkeep` if present.
- **P3 never required for acceptance:** credentials, model endpoint, Docker daemon, fixture download, and live generated corpus commits.

### §6.2 Strict dispatch sequence

Dispatch exactly:

`B01 → B02 → B03 → B04 → B05 → B06 → B07 → B08 → B09 → B10 → B11 → B12 → B13 → B14 → B15 → B16 → B17 → B18 → B19 → B20 → B21 → B22 → B23 → B24 → B25 → B26 → B27 → B28 → B29 → B30 → B31 → B32 → B33 → P2 → B34 → B35`.

This deliberately follows the SPEC §14 skeleton: scaffold/schema, data, retrieval, snippet, VFS, iterative budgeted agent, mock integration, real-client configuration, evaluation, intervention, report/plots, smoke, documentation.
B10 appears before concrete harnesses so fairness is not retrofitted.
B11 includes the real-client boundary before acceptance integration, but no live call.
B21 adds intervention assignment before the runner to prevent special-case runner branching later.

Parallel execution is disabled by default because cheap isolated agents benefit from seeing a green predecessor commit.
If the operator accepts merge/rebase overhead, only these pairs MAY run concurrently after all stated dependencies are done:

- B11 and B15: disjoint model-client and VFS-security paths.
- B19 and B21: disjoint scoring and design paths.
- B23 and B24: disjoint statistics and failure paths.
- B26 and B27 MUST NOT run concurrently despite disjoint files, because B27 relies on finalized output names/semantics from B26.
- No two cards that share `config.py`, `cli.py`, or an operator-owned file may run concurrently.

No fixture generation is operator prep: tests create temporary fixtures.
Generated default data is created only by `make generate` or direct CLI after implementation.

## §7 Out of scope and deferred

### §7.1 Repository must fully support

- Deterministic synthetic generation at 50 dev/200 eval defaults and configurable corpus/distractor/hop/seed values.
- Strict import of user corpus/task JSONL with exact provenance supplied by the user.
- Both primary harnesses on one index, one task set, and one total estimated-token ceiling.
- A real OpenAI-compatible endpoint configured through YAML and the three allowed environment variables.
- A zero-network scripted mock for every acceptance path.
- Full traces, resume, caching, scoring, paired statistics, strata, regression, failure analysis, tables, plots, and report.
- Tool-call-limit intervention at every prescribed limit and oracle candidate mode as separately labeled design cells.
- Static Dockerfile artifact and documented nonvalidation.

### §7.2 Operator-gated after build

- Actual live-model headline runs, provider credentials, chosen commercial model/version, rate-limit scheduling, and monetary price entries.
- Scientific validation that the synthetic benchmark predicts behavior on real enterprise corpora.
- Running the full 200-task x conditions experiment or the six-level intervention grid; implementation supports them, smoke does not pay for them.
- Running oracle cells at publication scale.
- Enabling or validating an LLM judge; the build reserves only disabled secondary fields and mock tests.
- Calibration against a provider tokenizer; the build documents and uses `regex_v1`, then records provider reconciliation.
- Docker build, run, image publication, or multi-architecture verification; host has no daemon and none may gate merge.
- CI configuration, branch protection, release automation, badges, website/dashboard, hosted results, or GitHub Pages.
- Publication claims, causal mediation claims, preregistration, human annotation, and external peer review.

### §7.3 Explicitly prohibited scope creep

- LangChain, LlamaIndex, an agent framework, database, vector search, neural reranker, browser UI, service daemon, or distributed runner.
- Automatic corpus fact extraction or gold-label guessing for imports.
- A third primary harness, prompt optimization per condition, condition-specific BM25/chunking, or unequal token ceilings.
- Logging hidden chain-of-thought, provider secrets, or raw unredacted authorization errors.
- Treating actual calls as randomized treatment or calling within-VFS regression mediation.

## §8 Open questions and brief flaws

Everything needed to build is given a default below; the operator can ratify this section without blocking dispatch unless an item is promoted to a requirement change.

- **[ASSUME] Dependency versions:** the brief asks for exact dependency names while explicitly saying assume latest resolvable versions. The defensible resolution is exact names in this plan plus exact versions/hashes in the operator-generated `uv.lock`; inventing current version numbers offline would be less reproducible.
- **[UNVERIFIED] Python host mismatch:** the stated target is Python 3.12.13, but the scratch shell reported Python 3.13.5. P0 must ensure `uv` installs/selects 3.12.13 before any acceptance claim.
- **[ASSUME] VFS materialization:** generated JSONL is authoritative, while runner materializes document content into a temporary corpus root for filesystem operations. It does not expose the JSONL or host repo through VFS.
- **[ASSUME] `grep` meaning:** the brief says search corpus and use BM25 where applicable. This plan makes it BM25 query ranking plus literal normalized-term line previews; it is not regex syntax despite the familiar name.
- **[ASSUME] Token fairness:** provider-style repeated conversation input is charged on every turn. This inherently spends more budget on VFS protocol/history; prompt overhead is reported as a validity threat rather than waived.
- **[ASSUME] Tool result before next call:** a local result consumes no LLM tokens until inserted into a request. The executor still truncates it prospectively so the following request can be admitted.
- **[ASSUME] Zero-call intervention:** max 0 still permits one initial answer call and forbids tool execution. Otherwise the cell could never produce an answer and would not test premature/no-search behavior.
- **[ASSUME] Intervention pairing:** each maximum-call cell contains an unchanged snippet record as a cached/repeated control plus one VFS record. This costs storage but makes paired cell analysis unambiguous.
- **[ASSUME] Oracle payload:** oracle mode supplies gold document references/paths, not gold chunks, facts, snippets, or answer. Supplying content would collapse evidence inspection into retrieval.
- **[ASSUME] Correctness:** exact normalized answer membership is viable because synthetic tasks are generated with structured short answers. Imported tasks must supply exhaustive acceptable variants or accept conservative false negatives.
- **[UNVERIFIED] Free-text user corpora:** exact fact spans, gold chunks, aliases, and no-single-chunk validation cannot be inferred from arbitrary files. The advertised import path therefore requires schema-complete corpus/tasks, not an unlabeled folder.
- **[ASSUME] Cost:** price defaults are zero because prices are provider/model/time-specific. Later live configs must set both per-million rates; zero cost is displayed as `not priced` rather than `free` in prose.
- **[UNVERIFIED] `rank-bm25` compatibility:** it is pure Python atop NumPy and is expected to work on Python 3.12/aarch64, but P0 lock resolution is the authoritative compatibility check because agents have no network.
- **[UNVERIFIED] `statsmodels` robust covariance:** exact call syntax can vary across releases. B23 targets current public GLM/Logit APIs available in the locked version and must record convergence/failure rather than silently downgrade covariance.
- **[ASSUME] Regression repeated templates:** HC3 handles heteroskedasticity but not template clustering. The report must call shared-template dependence a threat; a cluster-robust sensitivity analysis is deferred because few template clusters can make it unstable.
- **[ASSUME] Failure labels:** categories are nonexclusive. Forcing one mutually exclusive label would hide compound failures such as retrieval miss plus budget exhaustion.
- **[ASSUME] Latency determinism:** latency is measured and reported but excluded from deterministic equality. Mock latency may be zeroed in stable projections, never asserted byte-equal at runtime.
- **[ASSUME] Plot determinism:** semantic content and filenames are gated, not PNG hashes; renderer/library metadata makes binary plot equality a weak reproducibility test.
- **[UNVERIFIED] Docker base mutability:** the Dockerfile uses a named uv/Python image because the operator did not provide an image digest. For publication-grade container reproducibility, the operator should later pin a verified multi-arch digest; build remains explicitly untested here.
- **[UNVERIFIED] Docker and public source install:** `uv sync --frozen --no-dev` assumes P0's package metadata and lock include the local project correctly. Static tests cannot establish that the image builds.
- **[ASSUME] Generated artifacts in git:** `data/`, `results/`, and `reports/experiment.md` are runtime outputs and not committed by implementation tasks. `.gitkeep` ownership remains with the operator skeleton.
- **[ASSUME] Exact test count:** 196 named functions are planned; parametrization can raise collected cases. Acceptance requires zero failures, not a brittle exact collected-case total.
- **[FLAW] The brief requests `make setup` while saying the operator alone maintains Makefile.** Implementation cards cannot deliver that edit. P0/P2 explicitly place all Make targets with the operator, and B34 verifies them.
- **[FLAW] The brief requests README completion while reserving README to the operator.** B34 tests the required content, but the operator must author it in P2.
- **[FLAW] The brief asks for Docker support on a confirmed no-Docker host.** B31 supplies only a statically checked best-effort artifact and no acceptance command may imply validation.
- **[FLAW] Gold chunk IDs depend on chunking configuration, yet imports may use a different config.** Loader/run validation must reject a chunk fingerprint mismatch; users must regenerate task gold chunks for changed chunking rather than compare incompatible experiments.
- **[FLAW] Exact provider token counts usually arrive only after a call, so they cannot enforce a prospective hard ceiling.** This plan uses one documented estimator for enforcement and records provider deltas; a provider may report actual usage above the estimate without making the run silently comparable.
- **[FLAW] Same total ceiling does not make prompt affordances identical.** Tool schemas/history consume budget and snippet IDs/content consume budget differently. Component accounting and the prompt-overhead validity threat are mandatory.
- **[FLAW] Actual tool-call count is post-treatment and undefined in snippets.** Only assigned maximum calls support a causal dose interpretation; within-VFS call regression remains associative.
- **[FLAW] Shared templates violate naive independent-task assumptions.** Paired bootstrap over tasks is required by the brief but may understate uncertainty; the report must disclose this and future work may bootstrap template clusters.
- **[FLAW] The deterministic mock is gold-aware by design.** It validates software, not model capability. Records carry `mock_oracle_policy=true`, and reports must suppress scientific headline language for mock-only results.
- **[FLAW] A single exact answer string can penalize correct paraphrases.** Acceptable variants mitigate this for generated tasks; the optional secondary judge is intentionally not allowed to replace primary scoring.
- **[FLAW] Randomized execution order cannot fully control provider drift.** It balances order within task but live retries/cache timing may still differ; returned model ID, timestamps, cache hits, and order are therefore recorded.

Operator ratification default: accept every `[ASSUME]`, track every `[UNVERIFIED]` as a nonblocking P0 or post-build check, and preserve every `[FLAW]` in README/report limitations.
