# SPEC — BM25 VFS Workspace vs BM25 Snippet Ablation

Status: operator-authored brief, frozen 2026-09-19. Authoritative contract for this repository.
Execution plan: `PLAN.md`. Agent conventions: `AGENTS.md`. Sibling house-style reference: `rmax-ai/rsi-gym`.

---

Build a reproducible Python proof of concept titled:

"BM25 VFS Workspace vs BM25 Snippet Ablation"

Research question

Does giving an LLM a navigable virtual filesystem (VFS) with search/read tools improve multi-hop question answering over returning BM25 top-k snippets directly, when both conditions use the same corpus, BM25 index, model, query set, and total token budget?

Primary hypothesis

BM25 + VFS tools achieves higher end-to-end task success than BM25-only snippets on multi-hop tasks.

Mechanism hypothesis

The improvement is associated with adaptive tool use—especially the number and sequence of tool calls—and cannot be explained by first-stage retrieval recall alone.

Important epistemic constraint

This PoC may establish an association compatible with mediation, but it must not claim causal mediation from tool-call count without an explicit intervention. Include an optional controlled experiment that imposes tool-call limits to test the mechanism more directly.

## 1. Deliverable

Create a complete, runnable repository containing:

- Two agent harnesses over one shared corpus and BM25 index.
- A deterministic multi-hop task generator.
- A fair token-budget controller.
- Evaluation at task, evidence, retrieval, token, latency, and tool-use levels.
- Statistical comparison of both harnesses.
- An optional tool-call-budget intervention.
- CLI commands to generate data, run experiments, evaluate results, and produce plots.
- Unit and integration tests.
- A concise research report generated from experiment outputs.
- Docker support and a Makefile or equivalent task runner.

Use Python 3.12.

Prefer a small, inspectable implementation over a large framework. Use standard libraries plus focused dependencies such as:

- rank-bm25 or bm25s
- pydantic
- typer
- pandas
- numpy
- scipy
- statsmodels
- matplotlib or seaborn
- pytest
- an OpenAI-compatible HTTP client for model inference

Do not depend on LangChain or another agent framework unless strictly necessary.

## 2. Experimental conditions

Implement exactly two primary conditions.

### Condition A: BM25 snippet harness

For each task:

1. Submit the task query to the shared BM25 index.
2. Retrieve the top-k chunks.
3. Construct a prompt containing:
   - the question;
   - retrieved snippets;
   - stable document and chunk identifiers;
   - instructions to answer and cite supporting evidence.
4. Make one model call.
5. Do not expose filesystem or retrieval tools to the model.

Configuration:

- top_k
- maximum retrieval-context tokens
- maximum answer tokens
- BM25 parameters
- chunk size and overlap

### Condition B: BM25 VFS harness

For each task:

1. Expose the same indexed corpus as a read-only virtual filesystem.
2. Give the model a minimal initial context containing:
   - the question;
   - tool descriptions;
   - corpus manifest or initial BM25-ranked file candidates;
   - the remaining token budget.
3. Allow iterative use of these tools:
   - grep(query, path=None, max_results=N)
     Search indexed corpus content and return matching paths, line ranges, and compact previews. Rank results using the same BM25 index where applicable.
   - read(path, start_line, end_line)
     Return a bounded line range from one file.
   - cat(path)
     Return an entire file only if it fits within the remaining budget and configured per-call limit.
   - list(path="/")
     List files or directories. Include this only if required for navigation.
4. Record every tool request, tool result, token cost, and timestamp.
5. Stop when:
   - the model produces a final answer;
   - the total token budget is exhausted;
   - the maximum number of tool calls is reached;
   - an invalid-call threshold is reached.

The VFS must be sandboxed and read-only. Reject path traversal and access outside the corpus root.

## 3. Fairness and controlled variables

Hold constant across both conditions:

- model and model version;
- temperature and sampling parameters;
- question set;
- corpus;
- BM25 implementation and index;
- chunking policy;
- answer format;
- scoring logic;
- total per-task token budget;
- system instructions, except where tools require condition-specific instructions;
- retry policy;
- concurrency policy.

Randomize condition execution order per task to reduce time- and provider-related bias.

Cache model responses using a hash of:
- condition;
- model configuration;
- prompt/messages;
- tool schema;
- corpus version;
- experiment configuration.

Record the exact model identifier returned by the provider.

### Token-budget accounting

Define total task cost as:

- all input tokens;
- retrieved snippet or tool-result tokens;
- model output tokens across every turn;
- tool-call arguments serialized into the conversation.

Local BM25 computation does not consume LLM tokens but must be reported separately as latency.

The snippet harness and VFS harness must receive the same total LLM-token ceiling. Do not compare a one-call snippet run against an unconstrained iterative agent.

If exact provider token counts are unavailable during execution, use the model's tokenizer or a documented approximation to enforce the budget, then reconcile against provider usage metadata afterward.

## 4. Dataset and task generator

Build a synthetic but realistic versioned corpus that makes ground truth and evidence provenance exact.

Generate documents representing an enterprise environment, for example:

- service documentation;
- incident reports;
- architecture decisions;
- ownership records;
- deployment histories;
- policy documents;
- dependency manifests.

Generate multi-hop questions requiring evidence from 2–4 separate facts, preferably across multiple documents.

Example task:

- Service Atlas depends on library Quartz.
- An architecture decision states that Quartz versions below 4.2 use legacy encryption.
- A deployment record states Atlas currently runs Quartz 4.1.
- A policy says systems using legacy encryption require a security review.
- Question: "Does Atlas require a security review, and why?"

The correct answer requires composing facts rather than locating one matching sentence.

For every task, store:

- task_id
- question
- canonical answer
- acceptable answer variants
- required supporting fact IDs
- gold document IDs
- gold chunk IDs
- hop count
- task template/category
- distractor documents
- generator seed
- corpus version

Prevent trivial leakage:

- Do not repeat the exact question wording in a source document.
- Use aliases and paraphrases.
- Include plausible distractors sharing keywords.
- Separate linking facts across documents.
- Verify that no single chunk contains the complete answer.
- Split train/development tasks from final evaluation tasks by templates or entity combinations, not only random rows.

Implement at least:

- 50 development tasks for pipeline testing;
- 200 evaluation tasks by default;
- configurable corpus size, distractor count, hop count, and random seed.

Also support importing a user-provided corpus and JSONL task file.

## 5. Measurements

### Primary outcome

task_success: whether the final answer is correct.

Implement deterministic exact/structured scoring wherever possible. For free-text answers, extract a structured answer or use a rubric-based judge only as a secondary scorer.

Report:

- success rate by condition;
- paired difference in success rate;
- 95% confidence interval using paired bootstrap;
- McNemar's test for paired binary outcomes;
- results by hop count and task category.

### Retrieval and evidence measures

Measure separately:

- initial BM25 Recall@k for gold chunks;
- initial BM25 Recall@k for gold documents;
- final evidence recall after all VFS calls;
- evidence precision;
- number of distinct gold facts accessed;
- whether all required evidence was accessed;
- citation correctness, if citations are required.

Keep these separate from final task success. High retrieval recall does not imply correct multi-hop synthesis.

### Tool-use measures

For VFS runs record:

- total tool calls;
- calls by tool type;
- successful and invalid calls;
- repeated calls;
- unique files read;
- unique chunks/lines accessed;
- tokens returned per tool;
- tool-call sequence;
- time to first gold fact;
- time/calls to complete gold evidence coverage.

### Efficiency measures

Record:

- total input, output, and combined tokens;
- wall-clock latency;
- model latency;
- retrieval/tool latency;
- estimated monetary cost;
- answer success per 1,000 tokens;
- answer success by token-budget bucket.

## 6. Covariate and mechanism analysis

Tool-call count exists only in the VFS condition, so do not naively treat it as an ordinary mediator shared across both conditions.

Implement these analyses:

### A. Descriptive within-VFS analysis

Model VFS task success using:

- tool-call count;
- initial retrieval recall;
- hop count;
- task category;
- total tokens consumed;
- corpus difficulty or distractor count.

Use logistic regression with robust standard errors.

Include nonlinear tool-use effects using either:
- tool-call-count bins; or
- a quadratic/spline term.

This matters because zero calls may indicate premature answering, moderate calls may help, and excessive calls may indicate search failure.

Clearly label this analysis as associative.

### B. Matched/stratified comparisons

Compare success across:

- matched token-usage bands;
- initial retrieval-recall bands;
- hop counts;
- task categories.

Determine whether the VFS advantage remains when initial recall and token use are comparable.

### C. Controlled tool-call intervention

Add an optional experiment with VFS maximum tool calls in:

[0, 1, 2, 4, 8, 12]

Use the same tasks and token ceiling for every limit. This intervention is the stronger test of whether additional adaptive interaction causes improved success.

Report a tool-call dose-response curve while noting that the maximum permitted calls and actual calls are different variables.

### D. Optional retrieval-oracle analysis

Add an oracle mode in which both conditions are supplied with references to all gold documents, while only VFS can navigate adaptively. Use this to partially separate retrieval failure from evidence inspection and reasoning failure.

Keep oracle results separate from primary results.

## 7. Output schema

Write one JSONL record per task-condition run.

Include:

- experiment ID
- task ID
- condition
- model configuration
- seed
- corpus version
- question
- final answer
- parsed answer
- correctness
- scoring details
- citations
- initial retrieved documents/chunks and scores
- gold evidence
- accessed evidence
- tool trace
- tool-call count
- token accounting by turn
- total tokens
- latency breakdown
- termination reason
- errors
- prompt/config hashes

Store aggregate results in CSV or Parquet.

## 8. Repository structure

Use a structure similar to:

bm25-vfs-ablation/

  README.md
  pyproject.toml
  Makefile
  Dockerfile
  configs/
    default.yaml
    tool_call_intervention.yaml
  src/bm25_vfs_ablation/
    corpus/
      generator.py
      schema.py
      loader.py
    retrieval/
      bm25.py
      chunking.py
    vfs/
      filesystem.py
      tools.py
      security.py
    harnesses/
      base.py
      snippets.py
      vfs_agent.py
      budget.py
    models/
      client.py
      cache.py
      tokenization.py
    evaluation/
      answer_scoring.py
      evidence_scoring.py
      statistics.py
      plots.py
    experiment/
      runner.py
      schemas.py
      reporting.py
    cli.py
  tests/
  data/
    .gitkeep
  results/
    .gitkeep
  reports/
    .gitkeep

Adjust if a cleaner structure emerges, but retain separation between corpus, retrieval, harness, budget enforcement, and evaluation.

## 9. CLI

Provide commands equivalent to:

make setup
make test
make generate
make experiment
make evaluate
make report

And direct CLI usage:

python -m bm25_vfs_ablation generate \
  --tasks 200 \
  --seed 42

python -m bm25_vfs_ablation run \
  --config configs/default.yaml

python -m bm25_vfs_ablation run \
  --config configs/tool_call_intervention.yaml

python -m bm25_vfs_ablation evaluate \
  --runs results/runs.jsonl

python -m bm25_vfs_ablation report \
  --runs results/runs.jsonl \
  --output reports/experiment.md

Allow a smoke-test mode using a small local or mocked model.

## 10. Required plots and tables

Generate:

1. Success rate by harness with paired 95% confidence intervals.
2. Success rate by hop count.
3. Success rate versus actual tool-call count within VFS.
4. Success rate versus maximum permitted tool calls.
5. Initial retrieval recall versus task success.
6. Final evidence recall versus task success.
7. Tokens consumed versus task success.
8. Latency and estimated cost by condition.
9. Failure-mode distribution.
10. Paired task outcome table:
   - both succeed;
   - snippets only succeeds;
   - VFS only succeeds;
   - both fail.

## 11. Failure analysis

Classify failures using reproducible heuristics where possible:

- initial retrieval miss;
- failed exploration;
- incomplete evidence coverage;
- incorrect evidence composition;
- unsupported answer;
- correct evidence but wrong conclusion;
- budget exhaustion;
- excessive/repeated tool use;
- invalid tool call;
- malformed final answer.

Produce representative trace examples for:
- VFS wins;
- snippet wins;
- both fail;
- excessive tool use without benefit.

Do not send private chain-of-thought to logs. Store observable messages, tool calls, evidence accessed, answers, and concise model-provided justifications only.

## 12. Tests

Include tests for:

- identical corpus/index usage across conditions;
- deterministic dataset generation;
- BM25 ranking stability;
- VFS path traversal prevention;
- bounded read, cat, and grep;
- exact token-budget enforcement;
- correct accounting across multiple model turns;
- task scorer correctness;
- evidence recall calculation;
- JSONL schema validation;
- run resumption after interruption;
- paired statistical analysis;
- mocked end-to-end experiment.

## 13. Research report

Generate reports/experiment.md containing:

- executive summary;
- hypothesis and experimental design;
- controlled variables;
- dataset description;
- primary results;
- evidence/retrieval results;
- tool-use analysis;
- intervention results;
- efficiency trade-offs;
- failure analysis;
- threats to validity;
- conclusions.

The report must distinguish:
- observed results;
- statistical inference;
- causal evidence;
- speculation.

Explicitly discuss these validity threats:
- synthetic-task realism;
- prompt sensitivity;
- model/provider drift;
- tokenizer/accounting error;
- VFS interface quality;
- different prompt overhead between conditions;
- tool-call count as a post-treatment variable;
- repeated observations over shared templates;
- judge-model bias, if an LLM judge is used.

## 14. Implementation sequence

Work incrementally:

1. Scaffold repository and schemas.
2. Implement deterministic corpus/task generation.
3. Implement shared BM25 index.
4. Implement snippet harness.
5. Implement secure VFS tools.
6. Implement iterative VFS harness.
7. Implement token-budget enforcement.
8. Add mocked-model integration tests.
9. Add real OpenAI-compatible model configuration.
10. Implement evaluation and statistics.
11. Add intervention experiment.
12. Generate report and plots.
13. Run tests and a small smoke experiment.
14. Document exact reproduction commands.

At each stage, run relevant tests before continuing.

## 15. Acceptance criteria

The project is complete when:

- both harnesses operate over the exact same corpus and BM25 index;
- each task receives the same configured total token ceiling;
- the experiment is reproducible from a seed and committed configuration;
- every VFS interaction is observable and measurable;
- results include paired uncertainty estimates, not only averages;
- retrieval recall, evidence access, reasoning success, and efficiency are reported separately;
- tool-call analysis is described as associative unless based on the explicit call-limit intervention;
- the full smoke experiment runs locally with one documented command;
- tests pass;
- README explains setup, experiment design, limitations, and interpretation.

Before implementing, briefly restate the design, identify any ambiguity, and choose the simplest defensible interpretation. Then build the repository without waiting for confirmation unless credentials or a model endpoint are genuinely required.
