from __future__ import annotations

from types import SimpleNamespace

from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.corpus.schema import DocumentRecord, TaskRecord
from bm25_vfs_ablation.experiment.schemas import TerminationReason
from bm25_vfs_ablation.harnesses.base import HarnessContext
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.harnesses.vfs_agent import (
    VfsHarness,
    parse_model_action,
)
from bm25_vfs_ablation.models.client import (
    MockAction,
    MockPolicy,
    ModelRequest,
    ModelResponse,
    ScriptedMockClient,
)
from bm25_vfs_ablation.retrieval.bm25 import BM25Index
from bm25_vfs_ablation.retrieval.chunking import Chunker


def _config(
    *,
    token_ceiling: int = 2200,
    max_tool_calls: int = 8,
    invalid_call_limit: int = 3,
    oracle_mode: bool = False,
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": {
                "experiment_id": None,
                "seed": 42,
                "conditions": ["snippets", "vfs"],
                "token_ceiling": token_ceiling,
                "concurrency": 1,
                "oracle_mode": oracle_mode,
                "resume": True,
                "output_runs": "results/runs.jsonl",
            },
            "corpus": {
                "corpus_path": "data/generated/corpus.jsonl",
                "tasks_path": "data/generated/tasks.jsonl",
                "corpus_version": "synthetic-v1",
            },
            "chunking": {
                "chunk_size_tokens": 180,
                "chunk_overlap_tokens": 30,
            },
            "retrieval": {
                "implementation": "rank_bm25_okapi",
                "top_k": 8,
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
                "query_tokenizer": "unicode_word_v1",
            },
            "harness": {
                "retrieval_context_tokens": 1800,
                "max_answer_tokens": 64,
                "max_tool_calls": max_tool_calls,
                "invalid_call_limit": invalid_call_limit,
                "tool_result_tokens_per_call": 768,
                "initial_candidate_documents": 5,
            },
            "model": {
                "provider": "mock",
                "base_url": "http://placeholder.invalid/v1",
                "api_key": None,
                "model": "model-placeholder",
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "timeout_seconds": 60.0,
                "max_attempts": 2,
                "tokenizer": "regex_v1",
            },
            "evaluation": {
                "bootstrap_resamples": 10,
                "bootstrap_seed": 20250308,
                "confidence_level": 0.95,
                "judge_enabled": False,
                "estimated_input_usd_per_million": 0.0,
                "estimated_output_usd_per_million": 0.0,
            },
        }
    )


def _document(doc_id: str, category: str, content: str) -> DocumentRecord:
    return DocumentRecord(
        schema_version=1,
        corpus_version="synthetic-v1",
        doc_id=doc_id,
        path=f"/{category}/{doc_id}.md",
        category=category,
        title=f"{category} document",
        content=content,
        facts=[],
        generator_seed=42,
    )


def _task() -> TaskRecord:
    return TaskRecord(
        schema_version=1,
        task_id="task-eval-000001",
        split="eval",
        question="Which library does Atlas depend on?",
        canonical_answer="Quartz",
        acceptable_answer_variants=["Quartz"],
        required_fact_ids=["fact-000001", "fact-000002"],
        gold_document_ids=["doc-policies-0001", "doc-services-0001"],
        gold_chunk_ids=[
            "doc-policies-0001::c0000",
            "doc-services-0001::c0000",
        ],
        hop_count=2,
        task_template="dependency-policy",
        category="security",
        distractor_document_ids=["doc-notes-0001"],
        distractor_count=1,
        generator_seed=42,
        corpus_version="synthetic-v1",
    )


def _documents() -> list[DocumentRecord]:
    return [
        _document(
            "doc-policies-0001",
            "policies",
            "Security policy\nQuartz requires review.\n",
        ),
        _document(
            "doc-services-0001",
            "services",
            "Atlas service\nAtlas depends on Quartz.\n",
        ),
        _document(
            "doc-notes-0001",
            "notes",
            "Reference notes\nThe unrelated service mentions Atlas.\n",
        ),
    ]


def _context(
    model: object,
    *,
    config: AppConfig | None = None,
    documents: list[DocumentRecord] | None = None,
) -> HarnessContext:
    records = documents or _documents()
    chunks = Chunker().chunk(records)
    index = BM25Index(chunks)
    bundle = SimpleNamespace(
        documents=tuple(records),
        documents_by_id={record.doc_id: record for record in records},
        corpus_sha256="a" * 64,
    )
    selected_config = config or _config()
    return HarnessContext(
        bundle=bundle,  # type: ignore[arg-type]
        index=index,
        model=model,  # type: ignore[arg-type]
        cache=None,
        config=selected_config,
        budget_factory=lambda: BudgetController(selected_config.experiment.token_ceiling),
    )


_FINAL = (
    '{"answer":"Quartz","citations":["doc-services-0001::c0000"],'
    '"justification":"The service record names the dependency."}'
)


def _client(actions: list[MockAction], *, condition: str = "vfs") -> ScriptedMockClient:
    key = ("task-eval-000001", condition, "primary")
    return ScriptedMockClient(MockPolicy({key: actions}), *key)


def test_vfs_initial_context_has_paths_not_content() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.client = _client([MockAction(kind="final", answer=_FINAL)])

        def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            return self.client.complete(request)

    model = RecordingModel()
    context = _context(model)

    result = VfsHarness().run(_task(), context)

    assert result.termination is TerminationReason.ANSWERED
    assert result.tool_call_count == 0
    prompt = model.requests[0].messages[1]["content"]
    assert "/services/doc-services-0001.md" in prompt
    assert "score=" in prompt
    assert "Atlas depends on Quartz." not in prompt


def test_vfs_scripted_multiturn_trace() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.responses = [
                ModelResponse(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call-001",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": (
                                    '{"path":"/services/doc-services-0001.md",'
                                    '"start_line":2,"end_line":2}'
                                ),
                            },
                        }
                    ],
                    returned_model_id="model-placeholder",
                    latency_ms=0.0,
                ),
                ModelResponse(
                    content=_FINAL,
                    returned_model_id="model-placeholder",
                    latency_ms=0.0,
                ),
            ]

        def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            return self.responses.pop(0)

    model = RecordingModel()
    result = VfsHarness().run(_task(), _context(model))

    initial_prompt = model.requests[0].messages[1]["content"]
    assert "Which library does Atlas depend on?" in initial_prompt
    assert "/services/doc-services-0001.md" in initial_prompt
    assert "Atlas depends on Quartz." not in initial_prompt
    assert model.requests[0].tools
    assert result.termination is TerminationReason.ANSWERED
    assert result.tool_call_count == 1
    assert result.successful_tool_calls == 1
    assert len(result.trace) == 1
    assert result.trace[0].request_id == "tool-001"
    assert result.trace[0].result.content == "Atlas depends on Quartz.\n"
    assert result.trace[0].latency_ms >= 0


def test_vfs_budget_across_turns() -> None:
    model = _client(
        [
            MockAction(
                kind="tool_call",
                tool_name="read",
                arguments={
                    "path": "/services/doc-services-0001.md",
                    "start_line": 2,
                    "end_line": 2,
                },
            ),
            MockAction(kind="final", answer=_FINAL),
        ]
    )
    result = VfsHarness().run(_task(), _context(model))

    assert result.accounting is not None
    assert len(result.accounting.turns) == 2
    assert result.accounting.total_tokens_estimated <= result.accounting.limit
    assert result.accounting.turns[1].tool_result_input_tokens > 0


def test_vfs_invalid_call_threshold() -> None:
    model = _client([MockAction(kind="tool_call", tool_name="unknown", arguments={})])
    config = _config(invalid_call_limit=1)

    result = VfsHarness().run(_task(), _context(model, config=config))

    assert result.termination is TerminationReason.INVALID_CALL_LIMIT
    assert result.tool_call_count == 1
    assert result.invalid_tool_calls == 1
    assert result.trace[0].result.ok is False
    assert len(result.errors) == 1


def test_vfs_max_call_limit_zero() -> None:
    model = _client(
        [
            MockAction(
                kind="tool_call",
                tool_name="read",
                arguments={
                    "path": "/services/doc-services-0001.md",
                    "start_line": 1,
                    "end_line": 1,
                },
            )
        ]
    )
    config = _config(max_tool_calls=0)

    result = VfsHarness().run(_task(), _context(model, config=config))

    assert result.termination is TerminationReason.MAX_TOOL_CALLS
    assert result.tool_call_count == 1
    assert result.successful_tool_calls == 0
    assert result.trace[0].result.error_code == "MAX_TOOL_CALLS"


def test_vfs_repeated_call_count() -> None:
    action = MockAction(
        kind="tool_call",
        tool_name="read",
        arguments={
            "path": "/services/doc-services-0001.md",
            "start_line": 2,
            "end_line": 2,
        },
    )
    model = _client([action, action, MockAction(kind="final", answer=_FINAL)])

    result = VfsHarness().run(_task(), _context(model))

    assert result.termination is TerminationReason.ANSWERED
    assert result.tool_call_count == 2
    assert result.repeated_tool_calls == 1
    assert [entry.repeated for entry in result.trace] == [False, True]


def test_vfs_oracle_candidates_are_gold_only() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.request: ModelRequest | None = None

        def complete(self, request: ModelRequest) -> ModelResponse:
            self.request = request
            return ModelResponse(
                content=_FINAL,
                returned_model_id="model-placeholder",
                latency_ms=0.0,
            )

    model = RecordingModel()
    context = _context(model, config=_config(oracle_mode=True))

    result = VfsHarness().run(_task(), context)

    assert result.termination is TerminationReason.ANSWERED
    assert model.request is not None
    prompt = model.request.messages[1]["content"]
    assert "/services/doc-services-0001.md" in prompt
    assert "/policies/doc-policies-0001.md" in prompt
    assert "/notes/doc-notes-0001.md" not in prompt


def test_vfs_never_logs_hidden_reasoning() -> None:
    model = _client(
        [
            MockAction(
                kind="tool_call",
                tool_name="list",
                arguments={"path": "/"},
            ),
            MockAction(kind="final", answer=_FINAL),
        ]
    )

    result = VfsHarness().run(_task(), _context(model))

    observable = repr(result)
    assert "reasoning" not in str(observable).casefold()
    assert "chain-of-thought" not in str(observable).casefold()
    assert "hidden" not in observable.casefold()
    assert (
        parse_model_action(
            ModelResponse(
                content=_FINAL,
                returned_model_id="model-placeholder",
                latency_ms=0.0,
            )
        ).parsed_answer.answer
        == "Quartz"
    )
