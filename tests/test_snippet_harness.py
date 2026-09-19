from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.experiment.schemas import TerminationReason
from bm25_vfs_ablation.harnesses.base import HarnessContext
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.harnesses.snippets import SnippetHarness
from bm25_vfs_ablation.models.client import ModelRequest, ModelResponse
from bm25_vfs_ablation.retrieval.bm25 import SearchHit


def _config(
    *,
    token_ceiling: int = 512,
    retrieval_context_tokens: int = 32,
    top_k: int = 4,
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
                "oracle_mode": False,
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
                "top_k": top_k,
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
                "query_tokenizer": "unicode_word_v1",
            },
            "harness": {
                "retrieval_context_tokens": retrieval_context_tokens,
                "max_answer_tokens": 64,
                "max_tool_calls": 8,
                "invalid_call_limit": 3,
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


def _task() -> Any:
    from bm25_vfs_ablation.corpus.schema import TaskRecord

    return TaskRecord.model_validate(
        {
            "schema_version": 1,
            "task_id": "task-eval-000001",
            "split": "eval",
            "question": "Which owner manages quartz?",
            "canonical_answer": "Quartz",
            "acceptable_answer_variants": ["Quartz"],
            "required_fact_ids": ["fact-000001", "fact-000002"],
            "gold_document_ids": ["doc-ops-0001"],
            "gold_chunk_ids": ["doc-ops-0001::c0000"],
            "hop_count": 2,
            "task_template": "template-placeholder",
            "category": "ops",
            "distractor_document_ids": [],
            "distractor_count": 0,
            "generator_seed": 42,
            "corpus_version": "synthetic-v1",
        }
    )


def _hit(
    chunk_id: str,
    text: str,
    *,
    start_line: int = 1,
    end_line: int = 1,
) -> SearchHit:
    doc_id = chunk_id.split("::", maxsplit=1)[0]
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        path=f"/ops/{doc_id}.md",
        score=1.0,
        start_line=start_line,
        end_line=end_line,
        text=text,
    )


class _Index:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        self.calls.append((query, top_k))
        return self.hits[:top_k]


class _Model:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=self.content,
            returned_model_id="model-placeholder",
            latency_ms=0.0,
        )


def _context(
    index: _Index,
    model: _Model,
    config: AppConfig,
) -> HarnessContext:
    return HarnessContext(
        bundle=SimpleNamespace(corpus_sha256="a" * 64),
        index=index,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        cache=None,
        config=config,
        budget_factory=lambda: BudgetController(config.experiment.token_ceiling),
    )


_ANSWER = (
    '{"answer":"Quartz","citations":["doc-ops-0001::c0000"],'
    '"justification":"The retrieved record names Quartz."}'
)


def test_snippet_prompt_contains_stable_ids() -> None:
    hit = _hit("doc-ops-0001::c0000", "owner Quartz", start_line=3, end_line=4)
    model = _Model(_ANSWER)
    index = _Index([hit])

    result = SnippetHarness().run(_task(), _context(index, model, _config()))

    prompt = "\n".join(message["content"] for message in model.requests[0].messages)
    assert "doc_id=doc-ops-0001" in prompt
    assert "chunk_id=doc-ops-0001::c0000" in prompt
    assert "path=/ops/doc-ops-0001.md" in prompt
    assert "lines=3-4" in prompt
    assert result.termination is TerminationReason.ANSWERED


def test_snippet_uses_shared_index_once() -> None:
    hits = [
        _hit("doc-ops-0001::c0000", "owner Quartz"),
        _hit("doc-ops-0002::c0000", "owner Atlas"),
    ]
    model = _Model(_ANSWER)
    index = _Index(hits)
    config = _config(top_k=2)

    result = SnippetHarness().run(_task(), _context(index, model, config))

    assert index.calls == [("Which owner manages quartz?", 2)]
    assert result.initial_hits == tuple(hits)


def test_snippet_context_whole_chunk_truncation() -> None:
    first = _hit("doc-ops-0001::c0000", "alpha beta")
    second = _hit("doc-ops-0002::c0000", "gamma delta")
    model = _Model(_ANSWER)
    index = _Index([first, second])
    config = _config(retrieval_context_tokens=3)

    SnippetHarness().run(_task(), _context(index, model, config))

    prompt = model.requests[0].messages[-1]["content"]
    assert "alpha beta" in prompt
    assert "gamma delta" not in prompt
    assert "gamma" not in prompt


def test_snippet_exactly_one_model_call() -> None:
    model = _Model(_ANSWER)
    index = _Index([_hit("doc-ops-0001::c0000", "owner Quartz")])

    result = SnippetHarness().run(_task(), _context(index, model, _config()))

    assert len(model.requests) == 1
    assert result.accounting is not None
    assert len(result.accounting.turns) == 1


def test_snippet_budget_exhaustion_before_call() -> None:
    model = _Model(_ANSWER)
    index = _Index([_hit("doc-ops-0001::c0000", "owner Quartz")])
    config = _config(token_ceiling=1)

    result = SnippetHarness().run(_task(), _context(index, model, config))

    assert model.requests == []
    assert result.termination is TerminationReason.BUDGET_EXHAUSTED
    assert result.accounting is not None
    assert result.accounting.turns == []


def test_snippet_has_no_tools() -> None:
    model = _Model(_ANSWER)
    index = _Index([_hit("doc-ops-0001::c0000", "owner Quartz")])

    SnippetHarness().run(_task(), _context(index, model, _config()))

    assert model.requests[0].tools == []
