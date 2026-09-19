"""Orchestration, scoring, and resumable run-record construction."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bm25_vfs_ablation import Condition, ToolName
from bm25_vfs_ablation.config import AppConfig
from bm25_vfs_ablation.corpus.loader import CorpusBundle
from bm25_vfs_ablation.corpus.schema import TaskRecord
from bm25_vfs_ablation.evaluation.answer_scoring import score_answer
from bm25_vfs_ablation.evaluation.evidence_scoring import (
    complete_coverage_call_ordinal,
    derive_accessed_evidence,
    score_evidence,
    score_retrieval,
)
from bm25_vfs_ablation.experiment.artifacts import append_jsonl, canonical_json
from bm25_vfs_ablation.experiment.designs import (
    DesignAssignment,
    condition_order,
    expand_design,
)
from bm25_vfs_ablation.experiment.schemas import (
    AccessedEvidence,
    GoldEvidence,
    LatencyBreakdown,
    LineRange,
    ModelConfigSnapshot,
    RetrievedChunk,
    RetrievedDocument,
    RunRecord,
    TokenAccounting,
    ToolCallCounts,
    ToolTraceEntry,
)
from bm25_vfs_ablation.harnesses.base import HarnessContext, HarnessResult
from bm25_vfs_ablation.harnesses.budget import BudgetController
from bm25_vfs_ablation.harnesses.snippets import (
    SnippetHarness,
    _select_chunks,
)
from bm25_vfs_ablation.harnesses.vfs_agent import (
    VfsHarness,
    build_vfs_messages,
)
from bm25_vfs_ablation.models.cache import FileResponseCache
from bm25_vfs_ablation.models.client import ModelClient, ModelRequest, ModelResponse
from bm25_vfs_ablation.retrieval.bm25 import BM25Index, SearchHit
from bm25_vfs_ablation.retrieval.chunking import Chunker, ChunkRecord


def _expanded_path(path: Path | str) -> Path:
    return Path(os.path.expanduser(os.fspath(path)))


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The records written by one runner invocation and resume statistics."""

    records: tuple[RunRecord, ...]
    skipped_run_keys: tuple[str, ...] = ()

    @property
    def completed(self) -> int:
        """Return the number of records completed in this invocation."""

        return len(self.records)

    @property
    def completed_count(self) -> int:
        """Return the number of records completed in this invocation."""

        return self.completed

    @property
    def skipped(self) -> int:
        """Return the number of assignments skipped by resume."""

        return len(self.skipped_run_keys)

    @property
    def skipped_count(self) -> int:
        """Return the number of assignments skipped by resume."""

        return self.skipped


def _read_run_records(path: Path | str) -> list[RunRecord]:
    """Read and validate every complete line in an append-only run file."""

    target = _expanded_path(path)
    try:
        raw_bytes = target.read_bytes()
    except FileNotFoundError:
        return []

    if not raw_bytes:
        return []

    records: list[RunRecord] = []
    lines = raw_bytes.splitlines(keepends=True)
    for line_number, raw_line in enumerate(lines, start=1):
        is_final_without_newline = line_number == len(lines) and not raw_line.endswith(b"\n")
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            if is_final_without_newline:
                raise ValueError(
                    f"incomplete final JSON line in {target} at line {line_number}; "
                    "recovery: remove the incomplete final line or restore it "
                    "before resuming"
                ) from error
            raise ValueError(f"invalid JSON in {target} at line {line_number}: {error}") from error
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite)
        except json.JSONDecodeError as error:
            if is_final_without_newline:
                raise ValueError(
                    f"incomplete final JSON line in {target} at line {line_number}; "
                    "recovery: remove the incomplete final line or restore it "
                    "before resuming"
                ) from error
            raise ValueError(f"invalid JSON in {target} at line {line_number}: {error}") from error
        except ValueError as error:
            raise ValueError(f"invalid JSON in {target} at line {line_number}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"invalid JSON object in {target} at line {line_number}")
        try:
            record = RunRecord.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid run record in {target} at line {line_number}: {error}"
            ) from error
        records.append(record)
    return records


def load_completed_run_keys(path: Path | str) -> set[str]:
    """Validate an existing runs JSONL file and return its unique run keys."""

    records = _read_run_records(path)
    keys: set[str] = set()
    target = _expanded_path(path)
    for record in records:
        if record.run_key in keys:
            raise ValueError(f"duplicate run_key in {target}: {record.run_key}")
        keys.add(record.run_key)
    return keys


@dataclass(slots=True)
class _RunObservation:
    """Runtime values not exposed by the harness result contract."""

    requests: list[ModelRequest]
    responses: list[ModelResponse]
    cache_keys: list[str]
    cache_hit: bool = False

    @property
    def returned_model_id(self) -> str | None:
        if not self.responses:
            return None
        return self.responses[-1].returned_model_id


class _ObservedModel:
    """Record provider responses while preserving the model-client protocol."""

    def __init__(self, client: ModelClient, observation: _RunObservation) -> None:
        self._client = client
        self._observation = observation

    def complete(self, request: ModelRequest) -> ModelResponse:
        self._observation.requests.append(request.model_copy(deep=True))
        response = self._client.complete(request)
        self._observation.responses.append(response)
        return response


class _ObservedCache:
    """Record cache keys and hits while preserving atomic cache operations."""

    def __init__(self, cache: Any, observation: _RunObservation) -> None:
        self._cache = cache
        self._observation = observation

    def get(self, key: str) -> ModelResponse | None:
        self._observation.cache_keys.append(key)
        response = self._cache.get(key)
        if response is not None:
            self._observation.cache_hit = True
            self._observation.responses.append(response)
        return response

    def put(self, key: str, response: ModelResponse) -> None:
        self._cache.put(key, response)


def _config_payload(config: AppConfig) -> dict[str, Any]:
    if hasattr(config, "canonical_dict"):
        return config.canonical_dict()
    return config.model_dump(mode="json")


def _config_hash(config: AppConfig) -> str:
    return hashlib.sha256(canonical_json(_config_payload(config)).encode("utf-8")).hexdigest()


def _prompt_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _document_hits(index: object, hits: tuple[SearchHit, ...]) -> list[object]:
    method = getattr(index, "document_hits", None)
    if callable(method):
        return list(method(hits))

    grouped: dict[str, SearchHit] = {}
    for hit in hits:
        current = grouped.get(hit.doc_id)
        if current is None or (hit.score, hit.chunk_id) > (current.score, current.chunk_id):
            grouped[hit.doc_id] = hit
    return sorted(grouped.values(), key=lambda hit: (-hit.score, hit.doc_id))


def _document_path(bundle: CorpusBundle, doc_id: str) -> str | None:
    documents_by_id = getattr(bundle, "documents_by_id", None)
    if documents_by_id is not None:
        document = documents_by_id.get(doc_id)
        if document is not None:
            return str(document.path)
    for document in getattr(bundle, "documents", ()):
        if document.doc_id == doc_id:
            return str(document.path)
    return None


def _oracle_paths(bundle: CorpusBundle, task: TaskRecord) -> list[str]:
    return [
        path
        for doc_id in task.gold_document_ids
        if (path := _document_path(bundle, doc_id)) is not None
    ]


def _retrieved_documents(
    hits: tuple[SearchHit, ...],
    document_hits: list[object],
) -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            doc_id=str(_field(hit, "doc_id")),
            best_score=max(0.0, float(_field(hit, "best_score", _field(hit, "score", 0.0)))),
            rank=rank,
        )
        for rank, hit in enumerate(document_hits, start=1)
    ]


def _retrieved_chunks(hits: tuple[SearchHit, ...]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=str(_field(hit, "chunk_id")),
            doc_id=str(_field(hit, "doc_id")),
            score=max(0.0, float(_field(hit, "score", 0.0))),
            rank=rank,
            start_line=int(_field(hit, "start_line")),
            end_line=int(_field(hit, "end_line")),
        )
        for rank, hit in enumerate(hits, start=1)
    ]


def _model_snapshot(config: AppConfig) -> ModelConfigSnapshot:
    model = config.model
    return ModelConfigSnapshot(
        provider=model.provider,
        base_url=model.base_url,
        model=model.model,
        temperature=model.temperature,
        top_p=model.top_p,
        seed=model.seed,
        tokenizer=model.tokenizer,
    )


def _empty_accounting(config: AppConfig) -> TokenAccounting:
    return BudgetController(config.experiment.token_ceiling).accounting()


def _tool_counts(result: HarnessResult, trace: list[ToolTraceEntry]) -> ToolCallCounts:
    value = getattr(result, "tool_calls_by_type", None)
    if isinstance(value, ToolCallCounts):
        return value
    counts = {tool: 0 for tool in ToolName}
    for entry in trace:
        tool = entry.request.tool
        counts[tool] += 1
    return ToolCallCounts(
        grep=counts[ToolName.GREP],
        read=counts[ToolName.READ],
        cat=counts[ToolName.CAT],
        list=counts[ToolName.LIST],
    )


def _initial_prompt_and_snippets(
    assignment: DesignAssignment,
    task: TaskRecord,
    result: HarnessResult,
    context: HarnessContext,
    bundle: CorpusBundle,
    index: BM25Index,
) -> tuple[list[dict[str, Any]], tuple[ChunkRecord, ...]]:
    """Rebuild the initial deterministic prompt when the model was cached/skipped."""

    if assignment.condition is Condition.SNIPPETS:
        hits = tuple(result.initial_hits)
        budget = BudgetController(context.config.experiment.token_ceiling)
        selected, messages, _ = _select_chunks(task, hits, context, budget)
        return messages, selected

    candidates = _document_hits(index, tuple(result.initial_hits))
    oracle_paths = _oracle_paths(bundle, task) if assignment.oracle_mode else None
    messages = build_vfs_messages(
        task,
        candidates[: context.config.harness.initial_candidate_documents],
        context.config.experiment.token_ceiling,
        oracle_paths=oracle_paths,
        oracle_mode=assignment.oracle_mode,
    )
    return messages, ()


def _first_gold_fact_time_ms(
    task: TaskRecord,
    condition: Condition,
    accessed: AccessedEvidence,
    result: HarnessResult,
    corpus_chunks: tuple[ChunkRecord, ...],
) -> float | None:
    required = set(task.required_fact_ids)
    if condition is Condition.SNIPPETS:
        if required.intersection(accessed.fact_ids):
            return max(0.0, result.latency.retrieval_ms)
        return None

    elapsed_ms = max(0.0, result.latency.retrieval_ms)
    for entry in result.trace:
        elapsed_ms += max(0.0, entry.latency_ms)
        prefix = derive_accessed_evidence(
            task,
            trace=tuple(result.trace[: entry.call_ordinal]),
            corpus_chunks=corpus_chunks,
        )
        if required.intersection(prefix.fact_ids):
            return elapsed_ms
    return None


def _record_for_result(
    *,
    assignment: DesignAssignment,
    task: TaskRecord,
    result: HarnessResult,
    context: HarnessContext,
    bundle: CorpusBundle,
    index: BM25Index,
    chunks: tuple[ChunkRecord, ...],
    observation: _RunObservation,
    started_at: datetime,
    finished_at: datetime,
    config_hash: str,
) -> RunRecord:
    accounting = result.accounting
    if accounting is None:
        accounting = _empty_accounting(context.config)

    raw_for_scoring = result.raw_answer if isinstance(result.raw_answer, str) else ""
    answer_score = score_answer(task, raw_for_scoring)
    hits = tuple(result.initial_hits)
    document_hits = _document_hits(index, hits)
    initial_messages, selected_chunks = _initial_prompt_and_snippets(
        assignment,
        task,
        result,
        context,
        bundle,
        index,
    )

    if assignment.condition is Condition.SNIPPETS:
        accessed = derive_accessed_evidence(
            task,
            snippet_chunks=selected_chunks,
            corpus_chunks=chunks,
        )
        trace: list[ToolTraceEntry] = []
        max_tool_calls: int | None = None
        tool_call_count = 0
        successful_tool_calls = 0
        invalid_tool_calls = 0
        repeated_tool_calls = 0
        tool_counts = ToolCallCounts(grep=0, read=0, cat=0, list=0)
        unique_files: list[str] = []
        unique_ranges: list[LineRange] = []
        coverage_call = None
    else:
        trace = list(result.trace)
        accessed = derive_accessed_evidence(
            task,
            trace=trace,
            corpus_chunks=chunks,
        )
        max_tool_calls = assignment.max_tool_calls_permitted
        if max_tool_calls is None:
            max_tool_calls = context.config.harness.max_tool_calls
        tool_call_count = len(trace)
        successful_tool_calls = int(
            getattr(result, "successful_tool_calls", sum(entry.result.ok for entry in trace))
        )
        invalid_tool_calls = int(
            getattr(
                result,
                "invalid_tool_calls",
                sum(not entry.result.ok for entry in trace),
            )
        )
        repeated_tool_calls = int(
            getattr(result, "repeated_tool_calls", sum(entry.repeated for entry in trace))
        )
        tool_counts = _tool_counts(result, trace)
        unique_files = sorted({line_range.path for line_range in accessed.line_ranges})
        unique_ranges = list(accessed.line_ranges)
        coverage_call = complete_coverage_call_ordinal(
            task,
            (),
            trace,
            corpus_chunks=chunks,
        )

    retrieval_metrics = score_retrieval(
        task,
        hits,
        document_hits,
        k=context.config.retrieval.top_k,
    )
    evidence_metrics = score_evidence(
        task,
        accessed,
        citation_score=answer_score.citation_score,
    )
    returned_model_id = observation.returned_model_id
    model_snapshot = _model_snapshot(context.config)
    estimated_cost = (
        accounting.input_tokens_estimated
        * context.config.evaluation.estimated_input_usd_per_million
        + accounting.output_tokens_estimated
        * context.config.evaluation.estimated_output_usd_per_million
    ) / 1_000_000
    latency = result.latency
    if not isinstance(latency, LatencyBreakdown):
        raise TypeError("harness result latency must be a LatencyBreakdown")

    return RunRecord(
        schema_version=1,
        run_key=assignment.run_key,
        experiment_id=assignment.experiment_id,
        design_cell=assignment.design_cell,
        task_id=assignment.task_id,
        condition=assignment.condition,
        condition_order=assignment.condition_order,
        model_config=model_snapshot,
        requested_model_id=context.config.model.model,
        returned_model_id=returned_model_id,
        seed=context.config.experiment.seed,
        corpus_version=task.corpus_version,
        question=task.question,
        final_answer=result.raw_answer,
        parsed_answer=answer_score.parsed_answer,
        correctness=answer_score.correctness,
        secondary_judge=answer_score.secondary_judge,
        scoring_details=answer_score.scoring_details,
        citations=list(answer_score.citations),
        initial_retrieved_documents=_retrieved_documents(hits, document_hits),
        initial_retrieved_chunks=_retrieved_chunks(hits),
        gold_evidence=GoldEvidence(
            fact_ids=sorted(task.required_fact_ids),
            doc_ids=sorted(task.gold_document_ids),
            chunk_ids=sorted(task.gold_chunk_ids),
        ),
        accessed_evidence=accessed,
        retrieval_metrics=retrieval_metrics,
        evidence_metrics=evidence_metrics,
        tool_trace=trace,
        tool_call_count=tool_call_count,
        max_tool_calls_permitted=max_tool_calls,
        tool_calls_by_type=tool_counts,
        successful_tool_calls=successful_tool_calls,
        invalid_tool_calls=invalid_tool_calls,
        repeated_tool_calls=repeated_tool_calls,
        unique_files_read=unique_files,
        unique_line_ranges_accessed=unique_ranges,
        time_to_first_gold_fact_ms=_first_gold_fact_time_ms(
            task,
            assignment.condition,
            accessed,
            result,
            chunks,
        ),
        calls_to_complete_gold_coverage=coverage_call,
        token_accounting=accounting,
        total_tokens=accounting.total_tokens_estimated,
        latency=latency,
        estimated_cost_usd=max(0.0, estimated_cost),
        termination_reason=result.termination,
        errors=list(result.errors),
        prompt_sha256=_prompt_hash(initial_messages),
        config_sha256=config_hash,
        corpus_sha256=bundle.corpus_sha256,
        cache_key=observation.cache_keys[0] if observation.cache_keys else None,
        cache_hit=observation.cache_hit,
        started_at=started_at,
        finished_at=finished_at,
    )


class ExperimentRunner:
    """Run both harnesses over one shared corpus/index and append validated rows."""

    def __init__(
        self,
        config: AppConfig,
        bundle: CorpusBundle,
        index: BM25Index | None = None,
        model: ModelClient | None = None,
        cache: FileResponseCache | None = None,
        *,
        output_runs: Path | str | None = None,
    ) -> None:
        if index is not None and not hasattr(index, "search"):
            if model is not None and not hasattr(model, "complete") and cache is None:
                cache = model  # type: ignore[assignment]
                model = None
            if model is None:
                model = index  # type: ignore[assignment]
                index = None
        if model is None:
            raise TypeError("model client is required")
        self.config = config
        self.bundle = bundle
        self.model = model
        self.cache = cache
        self.output_runs = _expanded_path(
            output_runs if output_runs is not None else config.experiment.output_runs
        )
        if index is None:
            chunker = Chunker(
                config.chunking.chunk_size_tokens,
                config.chunking.chunk_overlap_tokens,
            )
            chunks = chunker.chunk(bundle.documents)
            index = BM25Index(
                chunks,
                k1=config.retrieval.k1,
                b=config.retrieval.b,
                epsilon=config.retrieval.epsilon,
            )
        self.index = index
        self.chunks = tuple(getattr(index, "chunks", ()))

    def _context(self, observation: _RunObservation) -> HarnessContext:
        model = _ObservedModel(self.model, observation)
        cache = _ObservedCache(self.cache, observation) if self.cache is not None else None
        return HarnessContext(
            bundle=self.bundle,
            index=self.index,
            model=model,  # type: ignore[arg-type]
            cache=cache,  # type: ignore[arg-type]
            config=self.config,
            budget_factory=lambda: BudgetController(self.config.experiment.token_ceiling),
        )

    def _run_assignment(
        self,
        assignment: DesignAssignment,
        task: TaskRecord,
        config_hash: str,
    ) -> RunRecord:
        observation = _RunObservation([], [], [])
        context = self._context(observation)
        started_at = datetime.now(UTC)
        if assignment.condition is Condition.SNIPPETS:
            harness: Any = SnippetHarness()
        else:
            harness = VfsHarness(
                max_tool_calls=assignment.max_tool_calls_permitted,
                oracle_mode=assignment.oracle_mode,
            )
        result = harness.run(task, context)
        finished_at = datetime.now(UTC)
        return _record_for_result(
            assignment=assignment,
            task=task,
            result=result,
            context=context,
            bundle=self.bundle,
            index=self.index,
            chunks=self.chunks,
            observation=observation,
            started_at=started_at,
            finished_at=finished_at,
            config_hash=config_hash,
        )

    def run(self) -> RunSummary:
        """Execute assignments, validating every row before appending it."""

        completed_keys = load_completed_run_keys(self.output_runs)
        if completed_keys and not self.config.experiment.resume:
            raise ValueError(
                f"resume is disabled but {self.output_runs} already contains run records"
            )
        assignments = expand_design(self.config, self.bundle.tasks)
        tasks_by_id = {task.task_id: task for task in self.bundle.tasks}
        config_hash = _config_hash(self.config)
        written: list[RunRecord] = []
        skipped: list[str] = []

        for assignment in assignments:
            if assignment.run_key in completed_keys:
                skipped.append(assignment.run_key)
                continue
            task = tasks_by_id[assignment.task_id]
            record = self._run_assignment(assignment, task, config_hash)
            payload = record.model_dump(mode="json", by_alias=True)
            append_jsonl(self.output_runs, payload)
            completed_keys.add(record.run_key)
            written.append(record)

        return RunSummary(records=tuple(written), skipped_run_keys=tuple(skipped))


__all__ = [
    "ExperimentRunner",
    "RunSummary",
    "condition_order",
    "load_completed_run_keys",
]
