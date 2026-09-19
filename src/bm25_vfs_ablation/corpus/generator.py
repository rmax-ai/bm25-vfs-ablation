"""Deterministic synthetic enterprise corpus and task generation."""

from __future__ import annotations

import hashlib
import os
import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bm25_vfs_ablation.corpus.schema import (
    DocumentRecord,
    FactSpan,
    Split,
    TaskRecord,
    validate_dataset,
)
from bm25_vfs_ablation.experiment.artifacts import write_jsonl_atomic

CHUNK_SIZE_TOKENS: Final = 180
CHUNK_OVERLAP_TOKENS: Final = 30

DOCUMENT_CATEGORIES: Final[tuple[str, ...]] = (
    "services",
    "incidents",
    "architecture-decisions",
    "ownership",
    "deployments",
    "policies",
    "dependency-manifests",
)

TEMPLATE_FAMILIES: Final[tuple[str, ...]] = tuple(f"template-{ordinal:02d}" for ordinal in range(4))

_SERVICE_BASES: Final[tuple[str, ...]] = (
    "Atlas",
    "Beacon",
    "Cedar",
    "Delta",
    "Ember",
    "Fable",
    "Garnet",
    "Harbor",
    "Ivory",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Mosaic",
    "Nova",
    "Orchid",
    "Prairie",
)
_LIBRARY_BASES: Final[tuple[str, ...]] = (
    "Quartz",
    "Nimbus",
    "Rivet",
    "Solace",
    "Trellis",
    "Vector",
    "Willow",
    "Yarrow",
    "Zephyr",
    "Cobalt",
    "Delta",
    "Ember",
    "Fjord",
    "Helix",
    "Indigo",
    "Lattice",
)
_TEAM_BASES: Final[tuple[str, ...]] = (
    "Platform Reliability",
    "Data Systems",
    "Service Foundations",
    "Runtime Operations",
    "Application Security",
    "Developer Productivity",
    "Network Services",
    "Storage Engineering",
)
_REGION_BASES: Final[tuple[str, ...]] = (
    "Northstar",
    "Lakeside",
    "Canyon",
    "Meadow",
    "Summit",
    "Harbor",
    "Ridge",
    "Valley",
)
_RELEASES: Final[tuple[str, ...]] = ("4.1", "4.2", "4.3", "5.0", "5.1")
_ENCRYPTION_MODES: Final[tuple[str, ...]] = (
    "legacy encryption",
    "modern encryption",
    "standard encryption",
)
_INCIDENT_IMPACTS: Final[tuple[str, ...]] = (
    "checkout delay",
    "queue saturation",
    "certificate warning",
    "replication lag",
    "request timeout",
)

# The tables are public so later generation and inspection code can use the same
# vocabulary without reconstructing aliases from rendered document text.
SERVICE_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    base: (
        f"{base} runtime",
        f"{base} API",
        f"the {base} application",
    )
    for base in _SERVICE_BASES
}
LIBRARY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    base: (
        f"{base} package",
        f"{base} dependency",
        f"the {base} component",
    )
    for base in _LIBRARY_BASES
}
TEAM_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    base: (
        f"{base} group",
        f"{base} crew",
        f"the {base} team",
    )
    for base in _TEAM_BASES
}
REGION_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    base: (
        f"{base} zone",
        f"{base} area",
        f"the {base} region",
    )
    for base in _REGION_BASES
}
ENTITY_ALIASES: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "service": SERVICE_ALIASES,
    "library": LIBRARY_ALIASES,
    "team": TEAM_ALIASES,
    "region": REGION_ALIASES,
}

_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:['-][^\W_]+)*|[^\w\s]",
    flags=re.UNICODE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True, init=False)
class GenerationConfig:
    """Validated settings for one deterministic generation run.

    The primary names mirror the ``generate`` CLI flags.  A few descriptive
    aliases are accepted for callers that construct the configuration directly.
    """

    seed: int
    corpus_version: str
    dev_tasks: int
    eval_tasks: int
    corpus_size: int
    distractors: int
    min_hops: int
    max_hops: int

    def __init__(
        self,
        seed: int = 42,
        corpus_version: str = "synthetic-v1",
        dev_tasks: int = 50,
        eval_tasks: int = 200,
        corpus_size: int = 80,
        distractors: int = 6,
        min_hops: int = 2,
        max_hops: int = 4,
        **aliases: object,
    ) -> None:
        alias_targets = {
            "dev_count": "dev_tasks",
            "eval_count": "eval_tasks",
            "num_dev_tasks": "dev_tasks",
            "num_eval_tasks": "eval_tasks",
            "dev_task_count": "dev_tasks",
            "eval_task_count": "eval_tasks",
            "tasks": "eval_tasks",
            "task_count": "eval_tasks",
            "entity_count": "corpus_size",
            "corpus_entities": "corpus_size",
            "num_entities": "corpus_size",
            "distractor_count": "distractors",
            "distractors_per_task": "distractors",
            "min_hop_count": "min_hops",
            "max_hop_count": "max_hops",
        }
        values: dict[str, object] = {
            "seed": seed,
            "corpus_version": corpus_version,
            "dev_tasks": dev_tasks,
            "eval_tasks": eval_tasks,
            "corpus_size": corpus_size,
            "distractors": distractors,
            "min_hops": min_hops,
            "max_hops": max_hops,
        }
        for alias, value in aliases.items():
            target = alias_targets.get(alias)
            if target is None:
                raise TypeError(f"unexpected GenerationConfig argument: {alias}")
            if values[target] != value and values[target] != {
                "dev_tasks": 50,
                "eval_tasks": 200,
                "corpus_size": 80,
                "distractors": 6,
                "min_hops": 2,
                "max_hops": 4,
            }.get(target):
                raise TypeError(f"conflicting GenerationConfig arguments: {alias} and {target}")
            values[target] = value

        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)
        self._validate()

    @property
    def dev_count(self) -> int:
        """Compatibility name for the requested development count."""

        return self.dev_tasks

    @property
    def eval_count(self) -> int:
        """Compatibility name for the requested evaluation count."""

        return self.eval_tasks

    @property
    def entity_count(self) -> int:
        """Compatibility name for the synthetic entity count."""

        return self.corpus_size

    @property
    def distractor_count(self) -> int:
        """Compatibility name for the configured distractor count."""

        return self.distractors

    def _validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.corpus_version, str) or not self.corpus_version.strip():
            raise ValueError("corpus_version must not be blank")
        for field_name in ("dev_tasks", "eval_tasks", "corpus_size", "distractors"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if (
            isinstance(self.min_hops, bool)
            or isinstance(self.max_hops, bool)
            or not isinstance(self.min_hops, int)
            or not isinstance(self.max_hops, int)
        ):
            raise TypeError("min_hops and max_hops must be integers")
        if self.corpus_size == 0 and self.dev_tasks + self.eval_tasks:
            raise ValueError("corpus_size must be positive when tasks are requested")
        if not 2 <= self.min_hops <= self.max_hops <= 4:
            raise ValueError("min_hops and max_hops must be between 2 and 4")
        if self.dev_tasks and self.eval_tasks and self.corpus_size < 2:
            raise ValueError("both splits require at least two disjoint entity tuples")
        if self.eval_tasks and not _split_entity_indices(self, Split.EVAL):
            raise ValueError("evaluation tasks require an evaluation entity tuple")
        if self.dev_tasks and not _split_entity_indices(self, Split.DEV):
            raise ValueError("development tasks require a development entity tuple")


@dataclass(frozen=True, slots=True)
class ArtifactHashes:
    """Paths and SHA-256 digests for the two generated JSONL artifacts."""

    corpus_sha256: str
    tasks_sha256: str
    corpus_path: Path
    tasks_path: Path

    @property
    def corpus(self) -> str:
        """Return the corpus digest using the short attribute name."""

        return self.corpus_sha256

    @property
    def tasks(self) -> str:
        """Return the task digest using the short attribute name."""

        return self.tasks_sha256


@dataclass(frozen=True, slots=True)
class _Slot:
    ordinal: int
    release: str
    encryption: str
    team: str
    region: str
    incident: str
    impact: str


@dataclass(frozen=True, slots=True)
class _Entity:
    ordinal: int
    service: str
    service_aliases: tuple[str, ...]
    library: str
    library_aliases: tuple[str, ...]
    team: str
    team_aliases: tuple[str, ...]
    region: str
    region_aliases: tuple[str, ...]
    slots: tuple[_Slot, ...]


@dataclass(frozen=True, slots=True)
class _FactDraft:
    key: tuple[str, int, int]
    subject: str
    predicate: str
    object: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class _DocumentDraft:
    entity_ordinal: int
    category: str
    doc_id: str
    path: str
    title: str
    content: str
    facts: tuple[_FactDraft, ...]
    generator_seed: int


@dataclass(frozen=True, slots=True)
class _ChunkWindow:
    chunk_index: int
    start_line: int
    end_line: int


def child_seed(seed: int, stage: str, ordinal: int) -> int:
    """Derive the frozen 64-bit child seed for one generation stage."""

    payload = f"{seed}:{stage}:{ordinal}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stage_rng(seed: int, stage: str, ordinal: int) -> random.Random:
    return random.Random(child_seed(seed, stage, ordinal))


def _normalise_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def _task_counts(config: GenerationConfig) -> int:
    return config.dev_tasks + config.eval_tasks


def _slots_per_entity(config: GenerationConfig) -> int:
    counts_and_pools: list[tuple[int, int]] = []
    dev_pool = (config.corpus_size + 1) // 2
    eval_pool = config.corpus_size // 2
    if config.dev_tasks:
        counts_and_pools.append((config.dev_tasks, dev_pool))
    if config.eval_tasks:
        counts_and_pools.append((config.eval_tasks, eval_pool))
    required = max(
        ((count + pool - 1) // pool for count, pool in counts_and_pools if pool),
        default=1,
    )
    return max(8, required)


def _split_entity_indices(config: GenerationConfig, split: Split) -> tuple[int, ...]:
    if split is Split.DEV and not config.eval_tasks:
        return tuple(range(config.corpus_size))
    if split is Split.EVAL and not config.dev_tasks:
        return tuple(range(config.corpus_size))
    midpoint = (config.corpus_size + 1) // 2
    if split is Split.DEV:
        return tuple(range(midpoint))
    return tuple(range(midpoint, config.corpus_size))


def _make_entities(config: GenerationConfig) -> tuple[_Entity, ...]:
    slot_count = _slots_per_entity(config)
    entities: list[_Entity] = []
    for entity_ordinal in range(config.corpus_size):
        rng = _stage_rng(config.seed, "entity", entity_ordinal)
        service_base = rng.choice(_SERVICE_BASES)
        library_base = rng.choice(_LIBRARY_BASES)
        team_base = rng.choice(_TEAM_BASES)
        region_base = rng.choice(_REGION_BASES)
        number = entity_ordinal + 1
        service = f"{service_base} service {number:02d}"
        library = f"{library_base} library {number:02d}"
        team = f"{team_base} team {number:02d}"
        region = f"{region_base} region {number:02d}"
        service_aliases = tuple(f"{alias} {number:02d}" for alias in SERVICE_ALIASES[service_base])
        library_aliases = tuple(f"{alias} {number:02d}" for alias in LIBRARY_ALIASES[library_base])
        team_aliases = tuple(f"{alias} {number:02d}" for alias in TEAM_ALIASES[team_base])
        region_aliases = tuple(f"{alias} {number:02d}" for alias in REGION_ALIASES[region_base])

        slots: list[_Slot] = []
        for slot_ordinal in range(slot_count):
            release = rng.choice(_RELEASES)
            encryption = rng.choice(_ENCRYPTION_MODES)
            slot_team_base = rng.choice(_TEAM_BASES)
            slot_region_base = rng.choice(_REGION_BASES)
            slot_team = f"{slot_team_base} team {number:02d}"
            slot_region = f"{slot_region_base} region {number:02d}"
            impact = rng.choice(_INCIDENT_IMPACTS)
            incident = f"{impact} event {number:02d}-{slot_ordinal + 1:02d}"
            slots.append(
                _Slot(
                    ordinal=slot_ordinal,
                    release=release,
                    encryption=encryption,
                    team=slot_team,
                    region=slot_region,
                    incident=incident,
                    impact=impact,
                )
            )
        entities.append(
            _Entity(
                ordinal=entity_ordinal,
                service=service,
                service_aliases=service_aliases,
                library=library,
                library_aliases=library_aliases,
                team=team,
                team_aliases=team_aliases,
                region=region,
                region_aliases=region_aliases,
                slots=tuple(slots),
            )
        )
    return tuple(entities)


def _document_header(category: str, entity: _Entity) -> tuple[str, list[str]]:
    display_names = {
        "services": "Service registry",
        "incidents": "Incident register",
        "architecture-decisions": "Architecture decisions",
        "ownership": "Ownership record",
        "deployments": "Deployment history",
        "policies": "Policy register",
        "dependency-manifests": "Dependency manifest",
    }
    title = f"{display_names[category]} for {entity.service}"
    lines = [
        f"# {title}",
        "This synthetic record captures one operational relationship per scenario.",
        "Rendered names are aliases; provenance is retained separately in metadata.",
    ]
    return title, lines


def _render_fact(
    category: str,
    entity: _Entity,
    slot: _Slot,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    service_alias = rng.choice(entity.service_aliases)
    library_alias = rng.choice(entity.library_aliases)
    team_alias = rng.choice(entity.team_aliases)
    region_alias = rng.choice(entity.region_aliases)
    scenario = f"scenario {slot.ordinal + 1:02d}"

    if category == "services":
        return (
            f"The {service_alias} runtime uses the {library_alias} package for {scenario}.",
            entity.service,
            "depends_on",
            entity.library,
        )
    if category == "incidents":
        return (
            f"The {service_alias} recorded a {slot.impact} during {scenario}; "
            "the event is tracked.",
            entity.service,
            "experienced",
            slot.incident,
        )
    if category == "architecture-decisions":
        return (
            f"The {library_alias} design record assigns {slot.encryption} to {scenario}.",
            entity.library,
            "uses_encryption",
            slot.encryption,
        )
    if category == "ownership":
        return (
            f"The {service_alias} support boundary is held by the {team_alias} "
            f"group for {scenario}.",
            entity.service,
            "owned_by",
            slot.team,
        )
    if category == "deployments":
        return (
            f"The {service_alias} currently runs {library_alias} release {slot.release} "
            f"in {region_alias}.",
            entity.service,
            "runs_release",
            f"{entity.library} {slot.release}",
        )
    if category == "policies":
        return (
            f"The governance register requires a security review for {slot.encryption} "
            f"in {scenario}.",
            slot.encryption,
            "requires",
            "security review",
        )
    if category == "dependency-manifests":
        return (
            f"The {library_alias} manifest records release {slot.release} for {scenario}.",
            entity.library,
            "release_version",
            slot.release,
        )
    raise ValueError(f"unsupported document category: {category}")


def _make_document_drafts(
    config: GenerationConfig,
    entities: Sequence[_Entity],
) -> tuple[tuple[_DocumentDraft, ...], dict[tuple[str, int, int], str]]:
    drafts: list[_DocumentDraft] = []
    fact_drafts: list[_FactDraft] = []
    for category_ordinal, category in enumerate(DOCUMENT_CATEGORIES):
        for entity in entities:
            document_ordinal = category_ordinal * config.corpus_size + entity.ordinal
            rng = _stage_rng(config.seed, "document", document_ordinal)
            title, lines = _document_header(category, entity)
            facts: list[_FactDraft] = []
            for slot in entity.slots:
                sentence, subject, predicate, object_value = _render_fact(
                    category,
                    entity,
                    slot,
                    rng,
                )
                lines.append(sentence)
                fact = _FactDraft(
                    key=(category, entity.ordinal, slot.ordinal),
                    subject=subject,
                    predicate=predicate,
                    object=object_value,
                    line_start=len(lines),
                    line_end=len(lines),
                )
                facts.append(fact)
                fact_drafts.append(fact)
            doc_id = f"doc-{category}-{entity.ordinal + 1:04d}"
            drafts.append(
                _DocumentDraft(
                    entity_ordinal=entity.ordinal,
                    category=category,
                    doc_id=doc_id,
                    path=f"/{category}/{doc_id}.md",
                    title=title,
                    content="\n".join(lines) + "\n",
                    facts=tuple(facts),
                    generator_seed=child_seed(config.seed, "document", document_ordinal),
                )
            )

    fact_ids: dict[tuple[str, int, int], str] = {}
    for fact_ordinal, fact in enumerate(fact_drafts, start=1):
        if fact.key in fact_ids:
            raise ValueError(f"duplicate fact tuple: {fact.key}")
        fact_ids[fact.key] = f"fact-{fact_ordinal:06d}"

    documents: list[DocumentRecord] = []
    for draft in drafts:
        facts = [
            FactSpan(
                fact_id=fact_ids[fact.key],
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                line_start=fact.line_start,
                line_end=fact.line_end,
            )
            for fact in draft.facts
        ]
        documents.append(
            DocumentRecord(
                schema_version=1,
                corpus_version=config.corpus_version,
                doc_id=draft.doc_id,
                path=draft.path,
                category=draft.category,
                title=draft.title,
                content=draft.content,
                facts=facts,
                generator_seed=draft.generator_seed,
            )
        )
    return tuple(documents), fact_ids


def _prospective_chunk_windows(
    content: str,
    size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> tuple[_ChunkWindow, ...]:
    """Return the frozen line-aware chunk windows without importing B08."""

    if size_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= size_tokens:
        raise ValueError("invalid prospective chunk settings")
    lines = content.splitlines()
    if not lines:
        return ()
    line_token_counts = [len(_tokens(line)) for line in lines]
    windows: list[_ChunkWindow] = []
    start = 0
    while start < len(lines):
        if line_token_counts[start] > size_tokens:
            step = size_tokens - overlap_tokens
            segment_start = 0
            while segment_start < line_token_counts[start]:
                segment_end = min(segment_start + size_tokens, line_token_counts[start])
                windows.append(
                    _ChunkWindow(
                        chunk_index=len(windows),
                        start_line=start + 1,
                        end_line=start + 1,
                    )
                )
                if segment_end == line_token_counts[start]:
                    break
                segment_start += step
            start += 1
            continue

        end = start
        total = 0
        while end < len(lines):
            line_count = line_token_counts[end]
            if end > start and total + line_count > size_tokens:
                break
            total += line_count
            end += 1
            if total >= size_tokens:
                break
        if end == start:
            end += 1
        windows.append(
            _ChunkWindow(
                chunk_index=len(windows),
                start_line=start + 1,
                end_line=end,
            )
        )

        if total <= overlap_tokens:
            start = end
            continue
        remaining = overlap_tokens
        overlap_start = end - 1
        while overlap_start >= start and remaining > 0:
            remaining -= line_token_counts[overlap_start]
            overlap_start -= 1
        next_start = overlap_start + 1
        start = next_start if next_start > start else end
    return tuple(windows)


def _fact_chunk_ids(
    document: DocumentRecord,
) -> dict[str, tuple[str, ...]]:
    windows = _prospective_chunk_windows(document.content)
    result: dict[str, list[str]] = {fact.fact_id: [] for fact in document.facts}
    for window in windows:
        chunk_id = f"{document.doc_id}::c{window.chunk_index:04d}"
        for fact in document.facts:
            if fact.line_start <= window.end_line and fact.line_end >= window.start_line:
                result[fact.fact_id].append(chunk_id)
    return {fact_id: tuple(chunk_ids) for fact_id, chunk_ids in result.items()}


def _validate_no_complete_fact_chunk(
    documents: Sequence[DocumentRecord],
    tasks: Sequence[TaskRecord],
) -> None:
    """Reject a prospective chunk that contains every fact for one task."""

    for document in documents:
        windows = _prospective_chunk_windows(document.content)
        for task in tasks:
            required_for_document = set(task.required_fact_ids) & {
                fact.fact_id for fact in document.facts
            }
            if not required_for_document:
                continue
            for window in windows:
                covered = {
                    fact.fact_id
                    for fact in document.facts
                    if fact.line_start <= window.end_line and fact.line_end >= window.start_line
                }
                if set(task.required_fact_ids).issubset(covered):
                    raise ValueError(f"task {task.task_id} has all required facts in one chunk")


def _template_ordinal(template: str) -> int:
    match = re.fullmatch(r"template-(\d+)", template)
    if match is None:
        raise ValueError(f"invalid generated template ID: {template}")
    return int(match.group(1))


def _task_fact_chain(
    template_ordinal: int,
    entity_ordinal: int,
    slot_ordinal: int,
) -> tuple[tuple[str, int, int], ...]:
    if template_ordinal in (0, 1):
        categories = (
            "services",
            "dependency-manifests",
            "architecture-decisions",
            "policies",
        )
    else:
        categories = ("ownership", "incidents", "deployments", "policies")
    return tuple((category, entity_ordinal, slot_ordinal) for category in categories)


def _question_and_answer(
    template_ordinal: int,
    hop_count: int,
    entity: _Entity,
    slot: _Slot,
) -> tuple[str, str, tuple[str, ...], str]:
    service_alias = entity.service_aliases[0]
    library_alias = entity.library_aliases[1]
    team_alias = entity.team_aliases[0]
    if template_ordinal in (0, 1):
        if hop_count == 2:
            if template_ordinal == 0:
                question = (
                    f"Which release is recorded for the {service_alias} dependency on "
                    f"{library_alias}?"
                )
            else:
                question = (
                    f"Name the active package version carried by {service_alias} through "
                    f"{library_alias}."
                )
            answer = slot.release
            variants = (f"release {slot.release}", f"version {slot.release}")
        elif hop_count == 3:
            if template_ordinal == 0:
                question = (
                    f"Which protection mode follows from the {library_alias} decision "
                    f"for {service_alias}?"
                )
            else:
                question = (
                    f"What encryption posture applies to the {service_alias} path using "
                    f"{library_alias}?"
                )
            answer = slot.encryption
            variants = (f"{slot.encryption} mode", f"uses {slot.encryption}")
        else:
            if template_ordinal == 0:
                question = (
                    f"Does the {service_alias} path require a security review for its "
                    f"{library_alias} release?"
                )
            else:
                question = (
                    f"Is a security review required for {service_alias} after the "
                    f"{library_alias} deployment?"
                )
            answer = "Yes, a security review is required"
            variants = ("yes", "security review required")
        category = "dependency-governance"
    else:
        if hop_count == 2:
            if template_ordinal == 2:
                question = f"Which event is recorded for {service_alias} under the {team_alias}?"
            else:
                question = (
                    f"Name the tracked operational event linked to {service_alias} and "
                    f"{team_alias}."
                )
            answer = slot.incident
            variants = (f"{slot.impact} event", slot.incident.replace(" event ", " "))
        elif hop_count == 3:
            if template_ordinal == 2:
                question = f"Which release was running when the {service_alias} event was recorded?"
            else:
                question = (
                    f"Identify the deployed version associated with the {service_alias} "
                    "operational event."
                )
            answer = slot.release
            variants = (f"release {slot.release}", f"version {slot.release}")
        else:
            if template_ordinal == 2:
                question = f"Does the {service_alias} operational path require a security review?"
            else:
                question = f"Is a security review required for the {service_alias} event path?"
            answer = "Yes, a security review is required"
            variants = ("yes", "security review required")
        category = "operational-governance"
    return question, answer, variants, category


def _preferred_distractor_documents(
    documents_by_entity_category: dict[tuple[int, str], DocumentRecord],
    entity_indices: Sequence[int],
    categories: Sequence[str],
    entity_ordinal: int,
    gold_document_ids: set[str],
) -> list[DocumentRecord]:
    preferred: list[DocumentRecord] = []
    for candidate_entity in entity_indices:
        for category in categories:
            document = documents_by_entity_category[(candidate_entity, category)]
            if document.doc_id not in gold_document_ids:
                preferred.append(document)
    if len(preferred) < 1:
        preferred = [
            document
            for (candidate_entity, _), document in documents_by_entity_category.items()
            if candidate_entity in entity_indices and document.doc_id not in gold_document_ids
        ]
    # Preserve a stable order before the caller applies its local shuffle.
    return sorted(preferred, key=lambda document: document.doc_id)


def _make_tasks(
    config: GenerationConfig,
    entities: Sequence[_Entity],
    documents: Sequence[DocumentRecord],
    fact_ids: dict[tuple[str, int, int], str],
) -> tuple[TaskRecord, ...]:
    documents_by_entity_category = {
        (entity_ordinal, document.category): document
        for document in documents
        for entity_ordinal in [
            next(entity.ordinal for entity in entities if entity.service in document.title)
        ]
    }
    documents_by_id = {document.doc_id: document for document in documents}
    fact_to_document = {
        fact.fact_id: document.doc_id for document in documents for fact in document.facts
    }
    tasks: list[TaskRecord] = []
    global_ordinal = 0
    for split, requested_count in (
        (Split.DEV, config.dev_tasks),
        (Split.EVAL, config.eval_tasks),
    ):
        if requested_count == 0:
            continue
        entity_indices = _split_entity_indices(config, split)
        if not entity_indices:
            raise ValueError(f"{split.value} tasks require a non-empty entity split")
        template_options = (0, 2) if split is Split.DEV else (1, 3)
        for sequence in range(requested_count):
            entity_ordinal = entity_indices[sequence % len(entity_indices)]
            slot_ordinal = sequence // len(entity_indices)
            task_rng = _stage_rng(config.seed, "task", global_ordinal)
            template_ordinal = template_options[
                (sequence + task_rng.randrange(len(template_options))) % len(template_options)
            ]
            hop_count = task_rng.randint(config.min_hops, config.max_hops)
            entity = entities[entity_ordinal]
            slot = entity.slots[slot_ordinal]
            question, answer, variants, category = _question_and_answer(
                template_ordinal,
                hop_count,
                entity,
                slot,
            )
            chain = _task_fact_chain(template_ordinal, entity_ordinal, slot_ordinal)
            required_fact_ids = sorted(fact_ids[key] for key in chain[:hop_count])
            gold_document_ids = sorted({fact_to_document[fact_id] for fact_id in required_fact_ids})
            if len(gold_document_ids) < 2:
                raise ValueError(f"task {sequence} does not span multiple documents")
            preferred = _preferred_distractor_documents(
                documents_by_entity_category,
                entity_indices,
                [key[0] for key in chain],
                entity_ordinal,
                set(gold_document_ids),
            )
            task_rng.shuffle(preferred)
            distractor_documents = preferred[: config.distractors]
            if len(distractor_documents) != config.distractors:
                raise ValueError(
                    f"cannot provide {config.distractors} distractors for {split.value} task"
                )
            distractor_document_ids = sorted(document.doc_id for document in distractor_documents)
            gold_chunk_ids: set[str] = set()
            for fact_id in required_fact_ids:
                document = documents_by_id[fact_to_document[fact_id]]
                gold_chunk_ids.update(_fact_chunk_ids(document)[fact_id])
            task = TaskRecord(
                schema_version=1,
                task_id=f"task-{split.value}-{sequence + 1:06d}",
                split=split,
                question=question,
                canonical_answer=answer,
                acceptable_answer_variants=list(variants),
                required_fact_ids=required_fact_ids,
                gold_document_ids=gold_document_ids,
                gold_chunk_ids=sorted(gold_chunk_ids),
                hop_count=hop_count,
                task_template=f"template-{template_ordinal:02d}",
                category=category,
                distractor_document_ids=distractor_document_ids,
                distractor_count=config.distractors,
                generator_seed=child_seed(config.seed, "task", global_ordinal),
                corpus_version=config.corpus_version,
            )
            tasks.append(task)
            global_ordinal += 1
    return tuple(tasks)


def _validate_records(
    documents: Sequence[DocumentRecord],
    tasks: Sequence[TaskRecord],
    config: GenerationConfig | None = None,
) -> None:
    """Validate cross-record IDs and all generator-specific leakage contracts."""

    if not documents and tasks:
        raise ValueError("tasks require at least one document")
    validate_dataset(documents, tasks, None)
    document_ids = {document.doc_id for document in documents}
    document_paths = {document.path for document in documents}
    if len(document_ids) != len(documents):
        raise ValueError("duplicate document IDs")
    if len(document_paths) != len(documents):
        raise ValueError("duplicate document paths")
    fact_ids = [fact.fact_id for document in documents for fact in document.facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("duplicate fact IDs")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs")

    documents_by_id = {document.doc_id: document for document in documents}
    fact_by_id = {
        fact.fact_id: (document, fact) for document in documents for fact in document.facts
    }
    split_document_ids: dict[Split, set[str]] = {Split.DEV: set(), Split.EVAL: set()}
    task_tuples: set[tuple[object, ...]] = set()
    for task in tasks:
        if task.task_template not in TEMPLATE_FAMILIES:
            raise ValueError(f"unknown task template: {task.task_template}")
        template_ordinal = _template_ordinal(task.task_template)
        if (task.split is Split.DEV and template_ordinal % 2 != 0) or (
            task.split is Split.EVAL and template_ordinal % 2 != 1
        ):
            raise ValueError("template parity does not match the task split")
        if not 2 <= task.hop_count <= 4 or len(task.required_fact_ids) != task.hop_count:
            raise ValueError("task hop count is outside the configured range")
        if len({fact_by_id[fact_id][0].doc_id for fact_id in task.required_fact_ids}) < 2:
            raise ValueError("task facts must span at least two documents")
        if _normalise_text(task.question) == "":
            raise ValueError("task question must not be blank")
        normalized_question = _normalise_text(task.question)
        if any(normalized_question in _normalise_text(document.content) for document in documents):
            raise ValueError(f"question text leaks into a document: {task.task_id}")
        required_document_ids = {
            fact_by_id[fact_id][0].doc_id for fact_id in task.required_fact_ids
        }
        if required_document_ids != set(task.gold_document_ids):
            raise ValueError(f"gold documents do not match facts for {task.task_id}")
        expected_chunk_ids: set[str] = set()
        for fact_id in task.required_fact_ids:
            document, _ = fact_by_id[fact_id]
            expected_chunk_ids.update(_fact_chunk_ids(document)[fact_id])
        if expected_chunk_ids != set(task.gold_chunk_ids):
            raise ValueError(f"gold chunks do not match facts for {task.task_id}")
        if set(task.gold_document_ids) & set(task.distractor_document_ids):
            raise ValueError(f"gold and distractor documents overlap for {task.task_id}")
        if task.distractor_count != len(task.distractor_document_ids):
            raise ValueError(f"distractor count is wrong for {task.task_id}")
        if any(document_id not in documents_by_id for document_id in task.distractor_document_ids):
            raise ValueError(f"missing distractor document for {task.task_id}")

        split_document_ids[task.split].update(task.gold_document_ids)
        split_document_ids[task.split].update(task.distractor_document_ids)
        task_tuple = (
            task.split.value,
            task.task_template,
            tuple(task.required_fact_ids),
            tuple(task.gold_document_ids),
        )
        if task_tuple in task_tuples:
            raise ValueError(f"duplicate task tuple for {task.task_id}")
        task_tuples.add(task_tuple)

    if split_document_ids[Split.DEV] & split_document_ids[Split.EVAL]:
        raise ValueError("development and evaluation entity tuples overlap")
    _validate_no_complete_fact_chunk(documents, tasks)

    if config is not None:
        expected_task_count = _task_counts(config)
        if len(tasks) != expected_task_count:
            raise ValueError("generated task count does not match the requested count")
        actual_counts = {
            Split.DEV: sum(task.split is Split.DEV for task in tasks),
            Split.EVAL: sum(task.split is Split.EVAL for task in tasks),
        }
        if (
            actual_counts[Split.DEV] != config.dev_tasks
            or actual_counts[Split.EVAL] != config.eval_tasks
        ):
            raise ValueError("generated split counts do not match the requested counts")
        expected_categories = set(DOCUMENT_CATEGORIES)
        if {document.category for document in documents} != expected_categories:
            raise ValueError("generated corpus is missing a document category")
        if len(documents) != config.corpus_size * len(DOCUMENT_CATEGORIES):
            raise ValueError("generated document count does not match corpus_size")


def generate_dataset(
    config: GenerationConfig,
) -> tuple[list[DocumentRecord], list[TaskRecord]]:
    """Generate one deterministic corpus and its multi-hop tasks."""

    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig")
    entities = _make_entities(config)
    documents, fact_ids = _make_document_drafts(config, entities)
    tasks = _make_tasks(config, entities, documents, fact_ids)
    ordered_documents = tuple(sorted(documents, key=lambda document: document.doc_id))
    ordered_tasks = tuple(
        sorted(
            tasks,
            key=lambda task: (0 if task.split is Split.DEV else 1, task.task_id),
        )
    )
    _validate_records(ordered_documents, ordered_tasks, config)
    return list(ordered_documents), list(ordered_tasks)


def write_dataset(
    output_dir: Path,
    documents: Iterable[DocumentRecord],
    tasks: Iterable[TaskRecord],
) -> ArtifactHashes:
    """Write validated corpus and task JSONL artifacts using B05 utilities."""

    target_dir = Path(os.path.expanduser(os.fspath(output_dir)))
    document_rows = tuple(sorted(documents, key=lambda document: document.doc_id))
    task_rows = tuple(
        sorted(
            tasks,
            key=lambda task: (0 if task.split is Split.DEV else 1, task.task_id),
        )
    )
    _validate_records(document_rows, task_rows)
    corpus_path = target_dir / "corpus.jsonl"
    tasks_path = target_dir / "tasks.jsonl"
    corpus_sha256 = write_jsonl_atomic(
        corpus_path,
        (document.model_dump(mode="json") for document in document_rows),
    )
    tasks_sha256 = write_jsonl_atomic(
        tasks_path,
        (task.model_dump(mode="json") for task in task_rows),
    )
    return ArtifactHashes(
        corpus_sha256=corpus_sha256,
        tasks_sha256=tasks_sha256,
        corpus_path=corpus_path,
        tasks_path=tasks_path,
    )


__all__ = [
    "ArtifactHashes",
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_SIZE_TOKENS",
    "DOCUMENT_CATEGORIES",
    "ENTITY_ALIASES",
    "GenerationConfig",
    "LIBRARY_ALIASES",
    "REGION_ALIASES",
    "SERVICE_ALIASES",
    "TEMPLATE_FAMILIES",
    "TEAM_ALIASES",
    "child_seed",
    "generate_dataset",
    "write_dataset",
]
