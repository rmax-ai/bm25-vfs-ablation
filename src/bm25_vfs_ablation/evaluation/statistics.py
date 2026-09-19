"""Deterministic paired inference and associative VFS analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest
from statsmodels.tools.sm_exceptions import PerfectSeparationError

type FrameLike = pd.DataFrame | Mapping[str, object] | Sequence[object]

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20250308

_MISSING = object()

_TASK_NAMES = ("task_id", "task.task_id")
_CONDITION_NAMES = ("condition", "condition_name")
_DESIGN_NAMES = ("design_cell", "design", "cell")
_SUCCESS_NAMES = (
    "correctness",
    "success",
    "task_success",
    "outcome",
)
_TOKEN_NAMES = (
    "total_tokens",
    "token_count",
    "tokens",
    "total_token_count",
    "token_accounting.total_tokens_estimated",
)
_RECALL_NAMES = (
    "initial_recall",
    "initial_chunk_recall",
    "initial_retrieval_recall",
    "retrieval_metrics.initial_chunk_recall",
)
_CALL_NAMES = (
    "tool_call_count",
    "calls",
    "actual_tool_calls",
    "vfs_tool_calls",
)
_HOP_NAMES = ("hop_count", "hops", "task_hop_count", "task.hop_count")
_CATEGORY_NAMES = ("category", "task_category", "task.category")
_DISTRACTOR_NAMES = (
    "distractor_count",
    "distractors",
    "task_distractor_count",
    "task.distractor_count",
)

_STRATA_ORDER = ("token_quartile", "recall_band", "hop", "category")
_STRATA_COLUMNS = (
    "stratum",
    "band",
    "n",
    "snippets_success_rate",
    "vfs_success_rate",
    "difference",
    "mcnemar_p_value",
    "missing_reason",
)
_REGRESSION_COLUMNS = (
    "term",
    "coefficient",
    "std_error",
    "ci_lower",
    "ci_upper",
    "p_value",
    "converged",
    "error",
    "associative",
    "covariance_type",
    "n",
    "reference_category",
    "status",
)


@dataclass(frozen=True, slots=True)
class PairedStats:
    """Primary paired success summary.

    ``difference`` is always VFS minus snippets.  The object intentionally
    stores the McNemar discordance counts as well as its exact p-value so that
    downstream tables do not need to reconstruct the paired outcomes.
    """

    n: int
    snippets_successes: int
    vfs_successes: int
    snippets_success_rate: float
    vfs_success_rate: float
    difference: float
    ci_lower: float
    ci_upper: float
    snippets_only: int
    vfs_only: int
    mcnemar_p_value: float
    resamples: int
    seed: int

    @property
    def n_pairs(self) -> int:
        """Compatibility name for the number of complete task pairs."""

        return self.n

    @property
    def vfs_minus_snippets(self) -> float:
        """Return the prespecified VFS-minus-snippets estimate."""

        return self.difference

    @property
    def snippets_rate(self) -> float:
        """Compatibility name for the snippets success rate."""

        return self.snippets_success_rate

    @property
    def vfs_rate(self) -> float:
        """Compatibility name for the VFS success rate."""

        return self.vfs_success_rate

    @property
    def success_difference(self) -> float:
        """Compatibility name for :attr:`difference`."""

        return self.difference

    @property
    def point_estimate(self) -> float:
        """Return the paired point estimate."""

        return self.difference

    @property
    def ci_low(self) -> float:
        """Compatibility name for the lower percentile bound."""

        return self.ci_lower

    @property
    def ci_high(self) -> float:
        """Compatibility name for the upper percentile bound."""

        return self.ci_upper

    @property
    def ci(self) -> tuple[float, float]:
        """Return the percentile interval as a pair."""

        return self.ci_lower, self.ci_upper

    @property
    def mcnemar_p(self) -> float:
        """Compatibility name for the exact McNemar p-value."""

        return self.mcnemar_p_value

    def as_dict(self) -> dict[str, object]:
        """Return a stable, JSON-friendly summary mapping."""

        return {
            "n": self.n,
            "snippets_successes": self.snippets_successes,
            "vfs_successes": self.vfs_successes,
            "snippets_success_rate": self.snippets_success_rate,
            "vfs_success_rate": self.vfs_success_rate,
            "difference": self.difference,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "snippets_only": self.snippets_only,
            "vfs_only": self.vfs_only,
            "mcnemar_p_value": self.mcnemar_p_value,
            "resamples": self.resamples,
            "seed": self.seed,
        }

    to_dict = as_dict

    def __getitem__(self, key: str) -> object:
        """Allow table-like access to the public summary fields."""

        aliases = {
            "n_pairs": "n",
            "vfs_minus_snippets": "difference",
            "success_difference": "difference",
            "point_estimate": "difference",
            "ci_low": "ci_lower",
            "ci_high": "ci_upper",
            "mcnemar_p": "mcnemar_p_value",
        }
        return self.as_dict()[aliases.get(key, key)]


@dataclass(frozen=True, slots=True)
class _Pair:
    task_id: str
    snippets: Mapping[str, object]
    vfs: Mapping[str, object]


def _is_missing(value: object) -> bool:
    if value is _MISSING or value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return attributes
    raise TypeError("statistics input rows must be mappings or model-like records")


def _records(frame: FrameLike) -> tuple[Mapping[str, object], ...]:
    if isinstance(frame, pd.DataFrame):
        return tuple(frame.to_dict(orient="records"))
    if isinstance(frame, Mapping):
        return (_as_mapping(frame),)
    if isinstance(frame, (str, bytes, bytearray)):
        raise TypeError("statistics input must not be a string")
    try:
        return tuple(_as_mapping(item) for item in frame)
    except TypeError as exc:
        raise TypeError("statistics input must be a DataFrame or row sequence") from exc


def _path_value(row: Mapping[str, object], path: str) -> object:
    if path in row:
        return row[path]
    current: object = row
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        else:
            current = getattr(current, part, _MISSING)
            if current is _MISSING:
                return _MISSING
    return current


def _value(row: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        candidate = _path_value(row, name)
        if not _is_missing(candidate):
            return candidate
    return _MISSING


def _number(value: object, *, nonnegative: bool = False) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number) or (nonnegative and number < 0):
        return None
    return number


def _integer(value: object, name: str) -> int:
    number = _number(value, nonnegative=True)
    if number is None or not number.is_integer():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(number)


def _outcome(row: Mapping[str, object]) -> bool:
    value = _value(row, _SUCCESS_NAMES)
    if _is_missing(value):
        raise ValueError("primary records must contain correctness/success")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    number = _number(value)
    if number in {0.0, 1.0}:
        return bool(number)
    raise ValueError("correctness/success must be boolean or binary")


def _condition(value: object) -> str:
    if _is_missing(value):
        raise ValueError("primary records must contain condition")
    enum_value = getattr(value, "value", value)
    normalized = str(enum_value).strip().casefold()
    if normalized.startswith("condition."):
        normalized = normalized.removeprefix("condition.")
    return normalized


def _design(value: object) -> str:
    if _is_missing(value):
        return "primary"
    enum_value = getattr(value, "value", value)
    normalized = str(enum_value).strip().casefold()
    if normalized.startswith("designcell."):
        normalized = normalized.removeprefix("designcell.")
    return normalized


def _task_id(row: Mapping[str, object]) -> str:
    value = _value(row, _TASK_NAMES)
    if _is_missing(value):
        raise ValueError("primary records must contain task_id")
    return str(value)


def _primary_pairs(frame: FrameLike) -> tuple[_Pair, ...]:
    rows = _records(frame)
    grouped: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if _design(_value(row, _DESIGN_NAMES)) != "primary":
            continue
        task_id = _task_id(row)
        condition = _condition(_value(row, _CONDITION_NAMES))
        if condition not in {"snippets", "vfs"}:
            raise ValueError(f"unsupported primary condition: {condition}")
        if condition in grouped[task_id]:
            raise ValueError(f"duplicate primary {condition} record for task {task_id}")
        grouped[task_id][condition] = row

    if not grouped:
        raise ValueError("no primary records supplied")

    pairs: list[_Pair] = []
    for task_id in sorted(grouped):
        conditions = grouped[task_id]
        if set(conditions) != {"snippets", "vfs"}:
            raise ValueError(f"task {task_id} does not have exactly one paired condition")
        pairs.append(_Pair(task_id, conditions["snippets"], conditions["vfs"]))
    return tuple(pairs)


def _discordance_counts(
    snippets: Sequence[bool],
    vfs: Sequence[bool],
) -> tuple[int, int]:
    if len(snippets) != len(vfs):
        raise ValueError("paired outcomes must have equal lengths")
    snippets_only = sum(
        1
        for snippets_success, vfs_success in zip(snippets, vfs, strict=True)
        if snippets_success and not vfs_success
    )
    vfs_only = sum(
        1
        for snippets_success, vfs_success in zip(snippets, vfs, strict=True)
        if vfs_success and not snippets_success
    )
    return snippets_only, vfs_only


def _count(value: object, name: str) -> int:
    return _integer(value, name)


def exact_mcnemar(
    b: int | Sequence[object] | Mapping[str, object] | pd.DataFrame | None = None,
    c: int | Sequence[object] | None = None,
    *,
    snippets_only: int | None = None,
    vfs_only: int | None = None,
) -> float:
    """Return the exact two-sided McNemar p-value.

    Integer arguments are the two discordant counts.  Two equal-length
    boolean sequences are also accepted and are counted as paired outcomes.
    The no-discordance case is explicitly defined as one.
    """

    if snippets_only is not None or vfs_only is not None:
        if b is not None or c is not None:
            raise TypeError("provide either b/c or snippets_only/vfs_only")
        if snippets_only is None or vfs_only is None:
            raise TypeError("both snippets_only and vfs_only are required")
        b = snippets_only
        c = vfs_only

    if isinstance(b, pd.DataFrame) and c is None:
        pairs = _primary_pairs(b)
        snippets = [_outcome(pair.snippets) for pair in pairs]
        vfs = [_outcome(pair.vfs) for pair in pairs]
        b_count, c_count = _discordance_counts(snippets, vfs)
    elif c is not None:
        if _is_outcome_sequence(b) and _is_outcome_sequence(c):
            snippets = [_coerce_outcome(item) for item in b]  # type: ignore[arg-type]
            vfs = [_coerce_outcome(item) for item in c]
            b_count, c_count = _discordance_counts(snippets, vfs)
        else:
            b_count = _count(b, "b")  # type: ignore[arg-type]
            c_count = _count(c, "c")
    elif isinstance(b, Mapping):
        b_value = b.get("b", b.get("snippets_only", _MISSING))
        c_value = b.get("c", b.get("vfs_only", _MISSING))
        if _is_missing(b_value) or _is_missing(c_value):
            raise ValueError("McNemar mapping must contain b/c or discordance keys")
        b_count = _count(b_value, "b")
        c_count = _count(c_value, "c")
    elif _is_count_pair(b):
        b_count = _count(b[0], "b")  # type: ignore[index]
        c_count = _count(b[1], "c")  # type: ignore[index]
    else:
        raise TypeError("exact_mcnemar expects two counts or two paired outcome sequences")

    discordant = b_count + c_count
    if discordant == 0:
        return 1.0
    return float(binomtest(min(b_count, c_count), discordant, 0.5).pvalue)


def _is_outcome_sequence(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return value.ndim == 1
    return isinstance(value, Sequence)


def _is_count_pair(value: object) -> bool:
    if not _is_outcome_sequence(value):
        return False
    values = list(value)  # type: ignore[arg-type]
    if len(values) != 2:
        return False
    return all(_number(item, nonnegative=True) is not None for item in values)


def _coerce_outcome(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    number = _number(value)
    if number in {0.0, 1.0}:
        return bool(number)
    raise ValueError("paired outcomes must be binary")


def paired_summary(
    frame: FrameLike,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedStats:
    """Compute the prespecified paired bootstrap and exact McNemar result."""

    resample_count = _count(resamples, "resamples")
    if resample_count < 1:
        raise ValueError("resamples must be positive")
    pairs = _primary_pairs(frame)
    snippets = [_outcome(pair.snippets) for pair in pairs]
    vfs = [_outcome(pair.vfs) for pair in pairs]
    differences = np.asarray(
        [
            float(vfs_success) - float(snippets_success)
            for snippets_success, vfs_success in zip(snippets, vfs, strict=True)
        ],
        dtype=float,
    )
    point = float(differences.mean())

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resample_count, len(differences)))
    bootstrap_estimates = differences[indices].mean(axis=1)
    ci_lower, ci_upper = np.percentile(bootstrap_estimates, (2.5, 97.5))
    snippets_only, vfs_only = _discordance_counts(snippets, vfs)

    return PairedStats(
        n=len(pairs),
        snippets_successes=sum(snippets),
        vfs_successes=sum(vfs),
        snippets_success_rate=float(np.mean(snippets)),
        vfs_success_rate=float(np.mean(vfs)),
        difference=point,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        snippets_only=snippets_only,
        vfs_only=vfs_only,
        mcnemar_p_value=exact_mcnemar(snippets_only, vfs_only),
        resamples=resample_count,
        seed=int(seed),
    )


def _row_number(row: Mapping[str, object], names: Sequence[str]) -> float | None:
    return _number(_value(row, names), nonnegative=True)


def _recall_band(value: float | None) -> str | None:
    if value is None or not 0 <= value <= 1:
        return None
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    return "(0,1)"


def _token_labels(
    pairs: Sequence[_Pair],
) -> tuple[dict[tuple[str, str], str], list[str], str | None]:
    values: list[float] = []
    raw: dict[tuple[str, str], float | None] = {}
    for pair in pairs:
        for condition, row in (("snippets", pair.snippets), ("vfs", pair.vfs)):
            value = _row_number(row, _TOKEN_NAMES)
            raw[(pair.task_id, condition)] = value
            if value is not None:
                values.append(value)

    if not values:
        return {}, ["q1"], "missing token count"

    quantiles = np.quantile(np.asarray(values, dtype=float), [0, 0.25, 0.5, 0.75, 1])
    edges: list[float] = []
    for edge in quantiles:
        edge = float(edge)
        if not edges or edge != edges[-1]:
            edges.append(edge)

    labels: dict[tuple[str, str], str] = {}
    if len(edges) == 1:
        for key, value in raw.items():
            if value is not None:
                labels[key] = "q1"
        return labels, ["q1"], None

    internal_edges = np.asarray(edges[1:-1], dtype=float)
    levels = [f"q{index}" for index in range(1, len(edges))]
    for key, value in raw.items():
        if value is not None:
            index = int(np.searchsorted(internal_edges, value, side="right"))
            labels[key] = levels[index]
    return labels, levels, None


def _pair_band_values(
    pairs: Sequence[_Pair],
) -> tuple[
    dict[tuple[str, str], str | int | None],
    dict[tuple[str, str], str | int | None],
    list[int],
    list[str],
]:
    recalls: dict[tuple[str, str], str | None] = {}
    hops: dict[tuple[str, str], int | None] = {}
    categories: dict[tuple[str, str], str | None] = {}
    hop_values: set[int] = set()
    category_values: set[str] = set()

    for pair in pairs:
        for condition, row in (("snippets", pair.snippets), ("vfs", pair.vfs)):
            key = (pair.task_id, condition)
            recalls[key] = _recall_band(_row_number(row, _RECALL_NAMES))
            hop_value = _row_number(row, _HOP_NAMES)
            if hop_value is not None and hop_value.is_integer():
                hops[key] = int(hop_value)
                hop_values.add(int(hop_value))
            else:
                hops[key] = None
            category_value = _value(row, _CATEGORY_NAMES)
            if not _is_missing(category_value) and str(category_value).strip():
                categories[key] = str(category_value)
                category_values.add(str(category_value))
            else:
                categories[key] = None

    values: dict[tuple[str, str], str | int | None] = {}
    values.update({("recall_band", *key): value for key, value in recalls.items()})
    values.update({("hop", *key): value for key, value in hops.items()})
    values.update({("category", *key): value for key, value in categories.items()})
    return values, {}, sorted(hop_values), sorted(category_values)


def _stratum_pairs(
    pairs: Sequence[_Pair],
    labels: Mapping[tuple[str, str], str | int | None],
    stratum: str,
    level: str | int,
) -> list[_Pair]:
    selected: list[_Pair] = []
    for pair in pairs:
        left = labels.get((stratum, pair.task_id, "snippets"))
        right = labels.get((stratum, pair.task_id, "vfs"))
        if left is not None and left == right == level:
            selected.append(pair)
    return selected


def _stratum_row(
    stratum: str,
    level: str | int,
    pairs: Sequence[_Pair],
    *,
    missing_reason: str | None,
) -> dict[str, object]:
    if pairs:
        snippets = [_outcome(pair.snippets) for pair in pairs]
        vfs = [_outcome(pair.vfs) for pair in pairs]
        snippets_rate = float(np.mean(snippets))
        vfs_rate = float(np.mean(vfs))
        snippets_only, vfs_only = _discordance_counts(snippets, vfs)
        return {
            "stratum": stratum,
            "band": level,
            "n": len(pairs),
            "snippets_success_rate": snippets_rate,
            "vfs_success_rate": vfs_rate,
            "difference": vfs_rate - snippets_rate,
            "mcnemar_p_value": exact_mcnemar(snippets_only, vfs_only),
            "missing_reason": None,
        }
    return {
        "stratum": stratum,
        "band": level,
        "n": 0,
        "snippets_success_rate": np.nan,
        "vfs_success_rate": np.nan,
        "difference": np.nan,
        "mcnemar_p_value": np.nan,
        "missing_reason": missing_reason or "no within-band pairs",
    }


def stratified_summaries(frame: FrameLike) -> pd.DataFrame:
    """Return deterministic paired summaries for the prespecified strata."""

    pairs = _primary_pairs(frame)
    token_labels, token_levels, token_missing = _token_labels(pairs)
    values, _, hop_levels, category_levels = _pair_band_values(pairs)
    labels: dict[tuple[str, str, str], str | int | None] = {}
    labels.update(
        {
            ("token_quartile", task_id, condition): value
            for (task_id, condition), value in token_labels.items()
        }
    )
    labels.update(
        {
            (stratum, task_id, condition): value
            for (stratum, task_id, condition), value in values.items()
        }
    )

    levels_by_stratum: dict[str, list[str | int]] = {
        "token_quartile": token_levels,
        "recall_band": ["0", "(0,1)", "1"],
        "hop": hop_levels or ["all"],
        "category": category_levels or ["all"],
    }
    rows: list[dict[str, object]] = []
    for stratum in _STRATA_ORDER:
        levels = levels_by_stratum[stratum]
        for level in levels:
            selected = _stratum_pairs(pairs, labels, stratum, level)
            if selected:
                reason = None
            elif not pairs:
                reason = "no paired observations"
            elif stratum == "token_quartile" and token_missing is not None:
                reason = token_missing
            elif stratum in {"hop", "category"} and levels == ["all"]:
                reason = f"missing {stratum}"
            else:
                reason = "no within-band pairs"
            rows.append(_stratum_row(stratum, level, selected, missing_reason=reason))

    result = pd.DataFrame(rows, columns=_STRATA_COLUMNS)
    result.attrs["sorted"] = True
    return result


def _vfs_rows(frame: FrameLike) -> tuple[Mapping[str, object], ...]:
    rows = _records(frame)
    selected: list[Mapping[str, object]] = []
    saw_condition = False
    for row in rows:
        design = _design(_value(row, _DESIGN_NAMES))
        if design != "primary":
            continue
        raw_condition = _value(row, _CONDITION_NAMES)
        if not _is_missing(raw_condition):
            saw_condition = True
            if _condition(raw_condition) != "vfs":
                continue
        selected.append(row)
    if saw_condition and not selected:
        return ()
    return tuple(selected)


def _regression_result(
    rows: Sequence[dict[str, object]],
    *,
    n: int,
    reference_category: str | None,
    error: str | None,
    converged: bool,
) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=_REGRESSION_COLUMNS)
    result.attrs.update(
        {
            "associative": True,
            "covariance_type": "HC3",
            "reference_category": reference_category,
        }
    )
    if error is not None and not rows:
        status = {
            "term": "__status__",
            "coefficient": np.nan,
            "std_error": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "converged": converged,
            "error": error,
            "associative": True,
            "covariance_type": "HC3",
            "n": n,
            "reference_category": reference_category,
            "status": "error",
        }
        result = pd.DataFrame([status], columns=_REGRESSION_COLUMNS)
        result.attrs.update(
            {
                "associative": True,
                "covariance_type": "HC3",
                "reference_category": reference_category,
            }
        )
    return result


def within_vfs_regression(
    frame: FrameLike,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Fit the prespecified associative within-VFS binomial model.

    Actual calls are post-treatment observations, so the returned table is
    explicitly labelled associative and is not a causal mediation estimate.
    """

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    rows = _vfs_rows(frame)
    if not rows:
        return _regression_result(
            [],
            n=0,
            reference_category=None,
            error="no primary VFS records",
            converged=False,
        )

    task_ids: set[str] = set()
    prepared: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        raw_task_id = _value(row, _TASK_NAMES)
        if not _is_missing(raw_task_id):
            task_id = str(raw_task_id)
            if task_id in task_ids:
                return _regression_result(
                    [],
                    n=len(prepared),
                    reference_category=None,
                    error=f"duplicate VFS task_id: {task_id}",
                    converged=False,
                )
            task_ids.add(task_id)
        else:
            task_id = f"row-{index:06d}"

        outcome = _value(row, _SUCCESS_NAMES)
        calls = _row_number(row, _CALL_NAMES)
        recall = _row_number(row, _RECALL_NAMES)
        hops = _row_number(row, _HOP_NAMES)
        tokens = _row_number(row, _TOKEN_NAMES)
        distractors = _row_number(row, _DISTRACTOR_NAMES)
        category = _value(row, _CATEGORY_NAMES)
        missing = [
            name
            for name, value in (
                ("correctness", outcome),
                ("calls", calls),
                ("initial recall", recall),
                ("hops", hops),
                ("tokens", tokens),
                ("distractors", distractors),
                ("category", category),
            )
            if _is_missing(value)
        ]
        if missing:
            return _regression_result(
                [],
                n=len(prepared),
                reference_category=None,
                error="missing regression covariates: " + ", ".join(missing),
                converged=False,
            )
        try:
            success = float(_outcome(row))
        except ValueError as exc:
            return _regression_result(
                [],
                n=len(prepared),
                reference_category=None,
                error=str(exc),
                converged=False,
            )
        category_text = str(category).strip()
        if not category_text:
            return _regression_result(
                [],
                n=len(prepared),
                reference_category=None,
                error="category must not be blank",
                converged=False,
            )
        assert calls is not None
        assert recall is not None
        assert hops is not None
        assert tokens is not None
        assert distractors is not None
        if not 0 <= recall <= 1:
            return _regression_result(
                [],
                n=len(prepared),
                reference_category=None,
                error="initial recall must be between zero and one",
                converged=False,
            )
        prepared.append(
            {
                "task_id": task_id,
                "success": success,
                "calls": calls,
                "calls_squared": calls * calls,
                "initial_recall": recall,
                "hops": hops,
                "tokens": tokens,
                "distractors": distractors,
                "category": category_text,
            }
        )

    categories = sorted({str(row["category"]) for row in prepared})
    reference_category = categories[0]
    terms = [
        "intercept",
        "calls",
        "calls_squared",
        "initial_recall",
        "hops",
        "tokens",
        "distractors",
    ] + [f"category_{category}" for category in categories[1:]]
    if len({row["success"] for row in prepared}) < 2:
        return _regression_result(
            [],
            n=len(prepared),
            reference_category=reference_category,
            error="binomial outcome requires both success classes",
            converged=False,
        )
    if len(prepared) <= len(terms):
        return _regression_result(
            [],
            n=len(prepared),
            reference_category=reference_category,
            error="insufficient observations for regression design",
            converged=False,
        )

    design_rows: list[dict[str, float]] = []
    for row in prepared:
        design_row = {
            "intercept": 1.0,
            "calls": float(row["calls"]),
            "calls_squared": float(row["calls_squared"]),
            "initial_recall": float(row["initial_recall"]),
            "hops": float(row["hops"]),
            "tokens": float(row["tokens"]),
            "distractors": float(row["distractors"]),
        }
        for category in categories[1:]:
            design_row[f"category_{category}"] = float(row["category"] == category)
        design_rows.append(design_row)
    design = pd.DataFrame(design_rows, columns=terms)
    outcome = np.asarray([float(row["success"]) for row in prepared], dtype=float)

    try:
        model = sm.GLM(outcome, design, family=sm.families.Binomial())
        fitted = model.fit(
            cov_type="HC3",
            maxiter=100,
            disp=0,
        )
        converged = bool(getattr(fitted, "converged", False))
        if not converged:
            return _regression_result(
                [],
                n=len(prepared),
                reference_category=reference_category,
                error="binomial regression did not converge",
                converged=False,
            )
        coefficients = np.asarray(fitted.params, dtype=float)
        standard_errors = np.asarray(fitted.bse, dtype=float)
        p_values = np.asarray(fitted.pvalues, dtype=float)
        confidence = np.asarray(
            fitted.conf_int(alpha=1 - confidence_level),
            dtype=float,
        )
        rows_out: list[dict[str, object]] = []
        for index, term in enumerate(terms):
            coefficient = float(coefficients[index])
            std_error = float(standard_errors[index])
            p_value = float(p_values[index])
            ci_lower = float(confidence[index, 0])
            ci_upper = float(confidence[index, 1])
            if not all(
                isfinite(value) for value in (coefficient, std_error, p_value, ci_lower, ci_upper)
            ):
                raise ValueError(f"non-finite estimate for {term}")
            rows_out.append(
                {
                    "term": term,
                    "coefficient": coefficient,
                    "std_error": std_error,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": p_value,
                    "converged": True,
                    "error": None,
                    "associative": True,
                    "covariance_type": "HC3",
                    "n": len(prepared),
                    "reference_category": reference_category,
                    "status": "coefficient",
                }
            )
        rows_out.append(
            {
                "term": "__status__",
                "coefficient": np.nan,
                "std_error": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "converged": True,
                "error": None,
                "associative": True,
                "covariance_type": "HC3",
                "n": len(prepared),
                "reference_category": reference_category,
                "status": "converged",
            }
        )
        return _regression_result(
            rows_out,
            n=len(prepared),
            reference_category=reference_category,
            error=None,
            converged=True,
        )
    except (
        FloatingPointError,
        np.linalg.LinAlgError,
        PerfectSeparationError,
        ValueError,
        RuntimeError,
    ) as exc:
        return _regression_result(
            [],
            n=len(prepared),
            reference_category=reference_category,
            error=f"{type(exc).__name__}: {exc}",
            converged=False,
        )


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "PairedStats",
    "exact_mcnemar",
    "paired_summary",
    "stratified_summaries",
    "within_vfs_regression",
]
