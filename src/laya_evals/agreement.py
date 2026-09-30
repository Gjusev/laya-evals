"""Inter-rater agreement: percent agreement, Cohen's kappa, weighted kappa.

Used to compare the laya judge against an LLM judge (or human gold) on the
same items. Inputs are parallel sequences of decisions — labels for choice
questions, legend levels for score questions, booleans for noul questions.
Nothing here imports laya.
"""

from collections.abc import Sequence
from typing import NamedTuple, Optional

__all__ = [
    "JudgeComparison",
    "cohens_kappa",
    "judge_comparison",
    "percent_agreement",
    "quadratic_weighted_kappa",
]


def _validated_decisions(a: Sequence, b: Sequence) -> list[tuple[object, object]]:
    """Validate the shared input contract and return zipped decision pairs."""
    if len(a) != len(b):
        raise ValueError(
            f"both raters' decisions must have the same length, "
            f"got {len(a)} and {len(b)}"
        )
    if len(a) == 0:
        raise ValueError("at least one pair of decisions is required")
    return list(zip(a, b))


def percent_agreement(a: Sequence, b: Sequence) -> float:
    """Fraction of items on which the two raters gave the same decision."""
    pairs = _validated_decisions(a, b)
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa for two raters over nominal categories.

    kappa = (p_o - p_e) / (1 - p_e) with p_o the observed agreement rate and
    p_e the agreement expected from the raters' marginal distributions.
    1 is perfect agreement, 0 is chance-level, negative is systematic
    disagreement. Undefined when both raters are constant on the same
    single category (p_e = 1), which raises rather than returning a silent
    0 or 1; raters constant on *different* categories yield the formula
    value 0.0.
    """
    pairs = _validated_decisions(a, b)
    n = len(pairs)
    marg_a: dict[object, int] = {}
    marg_b: dict[object, int] = {}
    observed = 0
    for x, y in pairs:
        marg_a[x] = marg_a.get(x, 0) + 1
        marg_b[y] = marg_b.get(y, 0) + 1
        if x == y:
            observed += 1
    p_o = observed / n
    p_e = sum(
        (count_a / n) * (marg_b.get(label, 0) / n)
        for label, count_a in marg_a.items()
    )
    if p_e == 1.0:
        raise ValueError(
            "kappa is undefined when both raters use a single category "
            "(expected agreement is 1); use percent_agreement instead"
        )
    return (p_o - p_e) / (1.0 - p_e)


def quadratic_weighted_kappa(a: Sequence, b: Sequence, levels: Sequence) -> float:
    """Quadratic-weighted kappa for ordinal ratings.

    Weight w_ij = (i - j)^2 / (k - 1)^2 over the positions i, j of the two
    ratings in ``levels`` (the ordered level list, e.g. the score legend
    levels 0..k-1), so being off by one level costs less than being off by
    two. 1 is perfect; 0 is chance-level. Level *order* is what matters,
    never the levels' own numeric values. Undefined — and raised, not
    silently scored — when both raters use a single level, mirroring
    ``cohens_kappa``.
    """
    if len(levels) < 2:
        raise ValueError(f"levels must contain at least two levels, got {list(levels)!r}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"levels must be unique, got {list(levels)!r}")
    pairs = _validated_decisions(a, b)
    index = {level: i for i, level in enumerate(levels)}
    n_levels = len(levels)

    def weight(i: int, j: int) -> float:
        return (i - j) ** 2 / (n_levels - 1) ** 2

    observed = 0.0
    marg_a = [0] * n_levels
    marg_b = [0] * n_levels
    used_levels = set()
    for x, y in pairs:
        if x not in index or y not in index:
            raise ValueError(
                f"decision {x!r}/{y!r} is not in levels {list(levels)!r}"
            )
        used_levels.add(x)
        used_levels.add(y)
        i, j = index[x], index[y]
        marg_a[i] += 1
        marg_b[j] += 1
        observed += weight(i, j)
    if len(used_levels) < 2:
        raise ValueError(
            "weighted kappa is undefined when both raters use a single "
            "level (expected weighted disagreement is 0); use "
            "percent_agreement instead"
        )
    n = len(pairs)
    observed /= n
    expected = sum(
        (marg_a[i] / n) * (marg_b[j] / n) * weight(i, j)
        for i in range(n_levels)
        for j in range(n_levels)
    )
    if expected == 0.0:
        return 1.0  # perfect agreement with diverse levels: nothing to weight
    return 1.0 - observed / expected


class JudgeComparison(NamedTuple):
    """Agreement and cost summary of the laya judge vs a reference judge.

    ``kappa`` is ``None`` when both judges used a single category (where
    kappa is undefined). ``weighted_kappa`` is present only for ordinal
    decisions with ``levels`` supplied. Accuracies are present only when
    gold answers were supplied, and cost-per-1k only when a measured total
    cost was supplied — unmeasured fields stay ``None`` rather than being
    invented (fill them with TODO(measure) results from a real run).
    """

    n: int
    percent_agreement: float
    kappa: Optional[float]
    weighted_kappa: Optional[float]
    laya_accuracy: Optional[float]
    reference_accuracy: Optional[float]
    laya_cost_per_1k: Optional[float]
    reference_cost_per_1k: Optional[float]


def judge_comparison(
    laya_decisions: Sequence,
    reference_decisions: Sequence,
    golds: Optional[Sequence] = None,
    *,
    levels: Optional[Sequence] = None,
    laya_cost: Optional[float] = None,
    reference_cost: Optional[float] = None,
) -> JudgeComparison:
    """Compare the laya judge with a reference judge on the same items.

    ``golds`` (optional) adds each judge's accuracy against gold.
    ``levels`` (optional, ordinal decisions) adds quadratic-weighted kappa.
    ``laya_cost`` / ``reference_cost`` are the measured totals for producing
    these ``n`` decisions and are scaled to cost per 1000 judgments.
    """
    pairs = _validated_decisions(laya_decisions, reference_decisions)
    n = len(pairs)
    if golds is not None and len(golds) != n:
        raise ValueError(
            f"golds must have the same length as the decisions, got "
            f"{len(golds)} and {n}"
        )
    try:
        kappa = cohens_kappa(laya_decisions, reference_decisions)
    except ValueError:
        kappa = None
    if levels is None:
        weighted = None
    else:
        try:
            weighted = quadratic_weighted_kappa(laya_decisions, reference_decisions, levels)
        except ValueError:
            weighted = None
    if golds is None:
        laya_accuracy = reference_accuracy = None
    else:
        laya_accuracy = sum(1 for d, g in zip(laya_decisions, golds) if d == g) / n
        reference_accuracy = sum(1 for d, g in zip(reference_decisions, golds) if d == g) / n

    def per_1k(total: Optional[float]) -> Optional[float]:
        return total / n * 1000 if total is not None else None

    return JudgeComparison(
        n=n,
        percent_agreement=percent_agreement(laya_decisions, reference_decisions),
        kappa=kappa,
        weighted_kappa=weighted,
        laya_accuracy=laya_accuracy,
        reference_accuracy=reference_accuracy,
        laya_cost_per_1k=per_1k(laya_cost),
        reference_cost_per_1k=per_1k(reference_cost),
    )
