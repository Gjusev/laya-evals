"""Calibration core: ECE, Brier score, reliability bins and coverage/accuracy curves.

All metrics operate on parallel sequences of per-example confidence (float in
[0, 1], the model's stated probability that its answer is correct) and outcome
(bool, whether the answer was actually correct). This is the contract laya
decision records map onto; nothing here imports laya, so the math stays
testable without checkpoints.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple, Optional

__all__ = [
    "CoveragePoint",
    "ReliabilityBin",
    "brier_score",
    "coverage_accuracy_curve",
    "ece",
    "reliability_bins",
]


class CoveragePoint(NamedTuple):
    """One point on the coverage/accuracy (selective prediction) curve.

    Keeping every example with confidence >= ``threshold`` retains
    ``coverage`` of the set with the given ``accuracy`` among the kept
    examples. Ties enter together, so there is one point per distinct
    confidence value.
    """

    threshold: float
    coverage: float
    accuracy: float


class ReliabilityBin(NamedTuple):
    """One equal-width confidence bin for a reliability diagram.

    Bins follow the standard convention (Guo et al. 2017, Murphy 1973):
    left-open, right-closed intervals (lower, upper], with the last bin
    closed at 1.0 and 0.0 defensively assigned to the first bin. Empty bins
    carry ``None`` confidence/accuracy.
    """

    lower: float
    upper: float
    count: int
    mean_confidence: Optional[float]
    accuracy: Optional[float]


def _validated_pairs(
    confidences: Sequence[float], outcomes: Sequence[bool]
) -> list[tuple[float, bool]]:
    """Validate the shared input contract and return zipped pairs."""
    if len(confidences) != len(outcomes):
        raise ValueError(
            f"confidences and outcomes must have the same length, "
            f"got {len(confidences)} and {len(outcomes)}"
        )
    if len(confidences) == 0:
        raise ValueError("at least one (confidence, outcome) pair is required")
    pairs = []
    for c, y in zip(confidences, outcomes):
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {c!r}")
        if y != 0 and y != 1:
            # Rejects NaN (never equal), fractions and strings up front:
            # truthiness-coercing a missing label would count it as correct
            raise ValueError(f"outcome must be a bool or 0/1, got {y!r}")
        pairs.append((float(c), bool(y)))
    return pairs


def brier_score(confidences: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error between confidence and binary outcome.

    Brier (1950): mean over examples of (confidence - outcome)^2 with the
    outcome encoded as 1 (correct) or 0 (incorrect). 0 is perfect, 1 is
    confidently wrong about everything.
    """
    pairs = _validated_pairs(confidences, outcomes)
    return sum((c - float(y)) ** 2 for c, y in pairs) / len(pairs)


def _bin_index(confidence: float, n_bins: int) -> int:
    """Index of the left-open, right-closed bin (i/n, (i+1)/n] holding c.

    The 1e-9 tolerance keeps decimal edge values (e.g. 0.3 with n_bins=10,
    whose double representation sits epsilon off the edge) on the edge they
    close, per the Guo et al. convention. The cost is bounded: any value
    within 1e-10 of a bin edge is treated as exactly on it, well below the
    4-decimal quantization laya publishes confidences at. 0.0 falls below
    bin 0 and is clamped into it.
    """
    return max(0, min(n_bins - 1, math.ceil(confidence * n_bins - 1e-9) - 1))


def reliability_bins(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    n_bins: int = 15,
) -> list[ReliabilityBin]:
    """Equal-width reliability bins over (0, 1] for a reliability diagram.

    Bin i covers (i / n_bins, (i + 1) / n_bins], the convention used by Guo
    et al. (2017); the default of 15 bins matches their experiments. Each
    non-empty bin reports its example count, mean confidence and empirical
    accuracy; empty bins report ``None`` for the statistics.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    pairs = _validated_pairs(confidences, outcomes)

    counts = [0] * n_bins
    conf_sums = [0.0] * n_bins
    correct_sums = [0.0] * n_bins
    for c, y in pairs:
        i = _bin_index(c, n_bins)
        counts[i] += 1
        conf_sums[i] += c
        correct_sums[i] += 1.0 if y else 0.0

    return [
        ReliabilityBin(
            i / n_bins,
            (i + 1) / n_bins,
            counts[i],
            conf_sums[i] / counts[i] if counts[i] else None,
            correct_sums[i] / counts[i] if counts[i] else None,
        )
        for i in range(n_bins)
    ]


def ece(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    Guo et al. (2017): sum over non-empty bins of the bin's mass fraction
    times |empirical accuracy - mean confidence|. Empty bins contribute
    nothing. 0 is perfectly calibrated.
    """
    bins = reliability_bins(confidences, outcomes, n_bins=n_bins)
    total = sum(b.count for b in bins)
    return sum(
        b.count / total * abs(b.accuracy - b.mean_confidence)
        for b in bins
        if b.count > 0
    )


def coverage_accuracy_curve(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
) -> list[CoveragePoint]:
    """Coverage/accuracy curve for selective prediction.

    Geifman & El-Yaniv (2017) SoftMax response: sweep the threshold over the
    observed confidences from high to low, keeping every example with
    confidence >= threshold. Returns one point per distinct confidence,
    ordered by increasing coverage; the last point always has coverage 1.0.
    There is deliberately no 0-coverage point (accuracy of an empty kept set
    is undefined).
    """
    pairs = sorted(_validated_pairs(confidences, outcomes), reverse=True)
    total = len(pairs)

    points = []
    kept = 0
    correct = 0
    i = 0
    while i < total:
        threshold = pairs[i][0]
        while i < total and pairs[i][0] == threshold:
            kept += 1
            correct += 1 if pairs[i][1] else 0
            i += 1
        points.append(
            CoveragePoint(threshold, kept / total, correct / kept)
        )
    return points
