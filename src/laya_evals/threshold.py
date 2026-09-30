"""Threshold advisor: min_confidence per question shape at a target accuracy.

Confidence thresholds do not transfer across question shapes — the same
numeric confidence means different things for a 2-option and a 20-option
question (upstream issue #394). The advisor groups (confidence, outcome)
pairs by whatever shape key the caller uses — option count, question type,
language — and recommends, per group, the lowest confidence threshold whose
kept examples still meet the target accuracy, maximizing coverage.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Optional

from laya_evals.calibration import coverage_accuracy_curve

__all__ = ["ThresholdAdvice", "advise_thresholds", "recommended_threshold"]


class ThresholdAdvice(NamedTuple):
    """Recommended confidence gate for one group at one target accuracy.

    ``threshold`` is the lowest observed confidence whose kept examples
    (confidence >= threshold) meet ``target_accuracy`` with the highest
    coverage among qualifying thresholds. ``achievable`` is False when no
    threshold meets the target — then ``threshold`` is None and
    ``accuracy``/``coverage`` report the best available point on the curve.
    """

    threshold: Optional[float]
    coverage: float
    accuracy: float
    target_accuracy: float
    achievable: bool
    n: int


def recommended_threshold(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    target_accuracy: float,
) -> ThresholdAdvice:
    """Lowest confidence gate meeting ``target_accuracy``, or the best available."""
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError(
            f"target_accuracy must be in (0, 1], got {target_accuracy!r}"
        )
    curve = coverage_accuracy_curve(confidences, outcomes)  # validates the pairs
    n = sum(1 for _ in confidences)

    best = None  # highest-coverage curve point meeting the target
    for point in curve:  # ordered by increasing coverage
        if point.accuracy >= target_accuracy:
            best = point
    if best is not None:
        return ThresholdAdvice(
            threshold=best.threshold,
            coverage=best.coverage,
            accuracy=best.accuracy,
            target_accuracy=target_accuracy,
            achievable=True,
            n=n,
        )
    # No gate meets the target: report the most accurate point honestly,
    # preferring the highest coverage when accuracies tie
    top = max(curve, key=lambda point: (point.accuracy, point.coverage))
    return ThresholdAdvice(
        threshold=None,
        coverage=top.coverage,
        accuracy=top.accuracy,
        target_accuracy=target_accuracy,
        achievable=False,
        n=n,
    )


def advise_thresholds(
    groups: Mapping[str, tuple[Sequence[float], Sequence[bool]]],
    target_accuracy: float,
) -> dict[str, ThresholdAdvice]:
    """Recommend a gate per group; group keys are the caller's shape labels."""
    if not groups:
        raise ValueError("at least one group is required")
    return {
        key: recommended_threshold(confidences, outcomes, target_accuracy)
        for key, (confidences, outcomes) in groups.items()
    }
