"""Regression gate: fail the build when accuracy or calibration regresses.

Compares a fresh measurement (JSON report) against a committed baseline at
explicit dotted-path metric specs. Accuracy-like metrics (goal ``max``)
regress when they drop more than the tolerance; calibration metrics
(goal ``min``, e.g. ECE) regress when they rise. Wall-clock and throughput
fields are deliberately not gateable — they drift between runs and hardware
and would make CI flaky.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Optional, Union

__all__ = ["Metric", "MetricResult", "RegressionCheck", "check_regression"]

DEFAULT_TOLERANCE = 0.02

# Wall-clock fields drift between runners and machines; gating them would
# make CI flaky, so the gate refuses them outright.
DRIFTING_FIELDS = frozenset({"wall_seconds", "decisions_per_second"})


class Metric(NamedTuple):
    """One gateable metric: a dotted path into the report JSON and a goal.

    ``goal="max"``: regression when current < baseline - tolerance.
    ``goal="min"``: regression when current > baseline + tolerance.
    """

    path: str
    goal: str = "max"
    tolerance: float = DEFAULT_TOLERANCE


class MetricResult(NamedTuple):
    path: str
    goal: str
    baseline: float
    current: float
    delta: float  # current - baseline
    ok: bool


class RegressionCheck(NamedTuple):
    regressed: bool
    results: list[MetricResult]

    def summary(self) -> str:
        lines = [
            f"{'OK ' if r.ok else 'REGRESS'} {r.path}: "
            f"{r.baseline:.4f} -> {r.current:.4f} (delta {r.delta:+.4f}, goal {r.goal})"
            for r in self.results
        ]
        return "\n".join(lines)


def _resolve(document: Mapping, path: str, source: str) -> float:
    """Resolve a dotted path against keys that may themselves contain dots.

    Report keys like ``"xnli.en"`` are literal keys with a dot in them, so
    each step greedily tries the longest run of remaining path segments as a
    single key before splitting further. When both a dotted key and nested
    keys exist, the dotted key wins.
    """
    parts = path.split(".")
    node: object = document
    i = 0
    while i < len(parts):
        for end in range(len(parts), i, -1):
            candidate = ".".join(parts[i:end])
            if isinstance(node, Mapping) and candidate in node:
                node = node[candidate]
                i = end
                break
        else:
            raise ValueError(f"{source} is missing {path} (failed at {'.'.join(parts[:i + 1])})")
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ValueError(f"{source} metric {path} is not numeric, got {node!r}")
    return float(node)


def check_regression(
    current: Mapping,
    baseline: Mapping,
    metrics: Sequence[Metric],
) -> RegressionCheck:
    """Gate a fresh report against a baseline; any metric failing regresses."""
    if not metrics:
        raise ValueError("at least one metric is required")
    results = []
    for metric in metrics:
        if metric.goal not in ("max", "min"):
            raise ValueError(
                f"metric {metric.path!r}: goal must be 'max' or 'min', "
                f"got {metric.goal!r}"
            )
        if metric.tolerance < 0:
            raise ValueError(
                f"metric {metric.path!r}: tolerance must be >= 0, "
                f"got {metric.tolerance}"
            )
        if metric.path.rsplit(".", 1)[-1] in DRIFTING_FIELDS:
            raise ValueError(
                f"metric {metric.path!r}: wall-clock fields drift between "
                "runs and are never gateable"
            )
        current_value = _resolve(current, metric.path, "current report")
        baseline_value = _resolve(baseline, metric.path, "baseline")
        delta = current_value - baseline_value
        ok = (
            delta >= -metric.tolerance
            if metric.goal == "max"
            else delta <= metric.tolerance
        )
        results.append(
            MetricResult(metric.path, metric.goal, baseline_value, current_value, delta, ok)
        )
    return RegressionCheck(regressed=any(not r.ok for r in results), results=results)
