"""laya-evals: cheap LLM-as-judge for CI plus a calibration auditor: score eval sets with a local System 1 decision model and audit confidence before automating on it."""

from laya_evals.agreement import (
    JudgeComparison,
    cohens_kappa,
    judge_comparison,
    percent_agreement,
    quadratic_weighted_kappa,
)
from laya_evals.calibration import (
    CoveragePoint,
    ReliabilityBin,
    brier_score,
    coverage_accuracy_curve,
    ece,
    reliability_bins,
)
from laya_evals.baseline import Metric, MetricResult, RegressionCheck, check_regression
from laya_evals.judge import Judge, Judgment, confidence_outcome_pairs
from laya_evals.threshold import ThresholdAdvice, advise_thresholds, recommended_threshold

__version__ = "0.1.0"

__all__ = [
    "CoveragePoint",
    "Judge",
    "JudgeComparison",
    "Judgment",
    "Metric",
    "MetricResult",
    "RegressionCheck",
    "ReliabilityBin",
    "ThresholdAdvice",
    "advise_thresholds",
    "brier_score",
    "check_regression",
    "cohens_kappa",
    "confidence_outcome_pairs",
    "coverage_accuracy_curve",
    "ece",
    "judge_comparison",
    "percent_agreement",
    "quadratic_weighted_kappa",
    "recommended_threshold",
    "reliability_bins",
    "__version__",
]
