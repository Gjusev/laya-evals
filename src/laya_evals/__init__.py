"""laya-evals: cheap LLM-as-judge for CI plus a calibration auditor: score eval sets with a local System 1 decision model and audit confidence before automating on it."""

from laya_evals.calibration import (
    CoveragePoint,
    ReliabilityBin,
    brier_score,
    coverage_accuracy_curve,
    ece,
    reliability_bins,
)
from laya_evals.judge import Judge, Judgment, confidence_outcome_pairs

__version__ = "0.1.0"

__all__ = [
    "CoveragePoint",
    "Judge",
    "Judgment",
    "ReliabilityBin",
    "brier_score",
    "confidence_outcome_pairs",
    "coverage_accuracy_curve",
    "ece",
    "reliability_bins",
    "__version__",
]
