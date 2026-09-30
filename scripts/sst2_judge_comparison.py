"""Run the laya judge on the public SST-2 dev set and record the measurement.

Opt-in script: it downloads the laya checkpoint on first use and the public
Kaggle dataset (no token needed for public datasets). The reference LLM
judge is not run here — those fields stay null/TODO(measure) until a real
LLM-judge run exists.

Usage:
    uv run --extra compare python scripts/sst2_judge_comparison.py [--limit N]
"""

import argparse
import json
import time
from pathlib import Path

import kagglehub
import pyarrow.parquet as pq

from laya import Agent

from laya_evals import (
    Judge,
    brier_score,
    cohens_kappa,
    confidence_outcome_pairs,
    coverage_accuracy_curve,
    ece,
    percent_agreement,
    recommended_threshold,
    reliability_bins,
)

ADVISE_TARGET = 0.95  # accuracy target for the recorded threshold advice

DATASET = "kanthetineha/sst2-sentiment-analysis"
LABELS = {1: "positive", 0: "negative"}
QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "Is the sentiment of the sentence positive or negative?",
        "criteria": {
            "positive": "The sentence expresses positive sentiment",
            "negative": "The sentence expresses negative sentiment",
        },
    }
}


def load_dev_split(limit=None):
    """The SST-2 dev split (872 balanced sentences) from the public dataset."""
    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    root = Path(kagglehub.dataset_download(DATASET))
    table = pq.read_table(root / "sst2_sentiment_dataset" / "sst2_valid.parquet")
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no rows loaded from the SST-2 dev split")
    return rows


def run(limit=None):
    rows = load_dev_split(limit)
    states = [row["sentence"] for row in rows]
    golds = [LABELS[int(row["label"])] for row in rows]

    judge = Judge(Agent())
    started = time.perf_counter()
    judged = judge.judge_batch(states, QUESTIONS, batch_size=64)
    wall_seconds = time.perf_counter() - started

    judgments = [item["sentiment"] for item in judged]
    decisions = [j.value for j in judgments]
    confidences, outcomes = confidence_outcome_pairs(judgments, golds)
    advice = recommended_threshold(confidences, outcomes, ADVISE_TARGET)

    return {
        "dataset": f"{DATASET} (SST-2 dev split)",
        "n": len(rows),
        "accuracy_vs_gold": percent_agreement(decisions, golds),
        "kappa_vs_gold": cohens_kappa(decisions, golds),
        "ece_15_bins": ece(confidences, outcomes),
        "brier_score": brier_score(confidences, outcomes),
        "coverage_accuracy_curve": [
            [point.coverage, point.accuracy]
            for point in coverage_accuracy_curve(confidences, outcomes)
        ],
        "reliability_bins_nonempty": [
            [bin_.count, bin_.mean_confidence, bin_.accuracy]
            for bin_ in reliability_bins(confidences, outcomes)
            if bin_.count
        ],
        "wall_seconds": wall_seconds,
        "decisions_per_second": len(rows) / wall_seconds,
        "threshold_advice": {
            "target_accuracy": ADVISE_TARGET,
            "threshold": advice.threshold,
            "coverage": advice.coverage,
            "accuracy": advice.accuracy,
            "achievable": advice.achievable,
        },
        # TODO(measure): fill from a real reference LLM-judge run on the same
        # items; null is honest, a placeholder number is not.
        "reference_judge_percent_agreement": None,
        "reference_judge_kappa": None,
        "laya_cost_per_1k_usd": None,
        "reference_cost_per_1k_usd": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="judge only the first N items")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path; defaults to results/sst2-comparison.json for a full "
        "run and results/sst2-smoke.json when --limit is set, so a smoke run "
        "never overwrites the measured artifact",
    )
    args = parser.parse_args()

    out = args.out or (
        Path("results/sst2-smoke.json") if args.limit else Path("results/sst2-comparison.json")
    )
    report = run(limit=args.limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"n={report['n']}")
    print(f"accuracy_vs_gold={report['accuracy_vs_gold']:.4f}")
    print(f"kappa_vs_gold={report['kappa_vs_gold']:.4f}")
    print(f"ece_15_bins={report['ece_15_bins']:.4f}")
    print(f"brier_score={report['brier_score']:.4f}")
    print(f"decisions_per_second={report['decisions_per_second']:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
