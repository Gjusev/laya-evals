"""Run the laya judge on the public SST-2 dev set and record the measurement.

Opt-in script: it downloads the laya checkpoint on first use and the public
Kaggle dataset (no token needed for public datasets). With --llm-judge it
also runs a reference LLM judge (ZAI GLM via the Z_AI_API_KEY environment
variable) over the same sentences and fills the agreement/accuracy/cost
fields; without it those fields stay null/TODO(measure).

Usage:
    uv run --extra compare python scripts/sst2_judge_comparison.py [--limit N] [--llm-judge]
"""

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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

ZAI_BASE_URL = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "glm-5.3-flash")
JUDGE_PROMPT = (
    "You are a strict evaluator. Classify the sentiment of the sentence.\n"
    "Answer with exactly one word: positive or negative.\n\n"
    "Sentence: {sentence}"
)


def _zai_complete(sentence: str) -> dict:
    """One ZAI chat call, temperature 0. glm-5.3 models always think; the
    verdict still lands in content. Returns decision + token usage."""
    payload = json.dumps({
        "model": LLM_JUDGE_MODEL,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(sentence=sentence)}],
        "temperature": 0,
        "max_tokens": 1024,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{ZAI_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {os.environ['ZAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
            content = body["choices"][0]["message"]["content"].lower()
            match = re.search(r"\b(positive|negative)\b", content)
            if match is None:
                raise ValueError(f"no verdict in judge reply: {content[:80]!r}")
            usage = body.get("usage", {})
            return {
                "decision": match.group(1),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as error:  # noqa: BLE001 - retry any transient failure
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ZAI judge failed after retries: {last_error}")


def run_llm_judge(sentences):
    """Judge every sentence with the reference LLM. Returns decisions + usage."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_zai_complete, sentences))
    return (
        [r["decision"] for r in results],
        sum(r["prompt_tokens"] for r in results),
        sum(r["completion_tokens"] for r in results),
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


def run(limit=None, llm_judge=False):
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

    reference = {
        "reference_judge_percent_agreement": None,
        "reference_judge_kappa": None,
        "reference_judge_model": None,
        "reference_judge_accuracy_vs_gold": None,
        "reference_judge_prompt_tokens": None,
        "reference_judge_completion_tokens": None,
        "reference_cost_per_1k_usd": None,
    }
    if llm_judge:
        if "ZAI_API_KEY" not in os.environ:
            raise SystemExit("--llm-judge requires the ZAI_API_KEY environment variable")
        print(f"judging {len(states)} sentences with {LLM_JUDGE_MODEL}...")
        ref_decisions, prompt_tokens, completion_tokens = run_llm_judge(states)
        reference = {
            "reference_judge_percent_agreement": percent_agreement(decisions, ref_decisions),
            "reference_judge_kappa": cohens_kappa(decisions, ref_decisions),
            "reference_judge_model": LLM_JUDGE_MODEL,
            "reference_judge_accuracy_vs_gold": percent_agreement(ref_decisions, golds),
            "reference_judge_prompt_tokens": prompt_tokens,
            "reference_judge_completion_tokens": completion_tokens,
            # Filled below only when ZAI_PRICING_PER_MTOK="in,out" is set to
            # verified published prices; otherwise null, never invented.
            "reference_cost_per_1k_usd": None,
        }
        pricing = os.environ.get("ZAI_PRICING_PER_MTOK")  # "in_usd,out_usd" per 1M tokens
        if pricing:
            price_in, price_out = (float(x) for x in pricing.split(","))
            n = len(states)
            reference["reference_cost_per_1k_usd"] = round(
                (prompt_tokens * price_in + completion_tokens * price_out)
                / 1_000_000 / n * 1000,
                4,
            )

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
        # The laya side runs on local CPU: a USD figure would need an
        # assumed machine rate, so it stays null rather than invented.
        "laya_cost_per_1k_usd": None,
        **reference,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="judge only the first N items")
    parser.add_argument("--llm-judge", action="store_true",
                        help="also run the reference ZAI GLM judge on the same sentences "
                             "(requires ZAI_API_KEY; fills the agreement/cost fields)")
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
    report = run(limit=args.limit, llm_judge=args.llm_judge)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"n={report['n']}")
    print(f"accuracy_vs_gold={report['accuracy_vs_gold']:.4f}")
    print(f"kappa_vs_gold={report['kappa_vs_gold']:.4f}")
    print(f"ece_15_bins={report['ece_15_bins']:.4f}")
    print(f"brier_score={report['brier_score']:.4f}")
    print(f"decisions_per_second={report['decisions_per_second']:.1f}")
    if report.get("reference_judge_model"):
        print(f"reference_judge={report['reference_judge_model']} "
              f"agreement={report['reference_judge_percent_agreement']:.4f} "
              f"kappa={report['reference_judge_kappa']:.4f} "
              f"accuracy={report['reference_judge_accuracy_vs_gold']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
