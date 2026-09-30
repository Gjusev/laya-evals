# laya-evals

> Cheap LLM-as-judge for CI plus a calibration auditor: score eval sets with a local System 1 decision model and audit confidence before automating on it.

Status: early development. Built on [laya](https://github.com/NandhaKishorM/laya),
the open-source System 1 decision engine (Apache 2.0).

## Why

- The gaps are filed as open issues in the upstream repo: confidence thresholds do not transfer across option counts (#394), the published ECE columns no longer reproduce (#208), and independent evals had to be hand-rolled (#555, #450).
- In one large independent eval, answers with confidence >= 0.9 were only 72.2% accurate: automating on confidence without auditing calibration is the classic silent failure.
- Unofficial community tool, not the laya-evals CLI shipped inside the upstream package.
- Every project in this portfolio needs honest evals; this one is that capability, productized.

## Roadmap

- [x] Calibration core: ECE, Brier, reliability bins and coverage/accuracy curves implemented from scratch, tested against synthetic distributions with known values
- [x] Judge: rubric-based score and choice questions as a drop-in cheap replacement for LLM judges, batched
- [x] Judge comparison tooling: percent agreement, Cohen's kappa, quadratic-weighted kappa and cost-per-1k scaling against a reference judge
- [ ] Public-set judge comparison: measured agreement and cost per 1k judgments vs an LLM judge on a public set (TODO(measure))
- [ ] Reproduction pack: re-run public benchmark claims (MASSIVE and XNLI subsets) and publish what reproduces
- [ ] Threshold advisor: recommended min_confidence per question shape at a target accuracy, by option count
- [ ] GitHub Action: fail the build when accuracy or calibration regresses

## Using the calibration core

The metrics operate on per-example pairs of confidence (float in `[0, 1]`,
the model's stated probability that its answer is correct — for laya
decisions, `answer_confidence`) and outcome (whether the answer was actually
correct):

```python
from laya_evals import brier_score, coverage_accuracy_curve, ece, reliability_bins

confidences = [0.9, 0.9, 0.9, 0.2]  # answer_confidence per question
outcomes = [True, True, True, False]  # predicted answer == reference answer

ece(confidences, outcomes)  # expected calibration error, 15 equal-width bins
brier_score(confidences, outcomes)  # mean squared confidence error
reliability_bins(confidences, outcomes)  # per-bin confidence/accuracy/count
coverage_accuracy_curve(confidences, outcomes)  # accuracy as coverage shrinks
```

Bins follow the standard convention (Guo et al. 2017): equal-width, left-open
right-closed `(lo, hi]`, 15 by default. Unit tests never touch a laya
checkpoint; tests that do are marked `slow` and skipped by default
(`pytest -m slow` to run them).

## Using the judge

`Judge` wraps a laya `Agent` (or anything shaped like it) and applies rubric
questions to a whole batch of states in one pass. All three laya question
types work — `choice` (pick a label), `score` (rate on ordered levels) and
`noul` (boolean verdict, true iff p(true) >= 0.5). Judgments carry the
calibrated `answer_confidence` and an optional `min_confidence` gate:

```python
from laya_evals import Judge, brier_score, confidence_outcome_pairs, ece

judge = Judge(laya_agent, min_confidence=0.8)
questions = {
    "relevance": {
        "type": "choice",
        "instructions": "Is the answer relevant to the question?",
        "criteria": {"relevant": "It addresses the question",
                     "irrelevant": "It does not"},
    },
    "quality": {
        "type": "score",
        "instructions": "Rate the answer quality",
        "criteria": ["bad", "weak", "ok", "good"],
    },
    "has_answer": {
        "type": "noul",
        "instructions": "Does the state contain an answer at all?",
    },
}
judged = judge.judge_batch(model_outputs, questions)  # one pass, batched

# Judge, then audit: gold labels turn judgments into calibration inputs
relevance = [j["relevance"] for j in judged]
confidences, outcomes = confidence_outcome_pairs(relevance, gold_relevance)
ece(confidences, outcomes)
brier_score(confidences, outcomes)
```

For score questions the judgment's `level` is the argmax over the emitted
probabilities — the same level laya's own `decide()` reports — not the
rounded expected score, so gold comparisons match laya's semantics on
spread-out distributions too.

## Comparing judges

`judge_comparison` summarizes the laya judge against a reference judge (an
LLM judge or human gold labels) on the same items: percent agreement,
Cohen's kappa, quadratic-weighted kappa for ordinal score levels, each
judge's accuracy against gold, and measured costs scaled to per-1k
judgments:

```python
from laya_evals import judge_comparison

comparison = judge_comparison(
    laya_decisions,       # e.g. [j.level for j in quality_judgments]
    reference_decisions,  # the LLM judge's decisions on the same items
    golds=gold_levels,
    levels=[0, 1, 2, 3],
    laya_cost=measured_laya_cost,          # TODO(measure) on a public set
    reference_cost=measured_reference_cost,  # TODO(measure)
)
# percent_agreement, kappa, weighted_kappa, laya_accuracy,
# reference_accuracy, laya_cost_per_1k, reference_cost_per_1k
```

Fields whose inputs were not supplied stay `None` rather than being
invented; the public-set numbers themselves are TODO(measure) until a real
run is recorded.

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip):

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).
