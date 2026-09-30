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
- [ ] Public-set judge comparison: measured agreement and cost per 1k judgments vs an LLM judge on a public set (laya side measured vs gold on SST-2, see below; the LLM-judge half is TODO(measure) until a reference run with an API key exists)
- [x] Reproduction pack: re-run public benchmark claims (MASSIVE and XNLI subsets) and publish what reproduces
- [x] Threshold advisor: recommended min_confidence per question shape at a target accuracy, by option count
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

## Measured: laya judge on SST-2 (public set)

One real run of the laya judge on the public SST-2 dev split (872 balanced
sentences, Kaggle dataset `kanthetineha/sst2-sentiment-analysis`), CPU
inference, batches of 64, laya 0.3.21 with the default
`convaiinnovations/laya` checkpoint. Sentiment was asked as a two-option
choice question; gold labels map 1 -> positive, 0 -> negative. Numbers are
from a single run on one machine and are recorded verbatim in
`results/sst2-comparison.json`:

| Metric | Value |
|---|---|
| Accuracy vs gold | 0.8968 |
| Cohen's kappa vs gold | 0.7938 |
| ECE (15 bins) | 0.0253 |
| Brier score | 0.0779 |
| Throughput | 6.4 decisions/s (wall-clock, batched, CPU) |

The reference LLM judge was not run (no API key): its agreement, kappa and
cost-per-1k fields stay null in the results file — TODO(measure), not
invented. The run also surfaces a laya RuntimeWarning that the checkpoint
ships invalid temperature buckets for `choice:11+` questions (11 or more
options); the two-option sentiment question used here is outside that
bucket. Reproduce with:

```bash
uv run --extra compare python scripts/sst2_judge_comparison.py
uv run --extra compare --extra dev pytest -m slow   # 16-item smoke, real checkpoint
```

## What reproduces: laya's public benchmark claims

`scripts/reproduction_pack.py` re-runs the upstream laya README's MASSIVE and
XNLI claims under the upstream benchmark notebook's exact protocol (seed 13,
gold + 19 sampled distractors = 20 options for MASSIVE, first 300 test rows
per language, identical state/instructions/criteria text), with checkpoints
pinned per claim the way the table specifies them. This run: CPU, single
machine; upstream ran one T4 GPU. Recorded verbatim in
`results/reproduction.json`:

| Claim (upstream README) | Claimed | Measured | Verdict |
|---|---|---|---|
| MASSIVE intent, English (English checkpoint) | 0.783 | **0.7833** | reproduces (delta +0.0003) |
| XNLI, English (English checkpoint) | 0.860 | **0.8600** | reproduces (delta 0.0000) |
| MASSIVE intent, non-English (multilingual) | 0.451 | 0.5067 | delta only — claim is a 13-language macro; measured on de/fr/es, which skew easier |
| XNLI, non-English (multilingual) | 0.731 | 0.7856 | delta only — claim is a 14-language macro; measured on de/fr/es, which skew easier |

Verdict rule: |measured − claimed| ≤ 0.05 at the claim's sample size
(300 per language); smaller runs are reported as deltas only, never
verdicts. Both English claims reproduce under an independent harness on
different hardware at the upstream sample size, with deltas of +0.0003
(exactly one item in 300) and 0.0000. The non-English rows are not
verdicts: our subset covers three high-resource languages while the
published numbers macro-average 13-14 languages including much harder
ones; per-language detail for de/fr/es is in the results file (e.g.
MASSIVE intent de 0.4633 / fr 0.5533 / es 0.5033; XNLI de 0.7900 /
fr 0.7667 / es 0.8000, with per-suite ECE and top-1 Brier).

Reproduce with:

```bash
uv run --extra compare python scripts/reproduction_pack.py --per-lang 300
```

## Threshold advisor

Confidence thresholds do not transfer across question shapes — the same
numeric confidence means different things for a 2-option and a 20-option
question (upstream issue #394). `recommended_threshold` finds, per group of
(confidence, outcome) pairs, the lowest confidence gate whose kept examples
still meet a target accuracy; `advise_thresholds` does it per shape key:

```python
from laya_evals import advise_thresholds

advice = advise_thresholds(
    {"options=2": (confs_2, outcomes_2), "options=20": (confs_20, outcomes_20)},
    target_accuracy=0.9,
)
# advice["options=20"].threshold, .coverage, .accuracy, .achievable
```

When no gate can reach the target, `achievable` is False and the best
available point on the coverage/accuracy curve is reported instead of an
invented threshold. Applied to the measured runs above (recorded in their
results files), the shapes need very different gates:

| Measured suite (question shape) | Target | Advised threshold | Coverage |
|---|---|---|---|
| SST-2 (2 options) | 0.95 | 0.8651 | 0.834 |
| XNLI English (3 options) | 0.90 | 0.8074 | 0.893 |
| MASSIVE intent English (20 options) | 0.90 | 0.9944 | 0.773 |

At 20 options, 90% accuracy requires gating at 0.9944 confidence — far
above the 3-option shape's own advised 0.8074, which already meets the
0.90 target at 0.893 coverage. A single global min_confidence cannot serve
both; advise per shape.

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip):

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).
