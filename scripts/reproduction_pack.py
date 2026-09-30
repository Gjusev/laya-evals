"""Reproduction pack: re-run laya's published MASSIVE and XNLI claims on subsets.

Follows the upstream benchmark notebook's exact protocol (seed 13, gold + 19
sampled distractors = 20 options for MASSIVE, first-N test rows per language,
identical state/instructions/criteria text) so differences come from the
model, not the harness. Language lists and per-language sizes match upstream
exactly (CPU here, GPU T4 upstream); every claim is reported with its n and
an explicit verdict, and nothing is extrapolated.

Claims are from the upstream README's benchmark table (17,416 questions, one
T4 GPU): MASSIVE intent English 0.783 (English checkpoint) / 13 other
languages 0.451 (multilingual); XNLI English 0.860 (English checkpoint) /
14 other languages 0.731 (multilingual). The language lists and per-language
sample size match upstream exactly, so every row — including the non-English
macros — is directly comparable and verdict-eligible.

Usage:
    uv run --extra compare python scripts/reproduction_pack.py [--per-lang N]
"""

import argparse
import json
import random
import time
from pathlib import Path

from datasets import load_dataset
from laya import Router

from laya_evals import brier_score, ece, recommended_threshold

ADVISE_TARGET = 0.90  # accuracy target for the recorded threshold advice

# Upstream protocol constants; ground truth cached at
# research/laya_benchmark_colab.ipynb (from NandhaKishorM/laya on GitHub,
# research/scripts/laya_benchmark_colab.ipynb, Apache 2.0)
SEED = 13
N_OPTS = 20
CLAIM_N = 300  # claims are per-language n=300 quantities (upstream PER_LANG)
XNLI_LANGS = ["en", "de", "fr", "es", "ru", "tr", "ar", "hi", "ur", "vi", "th", "el", "bg", "zh", "sw"]
MASSIVE_LANGS = ["en", "de", "fr", "es", "pt", "ru", "tr", "ar", "hi", "ta", "zh-CN", "ja", "ko", "sw"]

ENGLISH = "english"
MULTILINGUAL = "multilingual"

# Published claims (upstream README benchmark table). The non-English claims
# are 13/14-language macro averages over exactly the language lists above,
# so at per-language n=300 all four rows are directly comparable.
CLAIMS = {
    "massive_intent.en": {
        "model": ENGLISH, "claimed": 0.783, "verdict_eligible": True,
    },
    "massive_intent.non_en": {
        "model": MULTILINGUAL, "claimed": 0.451, "verdict_eligible": True,
    },
    "xnli.en": {
        "model": ENGLISH, "claimed": 0.860, "verdict_eligible": True,
    },
    "xnli.non_en": {
        "model": MULTILINGUAL, "claimed": 0.731, "verdict_eligible": True,
    },
}
TOLERANCE = 0.05  # |measured - claimed| within this reproduces

MASSIVE_INSTRUCTIONS = "What is the user asking for in `utterance`?"
XNLI_INSTRUCTIONS = "What is the relationship between `premise` and `hypothesis`?"
NLI_CRIT = {
    "entailment": "the premise implies the hypothesis is true",
    "neutral": "the premise neither implies nor contradicts the hypothesis",
    "contradiction": "the premise implies the hypothesis is false",
}


def build_choice_questions(rng, gold, all_labels, n_opts, instructions, qid="label"):
    """Verbatim port of the upstream sampler: gold + distractors, shuffled."""
    pool = [x for x in all_labels if x != gold]
    keys = [gold] + rng.sample(pool, min(n_opts - 1, len(pool)))
    rng.shuffle(keys)
    crit = {k: k.replace("_", " ").replace(".", ": ") for k in keys}
    return {qid: {"type": "choice", "instructions": instructions, "criteria": crit}}, keys.index(gold)


def massive_intent_suite(lang, per_lang):
    """Upstream MASSIVE intent suite for one language, first-N test rows.

    Each row gets its own 20-option draw, so questions differ per case.
    """
    if per_lang < 1:
        raise ValueError(f"per-lang must be >= 1, got {per_lang}")
    dataset = load_dataset("mteb/amazon_massive_intent", lang, split="test")
    labels = sorted(set(dataset["label_text"]))
    rng = random.Random(SEED)
    cases, golds = [], []
    for row in list(dataset)[:per_lang]:
        questions, gold_index = build_choice_questions(
            rng, row["label_text"], labels, N_OPTS, MASSIVE_INSTRUCTIONS
        )
        cases.append(({"utterance": row["text"]}, questions))
        golds.append(list(questions["label"]["criteria"])[gold_index])
    return cases, golds


def xnli_suite(lang, per_lang):
    """Upstream XNLI suite for one language, first-N test rows."""
    if per_lang < 1:
        raise ValueError(f"per-lang must be >= 1, got {per_lang}")
    dataset = load_dataset("facebook/xnli", lang, split="test")
    cases, golds = [], []
    for row in list(dataset)[:per_lang]:
        questions = {
            "relation": {
                "type": "choice",
                "instructions": XNLI_INSTRUCTIONS,
                "criteria": dict(NLI_CRIT),
            }
        }
        cases.append(
            ({"premise": row["premise"], "hypothesis": row["hypothesis"]}, questions)
        )
        golds.append(list(NLI_CRIT)[int(row["label"])])
    return cases, golds


def run_suite(router, cases, golds, model):
    """Run one suite on one pinned checkpoint; returns measured metrics."""
    requests = [
        {"state": state, "questions": questions, "model": model}
        for state, questions in cases
    ]
    started = time.perf_counter()
    results = router.predict_batch(requests, batch_size=32)
    wall_seconds = time.perf_counter() - started

    confidences, outcomes = [], []
    for result, gold in zip(results, golds):
        answer = result["answers"]["label" if "label" in result["answers"] else "relation"]
        decision = answer["choice"]
        confidences.append(float(answer["answer_confidence"]))
        outcomes.append(decision == gold)
    advice = recommended_threshold(confidences, outcomes, ADVISE_TARGET)
    return {
        "n": len(golds),
        "accuracy": sum(outcomes) / len(outcomes),
        "ece_15_bins": ece(confidences, outcomes),
        # Binary top-1 Brier on the chosen answer's confidence, NOT the
        # notebook benchmark's multiclass Brier summed over all options
        "brier_top1": brier_score(confidences, outcomes),
        "threshold_advice": {
            "target_accuracy": ADVISE_TARGET,
            "threshold": advice.threshold,
            "coverage": advice.coverage,
            "accuracy": advice.accuracy,
            "achievable": advice.achievable,
        },
        "wall_seconds": wall_seconds,
    }


def verdict_for(claim_key, measured, n):
    claim = CLAIMS[claim_key]
    delta = measured - claim["claimed"]
    entry = {
        "claimed": claim["claimed"],
        "measured": round(measured, 4),
        "delta": round(delta, 4),
        "model": claim["model"],
        "n": n,
    }
    if not claim["verdict_eligible"]:
        entry["verdict"] = "delta_only"
        entry["caveat"] = claim["caveat"]
    elif n != CLAIM_N:
        # Verdicts are only meaningful at the claim's sample size; a small-n
        # run is noise (at n=4 the accuracy grain alone is 0.25)
        entry["verdict"] = "delta_only"
        entry["caveat"] = (
            f"claim is an n={CLAIM_N} per-language quantity; measured at n={n}"
        )
    else:
        entry["verdict"] = (
            "reproduces" if abs(delta) <= TOLERANCE else "does not reproduce"
        )
    return entry


def run(per_lang):
    router = Router()
    suites = {}
    for family, builder, langs in (
        ("massive_intent", massive_intent_suite, MASSIVE_LANGS),
        ("xnli", xnli_suite, XNLI_LANGS),
    ):
        for lang in langs:
            claim_key = f"{family}.{lang if lang == 'en' else 'non_en'}"
            model = CLAIMS[claim_key]["model"]
            cases, golds = builder(lang, per_lang)
            print(f"running {family}.{lang} on {model} (n={len(golds)})...")
            measured = run_suite(router, cases, golds, model)
            suites[f"{family}.{lang}"] = measured
            print(
                f"  accuracy={measured['accuracy']:.4f} "
                f"ece={measured['ece_15_bins']:.4f} "
                f"({measured['wall_seconds']:.0f}s)"
            )

    claims = {}
    for family in ("massive_intent", "xnli"):
        claims[f"{family}.en"] = verdict_for(
            f"{family}.en", suites[f"{family}.en"]["accuracy"], per_lang
        )
        non_en = [suites[f"{family}.{lang}"] for lang in
                  [l for l in (MASSIVE_LANGS if family == "massive_intent" else XNLI_LANGS)
                   if l != "en"]]
        # Unweighted macro over languages, mirroring the notebook's np.mean
        macro = sum(s["accuracy"] for s in non_en) / len(non_en)
        claims[f"{family}.non_en"] = verdict_for(
            f"{family}.non_en", macro, per_lang
        )

    return {
        "meta": {
            "protocol": "upstream laya_benchmark_colab.ipynb, seed 13, "
            f"{N_OPTS} options for MASSIVE, first-N test rows per language",
            "per_lang": per_lang,
            "languages": sorted(set(MASSIVE_LANGS) | set(XNLI_LANGS)),
            "tolerance": TOLERANCE,
            "hardware_note": "CPU, single machine; upstream ran one T4 GPU",
        },
        "claims": claims,
        "suites": suites,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-lang", type=int, default=300,
                        help="rows per language (upstream uses 300)")
    parser.add_argument("--out", type=Path, default=Path("results/reproduction.json"))
    args = parser.parse_args()
    if args.per_lang < 1:
        parser.error("--per-lang must be >= 1")

    report = run(args.per_lang)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== What reproduces ===")
    for key, entry in report["claims"].items():
        line = (f"{key}: claimed {entry['claimed']:.3f} "
                f"measured {entry['measured']:.4f} ({entry['verdict']})")
        if "caveat" in entry:
            line += f" -- {entry['caveat']}"
        print(line)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
