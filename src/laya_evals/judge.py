"""Rubric judge: a drop-in cheap replacement for LLM judges on laya.

A ``Judge`` wraps anything with the laya ``Agent`` shape (a
``predict_batch(states, questions, **kwargs)`` returning one result dict
per state), applies rubric questions to a batch of states in one pass and
parses the answers into :class:`Judgment` records carrying the calibrated
``answer_confidence`` — the quantity the calibration core measures. The
module never imports laya, so tests run against deterministic fakes and
production passes a real ``laya.Agent`` (or ``Router``).

Question definitions are plain laya question dicts:

- choice: ``{"type": "choice", "instructions": ..., "criteria": {label: desc}}``
- score:  ``{"type": "score", "instructions": ..., "criteria": [level0, level1, ...]}``
- noul:   ``{"type": "noul", "instructions": ...}``
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple, Optional, Union

__all__ = ["Judge", "Judgment", "confidence_outcome_pairs"]


def _score_level(answer: Mapping, expected_score: float) -> int:
    """The legend level laya itself would report for a score answer.

    laya's ``decide()`` takes the argmax over the emitted probabilities and
    only falls back to rounding the expected score when they are absent.
    The two disagree exactly on spread-out distributions (mean rounds away
    from the mode), and the judge must match laya's own semantics for gold
    comparison. Rounding is Python's half-to-even.
    """
    probabilities = answer.get("probabilities")
    best_index = None
    best_prob = -1.0
    if isinstance(probabilities, Mapping):
        for key, prob in probabilities.items():
            try:
                index = int(key)
                prob = float(prob)
            except (TypeError, ValueError):
                return int(round(expected_score))
            if prob > best_prob:
                best_index, best_prob = index, prob
    if best_index is not None:
        return best_index
    return int(round(expected_score))


class Judgment(NamedTuple):
    """One judged question: the decision plus its calibrated confidence.

    ``value`` is the winning label for a choice question, the expected
    (probability-weighted) level for a score question, or the boolean
    verdict for a noul question (true iff p(true) >= 0.5). ``level`` is
    the argmax over the emitted probabilities for score questions when
    present (matching laya's own ``decide()``), else ``int(round(value))``
    with Python's half-to-even rounding; ``None`` for other types.
    ``answer_confidence`` is laya's calibrated confidence;
    ``low_confidence`` flags judgments the ``min_confidence`` gate would
    not automate on — either the Judge's own gate or one laya applied via
    ``predict(min_confidence=...)``.
    """

    question_id: str
    type: str
    value: Union[str, float, bool]
    level: Optional[int]
    answer_confidence: float
    low_confidence: bool


class Judge:
    """Batch rubric judge over a laya Agent-shaped object."""

    def __init__(self, agent, min_confidence: Optional[float] = None):
        if not callable(getattr(agent, "predict_batch", None)):
            raise TypeError(
                "agent must expose predict_batch(states, questions, **kwargs); "
                f"got {type(agent).__name__}"
            )
        if min_confidence is not None:
            if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
                raise ValueError(
                    f"min_confidence must be a number in [0, 1], got {min_confidence!r}"
                )
            if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
                raise ValueError(
                    f"min_confidence must be a finite number in [0, 1], got {min_confidence!r}"
                )
        self.agent = agent
        self.min_confidence = min_confidence

    def judge_batch(
        self,
        states: Sequence[object],
        questions: Mapping[str, Mapping],
        **predict_kwargs,
    ) -> list[dict[str, Judgment]]:
        """Judge every state against the same rubric questions, batched."""
        if isinstance(states, (str, bytes)):
            raise TypeError(
                "states must be a sequence of states, not a single string; "
                "wrap it in a list, or use judge() for one state"
            )
        if not states:
            return []
        results = self.agent.predict_batch(list(states), questions, **predict_kwargs)
        if not isinstance(results, list) or len(results) != len(states):
            raise ValueError(
                "agent.predict_batch must return one result per state; got "
                f"{len(results) if isinstance(results, list) else type(results).__name__} "
                f"for {len(states)} state(s)"
            )
        return [self._parse_result(result) for result in results]

    def judge(
        self,
        state: object,
        questions: Mapping[str, Mapping],
        **predict_kwargs,
    ) -> dict[str, Judgment]:
        """Judge a single state against the rubric questions."""
        return self.judge_batch([state], questions, **predict_kwargs)[0]

    def _parse_result(self, result) -> dict[str, Judgment]:
        answers = result.get("answers") if isinstance(result, Mapping) else None
        if not isinstance(answers, Mapping):
            raise ValueError(
                "agent result must be a mapping with an 'answers' dict, got "
                f"{type(result).__name__}"
            )
        judgments = {}
        for qid, answer in answers.items():
            judgments[qid] = self._parse_answer(qid, answer)
        return judgments

    def _parse_answer(self, qid: str, answer) -> Judgment:
        if not isinstance(answer, Mapping):
            raise ValueError(f"answer {qid!r}: must be a dict, got {type(answer).__name__}")
        qtype = answer.get("type")
        if qtype == "choice":
            if "choice" not in answer:
                raise ValueError(f"answer {qid!r}: no 'choice' label in the decision")
            value, level = answer["choice"], None
        elif qtype == "score":
            if "score" not in answer:
                raise ValueError(f"answer {qid!r}: no 'score' value in the decision")
            value = float(answer["score"])
            if not math.isfinite(value):
                raise ValueError(f"answer {qid!r}: score must be finite, got {value!r}")
            level = _score_level(answer, value)
        elif qtype == "noul":
            if "noul" not in answer:
                raise ValueError(f"answer {qid!r}: no 'noul' probability in the decision")
            probability = float(answer["noul"])
            if not math.isfinite(probability):
                raise ValueError(
                    f"answer {qid!r}: noul probability must be finite, got {probability!r}"
                )
            value, level = probability >= 0.5, None
        else:
            raise ValueError(
                f"answer {qid!r}: unknown type {qtype!r}; expected 'choice', "
                "'score' or 'noul'"
            )
        try:
            confidence = float(answer["answer_confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"answer {qid!r}: no usable 'answer_confidence' (the calibrated "
                f"quantity laya-evals gates and audits on), got "
                f"{answer.get('answer_confidence')!r}"
            ) from exc
        if not math.isfinite(confidence):
            raise ValueError(
                f"answer {qid!r}: answer_confidence must be finite, got {confidence!r}"
            )
        return Judgment(
            question_id=qid,
            type=qtype,
            value=value,
            level=level,
            answer_confidence=confidence,
            low_confidence=(
                bool(answer.get("low_confidence", False))
                or (self.min_confidence is not None and confidence < self.min_confidence)
            ),
        )


def confidence_outcome_pairs(
    judgments: Iterable[Judgment],
    golds: Iterable[object],
) -> tuple[list[float], list[bool]]:
    """Zip judgments with gold answers into calibration-core input.

    Gold for a choice question is the reference label, for a score question
    the reference legend level (compared against the judgment's rounded
    ``level``), for a noul question the reference boolean. Returns parallel
    ``(confidences, outcomes)`` lists ready for :mod:`laya_evals.calibration`.
    """
    judgments = list(judgments)
    golds = list(golds)
    if len(judgments) != len(golds):
        raise ValueError(
            f"judgments and golds must have the same length, "
            f"got {len(judgments)} and {len(golds)}"
        )
    if not judgments:
        raise ValueError("at least one (judgment, gold) pair is required")
    confidences = []
    outcomes = []
    for judgment, gold in zip(judgments, golds):
        decision = judgment.level if judgment.type == "score" else judgment.value
        confidences.append(judgment.answer_confidence)
        outcomes.append(decision == gold)
    return confidences, outcomes
