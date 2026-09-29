"""Tests for the judge: rubric questions over a fake laya-shaped agent.

The fake returns hand-built answer dicts in the exact shape laya 0.3.21
emits (agent.py _decode_answers), so no checkpoint is ever touched.
"""

import pytest

from laya_evals.calibration import brier_score, ece
from laya_evals.judge import Judge, Judgment, confidence_outcome_pairs

# One laya predict_batch result per state, matching the verified shapes:
# choice -> {"type","choice","probabilities","confidence","answer_confidence","action"}
# score  -> {"type","score","legend","probabilities","confidence","answer_confidence","action"}
# noul   -> {"type","noul","confidence","answer_confidence","action"}
FAKE_RESULTS = [
    {
        "model": "laya-rl-agent",
        "answers": {
            "relevance": {
                "type": "choice",
                "choice": "relevant",
                "probabilities": {"relevant": 0.87, "irrelevant": 0.13},
                "confidence": 0.75,
                "answer_confidence": 0.87,
                "action": {"act_probability": 0.5},
            },
            "quality": {
                "type": "score",
                # Honest laya output: score is the expectation of these
                # probabilities: 1*0.1 + 2*0.55 + 3*0.3 = 2.1
                "score": 2.1,
                "legend": {"0": "bad", "1": "weak", "2": "ok", "3": "good"},
                "probabilities": {"0": 0.05, "1": 0.1, "2": 0.55, "3": 0.3},
                "confidence": 0.62,
                "answer_confidence": 0.55,
                "action": {"act_probability": 0.5},
            },
            "has_answer": {
                "type": "noul",
                "noul": 0.91,
                "confidence": 0.91,
                "answer_confidence": 0.91,
                "action": {"act_probability": 0.5},
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    },
    {
        "model": "laya-rl-agent",
        "answers": {
            "relevance": {
                "type": "choice",
                "choice": "irrelevant",
                "probabilities": {"relevant": 0.2, "irrelevant": 0.8},
                "confidence": 0.68,
                "answer_confidence": 0.8,
                "action": {"act_probability": 0.5},
            },
            "quality": {
                "type": "score",
                # Honest laya output: 1*0.6 = 0.6
                "score": 0.6,
                "legend": {"0": "bad", "1": "weak", "2": "ok", "3": "good"},
                "probabilities": {"0": 0.4, "1": 0.6, "2": 0.0, "3": 0.0},
                "confidence": 0.3,
                "answer_confidence": 0.6,
                "action": {"act_probability": 0.5},
            },
            "has_answer": {
                "type": "noul",
                "noul": 0.3,
                "confidence": 0.7,
                "answer_confidence": 0.7,
                "action": {"act_probability": 0.5},
            },
        },
        "usage": {"input_tokens": 90, "output_tokens": 0},
    },
]

QUESTIONS = {
    "relevance": {
        "type": "choice",
        "instructions": "Is the answer relevant to the question?",
        "criteria": {"relevant": "It addresses the question", "irrelevant": "It does not"},
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


class FakeAgent:
    """Deterministic stand-in for laya.Agent: records calls, replays results."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((list(states), questions, kwargs))
        # A real laya Agent returns exactly one result per state
        return self.results[: len(states)]


@pytest.fixture
def agent():
    return FakeAgent([dict(r, answers=dict(r["answers"])) for r in FAKE_RESULTS])


class TestJudgeBatch:
    def test_parses_all_three_question_types(self, agent):
        judged = Judge(agent).judge_batch(
            ["output one", "output two"], QUESTIONS
        )
        assert len(judged) == 2

        r0, r1 = judged
        assert r0["relevance"].value == "relevant"
        assert r0["relevance"].type == "choice"
        assert r0["relevance"].answer_confidence == pytest.approx(0.87)
        assert r0["relevance"].level is None

        # score: expected-value float plus the argmax legend level
        assert r0["quality"].value == pytest.approx(2.1)
        assert r0["quality"].level == 2

        # noul: the boolean verdict, true iff p(true) >= 0.5
        assert r0["has_answer"].value is True
        assert r1["has_answer"].value is False

        assert r1["relevance"].value == "irrelevant"
        assert r1["quality"].level == 1

    def test_score_level_matches_laya_decide_argmax_not_mean(self):
        # laya's decide() reports the argmax over the emitted probabilities,
        # not the rounded mean: here E = 1*0.35 + 2*0.34 + 3*0.31 = 1.96, so
        # rounding the mean gives 2, but laya (and the judge) report level 1
        agent = FakeAgent(
            [
                {
                    "answers": {
                        "q": {
                            "type": "score",
                            "score": 1.96,
                            "probabilities": {"0": 0.0, "1": 0.35, "2": 0.34, "3": 0.31},
                            "answer_confidence": 0.35,
                        }
                    }
                }
            ]
        )
        judgment = Judge(agent).judge("s", {"q": QUESTIONS["quality"]})["q"]
        assert judgment.level == 1
        assert judgment.value == pytest.approx(1.96)

    def test_score_level_falls_back_to_rounding_without_probabilities(self):
        # Without probabilities the rounded expected score is the only signal;
        # Python half-to-even: round(2.5) == 2, round(1.5) == 2
        agent = FakeAgent(
            [{"answers": {"a": {"type": "score", "score": 2.5, "answer_confidence": 0.5},
                          "b": {"type": "score", "score": 1.5, "answer_confidence": 0.5}}}]
        )
        judged = Judge(agent).judge("s", QUESTIONS)
        assert judged["a"].level == 2
        assert judged["b"].level == 2

    def test_noul_boundary_prob_is_true(self):
        agent = FakeAgent(
            [{"answers": {"q": {"type": "noul", "noul": 0.5, "answer_confidence": 0.5}}}]
        )
        assert Judge(agent).judge("s", QUESTIONS)["q"].value is True

    def test_empty_answers_returns_empty_judgments(self):
        agent = FakeAgent(
            [{"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0}}]
        )
        assert Judge(agent).judge("s", QUESTIONS) == {}

    def test_score_fixtures_are_honest_laya_outputs(self, agent):
        # laya always emits score == sum(level * p_level): pin the fixtures to
        # that invariant so they can never silently drift into impossibility
        for result in FAKE_RESULTS:
            for answer in result["answers"].values():
                if answer["type"] != "score":
                    continue
                expectation = sum(i * p for i, p in enumerate(answer["probabilities"].values()))
                assert answer["score"] == pytest.approx(expectation)
                assert max(answer["probabilities"].values()) == pytest.approx(
                    answer["answer_confidence"]
                )

    def test_forwards_states_and_questions_to_the_agent(self, agent):
        Judge(agent).judge_batch(["a", "b"], QUESTIONS)
        states, questions, kwargs = agent.calls[0]
        assert states == ["a", "b"]
        assert questions is QUESTIONS
        assert kwargs == {}

    def test_predict_kwargs_are_forwarded(self, agent):
        Judge(agent).judge_batch(["a"], QUESTIONS, lang="en", batch_size=4)
        _, _, kwargs = agent.calls[0]
        assert kwargs == {"lang": "en", "batch_size": 4}

    def test_single_state_shortcut_matches_batch(self, agent):
        judge = Judge(agent)
        assert judge.judge("only state", QUESTIONS) == judge.judge_batch(["only state"], QUESTIONS)[0]

    def test_empty_state_list_returns_empty(self, agent):
        assert Judge(agent).judge_batch([], QUESTIONS) == []

    def test_bare_string_states_raise(self):
        # A single string is the one-sequence-of-states mistake everyone
        # makes: without a guard it is judged one character at a time
        agent = FakeAgent(FAKE_RESULTS[:1])
        with pytest.raises(TypeError, match="single string"):
            Judge(agent).judge_batch("one state", QUESTIONS)

    def test_kwargs_forwarding_recorded_on_judge_shortcut(self, agent):
        Judge(agent).judge("s", QUESTIONS, lang="en")
        states, questions, kwargs = agent.calls[-1]
        assert kwargs == {"lang": "en"}


class TestLowConfidenceChannels:
    def test_laya_injected_flag_is_honored(self):
        # laya's predict(min_confidence=...) flags answers in the result dict;
        # the Judgment must not silently discard that abstention signal
        agent = FakeAgent(
            [
                {
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "yes",
                            "answer_confidence": 0.6,
                            "low_confidence": True,
                        }
                    }
                }
            ]
        )
        judgment = Judge(agent).judge("s", QUESTIONS)["q"]
        assert judgment.low_confidence is True

    def test_constructor_gate_and_laya_flag_combine(self):
        agent = FakeAgent(
            [
                {
                    "answers": {
                        "low": {"type": "noul", "noul": 0.9, "answer_confidence": 0.9},
                        "flagged": {
                            "type": "choice",
                            "choice": "yes",
                            "answer_confidence": 0.99,
                            "low_confidence": True,
                        },
                    }
                }
            ]
        )
        judged = Judge(agent, min_confidence=0.95).judge("s", QUESTIONS)
        assert judged["low"].low_confidence is True  # constructor gate
        assert judged["flagged"].low_confidence is True  # laya's flag alone


class TestConstructorValidation:
    def test_min_confidence_must_be_in_unit_range(self):
        agent = FakeAgent(FAKE_RESULTS[:1])
        with pytest.raises(ValueError, match="min_confidence"):
            Judge(agent, min_confidence=1.5)
        with pytest.raises(ValueError, match="min_confidence"):
            Judge(agent, min_confidence=-0.1)

    def test_min_confidence_rejects_bool_and_nan_and_string(self):
        agent = FakeAgent(FAKE_RESULTS[:1])
        with pytest.raises(ValueError, match="min_confidence"):
            Judge(agent, min_confidence=True)
        with pytest.raises(ValueError, match="min_confidence"):
            Judge(agent, min_confidence=float("nan"))
        with pytest.raises(ValueError, match="min_confidence"):
            Judge(agent, min_confidence="high")


class TestMinConfidenceGate:
    def test_below_threshold_is_flagged(self, agent):
        judge = Judge(agent, min_confidence=0.6)
        r0, r1 = judge.judge_batch(["a", "b"], QUESTIONS)
        # 0.87, 0.55, 0.91 -> only the quality answer is below the gate
        assert r0["relevance"].low_confidence is False
        assert r0["quality"].low_confidence is True
        assert r0["has_answer"].low_confidence is False
        assert r1["relevance"].low_confidence is False  # 0.8 >= 0.6

    def test_threshold_is_inclusive(self):
        agent = FakeAgent(
            [
                {
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "yes",
                            "confidence": 0.5,
                            "answer_confidence": 0.6,
                        }
                    }
                }
            ]
        )
        judgment = Judge(agent, min_confidence=0.6).judge("s", {"q": QUESTIONS["has_answer"]})["q"]
        assert judgment.answer_confidence == pytest.approx(0.6)
        assert judgment.low_confidence is False  # equal confidence automates

    def test_no_threshold_disables_the_gate(self, agent):
        r0 = Judge(agent).judge_batch(["a"], QUESTIONS)[0]
        assert all(j.low_confidence is False for j in r0.values())


class TestMalformedAgentResults:
    def test_non_mapping_result_raises(self):
        agent = FakeAgent(["not a dict"])
        with pytest.raises(ValueError, match="answers"):
            Judge(agent).judge("s", QUESTIONS)

    def test_missing_answers_key_raises(self):
        agent = FakeAgent([{"model": "laya-rl-agent"}])
        with pytest.raises(ValueError, match="answers"):
            Judge(agent).judge("s", QUESTIONS)

    def test_missing_answer_confidence_raises(self):
        agent = FakeAgent(
            [{"answers": {"q": {"type": "choice", "choice": "yes"}}}]
        )
        with pytest.raises(ValueError, match="answer_confidence"):
            Judge(agent).judge("s", QUESTIONS)

    def test_unknown_answer_type_raises(self):
        agent = FakeAgent(
            [{"answers": {"q": {"type": "exotic", "answer_confidence": 0.9}}}]
        )
        with pytest.raises(ValueError, match="exotic"):
            Judge(agent).judge("s", QUESTIONS)

    def test_missing_decision_field_raises(self):
        # A choice answer without its label must not silently become None
        agent = FakeAgent([{"answers": {"q": {"type": "choice", "answer_confidence": 0.9}}}])
        with pytest.raises(ValueError, match="'choice'"):
            Judge(agent).judge("s", QUESTIONS)

    def test_missing_score_field_raises(self):
        agent = FakeAgent([{"answers": {"q": {"type": "score", "answer_confidence": 0.9}}}])
        with pytest.raises(ValueError, match="'score'"):
            Judge(agent).judge("s", QUESTIONS)

    def test_missing_noul_field_raises(self):
        agent = FakeAgent([{"answers": {"q": {"type": "noul", "answer_confidence": 0.9}}}])
        with pytest.raises(ValueError, match="'noul'"):
            Judge(agent).judge("s", QUESTIONS)

    def test_nan_score_raises_with_question_named(self):
        agent = FakeAgent(
            [{"answers": {"q": {"type": "score", "score": float("nan"), "answer_confidence": 0.9}}}]
        )
        with pytest.raises(ValueError, match="q.*finite"):
            Judge(agent).judge("s", QUESTIONS)

    def test_nan_noul_raises_with_question_named(self):
        agent = FakeAgent(
            [{"answers": {"q": {"type": "noul", "noul": float("nan"), "answer_confidence": 0.9}}}]
        )
        with pytest.raises(ValueError, match="q.*finite"):
            Judge(agent).judge("s", QUESTIONS)

    def test_nan_answer_confidence_raises(self):
        # NaN confidence would silently disable any confidence gate
        agent = FakeAgent(
            [{"answers": {"q": {"type": "choice", "choice": "yes",
                                "answer_confidence": float("nan")}}}]
        )
        with pytest.raises(ValueError, match="answer_confidence"):
            Judge(agent).judge("s", QUESTIONS)

    def test_none_answer_confidence_raises_value_error(self):
        agent = FakeAgent(
            [{"answers": {"q": {"type": "choice", "choice": "yes", "answer_confidence": None}}}]
        )
        with pytest.raises(ValueError, match="answer_confidence"):
            Judge(agent).judge("s", QUESTIONS)

    def test_wrong_result_count_raises(self):
        agent = FakeAgent(FAKE_RESULTS[:1])
        with pytest.raises(ValueError, match="one result per state"):
            Judge(agent).judge_batch(["a", "b"], QUESTIONS)

    def test_non_agent_constructor_raises(self):
        with pytest.raises(TypeError, match="predict_batch"):
            Judge(object())


class TestConfidenceOutcomePairs:
    def test_choice_gold_by_label(self):
        judgments = [
            Judgment("relevance", "choice", "relevant", None, 0.87, False),
            Judgment("relevance", "choice", "irrelevant", None, 0.8, False),
        ]
        confs, outs = confidence_outcome_pairs(judgments, ["relevant", "relevant"])
        assert confs == [0.87, 0.8]
        assert outs == [True, False]

    def test_score_gold_compares_rounded_level_not_float(self):
        # Expected score 2.6 means level 3 (legend index); the gold is an index
        judgments = [
            Judgment("quality", "score", 2.6, 3, 0.55, False),
            Judgment("quality", "score", 1.0, 1, 0.6, False),
        ]
        _, outs = confidence_outcome_pairs(judgments, [3, 2])
        assert outs == [True, False]

    def test_noul_gold_by_boolean(self):
        judgments = [
            Judgment("has_answer", "noul", True, None, 0.91, False),
            Judgment("has_answer", "noul", False, None, 0.7, False),
        ]
        _, outs = confidence_outcome_pairs(judgments, [True, False])
        assert outs == [True, True]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            confidence_outcome_pairs([Judgment("q", "noul", True, None, 0.9, False)], [])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            confidence_outcome_pairs([], [])

    def test_end_to_end_with_calibration_core(self, agent):
        # The portfolio's core promise: judge a batch, audit the confidence.
        # Judged confidences [0.87, 0.8] against outcomes [True, False]:
        # ECE (15 bins): both in distinct bins, gaps |1-0.87| and |0-0.8|
        #   weighted 1/2 each -> 0.065 + 0.4 = 0.465
        # Brier: (0.13^2 + 0.8^2) / 2 = (0.0169 + 0.64) / 2 = 0.32845
        judged = Judge(agent).judge_batch(["a", "b"], QUESTIONS)
        relevance = [j["relevance"] for j in judged]
        confs, outs = confidence_outcome_pairs(relevance, ["relevant", "relevant"])
        assert ece(confs, outs) == pytest.approx(0.465)
        assert brier_score(confs, outs) == pytest.approx(0.32845)
