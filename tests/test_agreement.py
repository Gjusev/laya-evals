"""Tests for agreement metrics: expected values are hand-derived literals."""

import pytest

from laya_evals.agreement import (
    cohens_kappa,
    judge_comparison,
    percent_agreement,
    quadratic_weighted_kappa,
)


class TestPercentAgreement:
    def test_hand_derived_two_of_three(self):
        # [a, a, b] vs [a, b, b]: matches at positions 0 and 2 -> 2/3
        assert percent_agreement(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)

    def test_perfect(self):
        assert percent_agreement([1, 2, 3], [1, 2, 3]) == 1.0

    def test_zero(self):
        assert percent_agreement([1, 2], [2, 1]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            percent_agreement([1], [1, 2])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            percent_agreement([], [])


class TestCohensKappa:
    def test_hand_derived_partial_agreement(self):
        # Hand-derived: A=[1,1,1,2,2,2], B=[1,1,2,2,2,2]
        # p_o = 5/6; p_e = P(A=1)P(B=1) + P(A=2)P(B=2)
        #      = (3/6)(2/6) + (3/6)(4/6) = 1/2
        # kappa = (5/6 - 1/2) / (1 - 1/2) = (1/3) / (1/2) = 2/3
        a = [1, 1, 1, 2, 2, 2]
        b = [1, 1, 2, 2, 2, 2]
        assert cohens_kappa(a, b) == pytest.approx(2 / 3)

    def test_perfect_agreement_is_one(self):
        assert cohens_kappa(["x", "y", "x"], ["x", "y", "x"]) == 1.0

    def test_independent_raters_are_zero(self):
        # p_o = 1/2 and p_e = 1/2 -> kappa = 0
        assert cohens_kappa([1, 2, 1, 2], [1, 1, 2, 2]) == pytest.approx(0.0)

    def test_systematic_disagreement_is_minus_one(self):
        # p_o = 0, p_e = 1/2 -> kappa = -1
        assert cohens_kappa([1, 1, 2, 2], [2, 2, 1, 1]) == pytest.approx(-1.0)

    def test_single_shared_category_raises(self):
        # p_e = 1 makes kappa 0/0: undefined, not silently 0 or 1
        with pytest.raises(ValueError, match="undefined"):
            cohens_kappa([1, 1], [1, 1])

    def test_labels_can_be_anything_hashable(self):
        # Choice judgments carry arbitrary labels
        a = ["relevant", "relevant", "irrelevant"]
        b = ["relevant", "irrelevant", "irrelevant"]
        # p_o = 2/3; p_e = (2/3)(1/3) + (1/3)(2/3) = 4/9
        # kappa = (2/3 - 4/9) / (1 - 4/9) = (2/9) / (5/9) = 2/5
        assert cohens_kappa(a, b) == pytest.approx(2 / 5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa([1], [1, 2])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            cohens_kappa([], [])


class TestQuadraticWeightedKappa:
    LEVELS = [0, 1, 2]

    def test_hand_derived_three_levels(self):
        # Hand-derived, 3 levels, weights w_ij = (i-j)^2 / (k-1)^2:
        # A=[0,1,2], B=[0,2,1] -> pairs (0,0) w=0, (1,2) w=1/4, (2,1) w=1/4
        # observed weighted disagreement = (0 + 1/4 + 1/4)/3 = 1/6
        # both marginals uniform -> expected = sum(w_ij)/9 = 3/9 = 1/3
        # QWK = 1 - (1/6)/(1/3) = 1/2
        assert quadratic_weighted_kappa([0, 1, 2], [0, 2, 1], self.LEVELS) == pytest.approx(0.5)

    def test_hand_derived_near_perfect(self):
        # A=[0,0,1,1,2,2], B=[0,1,1,1,2,2]:
        # observed = ((0,0)0 + (0,1)1/4 + (1,1)0 + (1,1)0 + (2,2)0 + (2,2)0)/6 = 1/24
        # margA = {0:2/6, 1:2/6, 2:2/6}, margB = {0:1/6, 1:3/6, 2:2/6}
        # expected = sum over cells of margA_i*margB_j*w_ij = 10.5/36 = 7/24
        # QWK = 1 - (1/24)/(7/24) = 1 - 1/7 = 6/7
        a = [0, 0, 1, 1, 2, 2]
        b = [0, 1, 1, 1, 2, 2]
        assert quadratic_weighted_kappa(a, b, self.LEVELS) == pytest.approx(6 / 7)

    def test_perfect_agreement_is_one(self):
        assert quadratic_weighted_kappa([0, 2, 1], [0, 2, 1], self.LEVELS) == 1.0

    def test_reduces_to_cohens_kappa_for_two_levels(self):
        # With 2 levels the only off-diagonal weight is 1, so QWK == kappa
        a = [1, 1, 1, 2, 2, 2]
        b = [1, 1, 2, 2, 2, 2]
        assert quadratic_weighted_kappa(a, b, [1, 2]) == pytest.approx(cohens_kappa(a, b))

    def test_respects_level_order_not_value(self):
        # Levels are ordered categories; their numeric values never enter
        assert quadratic_weighted_kappa(
            ["bad", "ok", "good"], ["bad", "good", "ok"], ["bad", "ok", "good"]
        ) == pytest.approx(0.5)

    def test_decision_outside_levels_raises(self):
        with pytest.raises(ValueError, match="not in levels"):
            quadratic_weighted_kappa([0, 5], [0, 1], self.LEVELS)

    def test_single_level_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            quadratic_weighted_kappa([1, 1], [1, 1], [1])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            quadratic_weighted_kappa([0], [0, 1], self.LEVELS)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            quadratic_weighted_kappa([], [], self.LEVELS)

    def test_duplicate_levels_raise(self):
        # A typo'd legend like [0, 1, 1, 2] used to silently renumber the
        # ordinal positions and return a confidently wrong number
        with pytest.raises(ValueError, match="unique"):
            quadratic_weighted_kappa([0, 1, 2], [0, 2, 1], [0, 1, 1, 2])

    def test_single_used_level_raises_like_cohens_kappa(self):
        # Both raters constant on one level: undefined, symmetric with
        # cohens_kappa's single-category raise — not a silent 1.0
        with pytest.raises(ValueError, match="undefined"):
            quadratic_weighted_kappa([1, 1], [1, 1], [1, 2])

    def test_perfect_agreement_with_diverse_levels_is_one(self):
        # All pairs equal but multiple levels in play: expected weighted
        # disagreement is genuinely 0 and the kappa is a real 1.0
        assert quadratic_weighted_kappa([0, 2, 1], [0, 2, 1], self.LEVELS) == 1.0


class TestJudgeComparison:
    def test_full_record_hand_derived(self):
        # laya=[1,1,1,2,2,2] vs reference=[1,1,2,2,2,2]:
        #   percent 5/6, kappa 2/3 (derived in TestCohensKappa)
        # gold=[1,2,1,2,2,2]: laya matches gold 5/6, reference matches 4/6
        # costs: 0.90 for 6 laya judgments -> 150.0 per 1k;
        #        1.20 for 6 reference judgments -> 200.0 per 1k
        comparison = judge_comparison(
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            golds=[1, 2, 1, 2, 2, 2],
            laya_cost=0.90,
            reference_cost=1.20,
        )
        assert comparison.n == 6
        assert comparison.percent_agreement == pytest.approx(5 / 6)
        assert comparison.kappa == pytest.approx(2 / 3)
        assert comparison.weighted_kappa is None  # no levels given
        assert comparison.laya_accuracy == pytest.approx(5 / 6)
        assert comparison.reference_accuracy == pytest.approx(4 / 6)
        assert comparison.laya_cost_per_1k == pytest.approx(150.0)
        assert comparison.reference_cost_per_1k == pytest.approx(200.0)

    def test_levels_enable_weighted_kappa(self):
        comparison = judge_comparison(
            [0, 1, 2], [0, 2, 1], levels=[0, 1, 2]
        )
        assert comparison.weighted_kappa == pytest.approx(0.5)

    def test_defaults_are_none_until_measured(self):
        # Without golds and costs the report stays honest: nothing invented
        comparison = judge_comparison([1, 0], [1, 1])
        assert comparison.laya_accuracy is None
        assert comparison.reference_accuracy is None
        assert comparison.laya_cost_per_1k is None
        assert comparison.reference_cost_per_1k is None

    def test_single_category_kappa_is_none_not_an_error(self):
        # A tolerant report: percent still meaningful, kappa simply undefined
        comparison = judge_comparison([1, 1], [1, 1])
        assert comparison.percent_agreement == 1.0
        assert comparison.kappa is None

    def test_single_level_weighted_kappa_is_none_not_an_error(self):
        # Symmetric: weighted kappa undefined on one used level, no crash,
        # no silent 1.0 in the record
        comparison = judge_comparison([1, 1], [1, 1], levels=[1, 2])
        assert comparison.kappa is None
        assert comparison.weighted_kappa is None

    def test_golds_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            judge_comparison([1], [1], golds=[1, 1])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            judge_comparison([1, 2], [1])
