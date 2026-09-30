"""Tests for the threshold advisor: hand-derived literals throughout."""

import pytest

from laya_evals.threshold import advise_thresholds, recommended_threshold


class TestRecommendedThreshold:
    def test_full_coverage_meets_target(self):
        # Curve: tau .9 -> acc 1.0 cov .25 | .8 -> 1.0 .5 | .7 -> 2/3 .75 | .6 -> .75 1.0
        # Highest coverage with accuracy >= 0.7 is full coverage (acc 3/4),
        # so the gate can stay at the lowest observed confidence
        advice = recommended_threshold([0.9, 0.8, 0.7, 0.6], [True, True, False, True], 0.7)
        assert advice.achievable is True
        assert advice.threshold == pytest.approx(0.6)
        assert advice.coverage == pytest.approx(1.0)
        assert advice.accuracy == pytest.approx(0.75)

    def test_gate_lifts_accuracy_to_target(self):
        # Only the most confident half is 100% right; target 0.9 needs it
        advice = recommended_threshold([0.9, 0.8], [True, False], 0.9)
        assert advice.achievable is True
        assert advice.threshold == pytest.approx(0.9)
        assert advice.coverage == pytest.approx(0.5)
        assert advice.accuracy == pytest.approx(1.0)

    def test_unreachable_target_reports_best_available(self):
        # Tied confidences enter together: no gate can beat 0.5 accuracy
        advice = recommended_threshold([0.9, 0.9], [True, False], 0.8)
        assert advice.achievable is False
        assert advice.threshold is None
        assert advice.coverage == pytest.approx(1.0)
        assert advice.accuracy == pytest.approx(0.5)

    def test_unreachable_tie_prefers_highest_coverage(self):
        # Two gates tie at the max accuracy 0.5: (tau 0.9, cov 0.5) and
        # (tau 0.5, cov 1.0). Discarding half the run buys nothing, so the
        # best available point is the full-coverage one.
        advice = recommended_threshold(
            [0.9, 0.9, 0.5, 0.5], [True, False, True, False], 0.7
        )
        assert advice.achievable is False
        assert advice.threshold is None
        assert advice.coverage == pytest.approx(1.0)
        assert advice.accuracy == pytest.approx(0.5)

    def test_accuracy_exactly_at_target_qualifies(self):
        # Full-coverage accuracy 0.5 == target 0.5 must qualify (>=, not >)
        advice = recommended_threshold([0.9, 0.6], [True, False], 0.5)
        assert advice.achievable is True
        assert advice.threshold == pytest.approx(0.6)
        assert advice.coverage == pytest.approx(1.0)

    def test_perfect_run_needs_no_gate(self):
        advice = recommended_threshold([0.55, 0.95], [True, True], 0.99)
        assert advice.achievable is True
        assert advice.threshold == pytest.approx(0.55)
        assert advice.coverage == pytest.approx(1.0)
        assert advice.accuracy == pytest.approx(1.0)

    def test_target_bounds_validated(self):
        with pytest.raises(ValueError, match="target_accuracy"):
            recommended_threshold([0.5], [True], 0.0)
        with pytest.raises(ValueError, match="target_accuracy"):
            recommended_threshold([0.5], [True], 1.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            recommended_threshold([], [], 0.9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            recommended_threshold([0.5, 0.5], [True], 0.9)


class TestAdviseThresholds:
    def test_advise_per_group(self):
        # Two option-count groups from the same eval; each gets its own gate
        groups = {
            "options=2": ([0.9, 0.8], [True, False]),
            "options=20": ([0.6, 0.7], [True, True]),
        }
        advice = advise_thresholds(groups, 0.9)
        assert advice["options=2"].threshold == pytest.approx(0.9)
        assert advice["options=2"].coverage == pytest.approx(0.5)
        assert advice["options=20"].threshold == pytest.approx(0.6)  # full coverage
        assert advice["options=20"].coverage == pytest.approx(1.0)

    def test_thresholds_need_not_transfer_across_groups(self):
        # The upstream #394 story: one pool, two shapes, different gates.
        # Group A: accuracy only at high confidence. Group B: accurate
        # throughout. A single global gate over- or under-shoots one of them.
        groups = {
            "A": ([0.9, 0.9, 0.5, 0.5], [True, True, False, False]),
            "B": ([0.5, 0.5], [True, True]),
        }
        advice = advise_thresholds(groups, 0.9)
        assert advice["A"].threshold == pytest.approx(0.9)
        assert advice["A"].coverage == pytest.approx(0.5)
        assert advice["B"].threshold == pytest.approx(0.5)
        assert advice["B"].coverage == pytest.approx(1.0)

    def test_empty_groups_raise(self):
        with pytest.raises(ValueError, match="at least one group"):
            advise_thresholds({}, 0.9)
