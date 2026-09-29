"""Tests for the calibration core: expected values are hand-derived literals,
independent of the implementation."""

import pytest

from laya_evals.calibration import (
    brier_score,
    coverage_accuracy_curve,
    ece,
    reliability_bins,
)


class TestBrierScore:
    def test_mixed_hand_derived(self):
        # Hand-derived: outcomes y=[1,0,1]
        # (1.0-1)^2 = 0, (0.5-0)^2 = 0.25, (0.0-1)^2 = 1 -> mean = (5/4)/3 = 5/12
        assert brier_score([1.0, 0.5, 0.0], [True, False, True]) == pytest.approx(5 / 12)

    def test_perfect_confidence(self):
        assert brier_score([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == 0.0

    def test_confidently_wrong(self):
        assert brier_score([1.0, 1.0], [False, False]) == 1.0

    def test_uniform_half_right(self):
        # (0.5-1)^2 = 0.25 and (0.5-0)^2 = 0.25 -> mean 0.25
        assert brier_score([0.5, 0.5], [True, False]) == 0.25

    def test_single_example(self):
        assert brier_score([0.8], [True]) == pytest.approx(0.04000000000000001)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            brier_score([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.5, 0.5], [True])

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            brier_score([1.5], [True])
        with pytest.raises(ValueError):
            brier_score([-0.1], [False])


class TestReliabilityBins:
    def test_three_examples_land_in_expected_bins(self):
        # Hand-derived, n_bins=10: 0.05 -> (0.0,0.1], 0.15 -> (0.1,0.2], 0.95 -> (0.9,1.0]
        bins = reliability_bins([0.05, 0.15, 0.95], [False, True, True], n_bins=10)
        assert len(bins) == 10
        b0, b1, b9 = bins[0], bins[1], bins[9]
        assert (b0.lower, b0.upper, b0.count, b0.mean_confidence, b0.accuracy) == (
            0.0, 0.1, 1, pytest.approx(0.05), 0.0,
        )
        assert (b1.lower, b1.upper, b1.count, b1.mean_confidence, b1.accuracy) == (
            0.1, 0.2, 1, pytest.approx(0.15), 1.0,
        )
        assert (b9.lower, b9.upper, b9.count, b9.mean_confidence, b9.accuracy) == (
            0.9, 1.0, 1, pytest.approx(0.95), 1.0,
        )
        empties = [b for i, b in enumerate(bins) if i not in (0, 1, 9)]
        assert all(b.count == 0 and b.mean_confidence is None and b.accuracy is None
                   for b in empties)
        assert sum(b.count for b in bins) == 3

    def test_bin_averages_within_bin(self):
        # Both in (0.9, 1.0]: mean conf (0.92+0.98)/2 = 0.95, accuracy 1/2
        bins = reliability_bins([0.92, 0.98], [True, False], n_bins=10)
        b9 = bins[9]
        assert b9.count == 2
        assert b9.mean_confidence == pytest.approx(0.95)
        assert b9.accuracy == pytest.approx(0.5)

    def test_interior_edges_are_right_closed(self):
        # Guo et al. bins are (lo, hi]: a value on an interior edge belongs to
        # the bin it closes, so 0.3 lands in (0.2,0.3], not (0.3,0.4]
        bins = reliability_bins([0.3, 0.31], [True, True], n_bins=10)
        assert bins[2].count == 1
        assert bins[3].count == 1

    def test_zero_confidence_falls_in_first_bin(self):
        # Strictly 0.0 lies outside (0.0, 0.1]; defensively it goes to bin 0
        bins = reliability_bins([0.0], [False])
        assert bins[0].count == 1

    def test_upper_edge_is_closed(self):
        # 1.0 lands in the last bin, never out of range
        bins = reliability_bins([1.0], [True], n_bins=10)
        assert bins[9].count == 1
        assert bins[9].upper == 1.0

    def test_custom_bin_count(self):
        # n_bins=2: bins (0.0,0.5] and (0.5,1.0]; 0.5 closes the first bin
        bins = reliability_bins([0.5, 0.7], [False, True], n_bins=2)
        assert len(bins) == 2
        assert bins[0] == (0.0, 0.5, 1, pytest.approx(0.5), 0.0)
        assert bins[1] == (0.5, 1.0, 1, pytest.approx(0.7), 1.0)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            reliability_bins([], [])

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError):
            reliability_bins([0.5], [True], n_bins=0)


class TestEce:
    def test_hand_derived_two_bins(self):
        # Hand-derived, n_bins=10: bin (0.8,0.9] (right-closed, so 0.9 lands
        # here) has 3 examples, mean conf 0.9, accuracy 1.0 -> gap 0.1
        # weighted 3/4 = 0.075; bin (0.1,0.2] has 1 example, conf 0.2,
        # accuracy 0.0 -> gap 0.2 weighted 1/4 = 0.05.
        # ECE = 0.075 + 0.05 = 0.125
        assert ece([0.9, 0.9, 0.9, 0.2], [True, True, True, False], n_bins=10) == pytest.approx(0.125)

    def test_perfectly_calibrated_synthetic(self):
        # n_bins=10. Bin (0.2,0.3]: conf 0.25, 4 examples, 1 correct ->
        # accuracy 0.25. Bin (0.7,0.8]: conf 0.75, 4 examples, 3 correct ->
        # accuracy 0.75. Every gap is zero.
        confs = [0.25] * 4 + [0.75] * 4
        outs = [True] + [False] * 3 + [True] * 3 + [False]
        assert ece(confs, outs, n_bins=10) == 0.0

    def test_confidently_wrong(self):
        assert ece([1.0, 1.0], [False, False]) == 1.0

    def test_perfectly_confident_and_right(self):
        assert ece([1.0, 1.0, 1.0], [True, True, True]) == 0.0

    def test_single_example(self):
        # n_bins=10, bin (0.5,0.6] (right-closed, so 0.6 lands here): conf
        # 0.6, accuracy 1.0 -> gap 0.4, weight 1
        assert ece([0.6], [True], n_bins=10) == pytest.approx(0.4)

    def test_weighted_by_bin_mass_not_bin_average(self):
        # n_bins=10. Bin (0.4,0.5] (right-closed, so 0.5 lands here): 4
        # examples, conf 0.5, accuracy (1+0+1+0)/4 = 0.5 -> gap 0. Bin
        # (0.9,1.0]: 1 example, conf 1.0, accuracy 0 -> gap 1.0.
        # ECE = 0*4/5 + 1.0*1/5 = 0.2, NOT the unweighted mean (0 + 1)/2 = 0.5
        assert ece([0.5, 0.5, 0.5, 0.5, 1.0], [True, False, True, False, False], n_bins=10) == pytest.approx(0.2)

    def test_default_bin_count_is_fifteen(self):
        # Guo et al. (2017) use 15 bins; pin the default so a silent change
        # to another count cannot pass unnoticed
        bins = reliability_bins([0.5], [True])
        assert len(bins) == 15

    def test_default_binning_discriminates_fifteen_bins(self):
        # Hand-derived under the default 15 bins: 0.05 -> (0, 1/15] with
        # accuracy 1.0 -> gap 0.95 weighted 1/2; 0.07 -> (1/15, 2/15] with
        # accuracy 0.0 -> gap 0.07 weighted 1/2. ECE = 0.475 + 0.035 = 0.51.
        # Under 10 bins both land in (0, 0.1] (mean conf 0.06, accuracy 0.5)
        # and ECE would be 0.44, so this literal pins the default bin count.
        assert ece([0.05, 0.07], [True, False]) == pytest.approx(0.51)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            ece([], [])

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError):
            ece([0.5], [True], n_bins=-1)


class TestCoverageAccuracyCurve:
    def test_geifman_el_yaniv_worked_example(self):
        # Worked example cross-checked against the selective-prediction
        # literature: sweeping the observed confidences as thresholds keeps
        # every example with confidence >= tau.
        curve = coverage_accuracy_curve(
            [0.91, 0.79, 0.61, 0.51, 0.40, 0.32],
            [True, True, False, True, False, True],
        )
        assert [(p.coverage, p.accuracy) for p in curve] == [
            (pytest.approx(1 / 6), pytest.approx(1.0)),
            (pytest.approx(2 / 6), pytest.approx(1.0)),
            (pytest.approx(3 / 6), pytest.approx(2 / 3)),
            (pytest.approx(4 / 6), pytest.approx(3 / 4)),
            (pytest.approx(5 / 6), pytest.approx(3 / 5)),
            (pytest.approx(1.0), pytest.approx(4 / 6)),
        ]
        # Thresholds descend as coverage grows
        thresholds = [p.threshold for p in curve]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_ties_enter_together(self):
        # Two examples share confidence 0.8: a single point keeps both,
        # coverage 1.0, accuracy 1/2. Ties are never split across a threshold.
        curve = coverage_accuracy_curve([0.8, 0.8], [True, False])
        assert len(curve) == 1
        assert curve[0].threshold == pytest.approx(0.8)
        assert curve[0].coverage == pytest.approx(1.0)
        assert curve[0].accuracy == pytest.approx(0.5)

    def test_input_order_is_irrelevant(self):
        shuffled = coverage_accuracy_curve(
            [0.32, 0.91, 0.51, 0.79, 0.40, 0.61],
            [True, True, True, True, False, False],
        )
        ordered = coverage_accuracy_curve(
            [0.91, 0.79, 0.61, 0.51, 0.40, 0.32],
            [True, True, False, True, False, True],
        )
        assert shuffled == ordered

    def test_single_example_curve(self):
        curve = coverage_accuracy_curve([0.7], [False])
        assert [(p.threshold, p.coverage, p.accuracy) for p in curve] == [(0.7, 1.0, 0.0)]

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            coverage_accuracy_curve([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            coverage_accuracy_curve([0.5], [True, False])


class TestSyntheticDistributions:
    """Larger synthetic sets whose metric values are known by construction."""

    def test_perfectly_calibrated_distribution(self):
        # Three groups of 4, each with empirical accuracy exactly equal to its
        # confidence: 0.25 with 1/4 correct, 0.5 with 2/4, 0.75 with 3/4.
        # ECE is 0 by construction. Brier, hand-derived:
        #   group 0.25: 0.75^2*1 + 0.25^2*3 = 0.75
        #   group 0.50: 0.50^2*4          = 1.00
        #   group 0.75: 0.75^2*1 + 0.25^2*3 = 0.75
        #   Brier = (0.75 + 1.00 + 0.75) / 12 = 5/24
        confs = [0.25] * 4 + [0.5] * 4 + [0.75] * 4
        outs = (
            [True, False, False, False]
            + [True, True, False, False]
            + [True, True, True, False]
        )
        assert ece(confs, outs) == pytest.approx(0.0)
        assert brier_score(confs, outs) == pytest.approx(5 / 24)
        # Every reliability gap is zero and counts sum to the full set
        bins = reliability_bins(confs, outs, n_bins=10)
        assert sum(b.count for b in bins) == 12
        assert all(
            b.accuracy == pytest.approx(b.mean_confidence) for b in bins if b.count
        )

    def test_systematically_overconfident_distribution(self):
        # Every example claims 1.0 and exactly half are right: ECE = 0.5,
        # Brier = 0.5, the top of the coverage curve shows the lie immediately
        confs = [1.0] * 8
        outs = [True] * 4 + [False] * 4
        assert ece(confs, outs) == pytest.approx(0.5)
        assert brier_score(confs, outs) == pytest.approx(0.5)
        curve = coverage_accuracy_curve(confs, outs)
        assert len(curve) == 1
        assert curve[0] == (1.0, 1.0, 0.5)


class TestInputContract:
    def test_nan_outcome_raises(self):
        # A missing gold label must not be silently truthiness-coerced into
        # "correct" and inflate accuracy
        with pytest.raises(ValueError):
            brier_score([0.7], [float("nan")])

    def test_fractional_outcome_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.7], [0.3])

    def test_string_outcome_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.7], ["yes"])

    def test_zero_and_one_ints_accepted(self):
        # 0/1 ints are the numeric spelling of False/True
        assert brier_score([0.5, 0.5], [1, 0]) == 0.25
        assert ece([0.5, 0.5], [1, 0], n_bins=10) == 0.0

    def test_numpy_arrays_accepted(self):
        # Eval data commonly arrives as numpy arrays; every operation the
        # metrics use is array-safe, so the module must not crash on one
        numpy = pytest.importorskip("numpy")
        confs = numpy.array([0.9, 0.9, 0.9, 0.2])
        outs = numpy.array([True, True, True, False])
        assert brier_score(confs, outs) == pytest.approx((0.01 * 3 + 0.04) / 4)
        assert ece(confs, outs, n_bins=10) == pytest.approx(0.125)
        assert len(coverage_accuracy_curve(confs, outs)) == 2
