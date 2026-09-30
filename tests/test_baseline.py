"""Tests for the regression gate: hand-derived scenarios throughout."""

import pytest

from laya_evals.baseline import Metric, check_regression

BASELINE = {
    "accuracy_vs_gold": 0.8968,
    "ece_15_bins": 0.0253,
    "suites": {"xnli.en": {"accuracy": 0.8600, "ece_15_bins": 0.0685}},
}
METRICS = [
    Metric(path="accuracy_vs_gold", goal="max"),
    Metric(path="ece_15_bins", goal="min"),
    Metric(path="suites.xnli.en.accuracy", goal="max"),
]


class TestCheckRegression:
    def test_unchanged_report_passes(self):
        check = check_regression(BASELINE, BASELINE, METRICS)
        assert check.regressed is False
        assert len(check.results) == 3
        assert all(r.ok for r in check.results)

    def test_small_drop_within_tolerance_passes(self):
        current = dict(BASELINE, accuracy_vs_gold=0.88)  # drop 0.0168 <= 0.02
        check = check_regression(current, BASELINE, METRICS)
        assert check.regressed is False

    def test_accuracy_drop_beyond_tolerance_fails(self):
        current = dict(BASELINE, accuracy_vs_gold=0.86)  # drop 0.0368 > 0.02
        check = check_regression(current, BASELINE, METRICS)
        assert check.regressed is True
        failed = [r for r in check.results if not r.ok]
        assert failed[0].path == "accuracy_vs_gold"
        assert failed[0].delta == pytest.approx(-0.0368, abs=1e-6)

    def test_ece_rise_beyond_tolerance_fails(self):
        # Calibration regression: ECE rising is the failure mode that lets a
        # model keep its accuracy while becoming unsafe to automate on
        current = dict(BASELINE, ece_15_bins=0.05)  # rise 0.0247 > 0.02
        check = check_regression(current, BASELINE, METRICS)
        assert check.regressed is True
        assert [r.path for r in check.results if not r.ok] == ["ece_15_bins"]

    def test_ece_improvement_is_not_a_regression(self):
        current = dict(BASELINE, ece_15_bins=0.01)  # ECE more than halved: fine
        check = check_regression(current, BASELINE, METRICS)
        assert check.regressed is False

    def test_nested_path_regression_fails(self):
        current = {**BASELINE,
                   "suites": {"xnli.en": {"accuracy": 0.80, "ece_15_bins": 0.0685}}}
        check = check_regression(current, BASELINE, METRICS)
        assert check.regressed is True
        assert [r.path for r in check.results if not r.ok] == ["suites.xnli.en.accuracy"]

    def test_improvement_passes_for_max_metrics(self):
        current = dict(BASELINE, accuracy_vs_gold=0.95)
        check = check_regression(current, BASELINE, METRICS)
        assert all(r.ok for r in check.results)

    def test_missing_path_in_current_raises(self):
        partial = {"accuracy_vs_gold": 0.9, "ece_15_bins": 0.02}
        with pytest.raises(ValueError, match="suites.xnli.en.accuracy"):
            check_regression(partial, BASELINE, METRICS)

    def test_missing_path_in_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline"):
            check_regression(BASELINE, {"accuracy_vs_gold": 0.9}, METRICS)

    def test_invalid_goal_raises(self):
        with pytest.raises(ValueError, match="goal"):
            check_regression(BASELINE, BASELINE, [Metric(path="accuracy_vs_gold", goal="sideways")])

    def test_no_metrics_raises(self):
        with pytest.raises(ValueError, match="at least one metric"):
            check_regression(BASELINE, BASELINE, [])

    def test_custom_tolerance_per_metric(self):
        # Strict 0.005 tolerance: a drop of 0.0068 regresses where the
        # default 0.02 would pass
        strict = [Metric(path="accuracy_vs_gold", goal="max", tolerance=0.005)]
        check = check_regression(
            dict(BASELINE, accuracy_vs_gold=0.890), BASELINE, strict
        )
        assert check.regressed is True
        loosened = check_regression(
            dict(BASELINE, accuracy_vs_gold=0.890),
            BASELINE,
            [Metric(path="accuracy_vs_gold", goal="max", tolerance=0.02)],
        )
        assert loosened.regressed is False

    def test_negative_tolerance_raises(self):
        # A sign typo must not silently turn the gate into an improvement
        # demand (delta >= +|tol|) that fails unchanged reports
        with pytest.raises(ValueError, match="tolerance"):
            check_regression(
                BASELINE, BASELINE, [Metric(path="accuracy_vs_gold", goal="max", tolerance=-0.01)]
            )

    def test_wall_clock_fields_are_never_gateable(self):
        # Wall-clock drifts between runners; gating it would be flaky CI.
        for path in ("wall_seconds", "decisions_per_second"):
            with pytest.raises(ValueError, match="drift"):
                check_regression(BASELINE, BASELINE, [Metric(path=path, goal="min")])

    def test_dotted_key_beats_nested_keys(self):
        # Report keys may contain dots; the greedy resolver must prefer the
        # literal key when both spellings exist
        doc = {"a.b": 1, "a": {"b": 2}}
        check = check_regression(doc, doc, [Metric(path="a.b", goal="max")])
        assert check.results[0].baseline == 1
