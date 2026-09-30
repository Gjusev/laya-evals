"""Opt-in test: runs the laya judge on real SST-2 items with a real checkpoint.

Marked slow and skipped by default (pyproject addopts). Run with:
    uv run --extra compare --extra dev pytest -m slow
(needs the compare extra for kagglehub/pyarrow and downloads the laya
checkpoint on first use).
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

SCRIPT = Path(__file__).parent.parent / "scripts" / "sst2_judge_comparison.py"


def _load_script():
    # Skip cleanly (not error) when the compare extra is not installed
    pytest.importorskip("kagglehub")
    pytest.importorskip("pyarrow")
    spec = importlib.util.spec_from_file_location("sst2_judge_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_limit_must_be_positive():
    module = _load_script()
    with pytest.raises(ValueError, match="limit"):
        module.run(limit=0)
    with pytest.raises(ValueError, match="limit"):
        module.run(limit=-4)


def test_sst2_small_run_produces_honest_report():
    module = _load_script()
    report = module.run(limit=16)
    assert report["n"] == 16
    assert 0.0 <= report["accuracy_vs_gold"] <= 1.0
    assert -1.0 <= report["kappa_vs_gold"] <= 1.0
    assert report["ece_15_bins"] >= 0.0
    assert report["brier_score"] >= 0.0
    assert report["decisions_per_second"] > 0
    # The unmeasured reference-judge and cost fields must stay null, not
    # invented
    assert report["reference_judge_percent_agreement"] is None
    assert report["reference_judge_kappa"] is None
    assert report["laya_cost_per_1k_usd"] is None
    assert report["reference_cost_per_1k_usd"] is None
    # The coverage curve ends at full coverage and its accuracy there equals
    # the overall accuracy
    assert report["coverage_accuracy_curve"][-1][0] == pytest.approx(1.0)
    assert report["coverage_accuracy_curve"][-1][1] == pytest.approx(
        report["accuracy_vs_gold"]
    )
