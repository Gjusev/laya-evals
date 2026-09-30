"""Opt-in test: reproduction-pack smoke with real checkpoints and datasets.

Marked slow and skipped by default. Run with:
    uv run --extra compare --extra dev pytest -m slow
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

SCRIPT = Path(__file__).parent.parent / "scripts" / "reproduction_pack.py"


def _load_module():
    pytest.importorskip("datasets")
    spec = importlib.util.spec_from_file_location("reproduction_pack", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_reproduction_run_produces_claim_report():
    module = _load_module()
    # One English XNLI suite at n=8: real checkpoint, real dataset, tiny n
    cases, golds = module.xnli_suite("en", 8)
    assert len(cases) == 8
    assert all(len(q) == 1 and "relation" in q for _, q in cases)

    from laya import Router

    measured = module.run_suite(Router(), cases, golds, module.ENGLISH)
    assert measured["n"] == 8
    assert 0.0 <= measured["accuracy"] <= 1.0
    assert measured["ece_15_bins"] >= 0.0
    assert measured["wall_seconds"] > 0
    advice = measured["threshold_advice"]
    assert advice["target_accuracy"] == module.ADVISE_TARGET
    assert advice["achievable"] is True or advice["threshold"] is None
    assert 0.0 <= advice["coverage"] <= 1.0

    # The verdict machinery is exercised by the fast tests; here just check
    # the claim table's English entries resolve
    for key in ("xnli.en", "massive_intent.en"):
        assert module.CLAIMS[key]["verdict_eligible"] is True
