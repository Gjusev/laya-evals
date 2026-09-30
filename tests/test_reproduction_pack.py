"""Fast tests for the reproduction pack's pure logic (no checkpoint, no dataset).

The slow path (real checkpoints + real datasets) lives in test_slow_reproduction.py.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "reproduction_pack.py"


@pytest.fixture(scope="module")
def module():
    pytest.importorskip("datasets")
    spec = importlib.util.spec_from_file_location("reproduction_pack", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestUpstreamSampler:
    def test_draws_twenty_options_with_gold_included(self, module):
        rng = __import__("random").Random(module.SEED)
        labels = [f"intent_{i}" for i in range(60)]
        gold = "intent_7"
        questions, gold_index = module.build_choice_questions(
            rng, gold, labels, module.N_OPTS, module.MASSIVE_INSTRUCTIONS
        )
        criteria = questions["label"]["criteria"]
        assert len(criteria) == module.N_OPTS
        assert list(criteria)[gold_index] == gold
        # Criteria text follows the upstream transforms
        assert all("_" not in text for text in criteria.values())

    def test_same_seed_same_draw(self, module):
        import random

        labels = [f"lbl.{i}" for i in range(30)]
        draws = [
            module.build_choice_questions(
                random.Random(module.SEED), "lbl.4", labels, 10, "instr"
            )
            for _ in range(2)
        ]
        assert draws[0][0] == draws[1][0]
        # The "." -> ": " transform is applied
        assert all("." not in text for text in draws[0][0]["label"]["criteria"].values())

    def test_pool_smaller_than_options_takes_whole_pool(self, module):
        import random

        labels = ["a", "b", "c"]
        questions, gold_index = module.build_choice_questions(
            random.Random(1), "b", labels, 20, "instr"
        )
        assert len(questions["label"]["criteria"]) == 3
        assert list(questions["label"]["criteria"])[gold_index] == "b"


class TestVerdictRules:
    def test_within_tolerance_reproduces_at_claim_n(self, module):
        entry = module.verdict_for("xnli.en", 0.845, module.CLAIM_N)  # |delta| 0.015
        assert entry["verdict"] == "reproduces"
        assert entry["delta"] == pytest.approx(-0.015)
        assert entry["n"] == module.CLAIM_N

    def test_outside_tolerance_does_not_reproduce(self, module):
        entry = module.verdict_for("xnli.en", 0.70, module.CLAIM_N)  # delta -0.16
        assert entry["verdict"] == "does not reproduce"

    def test_small_n_never_gets_a_verdict(self, module):
        # At n=8 the accuracy grain alone is 0.125: noise must not verdict
        entry = module.verdict_for("xnli.en", 0.845, 8)
        assert entry["verdict"] == "delta_only"
        assert "measured at n=8" in entry["caveat"]

    def test_non_english_claims_are_verdict_eligible(self, module):
        # Full upstream language coverage: the non-English macros get real
        # verdicts at the claim's sample size
        close = module.verdict_for("xnli.non_en", 0.75, module.CLAIM_N)
        assert close["verdict"] == "reproduces"  # |0.75 - 0.731| = 0.019
        far = module.verdict_for("xnli.non_en", 0.55, module.CLAIM_N)
        assert far["verdict"] == "does not reproduce"

    def test_per_lang_validation(self, module):
        with pytest.raises(ValueError, match="per-lang"):
            module.xnli_suite("en", 0)
