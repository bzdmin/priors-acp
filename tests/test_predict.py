"""Rules-v2: the frozen predictor, and the properties published about it.

The figures in the README and in public come from this code path. If any of
these fail, a published number has quietly stopped being true.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.features import DecisionView  # noqa: E402
from priors.predict import (  # noqa: E402
    BASE_RATE,
    MAX_DELTA_LOGIT,
    SHRINKAGE_K,
    Rule,
    predict,
)


def view(funded: int = 0, completed: int = 0, amount: int = 250_000) -> DecisionView:
    return DecisionView(
        job_id=0, provider="0xp", client="0xc", funding_block=0,
        funding_timestamp=0, amount=amount, budget_set_count=1,
        expiry_window=3600, funding_delay=0,
        provider_prior_funded=funded, provider_prior_completed=completed,
        provider_prior_rejected_by_self=0, provider_prior_expired=0,
        provider_median_completed_amount=None,
        provider_blocks_since_first_seen=None,
        provider_blocks_since_last_activity=None,
        provider_concurrent_open=0,
        client_prior_created=0, client_prior_funded=0,
        client_prior_funding_rate=None,
        repeat_pair=False, pair_prior_jobs=0, pair_prior_completed=0,
    )


def rule(rid: str, value: int, delta: float = -2.0) -> Rule:
    return Rule(rule_id=rid, subject="0xp", field="amount", op=">",
                value=value, delta_logit=delta, status="graduated")


class PublishedConstants(unittest.TestCase):
    def test_base_rate_matches_the_scan(self):
        # 14,597 of 18,367 funded jobs completed. Quoted everywhere.
        self.assertEqual(BASE_RATE, 0.7947)

    def test_shrinkage_and_cap_are_what_the_readme_says(self):
        self.assertEqual(SHRINKAGE_K, 10.0)
        self.assertEqual(MAX_DELTA_LOGIT, 2.0)


class StageOneSurvivesDeletion(unittest.TestCase):
    """With nothing recalled the prediction is the shrunk chain-only baseline."""

    def test_unknown_provider_gets_the_market_rate(self):
        self.assertAlmostEqual(predict(view()).p_final, BASE_RATE, places=4)

    def test_thin_history_is_pulled_toward_the_market_rate(self):
        # A provider with 3 jobs must not be read like one with 300.
        thin = predict(view(3, 0)).p0_shrunk
        self.assertGreater(thin, 0.5, "3 failures should not crater it")
        self.assertLess(thin, BASE_RATE)

    def test_real_volume_outruns_the_prior(self):
        # 110 of 168, the hero provider's record on the live runs.
        self.assertAlmostEqual(predict(view(168, 110)).p0_shrunk, 0.6626, places=4)

    def test_no_rules_means_no_memory_contribution(self):
        p = predict(view(168, 110))
        self.assertEqual(p.rules_applied, ())
        self.assertAlmostEqual(p.p_final, p.p0_shrunk, places=6)

    def test_default_threshold_is_one_half(self):
        self.assertEqual(predict(view()).hire_threshold, 0.5)


class RulesChangeTheCall(unittest.TestCase):
    def test_a_matching_rule_moves_the_prediction(self):
        p = predict(view(168, 110), rules=[rule("R-1", 50_000)])
        self.assertLess(p.p_final, p.p0_shrunk)
        self.assertEqual(p.decision, "DO NOT HIRE")
        self.assertEqual([a.rule_id for a in p.rules_applied], ["R-1"])

    def test_a_rule_that_does_not_match_is_not_applied(self):
        # amount is 250,000; this rule wants > 500,000.
        p = predict(view(168, 110), rules=[rule("R-9", 500_000)])
        self.assertEqual(p.rules_applied, ())

    def test_nested_thresholds_on_one_field_count_once(self):
        # R-1 and R-2 are the same evidence stated twice. Counting both was a
        # real bug: three nested rules drove 0.796 to 0.010.
        one = predict(view(168, 110), rules=[rule("R-1", 50_000)])
        both = predict(view(168, 110),
                       rules=[rule("R-1", 50_000), rule("R-2", 60_000)])
        self.assertAlmostEqual(one.p_final, both.p_final, places=6)
        self.assertEqual(len(both.rules_applied), 1)

    def test_the_shift_is_capped(self):
        huge = predict(view(168, 110), rules=[rule("R-1", 50_000, delta=-99.0)])
        self.assertGreater(huge.p_final, 0.0)


class DeterminismOnTheDecisionPath(unittest.TestCase):
    def test_the_same_input_gives_the_same_answer(self):
        a = predict(view(168, 110), rules=[rule("R-1", 50_000)])
        b = predict(view(168, 110), rules=[rule("R-1", 50_000)])
        self.assertEqual(a.p_final, b.p_final)
        self.assertEqual(a.decision, b.decision)


if __name__ == "__main__":
    unittest.main()
