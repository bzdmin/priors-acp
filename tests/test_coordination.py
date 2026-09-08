"""The response gate, and the properties the submission claims about it.

These are not coverage. Each one pins a specific claim made in the README or
in public, so that if the behaviour drifts the claim fails loudly rather than
quietly becoming untrue.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.coordination import (  # noqa: E402
    BLOCKS_PER_DAY,
    EXPIRY_BLOCKS,
    HALF_LIFE_BLOCKS,
    NEUTRAL_RESPONSE_PRIOR,
    RESPONSE_BAR,
    Observation,
    ResponseRecord,
    apply_response_gate,
)

NOW = 51_000_000


def rec(*obs: Observation, declared: int = 0) -> ResponseRecord:
    return ResponseRecord(
        provider="0xtest", declared=declared or len(obs), observations=obs
    )


class TheBarIsThePrior(unittest.TestCase):
    """The deletion case passes by construction, not by a tuned constant."""

    def test_bar_is_the_prior_itself(self):
        # Identity, not near-equality: if these ever drift apart, the claim
        # that an unknown counterparty is treated as typical stops being true.
        self.assertIs(RESPONSE_BAR, NEUTRAL_RESPONSE_PRIOR)

    def test_empty_record_returns_the_prior_exactly(self):
        # Computed shrinkage would land on 0.46690000000000004 and fail a
        # `>=` against 0.4669. The code returns the constant instead.
        self.assertEqual(rec().reliability(NOW), NEUTRAL_RESPONSE_PRIOR)

    def test_no_record_at_all_passes(self):
        g = apply_response_gate(0.8, None, declared_available=True, as_of_block=NOW)
        self.assertEqual(g.p_respond, NEUTRAL_RESPONSE_PRIOR)
        self.assertTrue(g.response_gate)
        self.assertTrue(g.proceed)

    def test_unknown_counterparty_sits_exactly_on_the_bar(self):
        g = apply_response_gate(0.8, rec(), declared_available=True, as_of_block=NOW)
        self.assertEqual(g.p_respond, g.response_bar)
        self.assertTrue(g.response_gate)


class EvidenceMovesIt(unittest.TestCase):
    def test_one_broken_promise_blocks(self):
        r = rec(Observation(NOW - 10, False))
        g = apply_response_gate(0.8, r, declared_available=True, as_of_block=NOW)
        self.assertLess(g.p_respond, RESPONSE_BAR)
        self.assertFalse(g.proceed)

    def test_one_kept_promise_after_a_broken_one_clears_it(self):
        r = rec(Observation(NOW - 20, False), Observation(NOW - 10, True))
        g = apply_response_gate(0.8, r, declared_available=True, as_of_block=NOW)
        self.assertGreaterEqual(g.p_respond, RESPONSE_BAR)
        self.assertTrue(g.proceed)

    def test_two_broken_against_one_kept_blocks(self):
        # The live sequence: 76935 kept, 76936 and 76941 broken.
        r = rec(
            Observation(NOW - 30, True),
            Observation(NOW - 20, False),
            Observation(NOW - 10, False),
        )
        g = apply_response_gate(0.8, r, declared_available=True, as_of_block=NOW)
        self.assertLess(g.p_respond, RESPONSE_BAR)
        self.assertFalse(g.proceed)

    def test_shrinkage_keeps_a_single_observation_near_the_prior(self):
        # One broken promise must not slam reliability to zero; the whole
        # point of shrinking is that one data point is not a verdict.
        r = rec(Observation(NOW, False))
        self.assertGreater(r.reliability(NOW), 0.3)

    def test_more_evidence_moves_further_from_the_prior(self):
        one = rec(Observation(NOW, False)).reliability(NOW)
        many = rec(*[Observation(NOW, False) for _ in range(10)]).reliability(NOW)
        self.assertLess(many, one)


class EvidenceExpires(unittest.TestCase):
    """Without expiry a blocked counterparty could never recover."""

    def test_weight_halves_each_day(self):
        o = Observation(NOW - BLOCKS_PER_DAY, False)
        self.assertAlmostEqual(o.weight(NOW), 0.5, places=6)

    def test_weight_is_one_when_fresh(self):
        self.assertAlmostEqual(Observation(NOW, False).weight(NOW), 1.0, places=6)

    def test_half_life_is_one_day(self):
        self.assertEqual(HALF_LIFE_BLOCKS, BLOCKS_PER_DAY)

    def test_expiry_is_five_days(self):
        self.assertEqual(EXPIRY_BLOCKS, 5 * BLOCKS_PER_DAY)

    def test_observation_is_dropped_at_expiry(self):
        self.assertEqual(Observation(NOW - EXPIRY_BLOCKS, False).weight(NOW), 0.0)

    def test_expired_record_returns_the_prior_exactly(self):
        # Decay alone asymptotes toward the prior from below and never
        # reaches it, which would leave a blocked counterparty blocked
        # forever. Expiry is what actually restores it.
        r = rec(Observation(NOW - EXPIRY_BLOCKS, False))
        self.assertEqual(r.reliability(NOW), NEUTRAL_RESPONSE_PRIOR)

    def test_a_blocked_counterparty_recovers_after_five_days(self):
        r = rec(Observation(NOW - EXPIRY_BLOCKS, False))
        g = apply_response_gate(0.8, r, declared_available=True, as_of_block=NOW)
        self.assertTrue(g.response_gate, "expiry must let a counterparty back in")

    def test_still_blocked_just_before_expiry(self):
        r = rec(Observation(NOW - EXPIRY_BLOCKS + 1, False))
        g = apply_response_gate(0.8, r, declared_available=True, as_of_block=NOW)
        self.assertFalse(g.response_gate)

    def test_a_future_dated_observation_does_not_gain_weight(self):
        self.assertAlmostEqual(Observation(NOW + 1000, False).weight(NOW), 1.0)


class TheGateOnlyEverBlocks(unittest.TestCase):
    """Coordination can decline what Rules-v2 approved, never the reverse."""

    def test_cannot_promote_a_job_rules_refused(self):
        g = apply_response_gate(0.2, None, declared_available=True, as_of_block=NOW)
        self.assertFalse(g.completion_gate)
        self.assertTrue(g.response_gate)
        self.assertFalse(g.proceed)

    def test_unavailable_declaration_stops_it_regardless(self):
        g = apply_response_gate(0.99, None, declared_available=False, as_of_block=NOW)
        self.assertTrue(g.completion_gate)
        self.assertFalse(g.proceed)

    def test_blocked_by_coordination_only_when_completion_passed(self):
        blocked = apply_response_gate(
            0.8, rec(Observation(NOW, False)), declared_available=True,
            as_of_block=NOW,
        )
        self.assertTrue(blocked.blocked_by_coordination)

        already_refused = apply_response_gate(
            0.2, rec(Observation(NOW, False)), declared_available=True,
            as_of_block=NOW,
        )
        self.assertFalse(already_refused.blocked_by_coordination)

    def test_completion_probability_passes_through_untouched(self):
        g = apply_response_gate(0.7334, None, declared_available=True, as_of_block=NOW)
        self.assertEqual(g.p_complete, 0.7334)


class CoverageIsSeparateFromReliability(unittest.TestCase):
    """A counterparty cannot bank a perfect record by declining everything."""

    def test_no_declarations_means_no_coverage(self):
        self.assertIsNone(rec(declared=0).coverage())

    def test_coverage_counts_declarations_not_just_observations(self):
        r = rec(Observation(NOW, True), declared=5)
        self.assertAlmostEqual(r.coverage(), 0.2)

    def test_dodging_does_not_improve_reliability(self):
        kept = rec(Observation(NOW, True), declared=1)
        dodger = rec(Observation(NOW, True), declared=20)
        self.assertEqual(kept.reliability(NOW), dodger.reliability(NOW))
        self.assertLess(dodger.coverage(), kept.coverage())


class TheDependencyClaim(unittest.TestCase):
    """The kill criterion, as a test: same inputs, different memory, different action."""

    def test_same_declaration_different_memory_different_action(self):
        blocked = rec(
            Observation(NOW - 30, True),
            Observation(NOW - 20, False),
            Observation(NOW - 10, False),
        )
        with_memory = apply_response_gate(
            0.7947, blocked, declared_available=True, as_of_block=NOW
        )
        without = apply_response_gate(
            0.7947, None, declared_available=True, as_of_block=NOW
        )
        self.assertEqual(with_memory.p_complete, without.p_complete)
        self.assertNotEqual(with_memory.proceed, without.proceed)
        self.assertFalse(with_memory.proceed)
        self.assertTrue(without.proceed)


if __name__ == "__main__":
    unittest.main()
