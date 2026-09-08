"""Event decoding, and the write-ownership separation.

The decoding tests exist because one of these bugs silently corrupted every
per-provider statistic in the project before it was caught. The ownership
tests exist because "nobody can write the record that judges their own claim"
is an architectural claim, and an architectural claim should fail a test if
someone adds a convenient method.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.events import (  # noqa: E402
    JOB_CREATED,
    JOB_FUNDED,
    TERMINAL_FAILURE,
    AcpEvent,
    decode_log,
    sort_events,
    usdc,
)
from priors.memory import (  # noqa: E402
    DeclarationWriter,
    PriorsWriter,
    ResolverWriter,
)

CLIENT = "0x58c8b031e963cea14e2440ddbcbb9bdb60c23a80"
PROVIDER = "0x5043147b8b666ac070e01ff659e1fbbbc2462bc7"


def topic(addr: str) -> str:
    return "0x" + "0" * 24 + addr[2:]


def word(n: int) -> str:
    return f"{n:064x}"


class AddressDecoding(unittest.TestCase):
    """A two-character slip here split one agent into two identities."""

    def test_addresses_are_full_length(self):
        log = {
            "topics": [JOB_CREATED, "0x" + word(1), topic(CLIENT), topic(PROVIDER)],
            "data": "0x" + word(0) + word(1_788_693_028),
            "blockNumber": "0x1", "logIndex": "0x0",
        }
        ev = decode_log(log)
        self.assertEqual(len(ev.fields["client"]), 42)
        self.assertEqual(len(ev.fields["provider"]), 42)

    def test_addresses_round_trip_exactly(self):
        log = {
            "topics": [JOB_CREATED, "0x" + word(1), topic(CLIENT), topic(PROVIDER)],
            "data": "0x" + word(0), "blockNumber": "0x1", "logIndex": "0x0",
        }
        ev = decode_log(log)
        self.assertEqual(ev.fields["client"], CLIENT)
        self.assertEqual(ev.fields["provider"], PROVIDER)

    def test_addresses_are_lowercased(self):
        log = {
            "topics": [JOB_CREATED, "0x" + word(1),
                       topic(CLIENT.upper().replace("0X", "0x")), topic(PROVIDER)],
            "data": "0x" + word(0), "blockNumber": "0x1", "logIndex": "0x0",
        }
        self.assertEqual(decode_log(log).fields["client"], CLIENT)


class EventDecoding(unittest.TestCase):
    def test_a_zero_evaluator_reads_as_absent(self):
        # The no-evaluator sentinel must not become an address, or the
        # published evaluator-independence figures are wrong.
        log = {
            "topics": [JOB_CREATED, "0x" + word(31495), topic(CLIENT), topic(PROVIDER)],
            "data": "0x" + word(0), "blockNumber": "0x1", "logIndex": "0x0",
        }
        self.assertIsNone(decode_log(log).fields["evaluator"])

    def test_job_funded_carries_the_amount(self):
        log = {
            "topics": [JOB_FUNDED, "0x" + word(76935), topic(CLIENT)],
            "data": "0x" + word(10_000), "blockNumber": "0x2", "logIndex": "0x1",
        }
        ev = decode_log(log)
        self.assertEqual(ev.kind, "JobFunded")
        self.assertEqual(ev.job_id, 76935)
        self.assertEqual(ev.fields["amount"], 10_000)

    def test_an_unmodelled_signature_is_ignored(self):
        log = {"topics": ["0x" + "ab" * 32, "0x" + word(1)],
               "data": "0x", "blockNumber": "0x1", "logIndex": "0x0"}
        self.assertIsNone(decode_log(log))

    def test_a_log_with_no_topics_is_ignored(self):
        self.assertIsNone(decode_log({"topics": [], "data": "0x"}))

    def test_usdc_conversion(self):
        self.assertAlmostEqual(usdc(10_000), 0.01)
        self.assertAlmostEqual(usdc(250_000), 0.25)

    def test_terminal_failures_are_the_three_expected(self):
        self.assertEqual(TERMINAL_FAILURE,
                         frozenset({"JobRejected", "Refunded", "JobExpired"}))


class Ordering(unittest.TestCase):
    """Two runs over the same data must produce identical decisions."""

    def test_total_order_is_block_then_log_index_then_job(self):
        a = AcpEvent(kind="JobFunded", job_id=2, block=10, log_index=1)
        b = AcpEvent(kind="JobFunded", job_id=1, block=10, log_index=0)
        c = AcpEvent(kind="JobFunded", job_id=9, block=9, log_index=99)
        self.assertEqual([e.job_id for e in sort_events([a, b, c])], [9, 1, 2])

    def test_sorting_is_stable_across_runs(self):
        evs = [AcpEvent(kind="JobFunded", job_id=i % 5, block=i // 3, log_index=i)
               for i in range(30)]
        self.assertEqual([e.order_key for e in sort_events(evs)],
                         [e.order_key for e in sort_events(list(reversed(evs)))])


class WriteOwnership(unittest.TestCase):
    """Nobody can write the record that judges their own claim."""

    def test_priors_cannot_write_an_episode(self):
        self.assertFalse(hasattr(PriorsWriter, "write_episode"))

    def test_priors_cannot_write_an_observation(self):
        self.assertFalse(hasattr(PriorsWriter, "record_response_observation"))

    def test_priors_cannot_declare(self):
        self.assertFalse(hasattr(PriorsWriter, "declare"))

    def test_the_resolver_cannot_write_a_decision(self):
        self.assertFalse(hasattr(ResolverWriter, "write_decision"))

    def test_the_resolver_cannot_declare(self):
        self.assertFalse(hasattr(ResolverWriter, "declare"))

    def test_the_counterparty_can_only_declare(self):
        public = {m for m in dir(DeclarationWriter) if not m.startswith("_")}
        self.assertEqual(public, {"declare", "pending_requests"})

    def test_the_counterparty_cannot_reach_its_own_score(self):
        self.assertFalse(hasattr(DeclarationWriter, "record_response_observation"))
        self.assertFalse(hasattr(DeclarationWriter, "recall_response_record"))

    def test_only_the_resolver_scores_declarations(self):
        self.assertTrue(hasattr(ResolverWriter, "record_response_observation"))

    def test_only_priors_requests(self):
        self.assertTrue(hasattr(PriorsWriter, "request_declaration"))
        self.assertFalse(hasattr(DeclarationWriter, "request_declaration"))


if __name__ == "__main__":
    unittest.main()
