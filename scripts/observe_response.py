"""Resolver-side: did the counterparty actually respond to this job?

The coordination loop has three writers and this is the third. Priors requests,
the counterparty declares, and this records what followed. It is a separate
script for a reason that is not tidiness: if Priors could write this, it would
be grading its own decision, and if the counterparty could write it, it would
be marking its own homework. Neither of those handles has a method that
reaches this record.

What counts as responding is ``BudgetSet``, the provider-side response step in
the ACP SDK. The event carries no setter address, so this is recorded as a
response rather than described as provider acceptance.

A job with no BudgetSet is only a broken promise once it can no longer get
one. While the job's own deadline is still ahead it is simply pending, and
this refuses to record anything, because scoring a live job as a failure would
manufacture the result the whole mechanism is supposed to measure.

The deadline comes from ``expired_at`` on the job's own JobCreated event, and
block time is chain data, so a missed window is observable without waiting for
anyone to send an expiry transaction. That is not a shortcut: nothing expires
these jobs automatically. Only 3,112 of the 40,161 scanned jobs that never
reached BudgetSet ever emitted JobExpired, so waiting for one would leave most
broken promises permanently unscoreable.

    python scripts/observe_response.py --job 76736 --from-block 50951600
    python scripts/observe_response.py --job 76736 --from-block 50951600 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.chain import ChainSource  # noqa: E402
from priors.memory import PriorsMemory, ResolverWriter  # noqa: E402

DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"
#: Terminal without a response: the job can no longer earn a BudgetSet.
DEAD = ("JobExpired", "JobRejected")


def scan(job_id: int, from_block: int, rpc: str | None) -> dict:
    """Every modelled event for one job, from ``from_block`` to the head.

    Runs at one confirmation rather than the usual thirty. This reads a job
    that has already terminated rather than deciding against fresh state, and
    the alternative is waiting a minute to observe something that has already
    happened.
    """
    source = ChainSource(from_block, rpc_url=rpc, confirmations=1)
    head = source.head()
    events = [e for e in source.iter_events() if e.job_id == job_id]
    events.sort(key=lambda e: (e.block, e.log_index))

    kinds = {e.kind for e in events}
    budget = next((e for e in events if e.kind == "BudgetSet"), None)
    dead = next((e for e in events if e.kind in DEAD), None)

    created = next((e for e in events if e.kind == "JobCreated"), None)
    expired_at = created.fields.get("expired_at") if created else None

    if budget is not None:
        state, block, confirmed = "responded", budget.block, True
    elif dead is not None:
        state, block, confirmed = "died unanswered", dead.block, False
    elif expired_at and source.head_timestamp() > int(expired_at):
        # The deadline is in the job's own JobCreated event and block time is
        # chain data, so a missed window is observable without anyone sending
        # an expiry transaction. That matters: nothing expires these jobs
        # automatically. Only 3,112 of the 40,161 scanned jobs that never
        # reached BudgetSet ever emitted JobExpired, so waiting for one would
        # leave most broken promises permanently unscoreable.
        state, block, confirmed = "deadline passed unanswered", head, False
    else:
        state, block, confirmed = "pending", None, None

    return {
        "events": events, "kinds": kinds, "state": state,
        "block": block, "confirmed": confirmed,
        "expired_at": expired_at,
        "provider": next(
            (e.fields.get("provider") for e in events if e.kind == "JobCreated"),
            None,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, required=True)
    ap.add_argument("--from-block", type=int, required=True,
                    help="the job's creation block, printed by the buyer")
    ap.add_argument("--provider", default=None,
                    help="defaults to the provider on the job's JobCreated")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--rpc", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be recorded, write nothing")
    args = ap.parse_args()

    found = scan(args.job, args.from_block, args.rpc)
    provider = (args.provider or found["provider"] or "").lower()

    print(f"job {args.job}")
    for e in found["events"]:
        print(f"  blk {e.block}  {e.kind}")
    if not found["events"]:
        print("  (no events found - is --from-block before the job was created?)")

    print()
    print(f"  provider : {provider or '(unknown)'}")
    print(f"  state    : {found['state']}")

    if found["confirmed"] is None:
        print()
        print("  PENDING - not recording. The job has neither responded nor "
              "died, so\n  there is nothing to observe yet. Re-run after it "
              "settles.")
        return

    if not provider:
        print("\n  cannot record without a provider address")
        sys.exit(1)

    verdict = "CONFIRMED" if found["confirmed"] else "CONTRADICTED"
    print(f"  verdict  : declaration {verdict} at block {found['block']}")

    if args.dry_run:
        print("\n  dry run - nothing written")
        return

    memory = PriorsMemory(args.db)
    record = ResolverWriter(memory).record_response_observation(
        provider, block=found["block"], confirmed=found["confirmed"],
        job_id=args.job,
    )
    acted, conf = record.weighted(found["block"])
    print()
    print(f"  recorded. declarations seen {record.declared}, "
          f"observations {len(record.observations)}")
    print(f"  reliability now {record.reliability(found['block']):.4f} "
          f"(weighted {conf:.2f} of {acted:.2f})")


if __name__ == "__main__":
    main()
