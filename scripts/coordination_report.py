"""What the coordination layer did, reported on its own terms.

Deliberately not part of scripts/ablation.py. That harness runs four
configurations over 18,360 historical jobs, and coordination cannot be
replayed over any of them: none of those jobs had a counterparty declaring
availability, because Digest did not exist when they ran. Its evidence base is
a handful of live jobs, and putting a three-observation result in a table
beside an eighteen-thousand-job result invites a comparison that is not there.

So this prints the coordination record with its sample size stated in the same
breath, and says what it proves. It proves that removing memory changes the
decision. It does not prove that Priors hires better because of it.

    python scripts/coordination_report.py --provider 0x5043...
    python scripts/coordination_report.py --provider 0x5043... --at-block 50966457
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.chain import ChainSource  # noqa: E402
from priors.coordination import (  # noqa: E402
    NEUTRAL_RESPONSE_PRIOR,
    RESPONSE_BAR,
    apply_response_gate,
)
from priors.memory import PriorsMemory, PriorsWriter  # noqa: E402

DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"
#: The Rules-v2 output the gate was asked about during the live runs. Passed
#: through untouched; coordination never modifies it.
P_COMPLETE = 0.7947


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--at-block", type=int, default=None,
                    help="evaluate the decay as of this block; default is the "
                         "current Base head")
    ap.add_argument("--p-complete", type=float, default=P_COMPLETE)
    args = ap.parse_args()

    provider = args.provider.lower()
    block = args.at_block if args.at_block is not None else ChainSource(0).head()

    memory = PriorsMemory(args.db)
    priors = PriorsWriter(memory)
    record = priors.recall_response_record(provider)

    print("=" * 72)
    print("COORDINATION RESULT")
    print(f"counterparty {provider}")
    print(f"evaluated at block {block:,}")
    print("=" * 72)

    if record is None:
        print("\n  no coordination history for this counterparty")
        return

    print()
    print(f"  declarations made        : {record.declared}")
    print(f"  of those, acted on       : {len(record.observations)}")
    confirmed = sum(1 for o in record.observations if o.confirmed)
    print(f"  kept                     : {confirmed}")
    print(f"  broken                   : {len(record.observations) - confirmed}")
    cov = record.coverage()
    print(f"  coverage                 : "
          f"{cov:.2f}" if cov is not None else "  coverage: n/a")

    print()
    print("  each observation, and what the decay leaves of it:")
    for o in sorted(record.observations, key=lambda x: x.block):
        age_days = (block - o.block) / 43_200
        print(f"    blk {o.block:,}  {'kept  ' if o.confirmed else 'broken'}"
              f"  {age_days:5.2f} days old  weight {o.weight(block):.3f}")

    acted, conf = record.weighted(block)
    print()
    print(f"  weighted                 : {conf:.2f} kept of {acted:.2f}")
    print(f"  marketplace prior        : {NEUTRAL_RESPONSE_PRIOR}")
    print(f"  shrunk reliability       : {record.reliability(block):.4f}")
    print(f"  bar                      : {RESPONSE_BAR}")

    # The dependency claim, both conditions, same everything else.
    with_memory = apply_response_gate(
        args.p_complete, record, declared_available=True, as_of_block=block
    )
    without = apply_response_gate(
        args.p_complete, None, declared_available=True, as_of_block=block
    )

    print()
    print("-" * 72)
    print("  THE DEPENDENCY TEST")
    print("  same counterparty, same declaration, same Rules-v2 output")
    print("-" * 72)
    for label, g in (("memory intact ", with_memory), ("memory deleted", without)):
        action = "CREATE JOB" if g.proceed else "DO NOT CREATE"
        print(f"  {label} : p_complete {g.p_complete:.4f} "
              f"{'PASS' if g.completion_gate else 'FAIL'} | "
              f"p_respond {g.p_respond:.4f} "
              f"{'PASS' if g.response_gate else 'FAIL'} -> {action}")

    changed = with_memory.proceed != without.proceed
    print()
    print(f"  memory changed the action: {'YES' if changed else 'no'}")

    print()
    print("  What this shows, and what it does not:")
    print(f"    n = {len(record.observations)} observations, one counterparty,")
    print("    including a deliberately injected fault. It demonstrates that")
    print("    the decision depends on what was remembered. It is not a")
    print("    measurement of whether Priors hires better, and it is not")
    print("    comparable to the {A,B,C,D} ablation, which runs over 18,360")
    print("    historical jobs that had no declarations to coordinate on.")


if __name__ == "__main__":
    main()
