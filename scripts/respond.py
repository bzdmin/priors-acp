"""The counterparty answering what it was asked.

Digest calls this on a timer. It reads the requests Priors has written for it
and answers each one, which is what makes the store the coordination surface
rather than a noticeboard: the counterparty is reacting to Priors' state, not
announcing itself whenever it likes.

Fault injection lives here rather than in the answer. ``--fault-refs`` names
the job references this agent should claim availability for and then fail to
act on, which is the controlled test of whether Priors notices a broken
promise. It is deliberate, listed by hand, and visible in the process
arguments. Nothing random decides it, so a judge re-running the demo gets the
same behaviour.

    python scripts/respond.py --agent 0x5043... --status available
    python scripts/respond.py --agent 0x5043... --status available --fault-refs 91002
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.chain import ChainSource  # noqa: E402
from priors.memory import DeclarationWriter, PriorsMemory  # noqa: E402

DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--status", default="available",
                    choices=("available", "unavailable"))
    ap.add_argument("--fault-refs", default="",
                    help="comma-separated job refs to claim availability for "
                         "and then deliberately not act on")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    faults = {r.strip() for r in args.fault_refs.split(",") if r.strip()}

    memory = PriorsMemory(args.db)
    writer = DeclarationWriter(memory, args.agent)
    pending = writer.pending_requests()

    if not pending:
        print(json.dumps({"answered": 0, "pending": 0}))
        return

    block = args.block if args.block is not None else ChainSource(0).head()

    answered = []
    for req in pending:
        ref = req["job_ref"]
        injected = ref in faults
        body = writer.declare(
            job_ref=ref,
            status=args.status,
            block=block,
            note="controlled fault injection: will not act on this claim"
            if injected else None,
        )
        answered.append({"job_ref": ref, "status": body["status"],
                         "fault_injected": injected})

    print(json.dumps({"answered": len(answered), "declarations": answered},
                     indent=2))


if __name__ == "__main__":
    main()
