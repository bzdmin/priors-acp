"""The counterparty's side of the bridge: write one claim, print it back.

Digest is TypeScript and the memory client is Python, so Digest shells out to
this the same way the buyer shells out to ``decide.py``. Writing a second
memory client in another language would give two implementations of the same
store to keep in step, and the coordination argument depends on there being
exactly one.

This holds a ``DeclarationWriter``, which has a single method. It cannot write
a decision, cannot write an observation, and cannot reach the record that
scores its own claims. That is the point: a counterparty may say whatever it
likes here, and what the claim is worth is settled elsewhere by what happens
next.

    python scripts/declare.py --agent 0x5043... --job-ref 76736 --status available
    python scripts/declare.py --agent 0x5043... --job-ref 76736 --status unavailable
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
STATUSES = ("available", "unavailable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="the declaring agent's address")
    ap.add_argument("--job-ref", required=True,
                    help="what this claim is about; scopes it to one request")
    ap.add_argument("--status", required=True, choices=STATUSES)
    ap.add_argument("--note", default=None)
    ap.add_argument("--block", type=int, default=None,
                    help="defaults to the current Base head")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args()

    # Time is block height here as everywhere else. Fetching the head costs a
    # round trip, so a caller that already knows it can pass it in.
    block = args.block if args.block is not None else ChainSource(0).head()

    memory = PriorsMemory(args.db)
    body = DeclarationWriter(memory, args.agent).declare(
        job_ref=args.job_ref,
        status=args.status,
        block=block,
        note=args.note,
    )
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
