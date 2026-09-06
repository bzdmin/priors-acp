"""Priors' side of the coordination step: ask, wait, then decide.

Called once by the buyer before it creates anything. It writes the request,
waits for the counterparty to answer it, reads what the store remembers about
that counterparty's earlier answers, and returns a verdict.

This is deliberately the only place the buyer touches coordination. The
alternative was four separate shell-outs from TypeScript, which would have put
the sequencing of a decision inside a script that is supposed to execute one.

The completion probability is an input, not something computed here. Rules-v2
produces it and this never modifies it: the gate can refuse a job Rules-v2
approved, and cannot approve one it refused.

    python scripts/coordinate.py --provider 0x5043... --p-complete 0.6626
    python scripts/coordinate.py --provider 0x5043... --p-complete 0.6626 --no-memory
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.chain import ChainSource  # noqa: E402
from priors.coordination import apply_response_gate  # noqa: E402
from priors.memory import PriorsMemory, PriorsWriter  # noqa: E402

DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--p-complete", type=float, required=True,
                    help="Rules-v2 output, passed through untouched")
    ap.add_argument("--job-ref", default=None,
                    help="defaults to the current head block")
    ap.add_argument("--timeout", type=float, default=45.0,
                    help="seconds to wait for the counterparty to answer")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--no-memory", action="store_true",
                    help="the deletion condition: ask, but recall nothing")
    args = ap.parse_args()

    provider = args.provider.lower()
    block = ChainSource(0).head()
    job_ref = args.job_ref or str(block)

    memory = PriorsMemory(args.db)
    priors = PriorsWriter(memory)

    # 1. Ask. The counterparty answers this reference or nothing.
    priors.request_declaration(provider, job_ref=job_ref, block=block)

    # 2. Wait. A counterparty that never answers is not available, and that is
    #    a finding rather than an error: it is the 53.31% of created jobs that
    #    never reach a response, caught before anything is created.
    deadline = time.time() + args.timeout
    declaration = None
    while time.time() < deadline:
        declaration = priors.recall_declaration(provider, job_ref)
        if declaration:
            break
        time.sleep(args.poll)

    declared_available = bool(
        declaration and declaration.get("status") == "available"
    )

    # 3. Recall. --no-memory is the deletion condition and takes the same path
    #    with nothing to recall, rather than a disabled branch.
    record = None if args.no_memory else priors.recall_response_record(provider)

    gate = apply_response_gate(
        args.p_complete, record,
        declared_available=declared_available,
        as_of_block=block,
    )

    print(json.dumps({
        "job_ref": job_ref,
        "provider": provider,
        "at_block": block,
        "declaration": (
            {"status": declaration["status"],
             "declared_at_block": declaration["declared_at_block"],
             "provenance": declaration["provenance"]}
            if declaration else None
        ),
        "waited_for_answer": declaration is not None,
        "p_complete": round(gate.p_complete, 4),
        "completion_gate": gate.completion_gate,
        "p_respond": round(gate.p_respond, 4),
        "response_bar": gate.response_bar,
        "response_gate": gate.response_gate,
        "proceed": gate.proceed,
        "blocked_by_coordination": gate.blocked_by_coordination,
        "memory_used": not args.no_memory,
        "memory_backed": gate.memory_backed,
        "record": (
            {"declared": record.declared,
             "observations": len(record.observations),
             "coverage": record.coverage()}
            if record else None
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
