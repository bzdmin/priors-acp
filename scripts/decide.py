"""Ask Priors whether to hire a provider, and print the answer as JSON.

This is the bridge between the decision and the transaction. The model lives
in Python; the ACP SDK is TypeScript. Rather than reimplement the predictor in
TS - two copies of the arithmetic that could drift apart - the buyer shells out
to this and reads one JSON object back.

The decision path here is the same ``predict`` the backtest and the live tail
call. Nothing about hiring for real takes a different route.

    python scripts/decide.py --provider 0xabc... --amount 0.02
    python scripts/decide.py --provider 0xabc... --amount 0.02 --no-memory

``--no-memory`` runs the identical call with an empty recall, so the buyer can
show both answers side by side before it spends anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.features import DecisionView  # noqa: E402
from priors.memory import PriorsMemory, PriorsWriter  # noqa: E402
from priors.predict import predict  # noqa: E402

DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"
USDC = 10**6


def build_view(provider: str, client: str, amount_raw: int,
               funded: int, completed: int) -> DecisionView:
    """The job about to be created, described the way a funded job would be.

    Chain counts are inputs, exactly as in replay: they are public and survive
    deletion. What memory supplies is the rules and the calibration.
    """
    return DecisionView(
        job_id=0, provider=provider, client=client,
        funding_block=0, funding_timestamp=0,
        amount=amount_raw, budget_set_count=1,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--client", default="0x" + "0" * 40)
    ap.add_argument("--amount", type=float, required=True, help="USDC")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--funded", type=int, default=0,
                    help="provider's prior funded jobs (public record)")
    ap.add_argument("--completed", type=int, default=0)
    ap.add_argument("--no-memory", action="store_true",
                    help="same call, empty recall - the deletion condition")
    args = ap.parse_args()

    provider = args.provider.lower()
    amount_raw = int(round(args.amount * USDC))

    rules = ()
    calibration = None
    recalled_from = None
    if not args.no_memory:
        memory = PriorsMemory(args.db)
        recalled = PriorsWriter(memory).recall_experience(
            provider, args.client.lower(), amount_raw
        )
        rules = recalled.rules
        calibration = recalled.calibration
        recalled_from = str(memory.path)
        if recalled.provider:
            # Memory's own record of this provider beats the caller's guess.
            args.funded = recalled.provider.get("funded_seen", args.funded)
            args.completed = recalled.provider.get(
                "completed_seen", args.completed
            )

    view = build_view(provider, args.client.lower(), amount_raw,
                      args.funded, args.completed)
    p = predict(view, rules=rules, calibration=calibration)
    # Always compute the deletion condition too. `memory_contributed` on the
    # Prediction is true whenever memory supplied anything at all, including a
    # calibration reading that is reported but never applied - so on its own it
    # would claim a contribution next to two identical probabilities. The
    # honest signal is whether the answer actually moved.
    bare = predict(view)

    print(json.dumps({
        "provider": provider,
        "amount_usdc": args.amount,
        "decision": p.decision,
        "hire": p.decision == "HIRE",
        "with_memory": {
            "p": round(p.p_final, 4),
            "decision": p.decision,
            "rules_applied": [
                {"rule_id": a.rule_id, "delta_logit": round(a.delta_logit, 4),
                 "weight": a.weight, "why": a.why}
                for a in p.rules_applied
            ],
        },
        "chain_only": {
            "p": round(bare.p_final, 4),
            "decision": bare.decision,
        },
        "memory_changed_probability": round(p.p_final, 4) != round(bare.p_final, 4),
        "memory_changed_decision": p.decision != bare.decision,
        "hire_threshold": p.hire_threshold,
        "memory_used": not args.no_memory,
        "provider_record": {"funded": args.funded, "completed": args.completed},
        "store": recalled_from,
    }, indent=2))


if __name__ == "__main__":
    main()
