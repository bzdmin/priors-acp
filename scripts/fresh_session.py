"""The eligibility gate, end to end (Sibyl pass/fail: persist, recall in a
fresh session, change a decision).

Two phases, deliberately separated by closing the store:

  Phase 1  replay history, and persist what Priors learned
  Phase 2  open a BRAND NEW client against that file, recall, and decide

Phase 2 constructs its own MemoryClient and never touches the replay dataset.
If Sibyl were empty it would have nothing at all - which is exactly what
`--empty` demonstrates.

    python scripts/fresh_session.py            # populated
    python scripts/fresh_session.py --empty    # deletion test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.events import LocalDatasetSource  # noqa: E402
from priors.features import DecisionView  # noqa: E402
from priors.memory import (  # noqa: E402
    PriorsMemory,
    PriorsWriter,
    ResolverWriter,
)
from priors.predict import predict  # noqa: E402
from priors.replay import run_replay  # noqa: E402

#: The scan ships with the repo, gzipped, so the figures here regenerate from
#: a clean clone. The uncompressed copy on the author's machine wins if it is
#: there, since it loads faster, but nothing depends on it existing.
def _dataset() -> Path:
    here = Path(__file__).resolve().parent.parent / "data" / "acp_scan_state_v2.json.gz"
    local = Path.home() / "OneDrive/Desktop/acp/acp_scan_state_v2.json"
    return local if local.exists() else here

DEFAULT_DATASET = _dataset()
DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"


def phase1_learn(dataset: Path, db: Path) -> dict:
    """Replay history and persist what was learned. Session one."""
    if db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    memory = PriorsMemory(db)   # policy follows tier: uncapped -> journal everything
    priors_writer = PriorsWriter(memory)
    resolver_writer = ResolverWriter(memory)

    events = list(LocalDatasetSource(dataset).iter_events())
    # Both processes write to the store as the replay runs, in the order they
    # would live: Priors records a decision before the outcome exists, the
    # resolver records an episode after it resolves.
    result = run_replay(
        events,
        learn_rules=True,
        use_calibration=True,
        priors_writer=priors_writer,
        resolver_writer=resolver_writer,
    )

    engine = result.engine
    assert engine is not None

    # Rules first: journalling policy keys off which providers have rules.
    for managed in engine.rules.values():
        resolver_writer.upsert_rule(managed.to_memory())

    resolver_writer.upsert_calibration(result.calibration)

    # Provider experience, rewritten in place so it stays bounded. The engine
    # already holds the per-provider tallies it mined patterns from, so this
    # is a rollup of what the resolver observed, not a second source of truth.
    by_provider: dict[str, dict] = {}
    for provider, seen in engine.provider_totals().items():
        resolver_writer.upsert_provider(
            provider,
            predictions=seen.observed,
            correct=seen.completed,
            funded_seen=seen.observed,
            completed_seen=seen.completed,
            last_block=seen.last_block,
            recent=seen.recent,
        )
        by_provider[provider] = {"n": seen.observed, "completed": seen.completed}

    journal = len(memory.client.read_events(limit=100_000))
    return {
        "journal": journal,
        "rules": len(engine.rules),
        "providers": len(by_provider),
        "episodes": len(result.episodes),
        "bytes": memory.size_bytes(),
        "budget": memory.budget(),
    }


def _job(provider: str, client: str, amount: int, *, funded: int, completed: int,
         budget_sets: int) -> DecisionView:
    """The job in front of Priors right now, plus its chain evidence.

    Chain counts are an input, exactly as they would be live - they are public
    and survive deletion. What memory supplies is the rules and calibration.
    """
    return DecisionView(
        job_id=0, provider=provider, client=client,
        funding_block=0, funding_timestamp=0,
        amount=amount, budget_set_count=budget_sets,
        expiry_window=3600, funding_delay=10,
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


def phase2_decide(db: Path, provider: str, client: str, amount: int) -> None:
    """A genuinely fresh session: new process state, new client, no dataset."""
    memory = PriorsMemory(db)
    priors = PriorsWriter(memory)

    recalled = priors.recall_experience(provider, client, amount)

    print(f"  memory returned      : {'experience' if recalled else 'NOTHING'}")
    funded = completed = 0
    if recalled.provider:
        p = recalled.provider
        funded, completed = p["funded_seen"], p["completed_seen"]
        print(f"  provider record      : {completed}/{funded} completed")
    print(f"  rules recalled       : {len(recalled.rules)}")
    for r in recalled.rules:
        print(
            f"      {r.rule_id}  {r.field} {r.op} {r.value}  "
            f"delta {r.delta_logit:+.2f}  weight {r.weight}  [{r.status}]"
        )
    if recalled.calibration:
        obs = recalled.calibration.observed_rate(0.75)
        print(f"  calibration in band  : "
              f"{obs:.3f}" if obs is not None else "  calibration: no support")

    # Chain evidence is public, so it is available either way. Hardcode the
    # figures for the deletion run so both runs face the identical job.
    if not funded:
        funded, completed = 694, 469
    view = _job(provider, client, amount, funded=funded, completed=completed,
                budget_sets=3)

    with_memory = predict(view, rules=recalled.rules,
                          calibration=recalled.calibration)
    chain_only = predict(view)

    print()
    print("  THIS JOB  amount %d, budget renegotiated 3x" % amount)
    print(f"    chain evidence only : {chain_only.p_final:.3f}  "
          f"{chain_only.decision}")
    print(f"    with memory         : {with_memory.p_final:.3f}  "
          f"{with_memory.decision}")
    changed = with_memory.decision != chain_only.decision
    print(f"    memory changed the decision: {'YES' if changed else 'no'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--empty", action="store_true", help="deletion test")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    if not args.empty:
        print("PHASE 1  learn from history and persist")
        stats = phase1_learn(args.dataset, args.db)
        b = stats["budget"]
        print(f"  journal entries      : {stats['journal']:,}")
        print(f"  rules persisted      : {stats['rules']}")
        print(f"  providers persisted  : {stats['providers']}")
        cap = (f"{b['pct']:.1%} of {b['cap'] / 1048576:.0f} MB cap"
               if b["cap"] else f"uncapped, tier {b['tier']}")
        print(f"  store size           : {b['bytes']:,} bytes ({cap})")
    else:
        if args.db.exists():
            args.db.unlink()
        print("PHASE 1  SKIPPED - memory deleted")

    print()
    print("PHASE 2  fresh session: new client, no dataset")
    provider = args.provider or "0xea0f80aac331a0aee486ee67d61f9f4ad7085ee8"
    phase2_decide(args.db, provider, "0x0000000000000000000000000000000000000000", 350_000)


if __name__ == "__main__":
    main()
