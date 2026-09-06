"""Follow ACP on Base and decide live, on the same code path as the backtest.

History and live are one stream. The scanned dataset is replayed first to
rebuild provider history and the rules Priors has earned, and the moment it
runs out the chain source takes over at the block the scan stopped at. Because
``run_replay`` groups by block and cannot tell the two apart, there is no
"live mode" branch to diverge from the tested one - the determinism guarantee
and the leakage rule hold for exactly the same reason they hold in replay.

Everything printed below the handover line is a decision made about a job
whose outcome did not exist when the decision was written.

    python scripts/live_tail.py                    # warm, then follow
    python scripts/live_tail.py --catch-up-only    # warm, drain to head, stop
    python scripts/live_tail.py --no-write         # decide, write nothing

Ctrl+C stops it. The checkpoint means a restart resumes at the last fully
processed block rather than re-deciding jobs already in the journal.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.chain import ChainSource  # noqa: E402
from priors.events import LocalDatasetSource, usdc  # noqa: E402
from priors.memory import PriorsMemory, PriorsWriter, ResolverWriter  # noqa: E402
from priors.replay import run_replay  # noqa: E402

DEFAULT_DATASET = Path.home() / "OneDrive/Desktop/acp/acp_scan_state_v2.json"
DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"
DEFAULT_CHECKPOINT = Path.home() / ".sibyl-memory/live_cursor.json"


def journal_high_water(memory: PriorsMemory) -> int:
    """Highest block already recorded in the journal, or 0 if it is empty.

    The warm-up has to replay history to rebuild provider counters and the
    rules Priors earned - there is nowhere else that state comes from. But the
    journal is append-only, so writing during the warm-up would record all
    18,367 historical decisions a second time on top of the ones already
    there, and every count published from the store would be wrong.

    So the replay runs, and the writers below drop anything at or below this
    block. Entity upserts are unaffected: those are keyed writes, not appends.
    """
    high = 0
    for ev in memory.iter_journal():
        for act in ev.get("acted") or []:
            blk = act.get("predicted_at_block") or act.get("resolved_at_block")
            if blk and blk > high:
                high = blk
    return high


class _GatedPriorsWriter:
    """Priors' writer, silent at or below ``after_block``."""

    def __init__(self, inner: PriorsWriter, after_block: int) -> None:
        self._inner, self._after = inner, after_block

    def recall_experience(self, *a, **kw):
        return self._inner.recall_experience(*a, **kw)

    def write_decision(self, view, prediction) -> None:
        if view.funding_block > self._after:
            self._inner.write_decision(view, prediction)


class _GatedResolverWriter:
    """The resolver's writer, silent at or below ``after_block``.

    Only ``write_episode`` is gated. The upserts are idempotent by key, so
    replaying them just rewrites the same record with the same content.
    """

    def __init__(self, inner: ResolverWriter, after_block: int) -> None:
        self._inner, self._after = inner, after_block

    def write_episode(self, episode, provider) -> None:
        if episode.resolved_at_block > self._after:
            self._inner.write_episode(episode, provider)

    def upsert_rule(self, body) -> None:
        self._inner.upsert_rule(body)

    def upsert_provider(self, *a, **kw) -> None:
        self._inner.upsert_provider(*a, **kw)

    def upsert_calibration(self, ledger) -> None:
        self._inner.upsert_calibration(ledger)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--rpc", default=None, help="override the Base RPC endpoint")
    ap.add_argument("--poll", type=float, default=12.0)
    ap.add_argument("--catch-up-only", action="store_true",
                    help="drain to the chain head and stop, instead of following")
    ap.add_argument("--no-write", action="store_true",
                    help="decide but persist nothing (dry run)")
    ap.add_argument("--since-block", type=int, default=None,
                    help="only journal blocks above this; default is the "
                         "journal's own high-water mark, so a writing run "
                         "cannot duplicate history it already recorded")
    args = ap.parse_args()

    local = LocalDatasetSource(args.dataset)
    handover = local.cursor()

    chain = ChainSource(
        handover,
        rpc_url=args.rpc,
        checkpoint=None if args.no_write else args.checkpoint,
    )
    head = chain.head()

    print(f"warming from  {args.dataset}")
    print(f"scan ends at  block {handover:,}")
    print(f"chain head    block {head:,}  ({head - handover:,} blocks behind)")
    print(f"following at  {chain.confirmations} confirmations deep")
    if args.no_write:
        print("DRY RUN       nothing will be written to memory or the checkpoint")
    print()

    priors_writer = resolver_writer = None
    memory = None
    if not args.no_write:
        memory = PriorsMemory(args.db)
        after = (args.since_block if args.since_block is not None
                 else journal_high_water(memory))
        print(f"journalling     blocks above {after:,}"
              f"{' (journal high-water)' if args.since_block is None else ''}")
        priors_writer = _GatedPriorsWriter(PriorsWriter(memory), after)
        resolver_writer = _GatedResolverWriter(ResolverWriter(memory), after)

    live = {"decisions": 0, "episodes": 0, "announced": False}

    def announce() -> None:
        if not live["announced"]:
            live["announced"] = True
            print("-" * 72)
            print("LIVE  past this line, no outcome existed when the decision "
                  "was written")
            print("-" * 72)

    def on_decision(view, prediction) -> None:
        # Historical decisions are the warm-up and are not worth printing;
        # 18,000 of them would bury the ones that matter.
        if view.funding_block <= handover:
            return
        announce()
        live["decisions"] += 1
        rules = ",".join(a.rule_id for a in prediction.rules_applied) or "-"
        print(f"  job {view.job_id:<7} block {view.funding_block:,}  "
              f"{usdc(view.amount):>8.2f} USDC  {view.provider[:10]}...")
        print(f"    chain-only {prediction.p0_shrunk:.3f}  ->  "
              f"{prediction.p_final:.3f} after [{rules}]   {prediction.decision}")

    def on_episode(episode, view) -> None:
        if episode.resolved_at_block <= handover:
            return
        announce()
        live["episodes"] += 1
        # Whether the call was right is `episode.correct`, not whether the job
        # succeeded - predicting a failure that then happens is a correct call.
        print(f"  job {episode.job_id:<7} RESOLVED {episode.outcome:<8} "
              f"call was {'right' if episode.correct else 'wrong'}"
              f"  (brier {episode.brier:.4f})")

    events = itertools.chain(
        local.iter_events(),
        chain.iter_events() if args.catch_up_only else chain.follow(args.poll),
    )

    try:
        result = run_replay(
            events,
            priors_writer=priors_writer,
            resolver_writer=resolver_writer,
            on_decision=on_decision,
            on_episode=on_episode,
        )
    except KeyboardInterrupt:
        print("\nstopped.")
        result = None

    print()
    print(f"live decisions written : {live['decisions']}")
    print(f"live outcomes resolved : {live['episodes']}")
    print(f"cursor now             : block {chain.cursor:,}")
    if result is not None and result.engine is not None and resolver_writer:
        # Persist the rule set as it stands, so the next run starts from what
        # this one learned rather than remining it.
        for managed in result.engine.rules.values():
            resolver_writer.upsert_rule(managed.to_memory())
        print(f"rules persisted        : {len(result.engine.rules)}")
    if memory is not None:
        print(f"store                  : {memory.size_bytes():,} bytes")


if __name__ == "__main__":
    main()
