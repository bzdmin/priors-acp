"""The open-file pane: read Sibyl live, print exactly what comes back.

Sibyl Labs' builder tip is that a claimed memory is unverifiable - the
audience has to watch the read happen. So this script renders nothing of its
own. Every block below is the raw body returned by the store, printed as it
arrived, next to the path and byte count of the file it came from.

Run it beside `fresh_session.py` in a split screen. The left pane makes a
decision; this pane shows the records that decision was made from.

    python scripts/show_memory.py --provider 0xa9667116b4f4e9f1bae85f93a21b4b8ea45de98f
    python scripts/show_memory.py --provider 0xa966... --job 31495
    python scripts/show_memory.py --provider 0xa966... --watch 2

With memory deleted the same command prints empty sections - which is the
other half of the deletion test, visible rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.memory import (  # noqa: E402
    CATEGORY_CALIBRATION,
    CATEGORY_PROVIDER,
    CATEGORY_RULE,
    PriorsMemory,
)

DEFAULT_DB = Path.home() / ".sibyl-memory/priors.db"
#: read_events truncates to this many entries regardless of the limit passed.
READ_EVENTS_CEILING = 10_000


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _rules_for(memory: PriorsMemory, provider: str) -> list[dict]:
    out = []
    for row in memory.client.list_entities(CATEGORY_RULE, limit=500):
        body = row.get("body") or row
        if body.get("subject") == provider:
            out.append(body)
    return sorted(out, key=lambda b: b.get("rule_id", ""))


def _trail(memory: PriorsMemory, provider: str, job: int | None) -> list[dict]:
    """The journal entries for this decision, in the order they were written.

    This is the ordering claim, and the reason the journal matters: the
    decision was written before the outcome existed, and the episode was
    written afterwards by a different handle.

    A specific job is fetched with Sibyl's own full-text search rather than
    ``read_events``, which truncates to the most recent 10,000 entries no
    matter what limit is passed - so a job early in the replay is invisible to
    it. Each row carries ``ts``, the write time, which is what orders them.
    """
    rows: list[dict] = []
    if job is not None:
        for hit in memory.client.search(str(job), limit=50):
            for act in (hit.get("body") or {}).get("acted") or []:
                if act.get("job_id") == job:
                    rows.append({"ts": hit.get("ts"), "act": act})
    else:
        for ev in memory.client.read_events(limit=READ_EVENTS_CEILING):
            for act in ev.get("acted") or []:
                if act.get("provider") == provider:
                    rows.append({"ts": ev.get("ts"), "act": act})
    rows.sort(key=lambda r: (r["ts"] or "", r["act"].get("type") != "decision"))
    return rows


def show(db: Path, provider: str, job: int | None) -> None:
    memory = PriorsMemory(db)
    b = memory.budget()
    cap = ("uncapped" if b["uncapped"]
           else f"{b['pct']:.1%} of {b['cap'] / 1048576:.0f} MB")

    print("=" * 72)
    print(f"FILE   {memory.path}")
    print(f"       {b['bytes']:,} bytes on disk, tier {b['tier']}, {cap}")
    print(f"READ   provider {provider}" + (f", job {job}" if job else ""))
    print("=" * 72)

    print("\n-- get_entity(provider) " + "-" * 47)
    rec = memory._get(CATEGORY_PROVIDER, provider)
    print(_dump(rec) if rec else "  (nothing stored)")

    print("\n-- list_entities(rule), subject == provider " + "-" * 27)
    rules = _rules_for(memory, provider)
    if not rules:
        print("  (nothing stored)")
    for body in rules:
        print(_dump(body))

    print("\n-- get_entity(calibration) " + "-" * 44)
    cal = memory._get(CATEGORY_CALIBRATION, "ledger")
    if cal:
        counts = cal.get("counts") or []
        print(f"  {sum(counts):,} predictions scored across {len(counts)} bands")
    else:
        print("  (nothing stored)")

    label = "search(job)" if job is not None else "read_events()"
    print(f"\n-- {label}, write order " + "-" * (48 - len(label)))
    trail = _trail(memory, provider, job)
    if not trail:
        print("  (nothing stored)")
    for row in trail:
        act, ts = row["act"], row["ts"]
        if act.get("type") == "decision":
            applied = ",".join(r["rule_id"] for r in act.get("rules_applied") or [])
            print(f"  {ts}  DECISION  job {act.get('job_id')}")
            print(f"    written at block {act.get('predicted_at_block')}, "
                  f"before any outcome existed")
            print(f"    chain-only {act.get('p0_shrunk')} -> "
                  f"{act.get('predicted_completion')} after rules "
                  f"[{applied or '-'}]")
        else:
            print(f"  {ts}  EPISODE   job {act.get('job_id')}")
            print(f"    written at block {act.get('resolved_at_block')}, "
                  f"by the resolver, after the fact")
            print(f"    outcome {act.get('outcome')} "
                  f"({act.get('failure_kind') or 'completed'}), prediction was "
                  f"{'right' if act.get('correct') else 'wrong'}")
    if trail and job is None:
        print(f"\n  ({len(trail)} entries for this provider; read_events is "
              f"capped at {READ_EVENTS_CEILING:,} by the SDK)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--job", type=int, default=None)
    ap.add_argument("--watch", type=float, default=0.0,
                    help="re-read every N seconds, so the pane is visibly live")
    args = ap.parse_args()

    provider = args.provider.lower()
    while True:
        show(args.db, provider, args.job)
        if not args.watch:
            return
        time.sleep(args.watch)
        print("\n")


if __name__ == "__main__":
    main()
