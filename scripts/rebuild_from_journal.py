"""Destroy the summary, keep the journal, and see whether the numbers survive.

Priors stores two things. The COLD journal is append-only: one decision per
funded job, one episode per resolved job, never edited. The WARM entities are a
summary: per-provider counters, rules, the calibration ledger, each rewritten in
place every time something changes.

The summary is not the truth. It is a cache of the truth, and the difference
matters because the journal is written first. If the process dies between the
two writes the record of what happened survives and the cache can be rebuilt.

Plenty of systems claim their log is authoritative and then write it last. This
checks the claim rather than making it: it reads only the journal, recomputes
the headline figures from the episodes in it, and compares them against what the
replay published. Nothing here reads a provider record, a rule, or any other
summary.

    python scripts/rebuild_from_journal.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.memory import PriorsMemory  # noqa: E402

DEFAULT_DB = Path.home() / ".sibyl-memory" / "priors.db"

#: What the replay published, from scripts/ablation.py run A.
PUBLISHED = {"accuracy": 0.9035, "brier": 0.0666, "decisions": 18367}


def bodies(event: dict):
    body = event.get("body") or event.get("acted") or event
    return body if isinstance(body, list) else [body]


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    print("=" * 68)
    print("REBUILD FROM JOURNAL")
    print(f"store {db}")
    print("reading the append-only journal, and nothing else")
    print("=" * 68)

    t0 = time.time()
    memory = PriorsMemory(db)
    episodes, decisions = [], 0
    for event in memory.iter_journal():
        for b in bodies(event):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "episode":
                episodes.append(b)
            elif b.get("type") == "decision":
                decisions += 1
    took = time.time() - t0

    if not episodes:
        print("\nno episodes in the journal, nothing to rebuild")
        return 1

    correct = sum(1 for e in episodes if e.get("correct"))
    brier = sum(float(e.get("brier", 0.0)) for e in episodes) / len(episodes)
    accuracy = correct / len(episodes)

    print(f"\n  journal events read     : {len(episodes) + decisions:,} in {took:.1f}s")
    print(f"  decisions               : {decisions:,}")
    print(f"  episodes                : {len(episodes):,}")
    print()
    print(f"{'':<26}{'from the journal':>18}{'published':>14}{'':>8}")
    print("-" * 68)

    rows = [
        ("accuracy", accuracy, PUBLISHED["accuracy"], 0.0005),
        ("Brier", brier, PUBLISHED["brier"], 0.0005),
        ("decisions", decisions, PUBLISHED["decisions"], 0),
    ]
    ok = True
    for name, got, want, tol in rows:
        if isinstance(want, int):
            agree = got == want
            print(f"  {name:<24}{got:>18,}{want:>14,}{'  match' if agree else '  DIFFERS':>8}")
        else:
            agree = abs(got - want) <= tol
            print(f"  {name:<24}{got:>18.4f}{want:>14.4f}{'  match' if agree else '  DIFFERS':>8}")
        ok = ok and agree

    print()
    if ok:
        print("  The summary can be deleted. These figures come from the journal,")
        print("  which nothing rewrites, and they are the ones already published.")
    else:
        print("  A figure does not match. Either the journal is incomplete or the")
        print("  published number is stale. Both are worth knowing.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
