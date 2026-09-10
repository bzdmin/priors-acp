"""Regenerate docs/DATA.md from the local ACP dataset.

Every figure quoted in the README, the spec, the Devpost entry, the research
post and the video should come from here rather than being typed by hand.
Re-run after any scanner catch-up:

    python scripts/dataset_stats.py --dataset <path to acp_scan_state_v2.json>

Numbers that live in prose go stale silently. Numbers that are generated
carry the block range that produced them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.events import LocalDatasetSource  # noqa: E402

#: The scan ships with the repo, gzipped, so the figures here regenerate from
#: a clean clone. The uncompressed copy on the author's machine wins if it is
#: there, since it loads faster, but nothing depends on it existing.
def _dataset() -> Path:
    here = Path(__file__).resolve().parent.parent / "data" / "acp_scan_state_v2.json.gz"
    local = Path.home() / "OneDrive/Desktop/acp/acp_scan_state_v2.json"
    return local if local.exists() else here

DEFAULT_DATASET = _dataset()
MIN_FUNDED_FOR_TABLE = 150


def collect(path: Path) -> dict:
    events = list(LocalDatasetSource(path).iter_events())

    created: dict[int, dict] = {}
    completed: dict[int, dict] = {}
    rejected: dict[int, dict] = {}
    funded: dict[int, dict] = {}
    evaluator_fees: dict[int, list[dict]] = defaultdict(list)
    refunded: set[int] = set()
    expired: set[int] = set()
    submitted: set[int] = set()
    budget_sets: Counter[int] = Counter()

    blocks = []
    for e in events:
        blocks.append(e.block)
        k, j = e.kind, e.job_id
        if k == "JobCreated":
            created[j] = e.fields
        elif k == "JobFunded":
            funded[j] = e.fields
        elif k == "JobCompleted":
            completed[j] = e.fields
        elif k == "JobRejected":
            rejected[j] = e.fields
        elif k == "Refunded":
            refunded.add(j)
        elif k == "JobExpired":
            expired.add(j)
        elif k == "JobSubmitted":
            submitted.add(j)
        elif k == "BudgetSet":
            budget_sets[j] += 1
        elif k == "EvaluatorFeePaid":
            evaluator_fees[j].append(e.fields)

    F, C = set(funded), set(completed)
    dead = F - C

    # Per-provider, funded jobs only.
    per_provider: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for j in F:
        c = created.get(j)
        if not c:
            continue
        row = per_provider[c["provider"]]
        row[0] += 1
        if j in C:
            row[1] += 1

    # Rejection attribution, across every rejection.
    rejector_roles: Counter[str] = Counter()
    for j, f in rejected.items():
        c = created.get(j, {})
        r = f.get("rejector")
        if r == c.get("provider"):
            rejector_roles["provider"] += 1
        elif r == c.get("client"):
            rejector_roles["client"] += 1
        else:
            rejector_roles["other"] += 1

    # Evaluator independence, over completed jobs.
    ev = Counter()
    for j, cf in completed.items():
        c = created.get(j)
        if not c:
            continue
        addr = cf.get("evaluator")
        if addr is None or int(addr, 16) == 0:
            ev["absent"] += 1
        elif addr == c["client"]:
            ev["client"] += 1
        elif addr == c["provider"]:
            ev["provider"] += 1
        else:
            ev["third_party"] += 1

    fee_total = fee_to_creator = 0
    for j, fees in evaluator_fees.items():
        c = created.get(j)
        if not c:
            continue
        for f in fees:
            fee_total += 1
            if f.get("recipient") == c["client"]:
                fee_to_creator += 1

    activity: Counter[str] = Counter()
    for f in created.values():
        activity[f["client"]] += 1
        activity[f["provider"]] += 1

    return {
        "blocks": (min(blocks), max(blocks)),
        "events": len(events),
        "created": len(created),
        "funded": len(F),
        "completed_of_funded": len(F & C),
        "dead": len(dead),
        "submitted": len(submitted),
        "failure": {
            "JobRejected": len(dead & set(rejected)),
            "JobExpired": len(dead & expired),
            "Refunded": len(dead & refunded),
        },
        "rejections_total": len(rejected),
        "rejector_roles": rejector_roles,
        "agents": len(activity),
        "agents_5plus": sum(1 for n in activity.values() if n >= 5),
        "agents_20plus": sum(1 for n in activity.values() if n >= 20),
        "top2_clients": sum(
            n for _, n in Counter(f["client"] for f in created.values()).most_common(2)
        ),
        "providers_with_funded": len(per_provider),
        "provider_table": sorted(
            (
                (p, f, c, c / f)
                for p, (f, c) in per_provider.items()
                if f >= MIN_FUNDED_FOR_TABLE
            ),
            key=lambda r: -r[1],
        ),
        "evaluator": ev,
        "evaluator_fee_total": fee_total,
        "evaluator_fee_to_creator": fee_to_creator,
    }


def render(s: dict) -> str:
    n_completed = sum(s["evaluator"].values())
    pct = lambda a, b: f"{a / b:.2%}" if b else "n/a"  # noqa: E731
    lo, hi = s["blocks"]

    rows = "\n".join(
        f"| `{p}` | {f:,} | {c:,} | {r:.2f} |" for p, f, c, r in s["provider_table"]
    )

    return f"""# Dataset facts

**Generated** {dt.date.today().isoformat()} by `scripts/dataset_stats.py`.
**Blocks** {lo:,} - {hi:,}. **Decoded events** {s["events"]:,}.

Do not hand-edit. Do not quote ACP figures anywhere - README, spec, Devpost,
video, research post - that did not come from this file. Re-run the script
after every scanner catch-up.

## Funnel

| | count |
|---|---|
| jobs created | {s["created"]:,} |
| jobs funded | {s["funded"]:,} ({pct(s["funded"], s["created"])} of created) |
| funded and completed | {s["completed_of_funded"]:,} |
| died after funding | {s["dead"]:,} |
| **base rate** P(completion \\| funded) | **{s["completed_of_funded"] / s["funded"]:.4f}** |

## Failure modes

Among the {s["dead"]:,} funded jobs that never completed. These overlap - a
rejection typically triggers a refund - and they are explanatory fields on a
failure episode, never competing labels.

| terminal event | count |
|---|---|
| JobRejected | {s["failure"]["JobRejected"]:,} |
| JobExpired | {s["failure"]["JobExpired"]:,} |
| Refunded | {s["failure"]["Refunded"]:,} |

Across **all** jobs, funded or not, there were {s["rejections_total"]:,}
rejections: {s["rejector_roles"]["provider"]:,} pressed by the provider,
{s["rejector_roles"]["client"]:,} by the client,
{s["rejector_roles"]["other"]:,} by neither.

## Agents

| | count |
|---|---|
| distinct agents (client or provider) | {s["agents"]:,} |
| with 5+ jobs | {s["agents_5plus"]:,} |
| with 20+ jobs | {s["agents_20plus"]:,} |
| providers with 1+ funded job | {s["providers_with_funded"]:,} |

The two busiest clients created {s["top2_clients"]:,} of {s["created"]:,} jobs
({pct(s["top2_clients"], s["created"])}), so the marketplace is heavily
concentrated on the demand side.

## Per-provider completion, funded jobs only

Providers with at least {MIN_FUNDED_FOR_TABLE} funded jobs.

| provider | funded | completed | rate |
|---|---|---|---|
{rows}

## Evaluator independence

Over {n_completed:,} completed jobs.

| evaluator is | count | share |
|---|---|---|
| the client who created the job | {s["evaluator"]["client"]:,} | {pct(s["evaluator"]["client"], n_completed)} |
| absent | {s["evaluator"]["absent"]:,} | {pct(s["evaluator"]["absent"], n_completed)} |
| the provider | {s["evaluator"]["provider"]:,} | {pct(s["evaluator"]["provider"], n_completed)} |
| **a genuine third party** | **{s["evaluator"]["third_party"]:,}** | **{pct(s["evaluator"]["third_party"], n_completed)}** |

Of {s["evaluator_fee_total"]:,} evaluator fee payments,
{s["evaluator_fee_to_creator"]:,}
({pct(s["evaluator_fee_to_creator"], s["evaluator_fee_total"])}) went to the
client who created the job.

ACP defines an evaluator role for independent verification. In practice it is
almost never independent.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "docs/DATA.md")
    args = ap.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"dataset not found: {args.dataset}")

    stats = collect(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(stats), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  blocks {stats['blocks'][0]:,} - {stats['blocks'][1]:,}")
    print(f"  base rate {stats['completed_of_funded'] / stats['funded']:.4f}")


if __name__ == "__main__":
    main()
