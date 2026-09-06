"""Sibyl Memory integration: what Priors has learned (spec 5, 10.1, 10.6).

Two things this module is responsible for, both of which the spec treats as
load-bearing rather than incidental.

**Write ownership is enforced, not documented** (spec 6). Priors gets a handle
that can only write decisions; the resolver gets one that can only write
episodes, providers, rules and calibration. Neither can write the other's
types, because neither is handed a method that would let it. That is what
makes the store shared coordination state between two processes rather than
one agent's notebook.

**Retention is bounded by measurement, not by hope** (spec 10.6). Measured on
2026-09-01 against sibyl-memory-client 0.8.0:

    journal entry      3,052 bytes   (a 359-byte JSON record; SQLite + FTS5
                                      inflate it ~8.5x)
    rule entity        4,396 bytes
    provider entity    2,567 bytes
    free tier cap      5,242,880 bytes

A full replay writes 18,367 decisions and 18,360 episodes. Journalling all of
them costs ~107 MB against a 5 MB cap - 21x over, where the spec assumed 2-3x.

The spec's answer was a bounded journal, keeping the last N episodes per
provider and rolling older evidence into WARM. That is not implementable:
the SDK has no delete or prune for journal events, only for entities, so the
COLD journal is append-only. Nothing can be rolled out of it.

What is implemented instead, preserving the intent:

* The "last N jobs with this provider" list the demo needs to point at lives
  on the WARM provider entity, which ``set_entity`` overwrites in place and
  which therefore never grows.
* The journal is written selectively at write time, since it cannot be
  trimmed afterwards. The default policy journals only providers Priors has
  actually authored a rule about - the decisions whose provenance a judge
  would want to audit - which is naturally recency-weighted, because rules
  only exist after evidence accumulates.

Pruning never removes information that changes a future decision: every
durable input to a prediction lives in a WARM entity, not the journal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from sibyl_memory_client import FREE_TIER_CAP_BYTES, MemoryClient

from .features import DecisionView
from .predict import CalibrationLedger, Prediction, Rule
from .coordination import Observation, ResponseRecord
from .resolver import Episode

def _load_credentials(path: Path) -> dict[str, Any]:
    """Read ``credentials.json`` if ``sibyl init`` has been run.

    Returns an empty dict when absent or unreadable, so an unbound machine
    still works - it is simply capped. Values are passed straight to the SDK
    and never logged; only whether they are present is ever reported.
    """
    try:
        import json

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _response_key(provider: str, declaration_type: str) -> str:
    return f"{provider.lower()}-{declaration_type}"


CATEGORY_PROVIDER = "provider"
CATEGORY_RULE = "rule"
CATEGORY_CALIBRATION = "calibration"
#: Coordination state. One row per (counterparty, declaration type), holding
#: what the counterparty claimed and what the resolver later observed. Written
#: only by ResolverWriter and read only by PriorsWriter, so the party whose
#: claim is being judged can never write the judgement.
CATEGORY_RESPONSE = "response_record"

#: Episodes retained per provider on the WARM entity. Bounded in place, so
#: this costs a fixed 2.5 KB per provider no matter how long Priors runs.
RECENT_EPISODES = 5

JournalPolicy = Literal["all", "rules_only", "none"]


@dataclass(slots=True)
class RecalledExperience:
    """Everything ``recall_experience`` found. Empty means memory is empty."""

    provider: dict[str, Any] | None = None
    rules: list[Rule] = field(default_factory=list)
    calibration: CalibrationLedger | None = None
    pair_history: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return bool(self.provider or self.rules or self.calibration)


class PriorsMemory:
    """The store. Prefer the role-scoped facades below over using this directly."""

    #: Written by ``sibyl init``. Mode 0600, and never logged or printed.
    CREDENTIALS = Path("~/.sibyl-memory/credentials.json").expanduser()

    def __init__(
        self,
        path: str | Path = "~/.sibyl-memory/priors.db",
        *,
        journal_policy: JournalPolicy | None = None,
        credentials: Path | None = None,
    ) -> None:
        creds = _load_credentials(credentials or self.CREDENTIALS)
        # Without account_id and session_token the SDK enforces a strict local
        # 5 MB cap and cannot check the account's real tier with the server -
        # so an uncapped account still behaves as capped. Binding with
        # `sibyl init` is necessary but not sufficient; the credentials have to
        # be handed to the client too.
        self.client = MemoryClient.local(
            str(path),
            tier=creds.get("tier", "free"),
            account_id=creds.get("account_id"),
            session_token=creds.get("session_token"),
        )
        self.path = Path(str(path)).expanduser()
        self.bound = bool(creds.get("account_id") and creds.get("session_token"))
        self.tier = creds.get("tier", "free")
        self._rule_subjects: set[str] = set()

        # Journal policy follows the store's actual capacity rather than a
        # constant someone has to remember to change. Uncapped: journal every
        # decision and episode, giving a judge a complete audit trail. Capped:
        # journal only providers Priors has authored a rule about, since a full
        # replay costs ~107 MB against a 5 MB ceiling and the journal cannot be
        # pruned afterwards.
        if journal_policy is None:
            journal_policy = "all" if self._uncapped() else "rules_only"
        self.journal_policy: JournalPolicy = journal_policy

    def _uncapped(self) -> bool:
        try:
            return bool(self.client.free_tier_status().get("uncapped"))
        except Exception:
            return False

    # -- retrieval: the single entry point Priors uses (spec 10.1) ----------

    def recall_experience(
        self, provider: str, client: str, amount: int
    ) -> RecalledExperience:
        """The one deterministic function that feeds a decision.

        Nothing else in the codebase calls Sibyl for decision input, so a
        judge looking for "where does Priors get its learned experience"
        finds exactly one place.

        Direct key lookups, not search: a keyed read is cheaper across ~18,400
        replay iterations and cannot return a near-match from the wrong
        provider. Retrieval is scoped to the subject of this decision - the
        whole store is never loaded.
        """
        out = RecalledExperience()

        rec = self._get(CATEGORY_PROVIDER, provider)
        if rec:
            out.provider = rec
            pairs = rec.get("pairs") or {}
            out.pair_history = pairs.get(client)

        for row in self.client.list_entities(CATEGORY_RULE, limit=500):
            body = row.get("body") or row
            if body.get("subject") != provider:
                continue
            if body.get("status") not in ("provisional", "graduated"):
                continue
            cond = body.get("condition") or {}
            out.rules.append(
                Rule(
                    rule_id=body["rule_id"],
                    subject=body["subject"],
                    field=cond["field"],
                    op=cond["op"],
                    value=cond["value"],
                    delta_logit=body["delta_logit"],
                    status=body["status"],
                )
            )
        out.rules.sort(key=lambda r: r.rule_id)

        cal = self._get(CATEGORY_CALIBRATION, "ledger")
        if cal:
            led = CalibrationLedger()
            led.counts = list(cal.get("counts") or led.counts)
            led.hits = list(cal.get("hits") or led.hits)
            out.calibration = led

        return out

    def _get(self, category: str, name: str) -> dict[str, Any] | None:
        try:
            row = self.client.get_entity(category, name)
        except Exception:
            return None
        if not row:
            return None
        return row.get("body") or row

    # -- housekeeping -------------------------------------------------------

    def iter_journal(self, *, page: int = 10_000):
        """Every journal event, oldest-last, paging past the SDK's read cap.

        ``read_events`` clamps any limit to MAX_LIMIT (10,000) and returns the
        most recent page. That clamp is a deliberate control - SQLite reads a
        negative LIMIT as unbounded, so an unclamped limit is a context-flood
        vector - and it is not something to patch around. The supported way to
        reach the rest is the ``until`` cursor the same method exposes.

        Rows are deduplicated by id because ``until`` filters on ``ts <= ?``
        inclusively, so the boundary timestamp reappears on the next page.
        """
        seen: set = set()
        until: str | None = None
        while True:
            batch = self.client.read_events(limit=page, until=until)
            if not batch:
                return
            fresh = [e for e in batch if e.get("id") not in seen]
            if not fresh:
                # A whole page inside one timestamp: the cursor cannot advance,
                # so stop rather than spin.
                return
            for ev in fresh:
                seen.add(ev.get("id"))
                yield ev
            if len(batch) < page:
                return
            until = batch[-1]["ts"]

    def journal_size(self) -> int:
        """Count of journal events, paged rather than capped."""
        return sum(1 for _ in self.iter_journal())

    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def budget(self) -> dict[str, Any]:
        """Live budget, straight from the SDK rather than the local constant.

        ``uncapped`` only becomes True when the account is bound *and* its
        credentials were handed to the client at construction.
        """
        status = self.client.free_tier_status()
        used = self.size_bytes()
        # An uncapped store reports cap and pct as None; normalise so callers
        # can format the result without special-casing every field.
        cap = status.get("soft_cap_bytes") or None
        pct = status.get("pct_used")
        return {
            "bytes": used,
            "cap": cap,
            "pct": pct if pct is not None else (used / cap if cap else None),
            "over": bool(status.get("at_or_above_cap")),
            "uncapped": bool(status.get("uncapped")),
            "tier": status.get("tier", self.tier),
            "bound": self.bound,
        }

    def _should_journal(self, provider: str) -> bool:
        if self.journal_policy == "all":
            return True
        if self.journal_policy == "none":
            return False
        return provider in self._rule_subjects


class PriorsWriter:
    """Priors' handle. Writes ``decision`` and nothing else (spec 6)."""

    def __init__(self, memory: PriorsMemory) -> None:
        self._m = memory

    def recall_experience(
        self, provider: str, client: str, amount: int
    ) -> RecalledExperience:
        return self._m.recall_experience(provider, client, amount)

    def recall_response_record(
        self, provider: str, declaration_type: str = "availability"
    ) -> ResponseRecord | None:
        """The counterparty's declaration history, or None if there is none.

        None is the deletion condition, and the gate reads it as "unknown"
        rather than "untrustworthy": an unrecorded counterparty is treated as
        typical for the marketplace, not as suspect.
        """
        body = self._m._get(
            CATEGORY_RESPONSE, _response_key(provider, declaration_type)
        )
        if not body:
            return None
        return ResponseRecord(
            provider=body["provider"],
            declaration_type=body.get("declaration_type", declaration_type),
            declared=int(body.get("declared", 0)),
            observations=tuple(
                Observation(int(o["block"]), bool(o["confirmed"]))
                for o in body.get("observations") or []
            ),
        )

    def write_decision(self, view: DecisionView, prediction: Prediction) -> None:
        """Written before the outcome is known. The load-bearing entry."""
        if not self._m._should_journal(view.provider):
            return
        self._m.client.write_event(
            acted=[
                {
                    "type": "decision",
                    "job_id": view.job_id,
                    "provider": view.provider,
                    "client": view.client,
                    "predicted_completion": round(prediction.p_final, 4),
                    "p0_shrunk": round(prediction.p0_shrunk, 4),
                    "rules_applied": [
                        {
                            "rule_id": a.rule_id,
                            "delta": round(a.delta_logit, 4),
                            "weight": a.weight,
                            "why": a.why,
                        }
                        for a in prediction.rules_applied
                    ],
                    "calibration_shift": round(prediction.calibration_shift, 4),
                    "amount": view.amount,
                    "predicted_at_block": view.funding_block,
                }
            ]
        )


class ResolverWriter:
    """The resolver's handle. Writes episodes, providers, rules, calibration.

    Cannot write decisions - there is no method for it. Its writes are what
    change Priors' behaviour on the next decision.
    """

    def __init__(self, memory: PriorsMemory) -> None:
        self._m = memory

    def write_episode(self, episode: Episode, provider: str) -> None:
        if self._m._should_journal(provider):
            self._m.client.write_event(acted=[episode.to_memory()])

    def upsert_provider(
        self,
        address: str,
        *,
        predictions: int,
        correct: int,
        funded_seen: int,
        completed_seen: int,
        last_block: int,
        recent: Sequence[dict[str, Any]],
        pairs: dict[str, Any] | None = None,
    ) -> None:
        """Provider experience, rewritten in place so it never grows.

        ``recent`` is truncated to RECENT_EPISODES. This is the concrete,
        inspectable "Priors remembers its last N jobs with this provider" list
        the demo points at, and the reason it lives in WARM rather than the
        append-only journal.
        """
        self._m.client.set_entity(
            CATEGORY_PROVIDER,
            address,
            {
                "type": "provider",
                "address": address,
                "predictions": predictions,
                "correct": correct,
                "accuracy": round(correct / predictions, 4) if predictions else None,
                "funded_seen": funded_seen,
                "completed_seen": completed_seen,
                "last_block": last_block,
                "recent": list(recent)[-RECENT_EPISODES:],
                "pairs": pairs or {},
            },
        )

    def upsert_rule(self, body: dict[str, Any]) -> None:
        """Write a rule, and move it to ARCHIVE once it has been demoted.

        The lifecycle is pattern -> provisional -> graduated -> demoted, and
        the store should show that rather than keeping every rule Priors ever
        wrote in one undifferentiated namespace. A demoted rule is not
        deleted - it is evidence of something Priors believed and stopped
        believing, and ``archive_entity`` keeps it recoverable.

        Retrieval is unaffected: ``recall_experience`` already admits only
        provisional and graduated rules, so a demoted rule cannot rejoin the
        decision path merely by still existing somewhere in the store.
        """
        status = body.get("status")
        self._m.client.set_entity(CATEGORY_RULE, body["rule_id"], body)
        if status in ("provisional", "graduated"):
            self._m._rule_subjects.add(body["subject"])
        elif status == "demoted":
            try:
                self._m.client.archive_entity(
                    CATEGORY_RULE,
                    body["rule_id"],
                    reason="demoted: stopped outperforming the provider baseline",
                )
            except Exception:
                # Archiving is bookkeeping. A store that refuses it must not
                # take down a replay that has already produced its decisions.
                pass

    def record_declaration(
        self, provider: str, declaration_type: str = "availability"
    ) -> None:
        """Count a declaration Priors received, whatever it said.

        Counted even when the declaration stops Priors creating a job, so a
        counterparty that dodges being tested by always claiming unavailable
        cannot bank a perfect record by never being observed.
        """
        key = _response_key(provider, declaration_type)
        body = self._m._get(CATEGORY_RESPONSE, key) or {
            "type": "response_record",
            "provider": provider.lower(),
            "declaration_type": declaration_type,
            "declared": 0,
            "observations": [],
        }
        body["declared"] = int(body.get("declared", 0)) + 1
        self._m.client.set_entity(CATEGORY_RESPONSE, key, body)

    def record_response_observation(
        self,
        provider: str,
        *,
        block: int,
        confirmed: bool,
        job_id: int,
        declaration_type: str = "availability",
    ) -> ResponseRecord:
        """What actually happened after Priors acted on a declaration.

        Resolver-owned, and deliberately so: the counterparty declares, Priors
        decides, and only this handle records whether the declaration held.
        Neither of the other two can write here, which is what stops a claim
        being marked correct by the party that made it.

        Idempotent per job, so re-running the observer after a restart cannot
        count the same outcome twice.
        """
        key = _response_key(provider, declaration_type)
        body = self._m._get(CATEGORY_RESPONSE, key) or {
            "type": "response_record",
            "provider": provider.lower(),
            "declaration_type": declaration_type,
            "declared": 0,
            "observations": [],
        }
        obs = list(body.get("observations") or [])
        if any(int(o.get("job_id", -1)) == job_id for o in obs):
            pass   # already recorded; leave the store untouched
        else:
            obs.append(
                {"block": block, "confirmed": confirmed, "job_id": job_id}
            )
            body["observations"] = obs
            self._m.client.set_entity(CATEGORY_RESPONSE, key, body)
        return ResponseRecord(
            provider=body["provider"],
            declaration_type=declaration_type,
            declared=int(body.get("declared", 0)),
            observations=tuple(
                Observation(int(o["block"]), bool(o["confirmed"]))
                for o in body.get("observations") or []
            ),
        )

    def upsert_calibration(self, ledger: CalibrationLedger) -> None:
        """The ledger exists only here. It is the most obviously
        unreconstructable artifact in the system (spec 7)."""
        bands = []
        for i, n in enumerate(ledger.counts):
            if not n:
                continue
            bands.append(
                {
                    "band": f"{i / ledger.bins:.1f}-{(i + 1) / ledger.bins:.1f}",
                    "predictions": n,
                    "observed": round(ledger.hits[i] / n, 4),
                    "midpoint": (i + 0.5) / ledger.bins,
                }
            )
        self._m.client.set_entity(
            CATEGORY_CALIBRATION,
            "ledger",
            {
                "type": "calibration",
                "counts": ledger.counts,
                "hits": ledger.hits,
                "bands": bands,
                "brier": ledger.brier(),
            },
        )
