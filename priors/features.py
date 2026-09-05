"""Decision-time feature extraction (spec 3) under the leakage rule (spec 2).

The leakage rule is structural here, not a check that could be forgotten.
Events are consumed in one chronological pass and folded into accumulating
state; a view is snapshotted at the moment a job is funded, from state that
has only ever seen earlier events. Nothing downstream is *allowed* to see the
outcome because, at the instant the view is built, the outcome has not been
read yet.

Two kinds of state, separated deliberately:

``JobFacts``
    A job's own attributes - who, how much, expiry. These may share a block
    with the funding event. They are the question being asked, not the answer,
    so they are absorbed before the view is built.

``History``
    Aggregate outcomes per provider, client and pair. This is where leakage
    would actually live, so it is held strictly to blocks earlier than the
    funding block, per spec 2.

Each block is therefore processed in three phases: absorb job attributes,
build views for that block's fundings, then fold in fundings and outcomes.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .events import AcpEvent

TERMINAL_KINDS = frozenset({"JobCompleted", "JobRejected", "Refunded", "JobExpired"})


@dataclass(slots=True)
class JobFacts:
    """What is knowable about a job from its own creation and negotiation."""

    job_id: int
    client: str | None = None
    provider: str | None = None
    evaluator: str | None = None
    expired_at: int | None = None
    created_block: int | None = None
    budget_set_count: int = 0
    last_budget: int = 0


@dataclass(slots=True)
class PartyHistory:
    """Outcome history for one address, as of some block."""

    first_seen_block: int | None = None
    last_active_block: int | None = None
    created: int = 0
    funded: int = 0
    completed: int = 0
    rejected_by_self: int = 0
    expired: int = 0
    open_jobs: set[int] = field(default_factory=set)
    completed_amounts: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DecisionView:
    """Everything Priors may see when deciding on one funded job.

    Contains no outcome for this job. Constructed only from events strictly
    earlier than ``funding_block`` (history) plus the job's own attributes.
    """

    job_id: int
    provider: str
    client: str
    funding_block: int
    funding_timestamp: int | None

    # Job
    amount: int
    budget_set_count: int
    expiry_window: int | None
    funding_delay: int | None

    # Provider, as of t
    provider_prior_funded: int
    provider_prior_completed: int
    provider_prior_rejected_by_self: int
    provider_prior_expired: int
    provider_median_completed_amount: float | None
    provider_blocks_since_first_seen: int | None
    provider_blocks_since_last_activity: int | None
    provider_concurrent_open: int

    # Client, as of t
    client_prior_created: int
    client_prior_funded: int
    client_prior_funding_rate: float | None

    # Relationship
    repeat_pair: bool
    pair_prior_jobs: int
    pair_prior_completed: int

    @property
    def provider_prior_rate(self) -> float | None:
        """Raw historical rate. Shrinkage happens in the model, not here."""
        if not self.provider_prior_funded:
            return None
        return self.provider_prior_completed / self.provider_prior_funded


class ReplayState:
    """Accumulating view of the marketplace as events stream past."""

    def __init__(self) -> None:
        self.jobs: dict[int, JobFacts] = {}
        self.providers: dict[str, PartyHistory] = defaultdict(PartyHistory)
        self.clients: dict[str, PartyHistory] = defaultdict(PartyHistory)
        self.pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.pair_completed: dict[tuple[str, str], int] = defaultdict(int)
        #: Jobs observed reaching JobFunded. Outcome counters are scoped to
        #: these, because some ACP jobs run JobCreated -> JobSubmitted ->
        #: JobCompleted with no JobFunded event at all. Counting those
        #: completions would put a numerator over an unrelated denominator and
        #: silently inflate the shrunk base rate of spec 4, stage 1, whose
        #: target is P(completion | funded).
        self.funded_jobs: set[int] = set()

    # -- phase 1: job attributes -------------------------------------------

    def absorb_attributes(self, ev: AcpEvent) -> None:
        """Fold in a job's own description. Never an outcome."""
        if ev.kind == "JobCreated":
            f = self.jobs.setdefault(ev.job_id, JobFacts(ev.job_id))
            f.client = ev.fields.get("client")  # type: ignore[assignment]
            f.provider = ev.fields.get("provider")  # type: ignore[assignment]
            f.evaluator = ev.fields.get("evaluator")  # type: ignore[assignment]
            f.expired_at = ev.fields.get("expired_at")  # type: ignore[assignment]
            f.created_block = ev.block
            if f.client:
                c = self.clients[f.client]
                c.created += 1
                if c.first_seen_block is None:
                    c.first_seen_block = ev.block
                c.last_active_block = ev.block
            if f.provider:
                p = self.providers[f.provider]
                if p.first_seen_block is None:
                    p.first_seen_block = ev.block
                p.last_active_block = ev.block

        elif ev.kind == "BudgetSet":
            f = self.jobs.setdefault(ev.job_id, JobFacts(ev.job_id))
            f.budget_set_count += 1
            f.last_budget = int(ev.fields.get("amount", 0))  # type: ignore[arg-type]

    # -- phase 2: snapshot --------------------------------------------------

    def build_view(self, funding: AcpEvent) -> DecisionView | None:
        """Snapshot features for a job being funded now.

        Returns None for a funding whose ``JobCreated`` we never saw, which
        happens only for jobs created before the dataset's first block.
        """
        f = self.jobs.get(funding.job_id)
        if f is None or not f.provider or not f.client:
            return None

        p = self.providers[f.provider]
        c = self.clients[f.client]
        pair = (f.client, f.provider)
        prior_pair = self.pairs.get(pair, [])

        amount = int(funding.fields.get("amount", 0))  # type: ignore[arg-type]

        expiry_window = None
        if f.expired_at is not None and funding.timestamp is not None:
            expiry_window = f.expired_at - funding.timestamp

        funding_delay = None
        if f.created_block is not None:
            funding_delay = funding.block - f.created_block

        return DecisionView(
            job_id=funding.job_id,
            provider=f.provider,
            client=f.client,
            funding_block=funding.block,
            funding_timestamp=funding.timestamp,
            amount=amount,
            budget_set_count=f.budget_set_count,
            expiry_window=expiry_window,
            funding_delay=funding_delay,
            provider_prior_funded=p.funded,
            provider_prior_completed=p.completed,
            provider_prior_rejected_by_self=p.rejected_by_self,
            provider_prior_expired=p.expired,
            provider_median_completed_amount=(
                statistics.median(p.completed_amounts) if p.completed_amounts else None
            ),
            provider_blocks_since_first_seen=(
                funding.block - p.first_seen_block if p.first_seen_block else None
            ),
            provider_blocks_since_last_activity=(
                funding.block - p.last_active_block if p.last_active_block else None
            ),
            provider_concurrent_open=len(p.open_jobs),
            client_prior_created=c.created,
            client_prior_funded=c.funded,
            client_prior_funding_rate=(c.funded / c.created if c.created else None),
            repeat_pair=bool(prior_pair),
            pair_prior_jobs=len(prior_pair),
            pair_prior_completed=self.pair_completed.get(pair, 0),
        )

    # -- phase 3: outcomes --------------------------------------------------

    def absorb_outcome(self, ev: AcpEvent) -> None:
        """Fold in a funding or a terminal event. Never called before a view
        for the same block has been built."""
        f = self.jobs.get(ev.job_id)
        if f is None or not f.provider or not f.client:
            return
        p = self.providers[f.provider]
        c = self.clients[f.client]
        pair = (f.client, f.provider)

        if ev.kind == "JobFunded":
            p.funded += 1
            c.funded += 1
            p.open_jobs.add(ev.job_id)
            p.last_active_block = ev.block
            c.last_active_block = ev.block
            self.pairs[pair].append(ev.job_id)
            self.funded_jobs.add(ev.job_id)
            f.last_budget = int(ev.fields.get("amount", 0))  # type: ignore[arg-type]

        elif ev.kind in TERMINAL_KINDS:
            p.open_jobs.discard(ev.job_id)
            p.last_active_block = ev.block
            # Outcomes only count for jobs we saw funded - see funded_jobs.
            if ev.job_id not in self.funded_jobs:
                return
            if ev.kind == "JobCompleted":
                p.completed += 1
                self.pair_completed[pair] += 1
                if f.last_budget:
                    p.completed_amounts.append(f.last_budget)
            elif ev.kind == "JobExpired":
                p.expired += 1
            elif ev.kind == "JobRejected":
                if ev.fields.get("rejector") == f.provider:
                    p.rejected_by_self += 1


def iter_decision_points(events: Iterable[AcpEvent]) -> Iterator[DecisionView]:
    """Walk the event stream and yield one view per funded job, in funding order.

    The three-phase per-block structure is what enforces spec 2: history state
    has provably not seen block ``B`` when a view for a funding in block ``B``
    is built.
    """
    block: int | None = None
    pending: list[AcpEvent] = []
    state = ReplayState()

    def flush() -> Iterator[DecisionView]:
        # 1. job attributes
        for e in pending:
            state.absorb_attributes(e)
        # 2. snapshot every funding in this block
        for e in pending:
            if e.kind == "JobFunded":
                view = state.build_view(e)
                if view is not None:
                    yield view
        # 3. fold in outcomes
        for e in pending:
            state.absorb_outcome(e)

    for ev in events:
        if block is not None and ev.block != block:
            yield from flush()
            pending.clear()
        block = ev.block
        pending.append(ev)

    if pending:
        yield from flush()
