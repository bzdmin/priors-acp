"""Pattern detection and the rule lifecycle (spec 6).

This is the product. Sibyl stores and retrieves; this logic is ours.

    pattern:  support >= 5  AND  abs(delta) >= 0.20
                  |
                  v
            PROVISIONAL          weight 0.5
                  |
            10 applications
                  |
          +-------+-------+
          |               |
       GRADUATE         DEMOTE     back to pattern,
       weight 1.0                  error_pattern written

Promotion is deliberately simple and fully visible: a judge has to be able to
read the evidence behind any rule in seconds. Statistical caution comes from
the provisional stage, not from a complex gate. Both thresholds are display
values, not hidden constants - "we chose 5" is a better answer than a number
buried in code.

Owned by the Outcome Resolver. Priors reads rules; it never writes them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterator, Literal

from .features import DecisionView
from .predict import MAX_DELTA_LOGIT, Rule, logit

#: Displayed in the UI. A subgroup needs this many observations before it is
#: allowed to become a pattern.
MIN_SUPPORT = 5

#: ...and must deviate from the provider's own baseline by at least this much.
MIN_DELTA = 0.20

#: A provisional rule is judged after this many applications.
TRIAL_APPLICATIONS = 10

#: Candidate thresholds for amount-based conditions, in raw USDC units.
#: Deliberately a fixed ladder rather than a fitted split: a fitted threshold
#: on few observations overfits, and a round number is explainable to a judge.
AMOUNT_THRESHOLDS = (10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000)

Status = Literal["pattern", "provisional", "graduated", "demoted"]


@dataclass(slots=True)
class Observation:
    """One resolved job, as evidence for pattern mining.

    Carries every decision-time field a condition may reference. Anything not
    computable strictly before the funding block must never appear here - the
    leakage rule is upstream, in DecisionView, and this must not widen it.
    """

    job_id: int
    provider: str
    amount: int
    repeat_pair: bool
    budget_set_count: int
    completed: bool
    block: int
    # Counterparty and timing context. The dataset's headline finding is that
    # client funding follow-through drives failure, yet the original condition
    # set could not express a single client-side rule.
    client_prior_funding_rate: float | None = None
    client_prior_created: int = 0
    pair_prior_jobs: int = 0
    provider_concurrent_open: int = 0
    expiry_window: int | None = None
    funding_delay: int | None = None


@dataclass(slots=True)
class ProviderTotals:
    """What the resolver has observed about one provider.

    ``recent`` is capped: this becomes the WARM entity's inspectable "last N
    jobs with this provider" list, and must not grow without bound.
    """

    observed: int = 0
    completed: int = 0
    last_block: int = 0
    recent: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class Candidate:
    """A conditional subgroup of one provider's jobs."""

    provider: str
    field: str
    op: str
    value: float | int | bool
    matched: int = 0
    matched_completed: int = 0
    first_block: int | None = None

    @property
    def conditional_rate(self) -> float | None:
        if not self.matched:
            return None
        return self.matched_completed / self.matched


@dataclass(slots=True)
class ManagedRule:
    """A rule plus its lifecycle state and performance record."""

    rule: Rule
    status: Status
    baseline_rate: float
    conditional_rate: float
    support: int
    created_at_block: int
    applications: int = 0
    hits: int = 0
    #: Outcomes of the trailing applications, newest last.
    recent: list[bool] = field(default_factory=list)

    @property
    def rule_id(self) -> str:
        return self.rule.rule_id

    def to_memory(self) -> dict:
        return {
            "type": "rule",
            "rule_id": self.rule_id,
            "subject": self.rule.subject,
            "condition": {
                "field": self.rule.field,
                "op": self.rule.op,
                "value": self.rule.value,
            },
            "delta_logit": round(self.rule.delta_logit, 4),
            "status": self.status,
            "baseline_rate": round(self.baseline_rate, 4),
            "conditional_rate": round(self.conditional_rate, 4),
            "support": self.support,
            "created_at_block": self.created_at_block,
            "applications": self.applications,
            "hits": self.hits,
        }


#: Client funding-reliability bands. The dataset's central finding is that
#: follow-through varies enormously by client, so a provider's completion rate
#: conditioned on who is hiring is the most promising unexploited signal.
CLIENT_RATE_THRESHOLDS = (0.25, 0.50, 0.75)

#: Concurrency: a provider juggling many open jobs may deliver differently.
CONCURRENCY_THRESHOLDS = (1, 3, 10)

#: Short expiry windows, in seconds - how much runway the job was given.
EXPIRY_THRESHOLDS = (3_600, 21_600, 86_400)


def _conditions(obs: Observation) -> Iterator[tuple[str, str, float | int | bool]]:
    """Candidate conditions this observation satisfies.

    Restricted, per spec 3, to amount, counterparty, timing and history. Task
    semantics stay excluded: the event stream carries no task type, so a rule
    phrased in those terms would not be supportable.

    Conditions are a fixed ladder of round thresholds rather than fitted
    splits. A fitted split on five observations overfits, and "above 0.2 USDC"
    is explainable to a judge in a way that "above 0.1873 USDC" is not.
    """
    # Job size
    for t in AMOUNT_THRESHOLDS:
        if obs.amount > t:
            yield ("amount", ">", t)

    # Negotiation history
    yield ("repeat_pair", "==", obs.repeat_pair)
    if obs.budget_set_count > 1:
        yield ("budget_set_count", ">", 1)

    # Who is hiring. A provider that completes reliably for committed clients
    # and stalls for flaky ones is a real pattern this could not previously
    # express.
    if obs.client_prior_funding_rate is not None:
        for t in CLIENT_RATE_THRESHOLDS:
            if obs.client_prior_funding_rate < t:
                yield ("client_prior_funding_rate", "<", t)
    if obs.client_prior_created == 0:
        yield ("client_prior_created", "==", 0)

    # Relationship depth, distinct from the binary repeat_pair.
    if obs.pair_prior_jobs >= 3:
        yield ("pair_prior_jobs", ">=", 3)

    # Load at decision time
    for t in CONCURRENCY_THRESHOLDS:
        if obs.provider_concurrent_open > t:
            yield ("provider_concurrent_open", ">", t)

    # Runway
    if obs.expiry_window is not None:
        for t in EXPIRY_THRESHOLDS:
            if obs.expiry_window < t:
                yield ("expiry_window", "<", t)


class RuleEngine:
    """Mines patterns from resolved jobs and manages the rule lifecycle."""

    def __init__(
        self,
        *,
        min_support: int = MIN_SUPPORT,
        min_delta: float = MIN_DELTA,
        trial: int = TRIAL_APPLICATIONS,
    ) -> None:
        self.min_support = min_support
        self.min_delta = min_delta
        self.trial = trial
        self._provider_n: dict[str, int] = defaultdict(int)
        self._provider_completed: dict[str, int] = defaultdict(int)
        self._totals: dict[str, ProviderTotals] = defaultdict(ProviderTotals)
        #: Episodes retained per provider for the WARM entity's display list.
        self.recent_cap = 5
        self._candidates: dict[tuple, Candidate] = {}
        self.rules: dict[str, ManagedRule] = {}
        self._next_id = 1
        self.demotions: list[str] = []

    # -- reading, for Priors ------------------------------------------------

    def provider_totals(self) -> dict[str, ProviderTotals]:
        """Per-provider observations, for the resolver to persist as WARM
        entities. Derived from episodes, never a competing source of truth."""
        return dict(self._totals)

    def active_rules(self) -> list[Rule]:
        """Rules Priors may apply. Patterns and demotions are not rules."""
        return [
            m.rule
            for m in sorted(self.rules.values(), key=lambda m: m.rule_id)
            if m.status in ("provisional", "graduated")
        ]

    # -- writing, resolver-owned -------------------------------------------

    def observe(self, obs: Observation) -> None:
        """Fold one resolved job into the evidence base, then re-check gates."""
        self._provider_n[obs.provider] += 1
        self._provider_completed[obs.provider] += int(obs.completed)

        t = self._totals[obs.provider]
        t.observed += 1
        t.completed += int(obs.completed)
        t.last_block = max(t.last_block, obs.block)
        t.recent.append(
            {
                "job_id": obs.job_id,
                "amount": obs.amount,
                "outcome": "completed" if obs.completed else "failed",
                "block": obs.block,
            }
        )
        if len(t.recent) > self.recent_cap:
            t.recent.pop(0)

        for fieldname, op, value in _conditions(obs):
            key = (obs.provider, fieldname, op, value)
            cand = self._candidates.get(key)
            if cand is None:
                cand = Candidate(obs.provider, fieldname, op, value)
                cand.first_block = obs.block
                self._candidates[key] = cand
            cand.matched += 1
            cand.matched_completed += int(obs.completed)

        self._promote(obs)

    def _baseline(self, provider: str) -> float | None:
        n = self._provider_n[provider]
        if not n:
            return None
        return self._provider_completed[provider] / n

    def _promote(self, obs: Observation) -> None:
        """Promote any candidate that now clears both gates."""
        baseline = self._baseline(obs.provider)
        if baseline is None:
            return

        for key, cand in self._candidates.items():
            if cand.provider != obs.provider:
                continue
            if cand.matched < self.min_support:
                continue
            rate = cand.conditional_rate
            if rate is None:
                continue
            delta = rate - baseline
            if abs(delta) < self.min_delta:
                continue
            if any(
                m.rule.subject == cand.provider
                and m.rule.field == cand.field
                and m.rule.op == cand.op
                and m.rule.value == cand.value
                for m in self.rules.values()
            ):
                continue

            # Fit the adjustment in logit space and clamp it, so no single
            # rule can saturate a prediction on thin evidence.
            delta_logit = max(
                -MAX_DELTA_LOGIT, min(MAX_DELTA_LOGIT, logit(rate) - logit(baseline))
            )
            rule_id = f"R-{self._next_id}"
            self._next_id += 1
            self.rules[rule_id] = ManagedRule(
                rule=Rule(
                    rule_id=rule_id,
                    subject=cand.provider,
                    field=cand.field,
                    op=cand.op,  # type: ignore[arg-type]
                    value=cand.value,
                    delta_logit=delta_logit,
                    status="provisional",
                ),
                status="provisional",
                baseline_rate=baseline,
                conditional_rate=rate,
                support=cand.matched,
                created_at_block=obs.block,
            )

    def record_application(
        self, rule_id: str, view: DecisionView, completed: bool
    ) -> None:
        """Score one application of a rule against what actually happened.

        A rule is 'right' when the direction it pushed matches the outcome:
        a negative rule that fired on a job that failed was correct.
        """
        m = self.rules.get(rule_id)
        if m is None or m.status not in ("provisional", "graduated"):
            return
        pushed_down = m.rule.delta_logit < 0
        correct = pushed_down != completed
        m.applications += 1
        m.hits += int(correct)
        m.recent.append(correct)
        if len(m.recent) > self.trial:
            m.recent.pop(0)

        if m.status == "provisional" and m.applications >= self.trial:
            self._judge(m, graduating=True)
        elif m.status == "graduated" and len(m.recent) == self.trial:
            self._judge(m, graduating=False)

    def _judge(self, m: ManagedRule, *, graduating: bool) -> None:
        """Apply the same test at graduation and at demotion.

        The rule must have been directionally right more often than the
        provider's own baseline would predict by chance.
        """
        window = m.recent[-self.trial :]
        if not window:
            return
        observed = sum(window) / len(window)
        # Chance level: how often the baseline alone would have got the
        # direction right.
        chance = 1.0 - m.baseline_rate if m.rule.delta_logit < 0 else m.baseline_rate

        if observed > chance:
            if graduating:
                m.status = "graduated"
                m.rule = replace(m.rule, status="graduated")
        else:
            m.status = "demoted"
            m.rule = replace(m.rule, status="provisional")
            self.demotions.append(m.rule_id)

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        by_status: dict[str, int] = defaultdict(int)
        for m in self.rules.values():
            by_status[m.status] += 1
        return {
            "rules_created": len(self.rules),
            **{k: by_status[k] for k in ("provisional", "graduated", "demoted")},
            "candidates_tracked": len(self._candidates),
            "demotions": len(self.demotions),
        }
