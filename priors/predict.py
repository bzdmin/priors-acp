"""The three-stage prediction (spec 4).

Pure arithmetic. No I/O, no clock, no randomness, and - binding, per spec
10.3 - no model calls. A language model may write the human-readable
explanation of a decision *after* ``p_final`` exists, and may draft candidate
pattern descriptions for the resolver, but nothing here may be produced or
adjusted by one. If a model could move the number, the replay would not be
reproducible and a judge re-running the build would get different answers.

Every stage records what it contributed. That record is what the demo card
renders, and it is precisely what vanishes when memory is empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

from .features import DecisionView

#: Global P(completion | funded). Regenerate with scripts/dataset_stats.py
#: rather than editing by hand; see docs/DATA.md.
BASE_RATE = 0.7947

#: Shrinkage strength for stage 1, in pseudo-observations. k = 10 means a
#: provider needs real volume before its own rate outruns the global one.
SHRINKAGE_K = 10.0

#: Rule deltas are clamped so no single rule can saturate a prediction.
MAX_DELTA_LOGIT = 2.0

PROVISIONAL_WEIGHT = 0.5
GRADUATED_WEIGHT = 1.0

#: Probabilities are held inside this band so logit() stays finite.
_EPS = 1e-6

RuleStatus = Literal["provisional", "graduated"]


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    # Branch to avoid overflow on large-magnitude inputs.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True, slots=True)
class Rule:
    """A promoted pattern (spec 5, 6).

    ``delta_logit`` is fitted on the pattern's supporting observations and
    clamped to +/- MAX_DELTA_LOGIT.
    """

    rule_id: str
    subject: str  # provider address
    field: str  # DecisionView attribute
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float | int | bool
    delta_logit: float
    status: RuleStatus = "provisional"

    @property
    def weight(self) -> float:
        return GRADUATED_WEIGHT if self.status == "graduated" else PROVISIONAL_WEIGHT

    def matches(self, view: DecisionView) -> bool:
        """Whether this rule applies to the job in ``view``.

        A rule is scoped to its subject provider and to one comparable field.
        A missing or non-comparable field means no match - never a guess.
        """
        if view.provider != self.subject:
            return False
        actual = getattr(view, self.field, None)
        if actual is None:
            return False
        try:
            if self.op == ">":
                return actual > self.value
            if self.op == ">=":
                return actual >= self.value
            if self.op == "<":
                return actual < self.value
            if self.op == "<=":
                return actual <= self.value
            if self.op == "==":
                return actual == self.value
            if self.op == "!=":
                return actual != self.value
        except TypeError:
            return False
        return False


@dataclass(frozen=True, slots=True)
class AppliedRule:
    """One rule's contribution, logged for the demo card."""

    rule_id: str
    delta_logit: float
    weight: float
    why: str

    @property
    def effective(self) -> float:
        return self.delta_logit * self.weight


@dataclass(frozen=True, slots=True)
class Prediction:
    """``p_final`` and the ordered list of adjustments that produced it.

    Stage 3 is reported, not applied (see ``predict``). ``p_final`` is the
    number Priors decides on; ``p_calibrated`` is what the calibration ledger
    would say about it, carried for display.
    """

    p_final: float
    p0_shrunk: float
    p_after_rules: float
    calibration_shift: float
    #: What stage 3 would produce if it were applied. Display only.
    p_calibrated: float = 0.0
    #: Observed completion rate in this probability band, or None when the
    #: band lacks support. This is the "I said 0.80, I observed 0.61" line on
    #: the demo card - unreconstructable from chain data, and the first thing
    #: to vanish when memory is empty.
    calibration_observed: float | None = None
    rules_applied: tuple[AppliedRule, ...] = ()
    #: The bar this decision was judged against. 0.5 is the naive default with
    #: no learned policy; anything else was derived from Priors' own resolved
    #: outcomes. This is memory changing the *policy*, not the estimate.
    hire_threshold: float = 0.5
    #: True when memory contributed anything at all - a rule, a calibration
    #: observation, or a learned bar. False is the deletion-test condition.
    memory_contributed: bool = False

    @property
    def decision(self) -> str:
        return "HIRE" if self.p_final >= self.hire_threshold else "DO NOT HIRE"

    @property
    def naive_decision(self) -> str:
        """What a fixed 0.5 bar would have said, for the counterfactual."""
        return "HIRE" if self.p_final >= 0.5 else "DO NOT HIRE"


def shrunk_base_rate(
    completed: int,
    funded: int,
    *,
    base_rate: float = BASE_RATE,
    k: float = SHRINKAGE_K,
) -> float:
    """Stage 1. Pull a provider's own rate toward the global one.

    Cold start is handled by trusting the global rate rather than three
    observations: a provider with 2 of 2 gets ~0.83, not 1.00.

    This is the only stage available with memory empty, and it uses chain
    counts alone.
    """
    a = base_rate * k
    b = (1.0 - base_rate) * k
    return (completed + a) / (funded + a + b)


def apply_rules(
    p0: float, view: DecisionView, rules: Iterable[Rule]
) -> tuple[float, tuple[AppliedRule, ...]]:
    """Stage 2. Logit-space adjustment from every matching rule.

    Rules are applied in ``rule_id`` order so that two runs with the same rule
    set produce the same number regardless of retrieval ordering (spec 10.3).
    """
    matching = sorted(
        (r for r in rules if r.matches(view)), key=lambda r: r.rule_id
    )
    if not matching:
        return p0, ()

    # Nested thresholds on the same field are not independent evidence. A
    # provider whose jobs above 50k are all also above 200k produces three
    # rules describing the same observations; applying all three triples the
    # weight of a single finding and was, before this guard, what drove some
    # predictions to 0.01 off five observations. Keep only the strongest rule
    # per (subject, field) - the one whose adjustment is largest in magnitude,
    # with rule_id as a deterministic tiebreak.
    strongest: dict[tuple[str, str], Rule] = {}
    for r in matching:
        key = (r.subject, r.field)
        held = strongest.get(key)
        if held is None or abs(r.delta_logit) > abs(held.delta_logit):
            strongest[key] = r
    matching = sorted(strongest.values(), key=lambda r: r.rule_id)

    total = logit(p0)
    applied: list[AppliedRule] = []
    for r in matching:
        delta = max(-MAX_DELTA_LOGIT, min(MAX_DELTA_LOGIT, r.delta_logit))
        total += delta * r.weight
        applied.append(
            AppliedRule(
                rule_id=r.rule_id,
                delta_logit=delta,
                weight=r.weight,
                why=f"{r.field} {r.op} {r.value}",
            )
        )
    return sigmoid(total), tuple(applied)


@dataclass(slots=True)
class CalibrationLedger:
    """Ten probability bins: predictions made, and how they actually resolved.

    Lives only in memory - it cannot be reconstructed from chain data, which
    is why it belongs on screen during the deletion segment (spec 7).
    """

    bins: int = 10
    counts: list[int] = field(default_factory=lambda: [0] * 10)
    hits: list[int] = field(default_factory=lambda: [0] * 10)
    #: Minimum observations in a bin before its gap is trusted enough to
    #: shift a live prediction.
    min_support: int = 20

    def bin_of(self, p: float) -> int:
        return min(int(p * self.bins), self.bins - 1)

    def record(self, predicted: float, completed: bool) -> None:
        i = self.bin_of(predicted)
        self.counts[i] += 1
        self.hits[i] += int(completed)

    def observed_rate(self, p: float) -> float | None:
        i = self.bin_of(p)
        if self.counts[i] < self.min_support:
            return None
        return self.hits[i] / self.counts[i]

    def bin_midpoint(self, p: float) -> float:
        return (self.bin_of(p) + 0.5) / self.bins

    def shift_for(self, p: float) -> float:
        """Stage 3. Logit-space correction for systematic bias in this band.

        The offset is measured against the band's midpoint, not against the
        individual prediction. Using ``logit(observed) - logit(p)`` would not
        correct a bias, it would *target* the observed rate - collapsing every
        prediction in the band onto one value and destroying the within-band
        discrimination stage 1 earned. What we want is "predictions around
        here run this much hot or cold", applied as a uniform nudge.

        Returns 0.0 when the band lacks support, so an empty or thin ledger
        cannot move a prediction on noise.
        """
        observed = self.observed_rate(p)
        if observed is None:
            return 0.0
        return logit(observed) - logit(self.bin_midpoint(p))

    def learned_threshold(
        self, *, target: float = 0.5, min_support: int = 200
    ) -> float | None:
        """The confidence level at which hiring has actually paid off.

        Not a tuned parameter. For each candidate bar, this asks what fraction
        of jobs Priors predicted *at or above* that bar actually completed, and
        returns the lowest bar where that fraction reaches ``target``.

        ``target`` is 0.5 because a hire should be likelier to succeed than
        fail; it is the definition of a sensible bar, not a knob fitted to the
        replay. Nothing here is searched against replay accuracy.

        The ledger only ever contains outcomes already resolved, so the bar
        available for a job is derived exclusively from jobs that finished
        before it - the temporal separation holds by construction, exactly as
        it does for rules.

        Returns None when there is not enough history to justify moving off
        the naive bar. That is the deletion-test condition: no learned policy,
        fall back to 0.5.
        """
        total = sum(self.counts)
        if total < min_support:
            return None

        for i in range(self.bins):
            n = sum(self.counts[i:])
            if n < min_support:
                break
            observed = sum(self.hits[i:]) / n
            if observed >= target:
                return i / self.bins
        return None

    def brier(self) -> float | None:
        """Overall Brier score across recorded predictions."""
        n = sum(self.counts)
        if not n:
            return None
        total = 0.0
        for i, c in enumerate(self.counts):
            if not c:
                continue
            mid = (i + 0.5) / self.bins
            rate = self.hits[i] / c
            total += c * (mid - rate) ** 2
        return total / n


def predict(
    view: DecisionView,
    *,
    rules: Sequence[Rule] = (),
    calibration: CalibrationLedger | None = None,
    base_rate: float = BASE_RATE,
    learn_threshold: bool = False,
) -> Prediction:
    """Run all three stages and return the number with its provenance.

    With ``rules`` empty and ``calibration`` None - the deletion-test
    condition - this reduces to stage 1 alone: the shrunk chain baseline, with
    ``memory_contributed`` False. It never silently reconstructs learned
    experience from the local dataset (spec 10.6, binding rule).
    """
    p0 = shrunk_base_rate(
        view.provider_prior_completed,
        view.provider_prior_funded,
        base_rate=base_rate,
    )

    p1, applied = apply_rules(p0, view, rules)

    # Stage 3 is measured and reported, but does NOT move the decision.
    #
    # Measured on the full replay: applying it flipped 458 hire/no-hire calls
    # and was right on 45% of them - worse than chance - while rules alone
    # flipped 96 and were right on 68%. The correction is real (in the 0.3-0.4
    # band Priors says 0.35 and observes 0.14) but it is a band-wide average
    # applied to individual jobs, and the hire threshold sits at 0.5, so a
    # correct average nudge pushes genuinely fine jobs across the line. It is
    # also learned from a few hundred observations in exactly the bands where
    # it moves decisions, against ten thousand where it does not.
    #
    # So: keep it on the card, keep it out of the call.
    shift = calibration.shift_for(p1) if calibration is not None else 0.0
    observed = calibration.observed_rate(p1) if calibration is not None else None
    p_calibrated = sigmoid(logit(p1) + shift) if shift else p1

    threshold = 0.5
    if learn_threshold and calibration is not None:
        learned = calibration.learned_threshold()
        if learned is not None:
            threshold = learned

    return Prediction(
        p_final=p1,
        p0_shrunk=p0,
        p_after_rules=p1,
        calibration_shift=shift,
        p_calibrated=p_calibrated,
        calibration_observed=observed,
        rules_applied=applied,
        hire_threshold=threshold,
        memory_contributed=(
            bool(applied) or observed is not None or threshold != 0.5
        ),
    )
