"""The response gate: will this provider actually answer if we create a job?

Rules-v2 answers a different question. It asks, given that a job gets funded,
how likely is it to complete. That leaves the largest failure in the
marketplace unaddressed: in the scanned history, 53.31% of created jobs never
reached a ``BudgetSet`` event at all, and only 24.38% were ever funded. A job
that nobody answers is not a completion risk, it is wasted before it starts.

So Priors asks the counterparty first. The counterparty declares whether it is
available, and Priors decides how much that declaration is worth using its own
record of whether that counterparty's earlier declarations held up.

Three properties make this different from taking the counterparty's word:

  The declaration is never trusted on arrival. A fresh counterparty sits
  exactly at the marketplace prior, so it is treated as neither honest nor
  dishonest but typical.

  Only observed outcomes move it. Priors writes the request, the counterparty
  writes the declaration, and the resolver writes what actually happened.
  Nobody can write the record that judges their own claim.

  Evidence expires. Without that, a blocked counterparty could never recover:
  the gate stops Priors creating jobs for it, and only a job can produce the
  observation that would clear it. See ``HALF_LIFE_BLOCKS``.

This module never touches Rules-v2. The completion probability it receives is
already final, and the gate can only decline a job the predictor approved. It
cannot promote one the predictor rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Share of created jobs that reached a BudgetSet event, 35,179 of 75,340 in
#: blocks 44,429,969 to 50,749,122. BudgetSet is the provider-side response
#: step in the ACP SDK, but the event carries no setter address, so this is a
#: response rate and is deliberately not described as provider acceptance.
#: Regenerate with scripts/dataset_stats.py rather than editing by hand.
NEUTRAL_RESPONSE_PRIOR = 0.4669

#: The bar is the prior itself: proceed only if this counterparty is at least
#: as likely to respond as a typical job on this marketplace. Setting them
#: equal is what makes an unknown counterparty pass, and what makes an empty
#: store fall back to passing, without anyone choosing a number to suit the
#: outcome.
RESPONSE_BAR = NEUTRAL_RESPONSE_PRIOR

#: Shrinkage strength in pseudo-observations, matching the stage 1 convention.
#: A counterparty needs a real record before its own rate outruns the prior.
RESPONSE_SHRINKAGE_K = 5.0

#: Measured, not assumed: 2.000 s/block across 6,319,153 blocks of scanned
#: history, so one day is 43,200 blocks. Time is block height here as it is
#: everywhere else - no wall clock touches a decision.
BLOCKS_PER_DAY = 43_200

#: A broken promise counts fully at first and halves each day. One day is a
#: long time on this marketplace: the median gap between a provider's
#: consecutive funded jobs is 196 blocks, about six and a half minutes, so a
#: day is roughly 220 missed opportunities rather than a brief pause.
HALF_LIFE_BLOCKS = BLOCKS_PER_DAY

#: And after five days an observation is dropped outright rather than merely
#: shrunk. Decay alone is not enough: a fading failure leaves the reliability
#: creeping toward the prior from below without ever reaching it, so a gate
#: that tests `>=` would stay shut forever. Expiry is what actually restores
#: the counterparty to unknown.
EXPIRY_BLOCKS = 5 * BLOCKS_PER_DAY


@dataclass(frozen=True, slots=True)
class Observation:
    """One declaration Priors acted on, and what followed.

    ``confirmed`` means the counterparty did what it said: the job reached the
    provider-side response step. ``block`` is when that was observed, and is
    what the decay is measured against.
    """

    block: int
    confirmed: bool

    def weight(self, as_of_block: int) -> float:
        age = max(0, as_of_block - self.block)
        if age >= EXPIRY_BLOCKS:
            return 0.0
        return 0.5 ** (age / HALF_LIFE_BLOCKS)


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    """What the store remembers about one counterparty's declarations.

    ``observations`` covers only declarations Priors acted on, because a
    declaration that never led to a job has no observation to be judged
    against. ``declared`` counts every declaration including the ones that
    stopped Priors creating a job, so a counterparty that avoids being tested
    by always declaring unavailable cannot bank a perfect record.
    """

    provider: str
    declaration_type: str = "availability"
    declared: int = 0
    observations: tuple[Observation, ...] = field(default_factory=tuple)

    def weighted(self, as_of_block: int) -> tuple[float, float]:
        """(acted_on, confirmed) after decay and expiry."""
        acted = conf = 0.0
        for o in self.observations:
            w = o.weight(as_of_block)
            if not w:
                continue
            acted += w
            if o.confirmed:
                conf += w
        return acted, conf

    def reliability(self, as_of_block: int) -> float:
        """P(this counterparty responds | it declared available), shrunk.

        With no live evidence this returns the prior constant itself rather
        than a computed value that merely rounds to it. The gate compares
        against that same constant, so the unknown and expired cases pass on
        `>=` exactly, not by floating-point luck.
        """
        acted, conf = self.weighted(as_of_block)
        if acted == 0.0:
            return NEUTRAL_RESPONSE_PRIOR
        return (conf + RESPONSE_SHRINKAGE_K * NEUTRAL_RESPONSE_PRIOR) / (
            acted + RESPONSE_SHRINKAGE_K
        )

    def coverage(self) -> float | None:
        """Share of declarations that were usable, or None with no history.

        Reported alongside reliability, never folded into it. A counterparty
        that is honest but perpetually unavailable is useless for this job,
        and that is a different fact from being untruthful.
        """
        if not self.declared:
            return None
        return len(self.observations) / self.declared


@dataclass(frozen=True, slots=True)
class GateResult:
    proceed: bool
    p_complete: float
    p_respond: float
    response_bar: float
    completion_gate: bool
    response_gate: bool
    memory_backed: bool
    record: ResponseRecord | None

    @property
    def blocked_by_coordination(self) -> bool:
        """The gate declined something Rules-v2 had approved."""
        return self.completion_gate and not self.response_gate


def apply_response_gate(
    p_complete: float,
    record: ResponseRecord | None,
    *,
    declared_available: bool,
    as_of_block: int = 0,
    completion_threshold: float = 0.5,
) -> GateResult:
    """Both gates. Rules-v2's probability is passed through untouched.

    ``record`` is None when the store is empty, which is the deletion
    condition: with nothing recalled the reliability is the prior, the prior
    equals the bar, and the response gate passes on `>=`. The same code takes
    the permissive path because it has nothing to know, not because a branch
    was disabled.
    """
    completion_gate = p_complete >= completion_threshold

    if not declared_available:
        # No claim to weigh. Priors does not create jobs for a counterparty
        # that has said it cannot take them.
        return GateResult(
            proceed=False, p_complete=p_complete,
            p_respond=0.0, response_bar=RESPONSE_BAR,
            completion_gate=completion_gate, response_gate=False,
            memory_backed=record is not None, record=record,
        )

    p_respond = (
        record.reliability(as_of_block) if record else NEUTRAL_RESPONSE_PRIOR
    )
    response_gate = p_respond >= RESPONSE_BAR
    return GateResult(
        proceed=completion_gate and response_gate,
        p_complete=p_complete,
        p_respond=p_respond,
        response_bar=RESPONSE_BAR,
        completion_gate=completion_gate,
        response_gate=response_gate,
        memory_backed=record is not None,
        record=record,
    )
