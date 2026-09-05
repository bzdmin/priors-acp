"""Outcome Resolver: terminal classification and episodes (spec 10.2, 5).

The second of the two processes. Write ownership is disjoint and enforced
here rather than merely documented (spec 6):

    Priors            writes  decision
    Outcome Resolver  writes  episode, pattern, rule, rule performance

Neither writes the other's types. The resolver's writes change what Priors
does on the next decision, which is what makes memory shared coordination
state between two processes rather than one agent's notebook.

Unresolved is not failure. Treating still-open jobs as failures would bias
every provider active near the end of the dataset, so they are counted,
displayed, and excluded - never quietly scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

from .events import AcpEvent
from .features import DecisionView
from .predict import CalibrationLedger, Prediction

Outcome = Literal["success", "failure", "unresolved"]

#: A prediction misses badly when it is this far from what happened.
ERROR_THRESHOLD = 0.5

TERMINAL_FAILURE_KINDS = ("JobRejected", "Refunded", "JobExpired")


@dataclass(frozen=True, slots=True)
class Episode:
    """Written when a job resolves. Pairs with the decision (spec 5)."""

    job_id: int
    outcome: Outcome
    predicted: float
    correct: bool
    brier: float
    resolved_at_block: int
    #: The bar this decision was actually judged against. 0.5 means no learned
    #: policy was available; anything else came from Priors' own history.
    hire_threshold: float = 0.5
    #: Explanatory only. Rejection, refund and expiry are fields on a failure
    #: episode, never competing labels, and they never change the target.
    failure_kind: str | None = None
    #: True when no terminal event was seen but the job is past expiredAt.
    inferred_expiry: bool = False

    def to_memory(self) -> dict:
        """Journal shape written to the COLD tier via write_event."""
        body = {
            "type": "episode",
            "job_id": self.job_id,
            "hire_threshold": round(self.hire_threshold, 3),
            "outcome": self.outcome,
            "predicted": round(self.predicted, 4),
            "correct": self.correct,
            "brier": round(self.brier, 4),
            "resolved_at_block": self.resolved_at_block,
        }
        if self.failure_kind:
            body["failure_kind"] = self.failure_kind
        if self.inferred_expiry:
            body["inferred_expiry"] = True
        return body


@dataclass(frozen=True, slots=True)
class ErrorPattern:
    """Where Priors' own reasoning failed - distinct from a provider failing."""

    job_id: int
    predicted: float
    actual: float
    provider: str
    conditions: dict
    note: str = ""

    def to_memory(self) -> dict:
        return {
            "type": "error_pattern",
            "job_id": self.job_id,
            "predicted": round(self.predicted, 4),
            "actual": self.actual,
            "provider": self.provider,
            "conditions": self.conditions,
            "note": self.note,
        }


@dataclass(slots=True)
class _Pending:
    view: DecisionView
    prediction: Prediction


class OutcomeResolver:
    """Consumes the same event stream and resolves what Priors predicted."""

    def __init__(self, calibration: CalibrationLedger | None = None) -> None:
        self.calibration = calibration if calibration is not None else CalibrationLedger()
        self._pending: dict[int, _Pending] = {}
        self._resolved: set[int] = set()
        self.episodes: list[Episode] = []
        self.error_patterns: list[ErrorPattern] = []
        self.unresolved_count: int = 0

    # -- Priors hands off, resolver never writes decisions -------------------

    def register_decision(self, view: DecisionView, prediction: Prediction) -> None:
        """Record that a prediction was made, so it can be scored later."""
        self._pending[view.job_id] = _Pending(view, prediction)

    # -- resolution ---------------------------------------------------------

    def observe(self, ev: AcpEvent) -> Episode | None:
        """Fold in one event; emit an episode if it resolves a predicted job."""
        if ev.job_id not in self._pending or ev.job_id in self._resolved:
            return None

        if ev.kind == "JobCompleted":
            return self._close(ev.job_id, "success", ev.block, None)

        if ev.kind in TERMINAL_FAILURE_KINDS:
            return self._close(ev.job_id, "failure", ev.block, ev.kind)

        return None

    def finalize(self, head_block: int, head_timestamp: int | None) -> list[Episode]:
        """Classify everything still open at the end of the stream.

        Past ``expiredAt`` with no completion and no explicit event is a
        failure, flagged ``inferred_expiry``. Anything still inside its window
        is genuinely unresolved: no episode, excluded from calibration and
        from pattern support.
        """
        out: list[Episode] = []
        for job_id in sorted(self._pending):
            if job_id in self._resolved:
                continue
            view = self._pending[job_id].view
            expired = (
                head_timestamp is not None
                and view.funding_timestamp is not None
                and view.expiry_window is not None
                and head_timestamp > view.funding_timestamp + view.expiry_window
            )
            if expired:
                ep = self._close(
                    job_id, "failure", head_block, "JobExpired", inferred=True
                )
                if ep:
                    out.append(ep)
            else:
                self.unresolved_count += 1
        return out

    # -- internals ----------------------------------------------------------

    def _close(
        self,
        job_id: int,
        outcome: Outcome,
        block: int,
        failure_kind: str | None,
        *,
        inferred: bool = False,
    ) -> Episode | None:
        pending = self._pending.get(job_id)
        if pending is None or job_id in self._resolved:
            return None
        self._resolved.add(job_id)

        p = pending.prediction.p_final
        actual = 1.0 if outcome == "success" else 0.0
        # Correctness is judged against the bar the decision was actually made
        # against, not a hardcoded 0.5 - otherwise a learned policy would be
        # scored on a rule it never used.
        hired = pending.prediction.decision == "HIRE"
        episode = Episode(
            job_id=job_id,
            outcome=outcome,
            predicted=p,
            correct=hired == (actual == 1.0),
            hire_threshold=pending.prediction.hire_threshold,
            brier=(p - actual) ** 2,
            resolved_at_block=block,
            failure_kind=failure_kind,
            inferred_expiry=inferred,
        )
        self.episodes.append(episode)
        # Record the pre-calibration probability, not p_final. The ledger must
        # measure how stages 1-2 behave, otherwise it is scoring its own
        # output and the correction compounds on itself run after run.
        self.calibration.record(
            pending.prediction.p_after_rules, outcome == "success"
        )

        if abs(p - actual) > ERROR_THRESHOLD:
            v = pending.view
            self.error_patterns.append(
                ErrorPattern(
                    job_id=job_id,
                    predicted=p,
                    actual=actual,
                    provider=v.provider,
                    conditions={
                        "amount": v.amount,
                        "repeat_pair": v.repeat_pair,
                        "budget_set_count": v.budget_set_count,
                        "provider_prior_funded": v.provider_prior_funded,
                    },
                    note=(
                        "overconfident"
                        if actual == 0.0
                        else "underconfident"
                    )
                    + (" on first-time pair" if not v.repeat_pair else ""),
                )
            )
        return episode

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        n = len(self.episodes)
        if not n:
            return {"episodes": 0, "unresolved": self.unresolved_count}
        successes = sum(1 for e in self.episodes if e.outcome == "success")
        return {
            "episodes": n,
            "unresolved": self.unresolved_count,
            "inferred_expiry": sum(1 for e in self.episodes if e.inferred_expiry),
            "success_rate": successes / n,
            "accuracy": sum(1 for e in self.episodes if e.correct) / n,
            "brier": sum(e.brier for e in self.episodes) / n,
            "error_patterns": len(self.error_patterns),
        }
