"""The replay harness (spec 8).

Runs both processes against the same event stream, in the order they would
run live. Write ownership holds here exactly as it does in production:
Priors decides and records a decision; the Outcome Resolver reveals what
happened and records an episode.

This is a backtest, and must be described as one. The predictions are real -
each was made before its outcome was read - but they were made against
historical data in fast-forward, not accumulated over months of live trading.
Stated plainly it is a standard method; unstated it looks like fabrication.

Each block is processed in three phases so that the leakage rule of spec 2
holds by construction rather than by inspection:

    1. absorb job attributes   (the question)
    2. predict every funding    (history is strictly older than this block)
    3. absorb outcomes          (the answer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .events import AcpEvent
from .features import DecisionView, ReplayState
from .predict import CalibrationLedger, Prediction, Rule, predict
from .resolver import Episode, OutcomeResolver
from .rules import Observation, RuleEngine


@dataclass(slots=True)
class ReplayResult:
    decisions: int = 0
    episodes: list[Episode] = field(default_factory=list)
    calibration: CalibrationLedger = field(default_factory=CalibrationLedger)
    resolver: OutcomeResolver | None = None
    engine: RuleEngine | None = None
    memory_backed_decisions: int = 0
    #: Decisions where memory flipped the HIRE / DO NOT HIRE call relative to
    #: the chain-only baseline. This, not aggregate Brier, is the claim the
    #: deletion test actually makes.
    reversals: int = 0
    reversals_correct: int = 0
    #: Jobs the learned bar rejected that a naive 0.5 bar would have hired,
    #: and how many of those actually failed. The number a judge understands
    #: immediately.
    policy_rejected: int = 0
    policy_rejected_failed: int = 0
    #: (block, threshold) whenever the learned bar moves - to show whether it
    #: behaves like something learned or snaps to a convenient constant.
    threshold_trajectory: list[tuple[int, float]] = field(default_factory=list)
    #: Detail of every reversal, for picking a real demo case rather than an
    #: illustrative one.
    reversal_detail: list[dict] = field(default_factory=list)

    def brier(self) -> float | None:
        if not self.episodes:
            return None
        return sum(e.brier for e in self.episodes) / len(self.episodes)

    def brier_naive(self, base_rate: float) -> float | None:
        if not self.episodes:
            return None
        return sum(
            (base_rate - (1.0 if e.outcome == "success" else 0.0)) ** 2
            for e in self.episodes
        ) / len(self.episodes)


def run_replay(
    events: Iterable[AcpEvent],
    *,
    learn_rules: bool = True,
    use_calibration: bool = True,
    learn_threshold: bool = False,
    calibration_min_support: int = 20,
    priors_writer=None,
    resolver_writer=None,
) -> ReplayResult:
    """Replay the stream, predicting before each outcome is revealed.

    ``learn_rules`` and ``use_calibration`` both False is the deletion-test
    condition (spec 10.4): the same code path, the same binary, with nothing
    in memory. There is no disabled branch - stages 2 and 3 simply have
    nothing to contribute, and the prediction falls back to the shrunk
    chain-only baseline of stage 1.
    """
    state = ReplayState()
    calibration = CalibrationLedger(min_support=calibration_min_support)
    resolver = OutcomeResolver(calibration)
    engine = RuleEngine()
    result = ReplayResult(
        calibration=calibration, resolver=resolver, engine=engine
    )

    block: int | None = None
    pending: list[AcpEvent] = []
    last_ts: int | None = None
    views: dict[int, DecisionView] = {}
    applied_by_job: dict[int, tuple[str, ...]] = {}
    baseline_p: dict[int, float] = {}
    calls: dict[int, tuple[str, str]] = {}   # job -> (actual call, naive call)
    last_threshold = 0.5

    nonlocal_threshold = [0.5]

    def flush() -> None:
        nonlocal last_ts, last_threshold
        for e in pending:
            state.absorb_attributes(e)

        for e in pending:
            if e.kind != "JobFunded":
                continue
            view = state.build_view(e)
            if view is None:
                continue
            prediction: Prediction = predict(
                view,
                rules=engine.active_rules() if learn_rules else (),
                calibration=calibration if use_calibration else None,
                learn_threshold=learn_threshold,
            )
            nonlocal_threshold[0] = prediction.hire_threshold
            calls[view.job_id] = (prediction.decision, prediction.naive_decision)
            resolver.register_decision(view, prediction)
            # Priors' own write, before the outcome exists. Write ownership
            # holds here exactly as it does live: this handle cannot write an
            # episode even if this code tried.
            if priors_writer is not None:
                priors_writer.write_decision(view, prediction)
            views[view.job_id] = view
            applied_by_job[view.job_id] = tuple(
                a.rule_id for a in prediction.rules_applied
            )
            # Stage 1 alone is what survives deletion; compare against it.
            baseline_p[view.job_id] = prediction.p0_shrunk
            result.decisions += 1
            if prediction.memory_contributed:
                result.memory_backed_decisions += 1

        for e in pending:
            state.absorb_outcome(e)
            episode = resolver.observe(e)
            if episode is not None:
                result.episodes.append(episode)
                _score(episode)
            if e.timestamp is not None:
                last_ts = e.timestamp
        if nonlocal_threshold[0] != last_threshold:
            last_threshold = nonlocal_threshold[0]
            result.threshold_trajectory.append((block or 0, last_threshold))

    def _score(episode: Episode) -> None:
        """Resolver-owned: feed the outcome back into the rule lifecycle."""
        view = views.pop(episode.job_id, None)
        if view is None:
            return
        completed = episode.outcome == "success"
        rules_for_job = applied_by_job.pop(episode.job_id, ())

        # The resolver's write, after the outcome is known.
        if resolver_writer is not None:
            resolver_writer.write_episode(episode, view.provider)

        # Did memory change the actual call?
        actual_call, naive_call = calls.pop(episode.job_id, ("HIRE", "HIRE"))
        if actual_call == "DO NOT HIRE" and naive_call == "HIRE":
            result.policy_rejected += 1
            if not completed:
                result.policy_rejected_failed += 1

        memory_call = actual_call == "HIRE"
        chain_only_p = baseline_p.pop(episode.job_id, episode.predicted)
        chain_call = chain_only_p >= 0.5
        if memory_call != chain_call:
            result.reversals += 1
            memory_right = memory_call == completed
            if memory_right:
                result.reversals_correct += 1
            result.reversal_detail.append(
                {
                    "job_id": episode.job_id,
                    "provider": view.provider,
                    "amount": view.amount,
                    "budget_set_count": view.budget_set_count,
                    "repeat_pair": view.repeat_pair,
                    "chain_p": round(chain_only_p, 4),
                    "memory_p": round(episode.predicted, 4),
                    "chain_call": "HIRE" if chain_call else "DO NOT HIRE",
                    "memory_call": "HIRE" if memory_call else "DO NOT HIRE",
                    "outcome": "completed" if completed else "failed",
                    "memory_right": memory_right,
                    "rules": list(rules_for_job),
                    "provider_funded": view.provider_prior_funded,
                    "provider_completed": view.provider_prior_completed,
                }
            )

        for rule_id in rules_for_job:
            engine.record_application(rule_id, view, completed)

        if learn_rules:
            engine.observe(
                Observation(
                    job_id=episode.job_id,
                    provider=view.provider,
                    amount=view.amount,
                    repeat_pair=view.repeat_pair,
                    budget_set_count=view.budget_set_count,
                    completed=completed,
                    block=episode.resolved_at_block,
                    client_prior_funding_rate=view.client_prior_funding_rate,
                    client_prior_created=view.client_prior_created,
                    pair_prior_jobs=view.pair_prior_jobs,
                    provider_concurrent_open=view.provider_concurrent_open,
                    expiry_window=view.expiry_window,
                    funding_delay=view.funding_delay,
                )
            )

    for ev in events:
        if block is not None and ev.block != block:
            flush()
            pending.clear()
        block = ev.block
        pending.append(ev)

    if pending:
        flush()

    if block is not None:
        result.episodes.extend(resolver.finalize(block, last_ts))

    return result
