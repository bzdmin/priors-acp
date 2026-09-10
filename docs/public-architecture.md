# Architecture, as designed and as built

This file was written on 2026-08-26, before any of the implementation existed.
It described an intended architecture and withheld the details.

It is kept, and rewritten here, as a record of what was predicted before the
code was written and what happened to each prediction. The original text is in
this repository's history at commit `Add public project foundation`.

The withholding is over. Every detail the original said would not be published
is now in `predict.py`, `rules.py`, `memory.py` and `coordination.py`, and the
README documents all of it.

---

## What the design said on 2026-08-26

Six claims, quoted or closely paraphrased from the original.

1. The decision process receives hiring context and provider signals, produces
   a deterministic prediction, records that prediction, and uses recalled
   experience when making later decisions.
2. The numerical decision path is pure code. Language models, when used, stay
   outside the decision arithmetic and may only generate explanatory text after
   a decision has been computed.
3. A separate deterministic worker observes completed ACP outcomes and writes
   outcome-related experience back to persistent state.
4. Write ownership is disjoint. Priors owns predictions and decision records.
   The resolver owns episodes, patterns, rules and rule performance. Neither
   writes the records owned by the other.
5. The prediction path is deterministic for a fixed input dataset and fixed
   configuration.
6. Deleting or replacing the experience store removes the learned component of
   the decision process while leaving the ACP integration and the deterministic
   decision machinery intact.

---

## What held

**All six.** Each one is now a property of the shipped code rather than an
intention, and most are enforced rather than promised.

**Claim 1** is `priors/predict.py`, three stages, with the recall step being a
single function, `PriorsMemory.recall_experience`. Nothing else in the codebase
reads Sibyl for decision input.

**Claim 2** held with room to spare. No model is called anywhere on the
decision path, and the only model in the project writes summary prose inside
the provider agent. The decision path also has no wall clock and no RNG, with
time expressed as a block number throughout, which was not part of the original
claim and turned out to be necessary for claim 5.

**Claim 3** is `priors/resolver.py`, and the separation is stronger than
described. The original expected two processes. There are three, since the
counterparty in the coordination layer needs to answer for itself without being
able to score itself.

**Claim 4** is enforced by the API surface rather than by convention. Priors has
no method capable of writing an episode, the resolver has none capable of
writing a decision, and the counterparty's handle exposes exactly one method.
Tests in `tests/test_events_and_ownership.py` fail if anyone adds a convenient
method to the wrong handle.

**Claim 5** required something the original did not anticipate. Determinism
across two runs of the same data needs a total ordering of events, since block
number alone is not unique. Ordering is now block, then log index, then job id,
and a test asserts two runs over reversed input produce identical order.

**Claim 6** is the deletion test, and it is the reason this file is worth
keeping. The property was written down before the code existed, so it shaped
the architecture rather than being discovered in it afterwards. Stage 1 reads
public chain history and survives deletion. Stages 2 and 3 recall from Sibyl
and have nothing to return when the store is empty. There is no
`if memory_enabled` branch anywhere, because deletion was designed to be the
absence of data rather than a mode.

---

## What changed

**The withholding was reversed.** The original listed the feature set, the
confidence equations, the memory schema, the rule thresholds and the
provider-specific scoring as details that would stay in an internal
specification. All of them are published. A prediction model whose arithmetic
cannot be inspected is not evidence of anything, and the figures in the README
are only checkable because the code that produced them is readable.

**Calibration was demoted from an input to a report.** The original design
treated recalled experience as one undifferentiated thing that would improve
predictions. In practice the three stages behaved differently. Rules earned the
right to move a decision. Calibration did not: applying it flipped 458
decisions and was right on 45% of them, worse than chance, because a band-wide
average is not evidence about an individual job. It is computed, stored and
reported, and never applied.

**A learned hiring threshold was built and rejected.** This was not in the
original design at all. It was tested on the theory that memory should be able
to change policy and not only estimates. Every bar above 0.5 scored worse
against the real outcome distribution, because failed ACP jobs are refunded
from escrow, so caution costs more than it saves. The code remains, defaulted
off.

---

## What the design missed entirely

**The largest failure in the marketplace is not bad work, it is silence.** The
original design asked one question, whether a funded job would complete, which
assumed jobs get funded. The scan showed that 40,161 of 75,340 created jobs
never draw a provider response at all, so the question the design was built to
answer applies to under a quarter of the market. The coordination layer in
`priors/coordination.py` exists because of that gap, and it is the largest
single thing built after this file was written.

**Nothing expires an unanswered job.** `JobExpired` requires someone to send a
transaction, and almost nobody does: only about 8% of jobs that never reached a
provider response ever emit one. An observer waiting for that event would wait
forever, so a broken availability promise could never be recorded. The observer
now reads the deadline from `expired_at` on the job's own `JobCreated` event
against block time.

**Evidence has to expire.** The gate stops Priors creating jobs for a
counterparty it has blocked, and only a job can produce the observation that
would clear it, so without expiry a blocked counterparty could never recover.
Decay alone does not solve this, because it approaches the prior without
reaching it. Observations are dropped entirely after five days.

**The store has operational limits that shape the design.** `read_events`
clamps any limit to 10,000 rows, so reading the full journal needs cursor
paging. The journal is append-only with no prune, which is why provider records
live in WARM entities rather than in the journal, and why a live run must not
re-journal history it has already written.

---

## Why this file is kept

The claim this project makes is that deleting the memory changes what the agent
does. A judge has no way to tell, from a finished repository, whether that
property was designed for or discovered afterwards and written up as though it
had been intended.

This file is dated evidence for the first reading. It is also a record of four
things the design did not foresee, which is the more useful half.
