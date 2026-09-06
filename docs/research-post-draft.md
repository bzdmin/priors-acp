# Research post: draft for X

Publish today. Every figure below comes from `docs/DATA.md`, regenerated from
the local dataset (blocks 44,429,969 to 50,749,122). Do not edit numbers by hand.

**Before posting:** confirm the handles. `@sibylcap` is from
`PROJECT_CONTEXT.md` and I have not verified it resolves; check the Virtuals
official account too rather than guessing.

---

## Thread

**1/**

I scanned every job on Virtuals Protocol's ACP marketplace, all 75,340 of them,
to find out whether agents hiring agents actually works.

One number stopped me.

ACP has an evaluator role, for independent verification of delivered work.

On completed jobs, a genuine third party fills it **0.12% of the time.**

**2/**

18 jobs. Out of 14,644 completed.

The rest:

- evaluator is the client who created the job: 59.1%
- no evaluator at all: 40.7%
- evaluator is the provider: 0%

Method: compare the evaluator address on each completed job against that
job's own client and provider addresses. Anyone can rerun it.

**3/**

The fee routing says the same thing from the other side.

7,356 evaluator fee payments.
7,338 of them, or **99.8%**, went back to the client who created the job.

The money for independent verification is, in practice, moving from a wallet
to itself.

**4/**

To be precise about what this does and doesn't show:

It does NOT show the work is bad, or that anyone is cheating.

It shows a mechanism the protocol built for verification is **not being used
for verification.** Two of the three roles collapse into one party.

**5/**

The funnel has the same shape.

Of 75,340 jobs created, only **24.4%** ever get funded.

And demand is concentrated: two clients account for 55,004 of those 75,340.

**6/**

Once a job IS funded, 79.47% complete. That sounds healthy until you look
per-provider.

Completion rates across providers with real volume run from **1.00 down to
0.05.**

The average tells you almost nothing about who you're about to hire.

**7/**

So: a marketplace where three quarters of jobs die unfunded, provider
reliability varies 20x, and the built-in verification mechanism is
self-refereed.

If you're an agent choosing who to hire here, the public record is not
enough.

**8/**

I'm building an agent for the @sibylcap hackathon in response.

It predicts whether a job will complete before it hires, records the
prediction, and after the job resolves records what actually happened, then
uses that accumulated experience on the next decision.

**9/**

The interesting part isn't accuracy. Chain data alone already predicts well.

It's that the agent learns where its own chain-based judgment *fails*, writes
that down as a rule, tests the rule against later decisions, and drops it when
it stops working.

**10/**

One case from the backtest.

Public record: this provider completed 83 of 114 jobs. 72.8%. Hire.

Its own memory: this provider collapses above a certain job size. Don't hire.

The job failed.

**11/**

Delete the memory and the same code hires. Keep it and it declines.

Across 18,367 replayed jobs, memory changed the call on 187 of them, and was
right on 67.4% where the public record was right on 32.6%.

Nearly all of them in one direction: blocking a provider the record endorsed.

**12/**

Full dataset, method and code going up with the submission.

If you're building on ACP: does the evaluator finding match what you've seen?
And has anyone found a signal for provider reliability that beats raw
completion rate? Genuinely asking, because I'd rather be corrected now than after I
ship.

---

## Notes for the poster

- **1/** and **3/** are the load-bearing tweets. If the thread gets truncated
  anywhere, those two carry it.
- **4/** exists to stop the "you're accusing people of fraud" reply. Do not
  cut it.
- **12/** is the PMF ask. The rules require publicly verifiable evidence of
  the pain point, and replies from ACP builders are that evidence. Ending on
  a real question rather than a call to action is what makes replies likely.
- Do not claim a Base partner bonus anywhere. That was dropped.
- If asked for the code before submission: the dataset scanner and figures are
  prior work and can be shared; the prediction model ships with the entry.
