# design.md

The source of truth for the Priors evidence page. Written before any markup.

## What this page is

Not a product frontend. Priors has no users to onboard and nothing to sign up
for. This page has exactly one job:

> A judge opens it cold, with no setup, and within sixty seconds believes that
> memory changed what this agent did.

Every decision below follows from that sentence. If an element does not serve
it, it does not ship.

## Who is reading it

A Sibyl Labs judge, on a laptop, on judging day, having already opened eleven
other submissions. They are tired, they are skeptical, and they have seen a
lot of demo videos claiming a lot of things.

Assume they will not run the code. Assume they will not read the README first.
Assume they scroll once and decide whether to keep reading.

What they are scoring, from the rubric:

| weight | category |
|---|---|
| 40 | memory is load-bearing |
| 15 | pitch and presentation |
| 10 | PMF bonus |

The page targets the 40 first and the 15 as a consequence. Presentation is
what happens when the evidence is laid out well, not a separate layer applied
on top.

## What it should feel like

Instrument panel, not landing page. The reference is a lab readout or a
BaseScan page: something that reports rather than sells.

Concretely, this means:

- No hero image, no gradient mesh, no illustration, no 3D render. Competitors
  in this space use a wireframe brain or a marble statue. Both are decoration
  standing in for evidence they do not have. We have the evidence.
- No marketing verbs. Nothing "revolutionises" or "unlocks" anything.
- Numbers are the largest type on the page. If a figure and a sentence compete
  for the same space, the figure wins.
- Monospace for anything measured. Sans for anything explained. The reader
  should be able to tell at a glance which is which.

## Voice

Lowercase, plain, and slightly understated, matching the build-in-public
thread. Short declarative sentences. The findings are strong enough that
overstating them only invites doubt.

Two rules:

- Never write a number without saying what it is out of. `40,161` alone is a
  boast; `40,161 of 75,340` is a measurement.
- State every limitation in the same typeface and size as every claim. The
  honesty section is not a footnote, and burying it is what a weaker
  submission would do.

## Structure

One page, one column, scrolling. No navigation, no tabs, no routing. A judge
should never have to find anything.

Order is deliberate. Strongest evidence first, because the scroll may stop at
any point:

1. **Identity.** What Priors is, in one line. Live on Base, registered on ACP.
2. **The deletion test.** The whole argument, side by side, above the fold.
   Same code, same provider, same claim, different memory, different action.
   This is the 40 points.
3. **Verify it yourself.** BaseScan links to the real jobs. The absence of a
   fourth transaction is itself the result, and the page says so.
4. **The problem.** 40,161 of 75,340 jobs never get a provider response. Why
   the gate exists at all.
5. **At scale.** 187 decisions changed across 18,367 replayed jobs, 126 right
   and 61 wrong. The aggregate case, with the win rate stated plainly.
6. **How it works.** Three stages, and which of them survives deletion.
7. **What Priors refuses to do.** Calibration and the learned threshold, both
   built, both measured, both switched off. Negative results shown on purpose.
8. **Honesty.** Backtest framing, self-dealing disclosure, evaluator caveat.
9. **Repo and links.**

## Type and colour

Tokens, so the page stays consistent without a framework:

```
ink          #07090d    background
paper        #f2f5f8    primary text
muted        #7d8b9c    secondary text
line         #1b2230    borders
amber        #ffb020    the accent, and the mark
red          #ff4d5a    memory intact / blocked
green        #2ee6a8    memory deleted / proceeded
```

Amber is the brand and marks the single most important figure in any section.
Red and green are never decorative: they mean blocked and proceeded, nothing
else, and they are used nowhere those words do not apply.

Type ramp:

```
display   clamp(56px, 9vw, 124px)   mono 800   the headline figures
figure    clamp(34px, 5vw, 50px)    mono 800   section figures
title     clamp(22px, 3vw, 33px)    sans 700   section headings
body      clamp(16px, 2vw, 19px)    sans 400   prose
label     14px                      mono 700   uppercase, 0.17em tracking
```

Spacing is a 4px scale. Section rhythm is 96px desktop, 56px mobile.

## The boring things, decided up front

- **Mobile.** A judge may open this on a phone from a Discord link. Single
  column throughout; the deletion test stacks red above green rather than
  shrinking to two unreadable columns.
- **Dark and light.** The viewer's theme wins. Tokens are defined on bare
  `:root`, redefined under `prefers-color-scheme: dark` and again under an
  explicit `[data-theme]`, so neither setting produces an unpainted body.
- **Wide content.** Any table or code block scrolls inside its own container.
  The page body never scrolls sideways.
- **No external requests.** No web fonts, no analytics, no CDN. System font
  stack. The page works offline and leaks nothing about who opened it.
- **Links.** Every external link says where it goes before it is clicked.
  BaseScan links show the truncated hash, not "click here".

## Explicitly out of scope

- Any interactive element. Nothing to toggle, query or configure. An
  interactive demo that half works on judging day is worse than a static one
  that reads perfectly.
- Any figure not regenerated and verified in the session that ships the page.
- Anything that changes Priors itself. The product is frozen.
