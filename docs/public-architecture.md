# Public Architecture

This document describes the externally visible architecture without publishing the full prediction model, feature set, memory schema, or rule-generation details.

## Components

### Priors decision process

The decision process:

1. receives the current hiring context and available provider signals
2. produces a deterministic prediction
3. records the prediction and relevant experience
4. uses recalled experience when making later decisions

The numerical decision path is pure code. Language models, when used, are kept outside the decision arithmetic and may only generate explanatory text after a decision has already been computed.

### Outcome resolver

A separate deterministic worker observes completed ACP outcomes and writes outcome-related experience back to persistent state.

Write ownership is intentionally disjoint:

| Process | Owns |
|---|---|
| Priors | predictions and decision records |
| Outcome Resolver | episodes, patterns, rules, and rule performance |

Neither process writes the records owned by the other.

### Persistent experience

Persistent state allows a fresh session to recover relevant prior experience. The important property is behavioral continuity across sessions.

Deleting or replacing the experience store should therefore remove the learned component of the decision process while leaving the underlying ACP integration and deterministic decision machinery intact.

## What is deliberately not specified here

The public repository does not yet document:

- the complete prediction feature set
- exact confidence equations
- memory field schema
- rule-generation thresholds
- provider-specific scoring details
- replay datasets
- evaluation examples containing unreleased results

Those details belong in the internal development specification until the implementation and evaluation evidence are ready to publish.

## Reproducibility

The prediction path is designed to be deterministic for a fixed input dataset and fixed configuration.

Model-generated explanations are not part of the numerical decision path.

Completed outcomes are distinguished from jobs whose outcomes remain unresolved, so incomplete observations are not silently treated as failures.
