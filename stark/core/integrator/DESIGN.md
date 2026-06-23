# Integrator Design Notes

The integrator package owns trajectory production. It does not own numerical
method logic.

This is one of the older parts of STARK and should be treated as a future
design-review target. This note documents the current boundary rather than
claiming the design is final.

## Current Role

The integrator layer coordinates:

- repeated calls to a stepper,
- interval advancement,
- mutable trajectory generation,
- checkpoint emission,
- endpoint/final-result helpers,
- optional progress checking.

The stepper bridges a scheme into the integration loop. The scheme decides what
step is accepted and how the state changes. The integrator decides how accepted
steps become a trajectory visible to callers.

## Mutating Semantics

Current trajectory paths are mutating by design. They reuse state objects where
possible and yield the current solver-owned state. This is efficient and honest,
but callers must not treat yielded state objects as immutable historical
snapshots unless the API explicitly promises snapshots.

## Checkpoints

Checkpoints are output requests, not method steps. An adaptive scheme may take
internal steps that do not align with checkpoint times. The integrator is the
right layer to mediate output timing without forcing the method to step on a
fixed grid.

## Progress Checks

Progress checks are a safety feature. They detect accepted steps that fail to
advance time. They should remain configurable and should not become method
policy.

## Future Review

When revisiting this package, ask:

- Are mutating and snapshotting paths named clearly enough?
- Is checkpoint interpolation/selection still in the right layer?
- Is `IntegratorStepper` still the right bridge shape?
- Can final-result helpers stay convenient without hiding mutability?
