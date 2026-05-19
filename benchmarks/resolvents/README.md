# Resolvent refactor smoke benchmarks

This directory contains manual, non-gating timing scripts for checking obvious
resolvent hot-path regressions during refactors.

These scripts are not CI tests and should not be interpreted as portable solver
performance claims. They are local before/after smoke checks for contributors.

## Resolvent layer refactor benchmark

Run from the repository root:

python -m benchmarks.resolvents.bench_resolvent_refactor

The benchmark covers representative one-stage and coupled-stage resolvent
paths:

one-stage Picard
one-stage Anderson
one-stage Broyden
one-stage Newton
coupled Picard
coupled Newton

The one-stage cases make shifted residual overhead visible. The coupled cases
exercise block residual construction, which is one of the places future
Algebraist-backed resolvent work should be able to help.

Command-line options
--repeat N
    Number of timed samples per case.
    Default: 7

--warmup N
    Number of untimed warmup samples per case.
    Default: 2

--solves-per-sample N
    Number of repeated resolvent solves inside one timed sample.
    Default: 200

--save-baseline NAME
    Save the current timing run as a named JSON baseline under
    benchmarks/resolvents/results/.

--compare-baseline NAME
    Compare the current timing run against a previously saved baseline.
    The comparison reports median timing ratios. Values above 1.00x are
    slower than the saved baseline.

## Example workflows

### Run a quick timing check:

python -m benchmarks.resolvents.bench_resolvent_refactor

### Run a slightly more stable local check:

python -m benchmarks.resolvents.bench_resolvent_refactor --warmup 3 --repeat 10 --solves-per-sample 500

### Save a baseline before a risky refactor:

python -m benchmarks.resolvents.bench_resolvent_refactor --warmup 3 --repeat 10 --solves-per-sample 500 --save-baseline pre-resolvent-algebraist

### Compare current timings against that baseline later:

python -m benchmarks.resolvents.bench_resolvent_refactor --warmup 3 --repeat 10 --solves-per-sample 500 --compare-baseline pre-resolvent-algebraist

Baseline files are written to:

benchmarks/resolvents/results/

These files are intended for local comparison. Commit only intentionally curated
reference baselines.
