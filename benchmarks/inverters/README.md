# Inverter refactor smoke benchmarks

This directory contains manual, non-gating timing scripts for checking obvious
inverter hot-path regressions during refactors.

These scripts are not CI tests and should not be interpreted as portable solver
performance claims. They are local before/after smoke checks for contributors.

## Inverter layer refactor benchmark

Run from the repository root:

python -m benchmarks.inverters.bench_inverter_refactor

The benchmark covers the current built-in iterative inverters on two small
systems:

scalar block: a one-item block with a scalar linear operator
coupled block: a small dense block operator written through STARK's block
operator contract

The scalar case makes call-routing and setup overhead visible. The coupled case
exercises the Krylov iteration workers.

Command-line options
--repeat N
    Number of timed samples per case.
    Default: 7

--warmup N
    Number of untimed warmup samples per case.
    Default: 2

--solves-per-sample N
    Number of repeated inverter solves inside one timed sample.
    Default: 200

--save-baseline NAME
    Save the current timing run as a named JSON baseline under
    benchmarks/inverters/results/.

--compare-baseline NAME
    Compare the current timing run against a previously saved baseline.
    The comparison reports median timing ratios. Values above 1.00x are
    slower than the saved baseline.

## Example workflows

### Run a quick timing check:

python -m benchmarks.inverters.bench_inverter_refactor

### Run a slightly more stable local check:

python -m benchmarks.inverters.bench_inverter_refactor --warmup 3 --repeat 10 --solves-per-sample 500

### Save a baseline before a risky refactor:

python -m benchmarks.inverters.bench_inverter_refactor --warmup 3 --repeat 10 --solves-per-sample 500 --save-baseline pre-inverter-refactor

### Compare current timings against that baseline later:

python -m benchmarks.inverters.bench_inverter_refactor --warmup 3 --repeat 10 --solves-per-sample 500 --compare-baseline pre-inverter-refactor

Baseline files are written to:

benchmarks/inverters/results/

These files are intended for local comparison. Commit only intentionally curated
reference baselines.
