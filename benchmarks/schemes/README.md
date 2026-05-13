# Scheme refactor smoke benchmarks

This directory contains manual, non-gating timing scripts for checking obvious
scheme hot-path regressions during refactors.

These scripts are not CI tests and should not be interpreted as portable solver
performance claims. They are local before/after smoke checks for contributors.

## Scheme-owned call refactor benchmark

Run from the repository root:

```powershell
python -m benchmarks.schemes.bench_scheme_refactor

The benchmark covers one representative from each currently refactored scheme
family:

fixed explicit: RK4
adaptive explicit: Bogacki-Shampine
fixed implicit: Backward Euler
adaptive implicit: Kvaerno3
adaptive IMEX: Kennedy-Carpenter32

The output reports best, median, and worst local timings in milliseconds.

Command-line options
--repeat N
    Number of timed runs per case.
    Default: 7

--warmup N
    Number of untimed warmup solves per case.
    Default: 2

--save-baseline NAME
    Save the current timing run as a named JSON baseline under
    benchmarks/results/.

--compare-baseline NAME
    Compare the current timing run against a previously saved baseline.
    The comparison reports median timing ratios. Values above 1.00x are
    slower than the saved baseline.

Example workflows

Run a quick timing check:

python -m benchmarks.schemes.bench_scheme_refactor

Run a slightly more stable local check:

python -m benchmarks.schemes.bench_scheme_refactor --warmup 3 --repeat 10

Save a baseline before a risky refactor:

python -m benchmarks.schemes.bench_scheme_refactor --save-baseline pre-support-audit

Compare current timings against that baseline later:

python -m benchmarks.schemes.bench_scheme_refactor --compare-baseline pre-support-audit

You can combine run-count options with baseline options:

python -m benchmarks.schemes.bench_scheme_refactor --warmup 3 --repeat 10 --compare-baseline pre-support-audit
Baseline files

Baseline files are written to:

benchmarks/results/

These files are intended for local comparison. Commit only intentionally curated
reference baselines.

For historical or repeatable benchmarking, use the ASV suite documented in
docs/benchmarking.md.