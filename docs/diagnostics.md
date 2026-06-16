# Diagnostics

Diagnostics help answer what the solver did, not just what value it returned.

## Monitors

Monitors observe solver internals such as:

- accepted and rejected steps;
- scheme stages;
- nonlinear residuals;
- inverter defects;
- iteration counts.

Monitoring is intentionally separate from normal timing. A monitored solve allocates and records more information, so it should not be compared directly with an unmonitored performance run.

See:

```powershell
python -m examples.features.monitor_vs_timing
```

## Comparison reports

Comparison reports live under `competition/`. They compare solvers on named problems and report timing and accuracy.

The timing tables distinguish:

- **preparation time**: setup plus warmup, including compilation or one-time preparation when it occurs;
- **warm run time**: repeated solves after setup and warmup;
- **total time**: preparation plus a measured solve.

This distinction is especially important for JIT-based or GPU-backed libraries, where a large amount of work may occur before the first warm run.

## Accuracy summaries

Comparison reports may show final error or RMS error relative to a reference. These summaries are useful for comparing configured runs, but they do not replace mathematical convergence analysis.

## Profiling

Use profiling for implementation work, not ordinary problem solving. A profile should normally be collected on an unmonitored solve unless the monitor itself is the target of the investigation.
