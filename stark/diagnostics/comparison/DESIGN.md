# Comparison Design Notes

The comparison package runs several configured `Method` entries on one prepared
`SystemIVP` problem and reports timing, diagnostics, profiles, and pairwise
differences.

It is a diagnostic tool, not a new integration layer.

## Run Separation

Comparison deliberately separates:

```text
setup      build stepper/integrator entry objects
observed   monitored run for behaviour summaries
warmup     unmeasured timing warmup
timed      repeated unmonitored runs
profiled   profiler run for approximate self-time categories
```

These runs answer different questions. Collapsing them into one run would make
the report easier to implement but less honest.

## Entry Boundary

Entries provide method choices. The runner prepares each method through the
same system, engine, dynamics, and configuration that an ordinary solve uses.
This keeps comparison useful for both built-in methods and contributor-written
schemes:

```python
ComparisonEntry("custom", Method(MyScheme))
```

The runner should not know the internal details of a scheme, resolvent, or
inverter. If custom entries need better profile categories, supply a category
function rather than teaching the runner about that external package.

## Design Rule

Comparison reports should make measurement conditions visible: setup, warmup,
repeat count, monitoring, profiling, checkpoints, and difference functions all
matter.
