# Diagnostics, monitors, and timing

This page is for users who want to understand what a solve did.

Diagnostics are useful, but they are not free. Keep observation and timing separate.

## Observe a solve with monitors

Use monitors when you want counts, summaries, residual history, or method behaviour.

```powershell
python -m examples.features.monitor_scheme_steps
python -m examples.features.monitoring_levels
python -m examples.features.compare_with_monitor_summary
```

A monitored run can answer:

```text
How many accepted steps?
How many rejected steps?
How many nonlinear iterations?
How many linear solves?
Did the method refresh linearization often?
```

## Do not time a monitored run as if it were raw solver speed

Monitoring adds calls, allocations, and recording work. It is diagnostic machinery.

Use:

```text
monitored run      understand behaviour
unmonitored run    measure speed
```

Run:

```powershell
python -m examples.features.monitor_vs_timing
```

## Read competition timing tables

Competition reports normally distinguish:

```text
Preparation time    setup + first warm solve
Warm run time       repeated solve after setup and warmup
Total time          preparation + one measured solve
```

This matters for JIT/GPU libraries. They may move substantial work into preparation.

Example commands:

```powershell
python -m competition.robertson.report
python -m competition.hires.report
python -m competition.allen_cahn_1d.report
```

## Which timing should I quote?

Use the timing that matches your use case.

| Use case | Timing |
|---|---|
| One solve from a cold setup | total time |
| Many repeated solves with same shape | warm run time |
| Measuring compile/setup overhead | preparation table |
| Comparing user-visible latency | total time |
| Comparing steady-state throughput | warm run time |

## Accuracy summaries

Comparison reports may include final error or RMS error. Treat them as regression/comparison tools, not mathematical proof.

If two methods use different tolerances, linear algebra, or checkpoint policies, compare both accuracy and timing.
