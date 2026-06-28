# Diagnostics Design Notes

Diagnostics observe and explain solves. They must not change the numerical
meaning of a solve.

## Role

Diagnostics own:

- monitors,
- comparison runners,
- timing summaries,
- profile summaries,
- display/writer helpers for diagnostic reports.

They may inspect method, resolvent, inverter, and integrator behaviour. They
should not become part of those hot paths unless explicitly attached.

## Monitoring Versus Timing

Monitoring is diagnostic machinery. It records behaviour such as accepted
steps, rejected steps, residuals, and inverter solves. It adds work.

Timing should normally use unmonitored runs. If a report needs both behaviour
and timing, run a monitored diagnostic pass separately from the measured pass.

## Comparison

Comparison helpers are for comparing method choices on the same problem. They
are useful user tools, but they should remain transparent about setup, warmup,
repeated timings, and profiling.

Competition reports and benchmarks may build on diagnostics, but they should
not collapse all timing categories into one headline.

## Design Rule

Diagnostics should answer:

```text
What happened, how often, and how long did the unobserved solve take?
```

They should not answer by quietly changing the solve they are measuring.
