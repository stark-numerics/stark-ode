# Monitor Design Notes

Monitors record what a solve did. They are attached deliberately and add work.

The monitor package is split by method layer:

```text
scheme      accepted/rejected step behaviour
resolvent   nonlinear iteration behaviour
inverter    linear solve behaviour
```

`Monitor` is the bundling object that lets a comparison or user-facing run
attach all three consistently.

## Hot Path Boundary

Monitoring must not be hidden inside unmonitored hot paths. Schemes,
resolvents, and inverters should choose monitored wrappers only when a monitor
is supplied.

This preserves two important properties:

```text
unmonitored solves stay lean
monitored solves make their extra work explicit
```

## Design Rule

A monitor should observe and summarise. It should not decide acceptance,
convergence, step size, or correction values.

## Beta Surface

`Monitor` is user-facing. The high-level summary objects are intended to remain
readable and stable enough for reports.

The low-level event records, such as individual scheme steps, resolvent solves,
and inverter solves, are diagnostic evidence. They are public enough to inspect
while debugging, but their exact fields may change before the diagnostics API is
declared stable. Prefer examples and docs that teach the summary-level workflow
unless the lesson is explicitly about low-level evidence.
