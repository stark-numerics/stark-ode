# Robertson

This benchmark uses the stiff Robertson chemical kinetics system:

```text
dy1/dt = -0.04 y1 + 1e4 y2 y3
dy2/dt = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
dy3/dt = 3e7 y2^2
```

with the initial condition:

```text
y(0) = [1, 0, 0]
```

The benchmark compares:

- STARK `SchemeKvaerno4` with the full Robertson right-hand side treated
  implicitly and a custom exact cubic resolvent
- SciPy `Radau` and `BDF`
- Diffrax `Kvaerno5` when Diffrax is installed

This makes Robertson a benchmark of the custom resolvent architecture as much as a raw solver race: the problem-specific cubic resolvent lets STARK treat the
whole Robertson right-hand side implicitly while still avoiding a generic
nonlinear iterative stage solve, so the report exposes what that exact stage
solve is buying against SciPy and Diffrax stiff solvers.







