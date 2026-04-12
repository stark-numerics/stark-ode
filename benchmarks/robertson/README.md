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

The benchmark compares fixed-step backward Euler in STARK, with both Picard and
Newton resolution, against SciPy's stiff solvers and Diffrax's `Kvaerno5` when
Diffrax is installed.
