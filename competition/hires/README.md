# HIRES Competition

This report solves the classic 8-variable HIRES stiff chemical kinetics system.
It is intended as a guardrail for the implicit dense stack: larger than
Robertson's 3-variable problem, but still small enough that dense inverse-action
paths are appropriate.

Run it with:

```powershell
python -m competition.hires.report
```

The STARK rows use Kvaerno5 with Newton, Chord, and VeryChord resolvents backed
by the dense inverter/nucleus path. SciPy Radau and BDF rows provide flat-array
baseline comparisons. Diffrax Kvaerno5 is included when Diffrax/JAX are locally
installed.

## Problem

HIRES is an 8-variable stiff chemical kinetics benchmark. The state is

\[
y(t) = (y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7),
\]

with initial condition

\[
y(0) = (1, 0, 0, 0, 0, 0, 0, 0.0057)
\]

and final time \(t = 321.8122\).

The ODE system is:

\[
r = 280 y_5 y_7
\]

\[
\begin{aligned}
y_0' &= -1.71y_0 + 0.43y_1 + 8.32y_2 + 0.0007, \\
y_1' &= 1.71y_0 - 8.75y_1, \\
y_2' &= -10.03y_2 + 0.43y_3 + 0.035y_4, \\
y_3' &= 8.32y_1 + 1.71y_2 - 1.12y_3, \\
y_4' &= -1.745y_4 + 0.43y_5 + 0.43y_6, \\
y_5' &= -r + 0.69y_3 + 1.71y_4 - 0.43y_5 + 0.69y_6, \\
y_6' &= r - 1.81y_6, \\
y_7' &= -r + 1.81y_6.
\end{aligned}
\]

The benchmark uses this problem to exercise small-but-not-tiny implicit dense
solves. Unlike Robertson, the dense inverse-action is 8-dimensional, so the
generic dense/nucleus path is tested rather than only the direct 3×3 kernel.