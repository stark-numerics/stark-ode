# FitzHugh-Nagumo 1D

This benchmark uses a stiff one-dimensional FitzHugh-Nagumo reaction-diffusion
system with periodic boundary conditions:

```text
u_t = D u_xx + u - u^3 / 3 - v
v_t = epsilon (u + a - b v)
```

on a periodic one-dimensional domain, discretized on a uniform grid.

The default parameters are:

```text
D = 1.0
epsilon = 0.08
a = 0.7
b = 0.8
```

The benchmark uses a localized excited initial condition and compares solver
outputs against a tight numerical reference solution.

It compares:

- STARK `SchemeKvaerno3` with `ResolverAnderson`, the current best local STARK
  combination on this problem
- SciPy `Radau` and `BDF`
- Diffrax `Kvaerno5` when Diffrax is installed

Run it with:

```powershell
python -m benchmarks.fitzhugh_nagumo_1d.report
```
