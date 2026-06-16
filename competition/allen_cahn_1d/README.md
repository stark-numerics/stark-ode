# Allen-Cahn 1D Krylov competition

This report stretches the matrix-free Krylov inverter path on a moderate
periodic PDE problem:

```text
u_t = D u_xx + u - u^3
```

STARK uses an implicit SDIRK21 Newton solve with `InverterKrylovArnoldi`.
SciPy rows use sparse Jacobians, so this report is also a useful reminder that
SciPy is the natural competitor when mature sparse linear algebra dominates.
Diffrax is included as an optional JAX/JIT row when available.

Run from a source checkout with:

```powershell
python -m competition.allen_cahn_1d.report
```
