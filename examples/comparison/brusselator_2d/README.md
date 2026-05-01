# 2D Brusselator

This benchmark uses a periodic semi-discretized 2D Brusselator
reaction-diffusion system:

```text
u_t = alpha (u_xx + u_yy) + A + u^2 v - (B + 1) u
v_t = alpha (v_xx + v_yy) + B u - u^2 v
```

on a square periodic domain, discretized on a uniform grid.

The default parameters are:

```text
alpha = 0.02
A = 1.0
B = 3.4
```

The benchmark uses a smooth perturbed initial condition around the homogeneous state and compares solver outputs against a tight numerical reference solution.







