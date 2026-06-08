# FPUT-beta

This benchmark uses the Fermi-Pasta-Ulam-Tsingou beta lattice with fixed endpoints:

```text
dq_i/dt = p_i
dp_i/dt = (q_{i+1} - 2 q_i + q_{i-1}) + beta[(q_{i+1} - q_i)^3 - (q_i - q_{i-1})^3]
```

for `i = 1, ..., N`, with `q_0 = q_{N+1} = 0`.

The default parameters are:

```text
beta = 0.25
amplitude = 0.1
```

The initial condition excites the first normal mode:

```text
q_i(0) = amplitude * sin(pi i / (N + 1))
p_i(0) = 0
```







