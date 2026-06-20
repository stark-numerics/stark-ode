# STARK documentation

This manual is organised by what you are trying to do.

## Solve a problem

1. [Getting started](getting-started.md): solve a small ODE with the high-level API.
2. [Define a problem](problem.md): use `System`, `Frame`, `Derivative`, and `Linearizer`.
3. [Choose a method](methods.md): choose schemes and customise numerical method components.
4. [Solve stiff problems](implicit.md): add linearizers, resolvents, inverters, and preconditioners.
5. [Use engines](engines.md): choose NumPy, Numba, JAX, or CuPy-backed storage/arithmetic.
6. [Use diagnostics](diagnostics.md): observe, compare, and time solves.

## Go deeper

- [Examples guide](examples.md): runnable scripts by topic.
- [Extending STARK](extending.md): write a scheme, resolvent, inverter, preconditioner, or monitor.
- [Foreign models](foreign-models.md): connect existing model objects through custom state and translation contracts.
- [Mathematical contracts](contracts_math.md): formal reference for the low-level model.

## Contributor and maintainer notes

These pages describe internal design paths and are more fragile than the user manual.

- [House style](contributing/house_style.md): naming and design conventions.
- [Algebraist backend paths](contributing/algebraist_backends.md): how generated/runtime algebra, engines, carriers, and accelerators connect.
- [Benchmarking](contributing/benchmarking.md): competitions, warm/total timing, and future benchmark work.

## Expected route through the manual

Most users should read only:

```text
getting-started -> problem -> methods
```

Read `implicit` when you need stiff or nonlinear implicit solves. Read `foreign-models` only when the high-level `Frame` path cannot represent your existing model naturally.
