# STARK documentation

This manual is organised by what you are trying to do.

```{toctree}
:hidden:
:maxdepth: 2

concepts
getting-started
problem
methods
implicit
engines
diagnostics
examples
extending
foreign-models
contract-maths
reference/index
contributing/README
contributing/house-style
```

## Solve a problem

1. [Concepts and terminology](concepts.md): learn STARK's names for ODE, IVP, state, derivative, linearizer, resolvent, inverter, and engine.
2. [Getting started](getting-started.md): solve a small ODE with the high-level API.
3. [Define a problem](problem.md): use `System`, `Frame`, `Derivative`, and `Linearizer`.
4. [Choose a method](methods.md): choose schemes and customise numerical method components.
5. [Solve stiff problems](implicit.md): add linearizers, resolvents, inverters, and preconditioners.
6. [Use engines](engines.md): choose NumPy, Numba, JAX, or CuPy-backed storage/arithmetic.
7. [Use diagnostics](diagnostics.md): observe, compare, and time solves.

## Go deeper

- [Examples guide](examples.md): runnable scripts by topic.
- [Extending STARK](extending.md): write a scheme, resolvent, inverter, preconditioner, or monitor.
- [Foreign models](foreign-models.md): connect existing model objects through custom state and translation contracts.
- [Contract maths](contract-maths.md): formal reference for the low-level model.
- [API reference](reference/index.md): generated reference pages for public
  modules and advanced surfaces.

## Contributor and maintainer notes

These pages describe internal design paths and are more fragile than the user manual.

- [Contributor notes](contributing/README.md): where contributor documentation lives and how to use local `DESIGN.md` files.
- [House style](contributing/house-style.md): naming and design conventions.

## Expected route through the manual

Most users should read only:

```text
concepts -> getting-started -> problem -> methods
```

Read `implicit` when you need stiff or nonlinear implicit solves. Read `foreign-models` only when the high-level `Frame` path cannot represent your existing model naturally.
