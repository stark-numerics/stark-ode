# STARK documentation

STARK has a layered interface. You do not need to understand every layer to solve an ODE.

## Recommended path

1. Read [Getting started](getting-started.md) for the high-level `System`, `Frame`, `Method`, and `Engine` path.
2. Read [Problem objects](problem.md) when you need derivatives, linearizers, or structured state fields.
3. Read [Methods](methods.md) when you need to choose or replace schemes, resolvents, inverters, predictors, or monitors.
4. Read [Implicit methods](implicit.md) when using Newton, Picard, dense inverters, Krylov inverters, or preconditioners.
5. Read [Foreign models](foreign-models.md) if your simulation already has its own state and tangent/translation representation.

## Pages

- [Getting started](getting-started.md) - the ordinary solver path.
- [Problem objects](problem.md) - `System`, `Frame`, `Derivative`, `Linearizer`, and IVPs.
- [Methods](methods.md) - `Method`, schemes, resolvents, inverters, and predictors.
- [Implicit methods](implicit.md) - nonlinear stage equations and linear correction solves.
- [Engines](engines.md) - Native, NumPy, JAX, CuPy, carriers, and acceleration boundaries.
- [Diagnostics](diagnostics.md) - monitors, comparisons, preparation timing, warm timing, and total timing.
- [Examples](examples.md) - a curated map of runnable examples.
- [Extending STARK](extending.md) - where to plug in custom numerical components.
- [Foreign models](foreign-models.md) - low-level state, translation, allocator, derivative, and linearizer integration.
- [Mathematical contracts](contracts_math.md) - the formal interpretation of STARK contracts.
- [Benchmarking](benchmarking.md) - comparison reports and ASV benchmarks.
- [House style](contributing/house_style.md) - contributor-facing naming and implementation conventions.

## The three surfaces

### Ordinary use

Most users start here:

```text
System + Frame + Method + Engine
```

STARK owns the translation objects, output buffers, and stepping machinery. You provide fields, an initial value, a derivative, and a method.

### Numerical intervention

Users who need method-level control can replace pieces:

```text
Scheme + Resolvent + Inverter + Predictor + Monitor
```

This keeps the high-level problem interface while allowing custom time-stepping or nonlinear/linear solve policy.

### Foreign-model integration

Users with existing simulation objects can bypass the `Frame` convenience layer and provide low-level contracts:

```text
State + Translation + Allocator + Derivative + Linearizer
```

This is the path for domain models that should not be flattened solely to call an ODE solver.
