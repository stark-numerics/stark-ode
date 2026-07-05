# Benchmark Design

The benchmark suite is built around three independent axes:

- problem definitions;
- method stacks;
- engines.

This mirrors the package architecture. A STARK performance result is rarely
"the speed of a scheme" in isolation. It is the behaviour of a scheme,
resolvent, inverter, engine, and problem shape working together.

## To Do

- Expand ASV coverage for scheme calls.
- Expand ASV coverage for backend comparisons.
- Expand ASV coverage for generated Algebraist kernels.
- Expand ASV coverage for dense/direct and Krylov inverter paths.
- Expand ASV coverage for representative end-to-end IVP runs.
- Add Torch to backend comparison benchmarks after a Torch engine exists.

## Catalogue Layers

`stark.methods.METHOD_CATALOGUE` owns package-level method-stack recipes. It
knows about schemes, resolvents, inverters, maturity, and benchmark tier.

`benchmarks.catalogue` owns benchmark-only metadata: representative problems,
engine entries, compatibility filters, and the benchmark axis object that joins
one problem, one method stack, and one engine.

Compatibility means more than matching state shape. Most current reusable
benchmark problems are NumPy-authored: they build NumPy initial arrays and use
into-style kernels that write into mutable outputs. Those are honest NumPy
benchmarks, but they are not automatically JAX or CuPy benchmarks. JAX needs
expression-style returning kernels, and CuPy needs device-resident initial values
plus GPU-shaped array operations. Add backend-specific problem definitions
before expanding those axes.

`benchmarks.problems` owns reusable problem definitions. A problem definition can
build a fresh `System`, initial values, interval, and optional final-state
reference. Problems do not know about timing or ASV.

`benchmarks.builders` owns the handshake between catalogues and runnable IVPs.
ASV classes should stay thin and delegate construction to this layer.

## Benchmark Tiers

Benchmark tiers keep the suite usable at different speeds:

- `smoke`: cheap enough to run during beta-release checks;
- `representative`: broader coverage for real performance work;
- `exhaustive`: cartesian-product coverage for dedicated benchmark runs.

The smoke tier should remain small. It is a release sanity check, not the full
performance story.

## ASV Shape

ASV benchmark classes should time separate stages:

- IVP setup;
- first solve;
- repeat solve after warmup;
- final-state error where a reference exists.

This separation matters because JAX, CuPy, Numba, generated Algebraist kernels,
and dense implicit stacks can move cost between preparation and repeated solves.

ASV parameters should be stable strings such as
`robertson/kvaerno5-newton-dense/numpy`. The builder resolves those strings back
to catalogue axes. This keeps ASV output readable and avoids serialising rich
objects as benchmark parameters.

## Coverage Targets

- FPUT-style coupled Hamiltonian chains are useful future representative
  problems for large coupled states and energy/invariant drift.
- Arity-based Algebraist linear-combine timing is useful for generated-kernel
  regressions.
- Dense/direct inverter timing should return as catalogue-driven ASV coverage,
  not as hand-rolled local scripts.
- Native-engine benchmarks should get native-shaped problem variants rather
  than reusing NumPy-array initial states.
- A tiny manual smoke layer may still be useful during dirty-tree refactors, but
  durable performance history should come from ASV.
