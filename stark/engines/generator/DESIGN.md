# Generator Design Notes

Last reviewed: 2026-07-13.

This package is a staging area for the engine-facing generation surface that
will eventually sit underneath Algebraist and downstream packages such as
stark-pde.

## Goal

The goal is not to refactor Algebraist for neatness. The goal is to expose the
engine decisions that make efficient generated code possible.

Algebraist currently prepares vector-space algebra for known `Frame` layouts.
stark-pde will need similar backend-aware generation for spatial operators,
derivative stencils, and symbolic PDE expressions. These should share a small
policy vocabulary without forcing PDE concepts into stark-ode.

## Generator Shape

A future `Generator` should be a backend-aware preparation object. It can own
families such as:

```text
Algebraist generation  linear combinations, norms, inner products, stencils
PDE generation         derivative stencils, fluxes, source expressions
runtime generation     flexible fallback for unknown or foreign state shapes
```

The default should be runtime-safe. A generator should choose emitted code only
when it has enough structure to do so honestly:

```text
unknown state shape       -> runtime providers
known Frame-like layout   -> generated algebra providers
known PDEFrame + lattice  -> generated PDE operators
```

This avoids the current failure mode where a concrete engine may silently fall
back from accelerated code generation to a different execution style.

## Policy

`GeneratorPolicy` describes source-shape decisions, not engine ownership.

It should answer questions such as:

```text
Should generated code mutate output objects or return new values?
Should fields be walked by Python loops, unrolled indices, vectorized array
expressions, or backend-native kernels?
How should backend scalar values cross into Python control code?
Is this path runtime-only or generated?
```

It should not own:

```text
frame
carriers
allocator
accelerator
translation factory
```

Those remain engine resources. The policy is the shared vocabulary that lets
Algebraist and future stark-pde generators agree on how code should be shaped.

## Relationship to Algebraist

Algebraist should continue working while this package grows. Once the policy
surface is stable, Algebraist wiring can move over in small steps:

```text
engine chooses GeneratorPolicy
engine chooses Algebraist target
Algebraist consumes policy + target
later, stark-pde consumes the same policy for PDEGenerator
```

Runtime Algebraist support should survive. It is the correct path for
user-defined state families where stark-ode cannot know enough structure to
emit good code.

## Open Questions

- Should a future `Generator` be a facade containing algebra/PDE families, or
  should `AlgebraistGenerator` and `PDEGenerator` be sibling consumers of the
  same policy?
- Should generated target selection remain separate from policy, or should
  policy grow enough metadata to choose targets automatically?
- How much carrier capability metadata should become part of the policy, and
  how much should remain on concrete carriers?
