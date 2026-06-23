# Derivative Design Notes

Derivatives are the user-side language for the ODE right-hand side.

This package exists because users should be able to write the equation in a
natural form, while schemes receive a consistent in-place worker that can reuse
translations.

## Mathematical Role

For an ODE:

```text
y' = f(t, y)
```

the derivative is `f`. In STARK's scheme-facing contract it writes into an
output translation:

```text
derivative(interval, state, out) -> None
```

## User-Facing Styles

`DerivativeStyle` adapts common user shapes into the scheme-facing contract:

- accepts an instant and writes,
- accepts an instant and returns,
- kernel variants that bind named fields and parameters,
- split derivatives for IMEX methods.

Decorator names should say what the callable accepts and whether it writes or
returns. This is intentionally a little long-winded; ambiguity here makes
examples dangerous.

## Kernel Variants

Kernel styles exist to reduce repeated field discovery and to give engines more
structure for prepared algebra/acceleration. They should not invent a second
derivative language. They are adapters for field-focused callables.

## Split Derivatives

Split derivatives belong here because an IMEX split is a property of the
problem statement:

```text
y' = implicit_part(t, y) + explicit_part(t, y)
```

The method decides how to use the split, but the user declares the split as
part of the derivative.

## Design Rule

Keep derivative adaptation in `problem`. Schemes should receive a prepared
worker and should not need to inspect user callback signatures.
