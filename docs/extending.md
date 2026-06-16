# Extending STARK

This page is for users who want to replace a numerical component while keeping the high-level `System` and `Frame` path.

For existing simulation objects with their own state and tangent representation, see [Foreign models](foreign-models.md).

## Extension map

| Goal | Component to provide |
|---|---|
| Change the right-hand side style | `Derivative` / `DerivativeStyle` |
| Add Jacobian actions | `Linearizer` / `LinearizerStyle` |
| Add a time-stepping method | `Scheme` |
| Add nonlinear stage solving policy | `Resolvent` |
| Add a linear correction solver | `Inverter` |
| Add a Krylov preconditioner | `InverterKrylovPreconditionerLike` |
| Add implicit initial-guess policy | `SchemePredictorLike` |
| Observe solver internals | monitor objects |
| Use existing state and translation objects | low-level contracts, see [Foreign models](foreign-models.md) |

## Custom derivatives

Derivatives can be in-place, return-style, or kernel-adapted. Prefer the style that matches your backend.

See:

```powershell
python -m examples.features.derivative_styles
```

## Custom linearizers

Implicit Newton-style methods require Jacobian actions. Provide a linearizer when using Newton, chord, very-chord, dense inverters, or matrix-free Krylov paths.

See:

```powershell
python -m examples.features.linearizer_styles
```

## Custom schemes

A scheme owns time-stepping structure. Implement a custom scheme when the tableau or step acceptance logic itself is new.

See:

```powershell
python -m examples.features.custom_scheme_fixed_explicit
```

## Custom resolvents

A resolvent owns the nonlinear solve for an implicit stage. Implement a custom resolvent when Newton/Picard/chord-style policies are not the right nonlinear iteration.

## Custom inverters

An inverter owns linear correction solves. Implement a custom inverter when dense, relaxation, or Krylov solvers are not the right linear algorithm.

The current inverter shape is request-based:

```text
inverter(request, output)
```

The request supplies the operator and residual information. The inverter writes or improves the output correction.

## Custom preconditioners

Preconditioners are normally owned by Krylov inverters. A preconditioner should exploit problem structure without making the Krylov algorithm itself problem-specific.

## Custom monitors

Monitors should observe, not control, solver execution. Keep monitored and unmonitored timing separate.
