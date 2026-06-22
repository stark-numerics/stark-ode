# Mathematical contracts

This is an advanced reference. Read [Foreign models](foreign-models.md) first if your immediate goal is to connect an existing model object.

STARK's user-friendly path hides most of these contracts behind `System`, `Frame`, `Method`, and `Engine`. This document names the lower-level mathematical objects so contributors and advanced users can reason about extensions.

## State and translation

STARK distinguishes the model state from a solver increment.

```text
state        x in a state space S
translation  v in a translation/tangent-like space T_x or T
```

A translation can be applied to a state to produce another state:

```text
x_new = x + v
```

For `Frame`-backed models, STARK generates state and translation carriers from named fields. For foreign models, the user provides these objects.

## Derivative

An ODE right-hand side is represented as a derivative:

```text
f(t, x) -> v
```

STARK's scheme-facing derivative contract writes into an output translation:

```text
derivative(interval, state, out) -> None
```

Adapters such as `DerivativeStyle.accepts_instant_returns` and `DerivativeStyle.kernel_accepts_instant_writes` let users write more natural problem-level functions while the solver receives a prepared derivative kernel.

## Linearizer

A linearizer represents the Jacobian action of the derivative:

```text
Df(t, x)[v]
```

Implicit Newton-style methods use linearizers to build correction equations. A linearizer may provide:

```text
operator action     apply J to a translation
dense fill          materialise J for dense inversion
```

Dense inverters can use dense fill. Krylov inverters can work with operator action alone.

## Blocks

Implicit schemes often solve product-space equations involving several stage translations. STARK represents those stage products with blocks.

```text
Block[T] = T x T x ... x T
```

Blocks let schemes, resolvents, and inverters work with stage vectors without flattening user state by default.

## Resolvents

A resolvent solves a nonlinear stage equation. It receives a request describing the equation and writes or improves a stage correction.

Examples:

```text
Picard       fixed-point iteration
Newton       linearized correction iteration
Chord        reused linearization
VeryChord    more aggressively reused linearization
```

The scheme owns stage structure. The resolvent solves the equation it is given.

## Inverters

An inverter solves a linear correction equation represented by an `InverterRequest`.

Current shape:

```text
inverter(request, output) -> None
```

The inverter should say whether it writes `output` or improves an existing guess through its output mode.

Examples:

```text
InverterDense            small dense systems
InverterKrylovArnoldi    matrix-free large systems
relaxation inverters     simple iterative baselines
```

## Preconditioners

A preconditioner is an inverter-side helper for Krylov-style solves. It applies an approximate inverse or easier correction solve to reduce Krylov iteration count.

Preconditioners should remain explicit. Do not hide problem-specific preconditioning inside a generic Krylov algorithm.

## Predictors

A scheme predictor seeds the initial guess for an implicit stage. It belongs to the scheme layer.

```text
SchemePredictorKnown
SchemePredictorZero
SchemePredictorPrevious
```

The predictor should not know about the linearizer or inverter.

## Engines and Algebraist

The engine chooses carriers, allocator behaviour, backend arithmetic, and acceleration. For `Frame`-backed states, STARK can derive an Algebraist frame and generate prepared kernels for operations such as linear combinations, norms, and state application.

For foreign states whose structure is not known, STARK may use runtime algebra instead. Runtime algebra is flexible but is not the preferred accelerated path for known `Frame` models.

## Summary table

| Mathematical role | STARK role |
|---|---|
| state space | state object / `Frame` state fields |
| tangent or increment space | translation object / `Frame` translation fields |
| right-hand side | derivative |
| Jacobian action | linearizer / operator |
| product of stage increments | block |
| nonlinear stage equation solver | resolvent |
| linear correction solver | inverter |
| approximate inverse | preconditioner |
| stage initial guess | scheme predictor |
| backend arithmetic | engine / carrier / Algebraist / accelerator |
