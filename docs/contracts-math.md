# Mathematical contracts

This page is the advanced reference for STARK's low-level contracts. It is most useful after reading [Foreign models](foreign-models.md).

STARK is built around one mathematical separation:

```text
state space:       nonlinear model values
translation space: linear increments/tangent values
```

The high-level `Frame` path creates these objects for ordinary named fields. Foreign-model integrations may provide them directly.

## State space

The state space is the set of model values being advanced by the ODE. A state may be a scalar, an array-backed field bundle, a dataclass, a mesh object, a particle system, or another domain object.

A state does not need to be a vector. The solver does not require arbitrary state addition. It only needs to be able to apply a translation to a state to obtain another state.

Mathematically, this is closer to an affine space than a vector space.

## Translation space

The translation space contains increments used by schemes. A translation must support linear operations:

```text
a * u + b * v
```

Schemes use this algebra to build stage combinations, embedded error estimates, and accepted updates.

Translations also need norms or inner products for adaptive error control, residual checks, and Krylov methods.

## Derivative

The derivative is the ODE right-hand side:

```text
f : time x state -> translation
```

In code, this may be in-place or return-style. The scheme-facing contract is that evaluating the derivative produces a translation representing the instantaneous velocity at the current state.

## Applying translations

A scheme constructs a translation update and applies it to the current state. Conceptually:

```text
state_next = state + translation
```

The `+` here is not required to be Python addition. It is the model-specific action of the translation space on the state space.

## Norms and adaptive control

Adaptive schemes estimate local error as a translation. A norm maps that translation to a scalar used by step acceptance and step-size control.

Frames can choose which fields contribute to the norm. Foreign models can provide custom norm behaviour through their translation algebra.

## Linearizers

For implicit Newton-style methods, the solver needs the derivative of the derivative:

```text
J(t, x) v = Df(t, x)[v]
```

The source and target of this action are translations. The linearizer does not advance the state; it describes how a derivative changes under a translation perturbation.

Dense inverters may ask for dense materialisation of the same operator. Krylov inverters normally need only the action.

## Blocks

Implicit Runge-Kutta methods often solve for several stage translations at once. STARK represents product spaces of translations using blocks.

If `T` is the translation space, a block may represent:

```text
T^m
```

where `m` is the number of coupled stage unknowns.

Blocks allow resolvents and inverters to work with coupled stage equations without requiring every user translation type to know about Runge-Kutta stages.

## Operators

An operator maps translations, or blocks of translations, to translations or blocks of translations. In implicit solves, operators usually represent correction systems derived from residual equations.

A matrix-free operator provides an `apply` action. A dense-capable operator can also fill a matrix representation for dense inverters.

## Resolvents

A resolvent solves a nonlinear equation, usually an implicit stage residual:

```text
F(delta) = 0
```

The scheme owns the stage equation. The resolvent owns the nonlinear iteration used to solve it.

Examples include Picard and Newton-style methods.

## Inverters

An inverter solves or improves a linear correction equation requested by a resolvent.

The current inverter shape is request-based:

```text
inverter(request, output)
```

The request supplies the operator, residual, and problem context. The inverter writes or improves the output correction according to its output mode.

Dense inverters materialise small operators. Krylov inverters use matrix-free operator actions and inner products. Relaxation inverters use simpler iterative updates.

## Preconditioners

A preconditioner approximately solves a related linear problem to improve Krylov convergence. It belongs near the inverter because it changes the linear solve, not the nonlinear residual or scheme stage definition.

## Component map

| Mathematical idea | STARK object |
|---|---|
| state space | state object or frame-backed state |
| translation space | translation object or frame-backed translation |
| right-hand side `f(t, x)` | `Derivative` / `DerivativeStyle` |
| Jacobian action `Df(t, x)[v]` | `Linearizer` / `LinearizerStyle` |
| affine update | allocator / engine translation application |
| norm or inner product | frame norm policy or translation algebra |
| product stage space | `Block` |
| linear action | `Operator` |
| implicit residual equation | resolvent request |
| nonlinear solver | `Resolvent` |
| linear correction solver | `Inverter` |
| Krylov acceleration aid | preconditioner |
| time-stepping structure | `Scheme` |

## Reading order

Most users should not start here. Read:

1. [Getting started](getting-started.md) for ordinary problems.
2. [Problem objects](problem.md) for frames, derivatives, and linearizers.
3. [Implicit methods](implicit.md) for resolvents and inverters.
4. [Foreign models](foreign-models.md) before implementing low-level contracts.
