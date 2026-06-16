# Problem objects

This page describes the problem layer: the objects that define the ODE independent of the numerical method used to solve it.

## System

`System` is the user-facing problem object. It owns:

- the derivative, which defines the right-hand side;
- the optional linearizer, which defines Jacobian actions for implicit methods;
- the frame, which describes state and translation fields.

A system can construct an IVP with an initial state, interval, method, engine, and configuration.

## Frame

`Frame` maps named state fields to named translation fields.

Example:

```python
Frame({
    "position": {"translation": "dposition", "shape": (3,)},
    "velocity": {"translation": "dvelocity", "shape": (3,)},
})
```

The state and the translation are intentionally different objects:

- the state is the nonlinear value being advanced;
- the translation is a linear increment or tangent value used by the solver.

This separation is the core STARK design decision. It lets schemes operate on translation algebra without requiring every state object to be a flat vector.

## Derivative

A derivative represents the ODE right-hand side:

```text
dx/dt = f(t, x)
```

In the common in-place style, the derivative writes a translation:

```python
def derivative(t, state, out):
    out.dy[:] = -0.5 * state.y
```

Return-style derivatives are supported through derivative adapters and are especially useful for JAX and other immutable array styles.

See:

```powershell
python -m examples.features.derivative_styles
```

## Linearizer

A linearizer represents the derivative of the derivative. It supplies Jacobian actions used by implicit methods:

```text
J(t, x) v = Df(t, x)[v]
```

Newton-style resolvents use the linearizer to build linear correction equations. Dense inverters may also ask for dense materialisation of the same operator.

See:

```powershell
python -m examples.features.linearizer_styles
```

## SystemIVP

An IVP couples a system to:

- an initial state;
- an interval;
- a method;
- an engine;
- optional configuration.

The IVP produces an integration iterator. Checkpoint and live-output behaviour belong at this layer, while individual accepted steps belong to the scheme and integrator machinery.

## Configuration

`Configuration` holds runtime policy such as tolerances, maximum iteration counts, progress checks, predictor policy, and other method-level settings.

The same configuration object can satisfy narrower configuration protocols used by schemes, resolvents, and inverters. This keeps components decoupled while letting users pass one high-level policy object.
