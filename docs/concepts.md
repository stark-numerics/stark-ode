# Concepts and terminology

This page is for readers who know they have a differential equation, but do
not yet know STARK's names for the pieces.

STARK is opinionated about one thing: the model you care about should stay
visible. The package should adapt to your state shape, not force every problem
to become a flat anonymous vector on day one.

## ODE

An ordinary differential equation describes how a state changes with time.

```text
y' = f(t, y)
```

Read this as:

```text
current time + current state -> rate of change
```

In STARK, the rate-of-change function is a `Derivative`.

## IVP

An initial value problem is an ODE plus a starting value and a time interval.

```text
y' = f(t, y)
y(t0) = y0
t0 <= t <= t1
```

In STARK, `System.ivp(...)` builds this runnable problem from:

```text
System      the equation and state shape
Interval    start time, first step guess, and stop time
Method      the numerical method
Engine      the array/object backend
```

## State

The state is the thing your model currently knows.

Examples:

```text
temperature field
position and velocity
chemical concentrations
a nested object from an existing simulation
```

In the high-level path, a `Frame` names the state fields:

```python
Frame.scalar("y", translation="dy")
Frame.vector("y", translation="dy", length=3)
Frame.array("u", translation="du", shape=(32, 32))
```

## Translation

A translation is a solver increment: something that can be applied to a state
to get another state.

For a simple vector problem, a translation is just another array. For a foreign
object model, it may be a custom "delta" object.

STARK keeps state and translation distinct because adaptive and implicit
methods do a lot of algebra on increments before accepting a new state.

## Frame

A `Frame` tells STARK how named state fields correspond to named translation
fields.

```text
state.y        current value
translation.dy derivative or increment for y
```

Use `Frame` when your model can be described as named scalar or array fields.
Use the foreign-model path only when your existing objects really need their
own allocation, copy, and increment behavior.

## Derivative

A derivative is the right-hand side of the ODE.

STARK's solver-facing derivative writes into an output translation:

```text
derivative(interval, state, out) -> None
```

User-facing adapters let you write more natural forms:

```text
DerivativeStyle.accepts_instant_writes
DerivativeStyle.accepts_instant_returns
DerivativeStyle.kernel_accepts_instant_writes
DerivativeStyle.kernel_accepts_instant_returns
```

Use the style that makes the mathematical right-hand side easiest to read.
Use kernel styles when you want a compact array-field function that can be
prepared once and reused.

## Method

A `Method` chooses how STARK advances the solution.

The method stack has several layers:

```text
scheme       takes time steps
resolvent    solves nonlinear implicit stage equations
inverter     solves linear correction equations
predictor    seeds implicit stage guesses
```

For non-stiff problems, a scheme may be all you need. Stiff problems often need
the full scheme/resolvent/inverter stack.

## Scheme

A scheme is the time-stepping formula: Euler, RK4, Cash-Karp, Kvaerno, IMEX,
and so on.

Use explicit adaptive schemes first unless the problem is stiff or split into
cheap and stiff parts.

## Stiffness

A stiff problem forces explicit methods to take tiny steps for stability even
when the solution itself looks smooth. Symptoms include:

```text
very small accepted steps
many rejected adaptive steps
explicit solves that are stable only at impractical step sizes
```

Implicit or IMEX methods are the usual response.

## IMEX and split derivatives

IMEX means implicit-explicit. It is useful when one part of the derivative is
stiff but structured, while another part is cheap and non-stiff.

```text
y' = implicit_part(t, y) + explicit_part(t, y)
```

In STARK, use `Derivative.split(...)` to declare the two parts.

## Linearizer

A linearizer represents the Jacobian action of a derivative.

If:

```text
y' = f(t, y)
```

then the Jacobian action is:

```text
J(t, y) v = Df(t, y)[v]
```

Newton-style implicit methods use this action to compute corrections. A
linearizer may also know how to fill a small dense matrix for dense inversion.

## Jacobian

The Jacobian is the derivative of your derivative with respect to the state.

For a vector-valued ODE, it is the matrix of sensitivities:

```text
how each component of f changes when each component of y changes
```

You do not always need to materialise this matrix. Krylov methods can often use
a matrix-free Jacobian action instead.

## Resolvent

A resolvent solves the nonlinear equation created inside an implicit scheme.

Common choices:

```text
Picard       simple fixed-point iteration
Newton       uses a linearizer and inverter
Chord        reuses a linearization for part of the solve
VeryChord    reuses a linearization more aggressively
```

Use Newton when you can provide a useful linearizer. Use simpler resolvents for
teaching or mild implicit equations.

## Inverter

An inverter solves a linear correction problem inside Newton-like methods.

Current public families:

```text
Dense       small dense systems
Krylov      matrix-free larger systems
Relaxation  simple iterative baselines and structured probes
```

Dense is usually best for very small stiff systems. Krylov becomes interesting
when dense matrices are too expensive or impossible to form.

## Engine

An engine chooses storage and arithmetic:

```text
NumPy
Native Python
CuPy
JAX
```

Engines own backend-specific state, translation, allocation, and generated
algebra choices. Start with NumPy unless you already know why another backend
matters for your project.

## Algebraist

The Algebraist is STARK's prepared algebra layer. For known `Frame`-backed
states, it can generate kernels for common solver operations such as linear
combinations and norms.

This matters because solver performance is often decided by repeated algebra
on translations, not by the one line that defines the derivative.

## Monitor

A monitor records what a solve did: accepted steps, rejected steps, nonlinear
iterations, inverter solves, and residual history.

Use monitors to understand a run. Do not use monitored timings as raw solver
speed.

## Where to go next

- [Getting started](getting-started.md): solve the first problem.
- [Define a problem](problem.md): learn `System`, `Frame`, and derivative styles.
- [Choose a method](methods.md): choose schemes and method components.
- [Solve stiff problems](implicit.md): use linearizers, resolvents, and inverters.
