# Getting started: solve one ODE

This page is for ordinary use: you have a state made of named fields and want STARK to solve an ODE for you.

You need four things:

```text
System   what derivative to solve
Frame    what fields the state and translation have
Method   which numerical scheme to use
Engine   where arrays and arithmetic live
```

## A minimal NumPy solve

Solve:

```text
y' = -0.5 y,   y(0) = 2
```

```python
from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def decay(t: float, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


frame = Frame.scalar("y", translation="dy")
system = System(
    derivative=decay,
    frame=frame,
)

ivp = system.ivp(
    initial={"y": np.array([2.0])},
    interval=Interval(present=0.0, step=0.1, stop=1.0),
    method=Method(SchemeCashKarp),
    engine=EngineNumpy,
)

for interval, state in ivp.integrate():
    print(f"t={interval.present:.1f}, y={state.y[0]:.6f}")
```

Run the maintained script version:

```powershell
python -m examples.getting_started.scalar_decay
```

## Change the state shape

A `Frame` maps user state fields to solver translation fields.

For a two-component oscillator:

```python
frame = Frame.vector("y", translation="dy", length=2)
```

The derivative writes into `out.dy` because `dy` is the translation field for `y`.

```python
def oscillator(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]
```

Run:

```powershell
python -m examples.getting_started.numpy_oscillator
```

## Use a return-style derivative

Some backends, especially JAX, work better when derivatives return a translation instead of mutating an output object. Use this route when the natural derivative is an expression.

```powershell
python -m examples.getting_started.returning_derivative
```

The important difference is:

```text
in-place derivative:   f(t, state, out) -> None
return derivative:     f(t, state) -> translation-like result
```

The scheme-facing contract is still prepared as an in-place kernel internally.

## Change the method

Use `Method(...)` to choose a scheme.

```python
from stark.methods import SchemeBogackiShampine, SchemeCashKarp

method = Method(SchemeCashKarp)
```

Run:

```powershell
python -m examples.getting_started.choose_scheme
```

## Ask for checkpoints

The solver may take internal adaptive steps that do not match the output times you want. Use checkpoints when you want output at specific times.

```powershell
python -m examples.getting_started.checkpoints
```

Internal steps are for accuracy and stability. Checkpoints are for output.

## Use another backend

Start with NumPy. Then try JAX or CuPy when you have a reason to use those array systems.

```powershell
python -m examples.backends.numpy
python -m examples.backends.jax
python -m examples.backends.cupy
```

JAX and CuPy support may be optional in your environment. The examples should report missing optional dependencies rather than fail mysteriously.

## Common mistakes

### Confusing state fields and translation fields

`state.y` is the current state. `out.dy` is the derivative/increment field. The `Frame` declares the relationship.

### Timing a monitored solve

Monitors add observation work. Use monitors to understand a solve. Use unmonitored solves for timings.

### Assuming JAX means whole-solver JIT

`EngineJax` supports JAX arrays and JAX-compatible kernels. It does not automatically mean the entire adaptive solver loop is compiled as one JAX program.

## Next

- [Define a problem](problem.md) when you need multiple fields, tolerances, or a linearizer.
- [Choose a method](methods.md) when you need a different scheme.
- [Solve stiff problems](implicit.md) when explicit schemes struggle.
