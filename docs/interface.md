# STARK problem layer

The `stark.problem` package is the high-level front door for ordinary named
field problems. A problem declaration combines:

- a `Frame`, which names state fields, translation fields, shapes, and norm
  policy;
- a `Derivative`, usually created through `DerivativeStyle`;
- optionally a `Linearizer`, usually created through `LinearizerStyle`, for
  Newton-like implicit methods;
- a `System`, which prepares an initial-value problem once a method, engine,
  interval, and initial values are supplied.

The lower-level core API remains available when a simulation already has custom
state and translation objects.

## Basic example

```python
import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


def exponential_decay(t, state, out):
    del t
    out.dy[0] = -0.5 * state.y[0]


system = System(
    derivative=DerivativeStyle.in_place(exponential_decay),
    frame=Frame({"y": {"translation": "dy", "shape": (1,)}}),
)

ivp = system.ivp(
    initial={"y": np.array([2.0])},
    interval=Interval(present=0.0, step=0.1, stop=1.0),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

for interval, state in ivp.integrate():
    print(interval.present, state.y[0])
```

`SystemIVP.integrate()` yields stable copied snapshots. Use
`SystemIVP.mutating_trajectory()` for low-overhead streaming or benchmarking
when you do not need copied states.

## Return-style derivatives

Return-style derivatives are useful when mutation is awkward or impossible, for
example in JAX-oriented code.

```python
from stark import DerivativeStyle


@DerivativeStyle.returning
def rhs(t, state):
    del t
    return {"dy": -0.5 * state.y}
```

The returned mapping is assigned to the translation fields declared by the
`Frame`.

## In-place derivatives

In-place derivatives receive the output translation object explicitly. This is
the most direct style for NumPy/native hot paths.

```python
from stark import DerivativeStyle


@DerivativeStyle.in_place
def rhs(t, state, out):
    del t
    out.dy[:] = -0.5 * state.y
```

## Field-level kernels

For backend acceleration, use field-level kernels that receive selected state
and translation fields directly.

```python
from stark import DerivativeStyle


@DerivativeStyle.kernel(state=("y",), translation=("dy",))
def rhs_kernel(y, dy):
    dy[:] = -0.5 * y
```

A returning kernel is also available:

```python
@DerivativeStyle.kernel_returning(state=("y",), translation=("dy",))
def rhs_kernel(y):
    return -0.5 * y
```

## Linearizers for implicit methods

Newton-like resolvents need a linearizer that configures the local Jacobian
operator. The user-facing `Linearizer` mirrors the derivative adapter: it
prepares a lower-level `linearizer(interval, state, operator)` callable for the
method stack.

For examples, see the Robertson and HIRES competition implementations:

```powershell
python -m competition.robertson.report
python -m competition.hires.report
```

## Example commands

From a source checkout:

```powershell
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
python -m examples.getting_started.in_place_derivative
python -m examples.getting_started.interface.native
python -m examples.getting_started.interface.numpy
```
