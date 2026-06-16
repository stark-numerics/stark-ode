# Getting started

This page is for ordinary users: you have a Python, NumPy, JAX, or CuPy state and want to integrate an ODE.

The high-level path is:

```text
System + Frame + Method + Engine
```

- `System` holds the derivative and optional linearizer.
- `Frame` describes named state fields and their translation fields.
- `Method` chooses the numerical method pieces.
- `Engine` chooses the storage and arithmetic backend.

## A small NumPy solve

```python
import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


def derivative(t, state, out):
    out.dy[:] = -0.5 * state.y


frame = Frame({"y": {"translation": "dy", "shape": (3,)}})
system = System(derivative=derivative, frame=frame)

ivp = system.ivp(
    initial={"y": np.array([2.0, 4.0, 8.0])},
    interval=Interval(present=0.0, step=0.1, stop=1.0),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
)

for interval, state in ivp.integrate():
    print(interval.present, state.y)
```

The derivative writes into `out.dy` because the frame says that state field `y` uses translation field `dy`.

## Return-style derivatives

For backends such as JAX, or for small examples where mutation is not useful, a derivative can return its result instead of writing into `out`.

See:

```powershell
python -m examples.getting_started.returning_derivative
python -m examples.getting_started.interface.jax
```

## Multiple fields

A frame can describe several state fields. Each field can have its own translation field.

See:

```powershell
python -m examples.getting_started.multiple_fields
```

Use this when your model is clearer as named parts rather than a single flat vector.

## Choosing a method

`Method` is a recipe. The simplest recipes choose a scheme:

```python
Method(scheme=SchemeCashKarp)
```

Implicit methods add resolvents and inverters. See [Methods](methods.md) and [Implicit methods](implicit.md).

## Optional backends

The same problem can be run with different engines when the derivative and state representation are compatible:

```text
EngineNative
EngineNumpy
EngineJax
EngineCupy
```

See [Engines](engines.md) for backend boundaries and optional dependency behaviour.

## What this page does not cover

You do not need to know about `Block`, `Operator`, `InverterRequest`, custom allocators, or custom translations for normal use. Those are described in [Foreign models](foreign-models.md) and [Mathematical contracts](contracts_math.md).
