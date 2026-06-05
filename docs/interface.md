# STARK Interface Layer

The `stark.interface` package is the front door for ordinary scalar and
array-valued initial-value problems.

It accepts ordinary Python values, array values, and carrier-backed values,
then prepares the derivative, state, carrier, routing, scheme, executor,
stepper, and integrator needed by the core STARK objects.

The explicit core API remains available for custom state objects, implicit
resolvents, inverters, accelerators, and problem-specific fast paths. The
interface layer is not a SciPy compatibility wrapper and does not provide a
free `solve(...)` helper.

## Basic example: exponential decay

```python
import numpy as np

from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=np.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=10.0),
)

for interval, state in ivp.integrate():
    print(interval.present, state.value)
```

Plain callables are treated as return-style derivatives by default.

`StarkIVP.integrate()` returns the existing STARK integration iterator. Each
yielded state is a `StarkVector`; the carried Python or array value is available
as `state.value`.

## Explicit return-style derivative

```python
import numpy as np

from stark import Interval
from stark.interface import StarkDerivative, StarkIVP


@StarkDerivative.returning
def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=np.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=10.0),
)

for interval, state in ivp.integrate():
    print(interval.present, state.value)
```

## In-place derivative

For performance-sensitive code, an in-place derivative can write into the
supplied output value.

```python
import numpy as np

from stark import Interval
from stark.interface import StarkDerivative, StarkIVP


@StarkDerivative.in_place
def exponential_decay(t, y, dy):
    dy[:] = -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=np.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=10.0),
)

for interval, state in ivp.integrate():
    print(interval.present, state.value)
```

## Explicit scheme

By default, `StarkIVP` uses `SchemeCashKarp`.

Pass an explicit scheme class when another stepping method is preferred.

```python
import numpy as np

from stark import Interval
from stark.interface import StarkIVP
from stark.schemes import SchemeDormandPrince


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=np.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=10.0),
    scheme=SchemeDormandPrince,
)

for interval, state in ivp.integrate():
    print(interval.present, state.value)
```

## Backend support levels

### Native Python values

Supported initial values:

- `int`
- `float`
- `list[int | float]`
- `tuple[int | float, ...]`

Native values use return/replacement routing.

```python
from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=2.0,
    interval=Interval(present=0.0, step=0.1, stop=2.0),
)
```
## Notes

- `Interval` is explicit and must satisfy STARK's interval-like contract: `present`, `step`, `stop`, `copy()`, and `increment(dt)`.
- The initial step is not inferred.
- Tuple/list `t_span`-style intervals are not accepted.
- Raw initial values are prepared through a carrier.
- Plain derivative callables are treated as return-style callables.
- Use `StarkDerivative.in_place` for explicit in-place derivative callables.
- `StarkIVP.integrate()` returns the core STARK integration result directly.
- The interface layer assembles the core STARK objects; it does not replace the core API.

## Runnable examples

From a source checkout with the package installed, run:

```powershell
python -m examples.interface.native
python -m examples.interface.numpy
python -m examples.interface.cupy
python -m examples.interface.jax
```

The CuPy and JAX examples skip cleanly when their optional dependencies are not
installed or usable.
