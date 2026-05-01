# STARK Interface Layer

The `stark.interface` package is the front door for ordinary scalar and
array-valued initial-value problems.

It accepts ordinary Python values, array values, and carrier-backed values,
then prepares the derivative, state, carrier, routing, scheme, executor,
marcher, and integrator needed by the core machinery.

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

## Explicit carrier

Most users do not need to provide a carrier. `StarkIVP` chooses one from the initial value through `CarrierLibrary.default()`.

Pass a carrier explicitly when you need carrier-specific options.

```python
import numpy as np

from stark import Interval
from stark.carriers import CarrierNumpy
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=np.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=10.0),
    carrier=CarrierNumpy(strict_shape=True),
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

### NumPy arrays

NumPy arrays are supported through `CarrierNumpy`.

NumPy arrays can be one-dimensional or multidimensional. For example, both a
packed state with shape `(2 * n,)` and a more natural state with shape `(2, n)`
are valid array states.

NumPy uses prefer-in-place vector routing where possible, so dense array states
are usually the best-performing public interface path. When a problem can be
represented clearly as one dense array, start there. Custom structured state
objects are useful when they make the model clearer or when the problem needs
specialized workbench, derivative, resolvent, or Algebraist behavior, but they
are not automatically faster than dense array carriers.

```python
import numpy as np

from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=np.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=2.0),
)
```

For coupled variables, a multidimensional array can keep the structure visible:

```python
initial = np.stack(
    (
        np.linspace(0.0, 1.0, 128),
        np.zeros(128),
    )
)


def oscillator_chain(t, y, dy):
    q = y[0]
    p = y[1]
    dy[0, :] = p
    dy[1, 1:-1] = q[:-2] - 2.0 * q[1:-1] + q[2:]
```

### CuPy arrays

CuPy arrays are supported when CuPy is installed and usable.

CuPy uses prefer-in-place vector routing where possible.

```python
import cupy as cp

from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=cp.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=2.0),
)
```

### JAX arrays

JAX arrays are supported as Python-level carried values.

JAX uses return/replacement routing and does not require in-place mutation.

```python
import jax.numpy as jnp

from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=jnp.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=2.0),
)
```

Initial JAX support means JAX arrays can pass through a Python-level STARK
solve without in-place array mutation.

It does not yet mean:

- `jax.jit` of the whole solve
- `jax.grad` through the whole solve
- `jax.vmap` over solves
- a fully functional JAX solver loop

## Routing

Routing controls local vector call paths:

- return/replacement
- in-place
- prefer in-place with return fallback

Carrier-backed interface translations use routing for translation application
and for linear-combination kernels exposed to the core scheme workspace. The
generic interface path provides named combinations through `combine12`, covering
the built-in explicit, implicit, and IMEX schemes without falling back to staged
pairwise accumulation.

Carriers recommend routing based on their value semantics.

Most users do not need to pass routing explicitly. Advanced callers can pass a
`Routing` object to override a policy family.

```python
from stark import Interval
from stark.interface import StarkIVP
from stark.routing import Routing, RoutingVectorReturn


ivp = StarkIVP(
    derivative=lambda t, y: -0.5 * y,
    initial=2.0,
    interval=Interval(present=0.0, step=0.1, stop=2.0),
    routing=Routing(vector=RoutingVectorReturn()),
)
```

## CarrierLibrary

`CarrierLibrary` selects a carrier for raw initial values.

`CarrierLibrary.default()` includes available built-in carriers:

- native Python carrier
- NumPy carrier
- CuPy carrier when CuPy is installed
- JAX carrier when JAX is installed

Use an explicit carrier or explicit carrier library only when the default
selection is not appropriate.

## Notes

- `Interval` is explicit and must satisfy STARK's interval-like contract: `present`, `step`, `stop`, `copy()`, and `increment(dt)`.
- The initial step is not inferred.
- Tuple/list `t_span`-style intervals are not accepted.
- Raw initial values are prepared through a carrier.
- Plain derivative callables are treated as return-style callables.
- Use `StarkDerivative.in_place` for explicit in-place derivative callables.
- `StarkIVP.integrate()` returns the existing STARK integration result directly.
- The interface layer assembles the existing STARK core machinery; it does not replace the core API.

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
