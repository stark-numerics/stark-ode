# STARK functionality guide

This page is a compact map of the public functionality in STARK. It is meant
to answer two questions:

- what can I use out of the box?
- where can I plug in my own problem-specific code?

For a worked narrative example, start with
[`examples/three_body_stark.ipynb`](../examples/three_body_stark.ipynb).

For the package's internal design language and coding conventions, see
[`docs/house_style.md`](house_style.md).

For the mathematical interpretation of the contracts, see
[`docs/contracts_math.md`](contracts_math.md).

## Core idea

STARK integrates rich mutable Python state objects by separating the nonlinear
state from the linear increments used by Runge-Kutta schemes.

The main pieces are:

- `State`: your own problem object.
- `Translation`: a linear update that can be added, scaled, measured, and
  applied to a state.
- `Derivative`: a callable `derivative(state, out)` that writes the time
  derivative into a translation object.
- `Workbench`: an object that allocates blank states/translations and copies
  states.
- `Scheme`: a one-step integration method.
- `Marcher`: couples a scheme to tolerances and performs one accepted step.
- `Integrator`: runs repeated `Marcher` calls over an interval.
- `Auditor`: checks that the objects satisfy the STARK contracts before a long
  run.

The standard import path is:

```python
from stark import Marcher, Auditor, Integrator, Interval, Tolerance
```

## Integration

Use `Interval` to describe the current time, proposed step, and stop time:

```python
interval = Interval(present=0.0, step=1.0e-3, stop=1.0)
```

Create a `Marcher` object from a scheme and tolerance:

```python
marcher = Marcher(scheme, tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6))
integrate = Integrator()
```

Then integrate in either snapshot mode or live mode:

```python
for interval_snapshot, state_snapshot in integrate(marcher, interval, state):
    ...
```

Snapshot mode yields copied states, so collected trajectories are stable. Live
mode yields the original mutable objects and is useful for tight loops:

```python
for live_interval, live_state in integrate.live(marcher, interval, state):
    ...
```

Both modes accept checkpoints. An integer gives equally spaced output times;
an iterable gives explicit absolute times:

```python
for output_interval, output_state in integrate(marcher, interval, state, checkpoints=100):
    ...

for output_interval, output_state in integrate.live(
    marcher,
    interval,
    state,
    checkpoints=[0.1, 0.25, 0.5, 1.0],
):
    ...
```

Checkpoints are useful for plots and animations: the solver may adapt internally
while the user only observes chosen output times.

## Built-in schemes

The scheme library is available from `stark.scheme_library`.

Adaptive embedded schemes:

| Class | Method |
| --- | --- |
| `SchemeBogackiShampine` | Bogacki-Shampine 3(2) |
| `SchemeCashKarp` | Cash-Karp 5(4) |
| `SchemeRKCK` | Alias for Cash-Karp |
| `SchemeDormandPrince` | Dormand-Prince 5(4) |
| `SchemeFehlberg45` | Fehlberg 4(5) |
| `SchemeTsitouras5` | Tsitouras 5(4) |

Fixed-step schemes:

| Class | Method |
| --- | --- |
| `SchemeEuler` | Forward Euler |
| `SchemeHeun` | Heun |
| `SchemeKutta3` | Kutta third-order |
| `SchemeMidpoint` | Explicit midpoint |
| `SchemeRalston` | Ralston |
| `SchemeRK4` | Classical fourth-order Runge-Kutta |
| `SchemeRK38` | 3/8-rule Runge-Kutta |
| `SchemeSSPRK33` | SSP RK33 |

Implicit schemes:

| Class | Method |
| --- | --- |
| `SchemeBackwardEuler` | Backward Euler |
| `SchemeSDIRK21` | ESDIRK 2(1) / SDIRK21 |
| `SchemeKvaerno3` | Kvaerno 3(2) |
| `SchemeKvaerno4` | Kvaerno 4(3) |
| `SchemeBDF2` | BDF2 |

For clarity, the physical subpackages are also importable:

```python
from stark.scheme_library.adaptive import SchemeDormandPrince
from stark.scheme_library.fixed_step import SchemeRK4
from stark.scheme_library.adaptive_implicit import SchemeKvaerno3
```

## Built-in resolvers and inverters

Resolvers live in `stark.resolver_library`:

| Class | Method |
| --- | --- |
| `ResolverPicard` | Fixed-point / Picard iteration |
| `ResolverAnderson` | Anderson-accelerated fixed-point iteration |
| `ResolverBroyden` | Broyden quasi-Newton iteration |
| `ResolverNewton` | Newton iteration |

Inverters live in `stark.inverter_library`:

| Class | Method |
| --- | --- |
| `InverterGMRES` | GMRES |
| `InverterFGMRES` | Flexible GMRES |
| `InverterBiCGStab` | BiCGStab |

## User state contracts

The `Translation` object is the main adapter between user code and STARK. A
translation must provide:

- `__call__(origin, result)`: apply the translation to `origin` and write into
  `result`.
- `norm()`: return a scalar size used for adaptive error control.
- `__add__(other)`: generic translation addition.
- `__rmul__(scalar)`: generic scalar multiplication.

The `Workbench` must provide:

- `allocate_state()`: return a blank state object.
- `copy_state(dst, src)`: overwrite `dst` with `src`.
- `allocate_translation()`: return a blank translation object.

The `Derivative` is usually the thinnest adapter:

```python
def derivative(state, out):
    out.position[:] = state.velocity
    out.velocity[:] = acceleration_from_existing_code(state)
```

Use `Auditor` before a long solve:

```python
audit = Auditor(
    state=state,
    derivative=derivative,
    translation=workbench.allocate_translation(),
    workbench=workbench,
    interval=interval,
    scheme=scheme,
    tolerance=tolerance,
)
print(audit)
audit.raise_if_invalid()
```

## Fast translation paths

The generic translation fallback uses `__add__` and `__rmul__`. That is simple
and expressive, but it may allocate many temporary objects. For array-backed
or performance-sensitive problems, translations can expose optimized
linear-combination kernels.

These kernels belong with the translation implementation for a particular
problem. In practice, define them in the same module as the translation class
or import compiled kernels into that module, then attach them to the
translation class with a `linear_combine` class attribute. STARK discovers them
from a translation instance allocated by the workbench:

```python
def scale_array(out, a, x):
    out.values[:] = a * x.values
    return out


def combine2_array(out, a0, x0, a1, x1):
    out.values[:] = a0 * x0.values + a1 * x1.values
    return out


class ArrayTranslation:
    linear_combine = [scale_array, combine2_array]

    def __init__(self, values):
        self.values = values

    def __call__(self, origin, result):
        result.values[:] = origin.values + self.values

    def norm(self):
        return float((self.values * self.values).sum() ** 0.5)

    def __add__(self, other):
        return ArrayTranslation(self.values + other.values)

    def __rmul__(self, scalar):
        return ArrayTranslation(scalar * self.values)
```

The workbench still returns ordinary translation objects:

```python
class ArrayWorkbench:
    def allocate_translation(self):
        return ArrayTranslation(np.zeros_like(self.prototype))
```

When a scheme is constructed, STARK asks the workbench for a translation,
inspects that translation for `linear_combine`, and stores the resolved kernels
inside the scheme. The rest of the integration code does not need to pass the
fast paths around explicitly.

The entries are:

- `linear_combine[0]`: `scale(out, a, x)`
- `linear_combine[1]`: `combine2(out, a0, x0, a1, x1)`
- `linear_combine[2]`: `combine3(out, a0, x0, a1, x1, a2, x2)`
- and so on up to `combine7`.

If only `scale` and `combine2` are supplied, STARK builds the higher-arity
combinations from `combine2` and scratch translations allocated by the
workbench. For best performance, provide fused `combine3`, `combine4`, ...
kernels directly. The benchmarks use this hook with Numba-jitted kernels.

Fast paths should obey the same aliasing rule as translation application: the
output buffer may be one of the input buffers, so kernels should be correct for
in-place use.

## Custom schemes

`Marcher` accepts any object that satisfies the `SchemeLike` contract. A custom
scheme does not need to inherit from a STARK base class.

The minimal scheme interface is:

```python
class MyScheme:
    def __call__(self, interval, state, tolerance):
        ...
        return accepted_dt

    def snapshot_state(self, state):
        ...

    def set_apply_delta_safety(self, enabled):
        ...
```

The `__call__` method should:

- choose a step no larger than `interval.stop - interval.present`;
- mutate `state` by one accepted step;
- update `interval.step` to the next proposed step;
- return the accepted step size.

`Marcher` will then increment `interval.present` by the returned step size.

For built-in-style schemes, the common pattern is to own a `SchemeWorkspace`
instance. `SchemeWorkspace` resolves the workbench, scratch states, scratch
translations, translation application, and linear-combination fast paths:

```python
from stark import SchemeWorkspace


class MyScheme:
    def __init__(self, derivative, workbench):
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, workbench.allocate_translation())
```

The richer `Scheme` protocol also exposes readable metadata, tableaus, and
string formatting. That is useful for library-quality schemes, but not required
for `Marcher`.

Run `Auditor(..., scheme=my_scheme)` to check a custom scheme alongside the
state, translation, workbench, interval, and tolerance objects.

## Implicit schemes

Built-in implicit schemes use the same `Marcher` and `Integrator` layer as the
explicit schemes, but they ask the user for extra structure.

The additional pieces depend on the resolver:

- `ResolverPicard` only needs residual evaluation.
- `ResolverAnderson` and `ResolverBroyden` also need an `InnerProduct`.
- `ResolverNewton` needs a `Linearizer` and an `InverterLike`.

The common implicit shape is:

```python
from stark import (
    InverterBiCGStab,
    InverterPolicy,
    InverterTolerance,
    Marcher,
    ResolverNewton,
    ResolverPolicy,
    ResolverTolerance,
    Tolerance,
)
from stark.scheme_library import SchemeKvaerno3

workbench = MyWorkbench()
derivative = MyDerivative()
linearizer = MyLinearizer()
inverter = InverterBiCGStab(
    workbench,
    my_inner_product,
    tolerance=InverterTolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=InverterPolicy(max_iterations=24),
)
resolver = ResolverNewton(
    workbench,
    inverter=inverter,
    tolerance=ResolverTolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=ResolverPolicy(max_iterations=24),
)
scheme = SchemeKvaerno3(
    derivative,
    workbench,
    linearizer=linearizer,
    resolver=resolver,
)
marcher = Marcher(scheme, tolerance=Tolerance(atol=1.0e-6, rtol=1.0e-5))
```

For Anderson or Broyden, replace `ResolverNewton(...)` with the chosen
resolver, pass an inner product directly to the resolver, and omit the
linearizer/inverter pair.

The `Auditor` is especially useful here because implicit methods rely on more
contracts at once:

- the `Translation` vector-space and norm structure;
- the `Linearizer` Jacobian action for Newton-like methods;
- the `InnerProduct` for Krylov and secant-based methods;
- the in-place linear-combination kernels required for strict implicit
  operator algebra.

For the mathematical meaning of those contracts, see
[`docs/contracts_math.md`](contracts_math.md).

## Benchmarks

The benchmark reports live under `benchmarks/` and are intended to be readable
examples of idiomatic STARK, SciPy, and Diffrax implementations of the same
problems.

Install benchmark dependencies with:

```powershell
python -m pip install -e ".[benchmarks]"
```

Run:

```powershell
python -m benchmarks.brusselator_2d.report
python -m benchmarks.fput.report
python -m benchmarks.fitzhugh_nagumo_1d.report
python -m benchmarks.robertson.report
```

The STARK benchmark implementations intentionally use fast translation paths so
the reports show both the generic interface and the performance-oriented
extension point.
