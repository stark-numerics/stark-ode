# STARK functionality guide

This page is a compact map of the public functionality in STARK. It is meant
to answer two questions:

- what is available out of the box?
- where can problem-specific code plug in?

For ordinary scalar or array-valued initial-value problems, start with the
public interface guide: [`docs/interface.md`](interface.md).

For a conceptual map of the object families in STARK, see
[`docs/object_map.md`](object_map.md).

For worked narrative examples using the core API, start with
[`examples/case_studies/three_body`](../examples/case_studies/three_body) and
[`examples/case_studies/allen_cahn`](../examples/case_studies/allen_cahn).

For the package's internal design language and coding conventions, see
[`docs/house_style.md`](house_style.md).

For the mathematical interpretation of the contracts, see
[`docs/contracts_math.md`](contracts_math.md).

For contributor performance-regression tracking and the ASV suite, see
[`docs/benchmarking.md`](benchmarking.md).

## Core idea

The front door for Python scalars, sequences, NumPy arrays, CuPy arrays, and
Python-level JAX array solves is `stark.interface.StarkIVP`. It prepares the
carrier-backed state and derivative, chooses routing, builds the core objects,
and returns the existing STARK integration result. Use the rest of this guide
when you need custom state objects, handwritten translation types, implicit
resolvents, inverters, accelerators, or problem-specific fast paths.

Dense NumPy arrays are the recommended starting point for performance-sensitive
ordinary problems. They may be one-dimensional or multidimensional, and the
interface layer uses in-place carrier routing where possible. Move to custom
structured state objects when the domain model or a specialized fast path needs
that extra structure.

STARK integrates rich mutable Python state objects by separating the nonlinear
state from the linear increments used by Runge-Kutta schemes.

The main pieces are:

- `State`: your own problem object.
- `Translation`: a linear update that can be added, scaled, measured, and
  applied to a state.
- `Derivative`: a callable `derivative(interval, state, out)` that writes the
  time derivative into a translation object.
- `Allocator`: an object that allocates blank states/translations and copies
  states.
- `Scheme`: a one-step integration method.
- `Marcher`: couples a scheme to tolerances and performs one accepted step.
- `Integrator`: runs repeated `Marcher` calls over an interval.
- `ComparisonRunner`: compares two or more marcher setups on the same problem.
- `Auditor`: checks that the objects satisfy the STARK contracts before a long
  run.

The standard import path is:

```python
from stark import Executor, Marcher, Auditor, Integrator, Interval, ExecutorTolerance
from stark.accelerators import AcceleratorAbsent
```

The standard interface import path is:

```python
from stark import Interval
from stark.interface import StarkIVP, StarkDerivative
```

## Integration

Use `Interval` to describe the current time, proposed step, and stop time:

```python
interval = Interval(present=0.0, step=1.0e-3, stop=1.0)
```

Create a `Marcher` object from a scheme and an `Executor`:

```python
executor = Executor(
    ExecutorTolerance=ExecutorTolerance(atol=1.0e-8, rtol=1.0e-6),
    accelerator=AcceleratorAbsent(),
)
marcher = Marcher(scheme, executor)
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

Acceleration is configured on the objects that use it, such as Algebraist
providers, resolvents, and inverters. `Executor` deliberately carries only
execution policy for a run.

Checkpoints are useful for plots and animations: the solver may adapt internally
while the user only observes chosen output times.

## ComparisonRunner

`ComparisonRunner` is a small comparison tool for development work on custom schemes.

It takes:

- one `ComparisonProblem`
- two or more `ComparisonEntry` objects

and reports:

- pairwise final-state differences
- optional pairwise checkpoint-trajectory differences
- setup, warmup, and repeated runtime timings
- optional problem-supplied final-state diagnostics
- a bucketed `cProfile` breakdown showing approximate shares of problem work,
  scheme work, resolvent work, inverter work, and framework overhead

This is meant for A/B testing schemes on one problem, not for replacing the
problem-specific comparison reports under `examples/comparison/`.

The returned `ComparisonReport` is also structured data, not just a printable
report. Advanced users can inspect:

- `report.results_by_name()`
- `report.timings_by_name()`
- `report.diagnostics_by_name()`
- `report.final_difference_map()`
- `report.trajectory_difference_map()`
- `report.profiles_by_name()`
- `report.as_dict()`

`ComparisonEntry` also accepts an optional `profile_category(filename, lineno, function_name)`
hook so a custom entry can teach the ComparisonRunner how to bucket its own profiled
code into `problem`, `scheme`, `resolvent`, `inverter`, `framework`, or `other`.

For a worked example, see
[`examples/case_studies/allen_cahn`](../examples/case_studies/allen_cahn).

## Built-in schemes

Built-in schemes are available from `stark.schemes`.

Adaptive embedded schemes:

| Class | Method |
| --- | --- |
| `SchemeBogackiShampine` | Bogacki-Shampine 3(2) |
| `SchemeCashKarp` | Cash-Karp 5(4) |
| `SchemeDormandPrince` | Dormand-Prince 5(4) |
| `SchemeFehlberg45` | Fehlberg 4(5) |
| `SchemeTsitouras5` | Tsitouras 5(4) |

Adaptive IMEX schemes:

| Class | Method |
| --- | --- |
| `SchemeKennedyCarpenter32` | Kennedy-Carpenter 3(2) |
| `SchemeKennedyCarpenter43_6` | Kennedy-Carpenter 4(3), 6-stage |
| `SchemeKennedyCarpenter43_7` | Kennedy-Carpenter 4(3), 7-stage |
| `SchemeKennedyCarpenter54` | Kennedy-Carpenter 5(4) |
| `SchemeKennedyCarpenter54b` | Kennedy-Carpenter 5(4) b |

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

Fixed-step IMEX schemes:

| Class | Method |
| --- | --- |
| `SchemeIMEXEuler` | IMEX Euler |

Implicit schemes:

| Class | Method |
| --- | --- |
| `SchemeBackwardEuler` | Backward Euler |
| `SchemeImplicitMidpoint` | Implicit midpoint |
| `SchemeCrankNicolson` | Crank-Nicolson / trapezoid |
| `SchemeCrouzeixDIRK3` | Crouzeix DIRK3 |
| `SchemeGaussLegendre4` | Gauss-Legendre 4 |
| `SchemeLobattoIIIC4` | Lobatto IIIC 4 |
| `SchemeRadauIIA5` | Radau IIA 5 |
| `SchemeSDIRK21` | ESDIRK 2(1) / SDIRK21 |
| `SchemeKvaerno3` | Kvaerno 3(2) |
| `SchemeKvaerno4` | Kvaerno 4(3) |
| `SchemeBDF2` | BDF2 |

For clarity, the physical subpackages are also importable:

```python
from stark.schemes.explicit.adaptive import SchemeDormandPrince
from stark.schemes.explicit.fixed import SchemeRK4
from stark.schemes.implicit.adaptive import SchemeKvaerno3
from stark.schemes.imex.adaptive import SchemeKennedyCarpenter43_7
```

## Built-in resolvents and inverters

Resolvents live in `stark.resolvents`:

| Class | Method |
| --- | --- |
| `ResolventPicard` | Fixed-point / Picard implicit resolution |
| `ResolventAnderson` | Anderson-accelerated implicit resolution |
| `ResolventBroyden` | Broyden quasi-Newton implicit resolution |
| `ResolventNewton` | Newton implicit resolution |

Inverters live in `stark.inverters`:

| Class | Method |
| --- | --- |
| `InverterGMRES` | GMRES |
| `InverterFGMRES` | Flexible GMRES |
| `InverterBiCGStab` | BiCGStab |

Each Krylov inverter also accepts an optional `preconditioner=` worker. The
preconditioner follows the same bind-then-call shape as the inverters
themselves, so users can supply problem-specific approximate inverse actions
without forcing the rest of the solver stack to change shape.

## User state contracts

The `Translation` object is the main adapter between user code and STARK. A
translation must provide:

- `__call__(origin, result)`: apply the translation to `origin` and write into
  `result`.
- `norm()`: return a scalar size used for adaptive error control.
- `__add__(other)`: generic translation addition.
- `__rmul__(scalar)`: generic scalar multiplication.

The `Allocator` must provide:

- `allocate_state()`: return a blank state object.
- `copy_state(source, out)`: overwrite `out` with `source`.
- `allocate_translation()`: return a blank translation object.

The `Derivative` is usually the thinnest adapter:

```python
def derivative(interval, state, out):
    t = interval.present
    out.position[:] = state.velocity
    out.velocity[:] = acceleration_from_existing_code(t, state)
```

For IMEX work, STARK provides a small split carrier:

```python
from stark import DerivativeIMEX

imex = DerivativeIMEX(
    implicit=implicit_derivative,
    explicit=explicit_derivative,
)
```

The field names are intentionally literal. Users usually think "this is the
implicit part" and "this is the explicit part", so STARK preserves that
language directly instead of hiding it behind a more abstract wrapper.

Use `Auditor` before a long solve:

```python
audit = Auditor(
    state=state,
    derivative=derivative,
    translation=allocator.allocate_translation(),
    allocator=allocator,
    interval=interval,
    scheme=scheme,
    ExecutorTolerance=ExecutorTolerance,
)
print(audit)
audit.raise_if_invalid()
```

Or, for an IMEX split:

```python
Auditor.require_imex_scheme_inputs(imex, allocator, allocator.allocate_translation())
```

## Fast translation paths

The generic translation fallback uses `__add__` and `__rmul__`. That is simple
and expressive, but it may allocate many temporary objects. For array-backed or
performance-sensitive problems, translations can expose optimized
linear-combination kernels.

Algebraist is STARK's generated-algebra layer. It has two common roles:

- `AlgebraistGeneratorGeneral` provides arity-based translation combinations
  such as `out = a0 * x0 + a1 * x1`.
- `AlgebraistGeneratorSpecialist` provides fixed-coefficient scheme kernels
  for stages, accepted increments, and embedded error estimates.

Both providers are explicit objects. The allocator installs them when the
problem wants generated kernels:

```python
from stark.algebraist import (
    AlgebraistArity,
    AlgebraistGeneratorGeneral,
    AlgebraistGeneratorSpecialist,
    AlgebraistLayout,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
)

LAYOUT = AlgebraistLayout(
    fields=(
        AlgebraistLayoutField("du", "u", policy=AlgebraistLayoutLooped(rank=2)),
        AlgebraistLayoutField("dv", "v", policy=AlgebraistLayoutLooped(rank=2)),
    ),
)


class ArrayTranslation:
    __slots__ = ("du", "dv")

    linear_combine = ()

    def __init__(self, du, dv):
        self.du = du
        self.dv = dv

    def __add__(self, other):
        return ArrayTranslation(self.du + other.du, self.dv + other.dv)

    def __rmul__(self, scalar):
        return ArrayTranslation(scalar * self.du, scalar * self.dv)
```

The allocator still returns ordinary translation objects. It can also attach
generated combination kernels to the translation class and keep a specialist
provider for schemes:

```python
class ArrayAllocator:
    _specialist = None

    def __init__(self, shape, accelerator):
        self.shape = shape
        self.accelerator = accelerator
        self._install_algebraist()

    def allocate_translation(self):
        return ArrayTranslation(
            np.zeros(self.shape),
            np.zeros(self.shape),
        )

    @property
    def specialist(self):
        return self.__class__._specialist

    def _install_algebraist(self):
        general = AlgebraistGeneratorGeneral(
            translation=self.allocate_translation(),
            allocator=self,
            layout=LAYOUT,
            accelerator=self.accelerator,
        )
        ArrayTranslation.linear_combine = tuple(
            general.provide(AlgebraistArity(arity))
            for arity in range(1, 13)
        )
        self.__class__._specialist = AlgebraistGeneratorSpecialist(
            translation=self.allocate_translation(),
            allocator=self,
            layout=LAYOUT,
            accelerator=self.accelerator,
        )
```

Pass the specialist to schemes that can use generated fixed-coefficient stage
algebra:

```python
scheme = SchemeDormandPrince(
    derivative,
    allocator,
    specialist=allocator.specialist,
)
```

Generated scheme algebra does not generate nonlinear solver loops,
convergence checks, resolvent logic, inverter internals, or preconditioners.
Implicit and IMEX schemes use it for scheme-owned algebra such as known stage
shifts, final increments, and embedded error combinations.

The general `linear_combine` tuple is one-based by arity:

- `linear_combine[0]`: one-source combination, `out = a0 * x0`
- `linear_combine[1]`: two-source combination, `out = a0 * x0 + a1 * x1`
- `linear_combine[2]`: three-source combination
- and so on through the fused arity supplied by the translation.

Fast paths should obey the same aliasing rule as translation application: the
output buffer may be one of the input buffers, so kernels should be correct for
in-place use.

## Accelerators

Accelerators are configured workers in the same sense as schemes, resolvents,
and inverters. STARK ships a small built-in library under `stark.accelerators`
and audits user-defined accelerators through `AcceleratorAudit` in
`stark.contracts.accelerator`.

The built-in import path is:

```python
from stark.accelerators import AcceleratorAbsent, AcceleratorJax, AcceleratorNumba
```

Users who want a custom accelerator implement the public accelerator protocol
and use `compile(...)` for plain callable kernels:

```python
from stark.accelerators import AcceleratorNumba

accelerator = AcceleratorNumba()

@accelerator.compile
def rhs_kernel(values, out):
    ...
```

Higher-level STARK entry points should remain the usual starting point. Custom
compiled kernels and custom worker objects are available when a problem needs a
performance-specific implementation.

`Auditor(..., accelerator=my_accelerator)` checks that a custom accelerator is
conformant before a solve, just as `Auditor(..., scheme=...)` and
`Auditor(..., marcher=...)` check the rest of the configured-worker stack.

## Custom schemes

`Marcher` accepts any object that satisfies the `SchemeLike` contract. A custom
scheme does not need to inherit from a STARK base class.

The minimal scheme interface is:

```python
class MyScheme:
    def __call__(self, interval, state, executor):
        ...
        return accepted_dt

    def snapshot_state(self, state):
        ...
```

The `__call__` method should:

- choose a step no larger than `interval.stop - interval.present`;
- mutate `state` by one accepted step;
- update `interval.step` to the next proposed step;
- return the accepted step size.

`Marcher` will then increment `interval.present` by the returned step size.

For built-in-style explicit schemes, the common pattern is to use the helpers under
`stark.schemes.explicit._support`. They install the non-algorithmic parts that would
otherwise clutter a scheme body: workspace construction, state snapshots,
adaptive step control, monitoring hooks, and display metadata.

```python
from stark.schemes.explicit._support import initialise_explicit_support


class MyScheme:
    def __init__(self, derivative, allocator):
        initialise_explicit_support(self, derivative, allocator)

    def __call__(self, interval, state, executor):
        ...
```

The richer `Scheme` protocol also exposes readable metadata, tableaus, and
string formatting. That is useful for library-quality schemes, but not required
for `Marcher`.

Run `Auditor(..., scheme=my_scheme)` to check a custom scheme alongside the
state, translation, allocator, interval, and ExecutorTolerance objects.

## Implicit and IMEX schemes

Built-in implicit and IMEX schemes use the same `Marcher` and `Integrator`
layer as the explicit schemes, but they ask the user for extra structure.

The scheme-facing object is a `Resolvent`. The additional pieces depend on
which resolvent you choose:

- `ResolventPicard` only needs the implicit derivative.
- `ResolventAnderson` and `ResolventBroyden` also need an `InnerProduct`.
- `ResolventNewton` needs a `Linearizer` and an `LegacyInverterLike`.

The common implicit shape is:

```python
from stark import Executor, Marcher, ExecutorTolerance
from stark.accelerators import AcceleratorAbsent
from stark.inverters import InverterBiCGStab
from stark.inverters import InverterPolicy, InverterTolerance
from stark.resolvents import ResolventNewton
from stark.resolvents import ResolventPolicy, ResolventTolerance
from stark.schemes import SchemeKvaerno3

allocator = MyAllocator()
derivative = MyDerivative()
linearizer = MyLinearizer()
accelerator = AcceleratorAbsent()
inverter = InverterBiCGStab(
    allocator,
    my_inner_product,
    ExecutorTolerance=InverterTolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=InverterPolicy(max_iterations=24),
    accelerator=accelerator,
)
resolvent = ResolventNewton(
    allocator,
    linearizer=linearizer,
    inverter=inverter,
    ExecutorTolerance=ResolventTolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=ResolventPolicy(max_iterations=24),
    accelerator=accelerator,
)
scheme = SchemeKvaerno3(
    derivative,
    allocator,
    resolvent=resolvent,
)
executor = Executor(tolerance=ExecutorTolerance(atol=1.0e-6, rtol=1.0e-5))
marcher = Marcher(scheme, executor)
```

For Anderson or Broyden, replace `ResolventNewton(...)` with
`ResolventAnderson(...)` or `ResolventBroyden(...)`, pass an inner product
directly to that resolvent, and omit the linearizer/inverter pair.

The same resolvent layer is what makes IMEX schemes fit naturally into the
package: the explicit part advances directly, while the implicit diagonal stage
corrections are handed to the chosen resolvent.

The current package layout mirrors that structure directly:

- `stark.accelerators`
- `stark.schemes`
- `stark.resolvents`
- `stark.inverters`
- `stark.executor`
- `stark.comparison`
- `stark.contracts`

The `Auditor` is especially useful here because implicit methods rely on more
contracts at once:

- the `Translation` vector-space and norm structure;
- the `Linearizer` Jacobian action for Newton-like resolvents;
- the `InnerProduct` for Krylov and secant-based methods;
- the in-place linear-combination kernels required for strict implicit
  operator algebra.

For the mathematical meaning of those contracts, see
[`docs/contracts_math.md`](contracts_math.md).

## Comparison Reports

The comparison reports live under `examples/comparison/` and are intended to be readable
examples of idiomatic STARK, SciPy, and Diffrax implementations of the same
problems.

Install comparison dependencies with:

```powershell
python -m pip install -e ".[comparison]"
```

Run:

```powershell
python -m examples.comparison.brusselator_2d.report
python -m examples.comparison.fput.report
python -m examples.comparison.fitzhugh_nagumo_1d.report
python -m examples.comparison.robertson.report
```

The STARK comparison implementations intentionally use `Algebraist`-generated
fast translation paths so the reports show both the generic interface and the
performance-oriented extension point.

Formal performance-regression tracking belongs in an ASV suite, not in these
comparison reports. See [`docs/benchmarking.md`](benchmarking.md).


