# STARK functionality guide

This page is a compact map of the public functionality in STARK. It is meant
to answer two questions:

- what can I use out of the box?
- where can I plug in my own problem-specific code?

For ordinary scalar or array-valued initial-value problems, start with the
public interface guide: [`docs/interface.md`](interface.md).

For a worked narrative example using the explicit core API, start with
[`examples/three_body_stark.ipynb`](../examples/three_body_stark.ipynb).

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
- `Workbench`: an object that allocates blank states/translations and copies
  states.
- `Scheme`: a one-step integration method.
- `Marcher`: couples a scheme to tolerances and performs one accepted step.
- `Integrator`: runs repeated `Marcher` calls over an interval.
- `Comparator`: compares two or more marcher setups on the same problem.
- `Auditor`: checks that the objects satisfy the STARK contracts before a long
  run.

The standard import path is:

```python
from stark import Executor, Marcher, Auditor, Integrator, Interval, Tolerance
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
    tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
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

`Executor` also carries the selected `Accelerator`. Accelerators live under
`stark.accelerators`, with contracts in `stark.contracts`, so the extension
point is explicit and auditable rather than hidden behind backend checks. The
default is `AcceleratorAbsent()`, so acceleration remains opt-in.

Checkpoints are useful for plots and animations: the solver may adapt internally
while the user only observes chosen output times.

## Comparator

`Comparator` is a small comparison tool for development work on custom schemes.

It takes:

- one `ComparatorProblem`
- two or more `ComparatorEntry` objects

and reports:

- pairwise final-state differences
- optional pairwise checkpoint-trajectory differences
- setup, warmup, and repeated runtime timings
- optional problem-supplied final-state diagnostics
- a bucketed `cProfile` breakdown showing approximate shares of problem work,
  scheme work, resolvent work, inverter work, and framework overhead

This is meant for A/B testing schemes on one problem, not for replacing the
problem-specific comparison reports under `examples/comparison/`.

The returned `ComparatorReport` is also structured data, not just a printable
report. Advanced users can inspect:

- `report.results_by_name()`
- `report.timings_by_name()`
- `report.diagnostics_by_name()`
- `report.final_difference_map()`
- `report.trajectory_difference_map()`
- `report.profiles_by_name()`
- `report.as_dict()`

`ComparatorEntry` also accepts an optional `profile_category(filename, lineno, function_name)`
hook so a custom entry can teach the comparator how to bucket its own profiled
code into `problem`, `scheme`, `resolvent`, `inverter`, `framework`, or `other`.

For a worked example, see
[`examples/allen_cahn.ipynb`](../examples/allen_cahn.ipynb).

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
from stark.schemes.explicit_adaptive import SchemeDormandPrince
from stark.schemes.explicit_fixed import SchemeRK4
from stark.schemes.implicit_adaptive import SchemeKvaerno3
from stark.schemes.imex_adaptive import SchemeKennedyCarpenter43_7
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

The `Workbench` must provide:

- `allocate_state()`: return a blank state object.
- `copy_state(dst, src)`: overwrite `dst` with `src`.
- `allocate_translation()`: return a blank translation object.

The `Derivative` is usually the thinnest adapter:

```python
def derivative(interval, state, out):
    t = interval.present
    out.position[:] = state.velocity
    out.velocity[:] = acceleration_from_existing_code(t, state)
```

For IMEX work, STARK now also provides a small split carrier:

```python
from stark import ImExDerivative

imex = ImExDerivative(
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
    translation=workbench.allocate_translation(),
    workbench=workbench,
    interval=interval,
    scheme=scheme,
    tolerance=tolerance,
)
print(audit)
audit.raise_if_invalid()
```

Or, for an IMEX split:

```python
Auditor.require_imex_scheme_inputs(imex, workbench, workbench.allocate_translation())
```

## Fast translation paths

The generic translation fallback uses `__add__` and `__rmul__`. That is simple
and expressive, but it may allocate many temporary objects. For array-backed
or performance-sensitive problems, translations can expose optimized
linear-combination kernels.

The easiest way to do that is with `Algebraist`, which generates inspectable
fast paths from field metadata:

```python
from stark.algebraist import Algebraist, AlgebraistField, AlgebraistLooped

ALGEBRAIST = Algebraist(
    fields=(
        AlgebraistField("du", "u", policy=AlgebraistLooped(rank=2)),
        AlgebraistField("dv", "v", policy=AlgebraistLooped(rank=2)),
    ),
    accelerator=ACCELERATOR,
    generate_norm="rms",
)


class ArrayTranslation:
    __slots__ = ("du", "dv")

    def __init__(self, du, dv):
        self.du = du
        self.dv = dv

    def __add__(self, other):
        return ArrayTranslation(self.du + other.du, self.dv + other.dv)

    def __rmul__(self, scalar):
        return ArrayTranslation(scalar * self.du, scalar * self.dv)

    linear_combine = ALGEBRAIST.linear_combine
    __call__ = ALGEBRAIST.apply
    norm = ALGEBRAIST.norm
```

The workbench still returns ordinary translation objects:

```python
class ArrayWorkbench:
    def allocate_translation(self):
        return ArrayTranslation(np.zeros_like(self.prototype))
```

If you are using an accelerator, the workbench can also precompile the emitted
kernels on probe data:

```python
class ArrayWorkbench:
    def __init__(self, prototype):
        self.prototype = prototype
        probe = np.zeros_like(prototype)
        ALGEBRAIST.compile_examples(probe, probe)

    def allocate_translation(self):
        return ArrayTranslation(
            np.zeros_like(self.prototype),
            np.zeros_like(self.prototype),
        )
```

When a scheme is constructed, STARK asks the workbench for a translation,
inspects that translation for `linear_combine`, and stores the resolved kernels
inside the scheme. The rest of the integration code does not need to pass the
fast paths around explicitly.

`Algebraist` keeps the generated source on `sources`, `kernel_sources`, and
`wrapper_sources`, so users can inspect exactly what it emitted before trusting
it. If a problem is better served by handwritten code, users can override the
generated `linear_combine`, `__call__`, or `norm` methods directly.

Explicit schemes can also receive the same object with `algebraist=ALGEBRAIST`.
That asks STARK to generate scheme-specific state-update kernels for explicit
Runge-Kutta stages and final updates. It is useful when the problem is large
enough, or repeated often enough, to offset accelerator warmup and compilation
costs. For small one-off solves, constructing the scheme without `algebraist=`
may still be faster.

The entries are:

- `linear_combine[0]`: `scale(out, a, x)`
- `linear_combine[1]`: `combine2(out, a0, x0, a1, x1)`
- `linear_combine[2]`: `combine3(out, a0, x0, a1, x1, a2, x2)`
- and so on through the fused arity supplied by the translation.

If only `scale` and `combine2` are supplied, STARK builds the higher-arity
combinations from `combine2` and scratch translations allocated by the
workbench. The generic scheme workspace has named combination slots through
`combine12`, which covers the current built-in explicit, implicit, and IMEX
schemes. For best performance, provide fused `combine3`, `combine4`, ...
kernels directly. `Algebraist` can generate fused kernels according to its
configured `fused_up_to`, and users can still attach handwritten kernels when
that is a better fit.

Fast paths should obey the same aliasing rule as translation application: the
output buffer may be one of the input buffers, so kernels should be correct for
in-place use.

## Accelerators

Accelerators are configured workers in the same sense as schemes, resolvents,
and inverters. STARK ships a small built-in library under `stark.accelerators`
and audits user-defined accelerators through `AcceleratorAudit` in
`stark.contracts.acceleration`.

The built-in import path is:

```python
from stark.accelerators import AcceleratorAbsent, AcceleratorJax, AcceleratorNumba
```

Users who want a custom accelerator implement the public accelerator protocol
and, when a worker has an accelerated form, expose it with an
`accelerated(accelerator, request)` hook:

```python
from stark.contracts.acceleration import AccelerationRequest, AccelerationRole


class MyDerivative:
    def __call__(self, interval, state, out):
        ...

    def accelerated(self, accelerator, request: AccelerationRequest):
        if request.role is AccelerationRole.DERIVATIVE and accelerator.name == "numba":
            return MyNumbaDerivative(...)
        return self
```

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
from stark.machinery.stage_solve.workspace import SchemeWorkspace


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

## Implicit and IMEX schemes

Built-in implicit and IMEX schemes use the same `Marcher` and `Integrator`
layer as the explicit schemes, but they ask the user for extra structure.

The scheme-facing object is now a `Resolvent`. The additional pieces depend on
which resolvent you choose:

- `ResolventPicard` only needs the implicit derivative.
- `ResolventAnderson` and `ResolventBroyden` also need an `InnerProduct`.
- `ResolventNewton` needs a `Linearizer` and an `InverterLike`.

The common implicit shape is:

```python
from stark import Executor, Marcher, Tolerance
from stark.accelerators import AcceleratorAbsent
from stark.inverters import InverterBiCGStab
from stark.inverters.policy import InverterPolicy
from stark.inverters.tolerance import InverterTolerance
from stark.resolvents import ResolventNewton
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.tolerance import ResolventTolerance
from stark.schemes import SchemeKvaerno3

workbench = MyWorkbench()
derivative = MyDerivative()
linearizer = MyLinearizer()
accelerator = AcceleratorAbsent()
inverter = InverterBiCGStab(
    workbench,
    my_inner_product,
    tolerance=InverterTolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=InverterPolicy(max_iterations=24),
    accelerator=accelerator,
)
resolvent = ResolventNewton(
    derivative,
    workbench,
    linearizer=linearizer,
    inverter=inverter,
    tolerance=ResolventTolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=ResolventPolicy(max_iterations=24),
    accelerator=accelerator,
)
scheme = SchemeKvaerno3(
    derivative,
    workbench,
    resolvent=resolvent,
)
executor = Executor(tolerance=Tolerance(atol=1.0e-6, rtol=1.0e-5), accelerator=accelerator)
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
- `stark.execution`
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











