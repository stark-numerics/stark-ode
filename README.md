# stark-ode

**State Translation Adaptive Runge-Kutta** for Python values, arrays, and
user-defined state types.

STARK is an ODE integration package with a friendly interface for ordinary
scalar and array initial-value problems, plus an explicit core API for problems
whose state is not naturally a single flat vector.

Start with `stark.problem.StarkIVP` when your state is a Python scalar,
sequence, NumPy array, CuPy array, or JAX array. Move to the core STARK objects
when a simulation already has its own domain model: particles, fields,
lattices, structured arrays, nested dataclasses, or other objects where
flattening everything solely to call a solver would obscure the code.

For performance-sensitive ordinary array problems, dense NumPy arrays are a
good first choice. They can be one-dimensional or multidimensional, and the
interface layer uses in-place array routing where possible.

## What STARK provides

- Adaptive embedded Runge-Kutta schemes, including Cash-Karp,
  Dormand-Prince, Fehlberg 4(5), Bogacki-Shampine, and Tsitouras 5.
- Fixed-step schemes, including Euler, Heun, midpoint, Kutta
  third-order, RK4, RK38, Ralston, and SSP RK33.
- Implicit schemes, including backward Euler, implicit midpoint,
  Crank-Nicolson, Crouzeix DIRK3, Gauss-Legendre 4, Lobatto IIIC 4, Radau IIA 5,
  SDIRK21, Kvaerno3, Kvaerno4, and BDF2.
- IMEX schemes, including IMEX Euler and Kennedy-Carpenter adaptive ARK
  pairs from orders 3(2) through 5(4).
- Built-in nonlinear resolvents, including Picard, Anderson, Broyden, and
  Newton.
- Built-in linear inverters, including GMRES, FGMRES, and BiCGStab.
- A public `stark.problem` layer for ordinary Python values and array-backed
  initial conditions.
- Snapshot and live integration loops.
- A `ComparisonRunner` helper for comparing two or more scheme setups on the same
  problem, including timing and `cProfile` summaries.
- Optional checkpoints for evenly spaced outputs or user-specified output
  times.
- An auditor that checks whether user objects satisfy the STARK contracts.
- Extension points for custom schemes and problem-specific fast translation
  kernels, including Algebraist providers for inspectable generated stage
  algebra.
- A `Derivative.split(...)` helper for splitting a right-hand side into implicit
  and explicit parts ahead of IMEX schemes.

Performance-sensitive custom objects can expose Algebraist-backed
`linear_combine` kernels on their translation type and can pass a scheme
specialist into built-in schemes. This is a performance option for large
states, long-running integrations, or repeated solves where the compiled
kernels are reused many times. Accelerated backends may need a noticeable
warmup or compilation pass, so tiny one-off integrations can be faster on the
plain path.

## Installation

Install directly from GitHub:

```powershell
python -m pip install git+https://github.com/stark-numerics/stark-ode.git
```

For a local editable checkout:

```powershell
git clone https://github.com/stark-numerics/stark-ode.git
cd stark-ode
python -m pip install -e .
```

Optional extras are available by task:

```powershell
python -m pip install -e ".[accelerators]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[comparison]"
```

- Core install:
  all built-in schemes, resolvents, inverters, auditing, and comparison tools.
- `.[accelerators]`:
  `AcceleratorNumba` and `AcceleratorJax`.
- `.[examples]`:
  plotting and accelerator dependencies used by the script-style examples.
- `.[comparison]`:
  SciPy, Diffrax, JAX, and accelerator dependencies used by the comparison
  reports.

## Quick Start

For scalar or array-valued problems, use the interface layer:

```python
import numpy as np

from stark import Interval
from stark.problem import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=np.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=2.0),
)

for interval, state in ivp.integrate():
    print(interval.present, state.value)
```

Plain derivative callables use the familiar return style `f(t, y) -> dy`.
For performance-sensitive array code, STARK also supports explicit in-place
derivatives with `@StarkDerivative.in_place`.

The interface layer chooses a carrier for the initial value, wraps the state,
selects routing appropriate to the value semantics, builds a default scheme,
and runs the core STARK integration objects. It supports:

- native Python `int`, `float`, `list`, and `tuple` values;
- NumPy arrays;
- CuPy arrays when CuPy is installed and usable;
- JAX arrays in a Python-level solve. This does not yet mean whole-solver
  `jax.jit`, `jax.grad`, or `jax.vmap`.

See the interface guide for return-style and in-place derivatives, explicit
carrier selection, routing, backend support levels, and examples:
[`docs/interface.md`](docs/interface.md).

From a source checkout with the package installed, the script-style interface
examples can be run in module mode:

```powershell
python -m examples.interface.native
python -m examples.interface.numpy
python -m examples.interface.cupy
python -m examples.interface.jax
```

## Core Shape

A STARK integration usually has five user-side objects:

```python
from stark import Marcher, Integrator, Interval, Tolerance
from stark.engines.shared.accelerators import AcceleratorNone
from stark.methods.schemes import SchemeDormandPrince

allocator = MyAllocator()
derivative = MyDerivative()
scheme = SchemeDormandPrince(derivative, allocator)
Configuration = Configuration(
    tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
)
marcher = Marcher(scheme)
integrate = Integrator()

state = initial_state()
interval = Interval(present=0.0, step=1.0e-3, stop=1.0)

for output_interval, output_state in integrate(marcher, interval, state, checkpoints=100):
    observe(output_interval, output_state)
```

The user provides:

- a state object;
- a translation object that can be applied, scaled, added, and measured;
- an allocator that allocates and copies states/translations;
- a derivative callable `derivative(interval, state, out)` that writes the
  time derivative into a translation.
- an `Configuration` that carries runtime Tolerance, and ConfigurationAdaptivity
  policy.

Use this explicit core shape when the interface layer is not enough: custom
state objects, custom translation types, implicit or IMEX method setup,
problem-specific fast paths, or detailed control over schemes, resolvents, and
inverters.

Accelerators are passed directly to the objects that use them, such as
Algebraist providers, resolvents, or inverters. `Configuration` deliberately carries
only runtime execution policy.

For split problems, declare the implicit and explicit parts through
`Derivative.split(...)`:

```python
from stark import Derivative

imex = Derivative.split(
    implicit=implicit_derivative,
    explicit=explicit_derivative,
)
```

See the functionality guide for the full contract surface and built-in worker
inventory.

## Implicit shape

Implicit and IMEX schemes add a few more moving parts. Alongside the
allocator and derivative, users may provide:

- a stage `Resolvent`, such as `ResolventNewton` or `ResolventAnderson`;
- for Newton-backed resolvents, a `LinearizerLike` that supplies the Jacobian
  action of the derivative;
- for Newton-backed resolvents, an `Inverter`, such as `InverterBiCGStab`.

For example:

```python
from stark import Marcher, Tolerance
from stark.engines.shared.accelerators import AcceleratorNone
from stark.methods.inverters import InverterBiCGStab
from stark.methods.inverters import InverterPolicy, Tolerance
from stark.methods.resolvents import ResolventNewton
from stark.methods.resolvents.policy import Configuration
from stark.methods.resolvents.Tolerance import Tolerance
from stark.methods.schemes import SchemeKvaerno3

allocator = MyAllocator()
derivative = MyDerivative()
linearizer = MyLinearizer()
accelerator = AcceleratorNone()
inverter = InverterBiCGStab(
    allocator,
    my_inner_product,
    Tolerance=Tolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=InverterPolicy(max_iterations=24),
    accelerator=accelerator,
)
resolvent = ResolventNewton(
    derivative,
    allocator,
    linearizer=linearizer,
    inverter=inverter,
    Tolerance=Tolerance(atol=1.0e-7, rtol=1.0e-7),
    policy=Configuration(max_iterations=24),
    accelerator=accelerator,
)
scheme = SchemeKvaerno3(
    derivative,
    allocator,
    resolvent=resolvent,
)
Configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-6, rtol=1.0e-5))
marcher = Marcher(scheme)
```

Anderson- or Broyden-backed resolvents are similar, but they do not need a
linearizer or inverter. They do need an inner product on translations.

## Documentation

Start with the interface guide if you are solving ordinary scalar or array
initial-value problems:
[`docs/interface.md`](docs/interface.md).

The compact functionality guide is [`docs/README.md`](docs/README.md). It
maps the explicit core API: integration objects, built-in schemes, resolvents,
inverters, accelerators, execution tools, auditing hooks, custom scheme
contracts, `Algebraist`, and translation fast paths.

For a conceptual guide to the main object families and extension points, see
[`docs/object_map.md`](docs/object_map.md).

Accelerators follow the same philosophy. Built-in workers live under
`stark.engines.shared.accelerators`, the contracts live under `stark.core.contracts`, and custom
accelerators can be checked with `Auditor(..., accelerator=...)` before a run.

For a terminology-first map of the main ideas, see
[`docs/concepts.md`](docs/concepts.md). For the mathematical view of the
low-level contracts, see [`docs/contract-maths.md`](docs/contract-maths.md).

## Examples

Small executable examples live under [`examples/getting_started/`](examples/getting_started/)
and concept folders such as [`examples/problem/`](examples/problem/),
[`examples/methods/`](examples/methods/), and
[`examples/diagnostics/`](examples/diagnostics/). Each example is a focused
teaching script: if a longer story needs several concepts, it should be split
into smaller runnable lessons.

From a source checkout:

```powershell
python -m pip install -e ".[examples]"
python -m examples.getting_started
python -m examples.problem
python -m examples.methods
python -m examples.diagnostics
```

Useful starting points include:

```powershell
python -m examples.problem.foreign_model_plug_in_solver
python -m examples.problem.reaction_diffusion_array
python -m examples.methods.imex_with_custom_spectral_resolvent
python -m examples.methods.matrix_free_jacobian
python -m examples.diagnostics.error_ratio_trace
```

## Competition Reports

Competition reports live under [`competition/`](competition/). They compare STARK,
SciPy, and Diffrax implementations of the same problems while keeping each
implementation close to its native idiom.

```powershell
python -m pip install -e ".[comparison]"
python -m competition.brusselator_2d.report
python -m competition.fput.report
python -m competition.fitzhugh_nagumo_1d.report
python -m competition.robertson.report
```

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

Contributors working on performance-sensitive internals can use the ASV suite
described in [`docs/benchmarking.md`](docs/benchmarking.md). ASV is contributor
tooling; ordinary solver use does not require it.

## Citation

If you use `stark-ode` in research or published work, please cite the package
repository. Citation metadata is provided in [`CITATION.cff`](CITATION.cff), and
GitHub should render it as a ready-to-copy citation.

## License

`stark-ode` is released under the MIT license. See [`LICENSE`](LICENSE).
