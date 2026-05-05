# stark-ode

**State Translation Adaptive Runge-Kutta** for Python values, arrays, and
user-defined state types.

STARK is an ODE integration package with a friendly interface for ordinary
scalar and array initial-value problems, plus an explicit core API for problems
whose state is not naturally a single flat vector.

Start with `stark.interface.StarkIVP` when your state is a Python scalar,
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
- Classic fixed-step schemes, including Euler, Heun, midpoint, Kutta
  third-order, RK4, RK38, Ralston, and SSP RK33.
- Implicit schemes, including backward Euler, implicit midpoint,
  Crank-Nicolson, Crouzeix DIRK3, Gauss-Legendre 4, Lobatto IIIC 4, Radau IIA 5,
  SDIRK21, Kvaerno3, Kvaerno4, and BDF2.
- IMEX schemes, including IMEX Euler and Kennedy-Carpenter adaptive ARK
  pairs from orders 3(2) through 5(4).
- Built-in nonlinear resolvents, including Picard, Anderson, Broyden, and
  Newton.
- Built-in linear inverters, including GMRES, FGMRES, and BiCGStab.
- A public `stark.interface` layer for ordinary Python values and array-backed
  initial conditions.
- Snapshot and live integration loops.
- A `Comparator` helper for comparing two or more scheme setups on the same
  problem, including timing and `cProfile` summaries.
- Optional checkpoints for evenly spaced outputs or user-specified output
  times.
- An auditor that checks whether user objects satisfy the STARK contracts.
- Extension points for custom schemes and problem-specific fast translation
  kernels, including `Algebraist` for inspectable generated boilerplate.
- An `ImExDerivative` helper for splitting a right-hand side into implicit and
  explicit parts ahead of IMEX schemes.

Explicit schemes can also take an `algebraist=` argument. When supplied, STARK
uses the Algebraist to generate scheme-specific state-update kernels for
explicit Runge-Kutta stages, reducing repeated translation-combine and
translation-apply work. This is a performance option for large states,
long-running integrations, or repeated solves where the compiled kernels are
reused many times. Accelerated backends may need a noticeable warmup or
compilation pass, so tiny one-off integrations can be faster without an
Algebraist.

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
python -m pip install -e ".[notebooks]"
python -m pip install -e ".[comparison]"
```

- Core install:
  all built-in schemes, resolvents, inverters, auditing, and comparison tools.
- `.[accelerators]`:
  `AcceleratorNumba` and `AcceleratorJax`.
- `.[examples]`:
  plotting and accelerator dependencies used by the script-style examples.
- `.[notebooks]`:
  Jupyter and plotting dependencies for the public notebooks.
- `.[comparison]`:
  SciPy, Diffrax, JAX, and accelerator dependencies used by the comparison
  reports.

## Quick Start

For scalar or array-valued problems, use the interface layer:

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

for interval, state in ivp.integrate():
    print(interval.present, state.value)
```

Plain derivative callables use the familiar return style `f(t, y) -> dy`.
For performance-sensitive array code, STARK also supports explicit in-place
derivatives with `@StarkDerivative.in_place`.

The interface layer chooses a carrier for the initial value, wraps the state,
selects routing appropriate to the value semantics, builds a default scheme,
and runs the existing STARK integration machinery. It supports:

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
from stark import Executor, Marcher, Integrator, Interval, Tolerance
from stark.accelerators import AcceleratorAbsent
from stark.schemes import SchemeDormandPrince

workbench = MyWorkbench()
derivative = MyDerivative()
scheme = SchemeDormandPrince(derivative, workbench)
executor = Executor(
    tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
    accelerator=AcceleratorAbsent(),
)
marcher = Marcher(scheme, executor)
integrate = Integrator()

state = initial_state()
interval = Interval(present=0.0, step=1.0e-3, stop=1.0)

for output_interval, output_state in integrate(marcher, interval, state, checkpoints=100):
    observe(output_interval, output_state)
```

The user provides:

- a state object;
- a translation object that can be applied, scaled, added, and measured;
- a workbench that allocates and copies states/translations;
- a derivative callable `derivative(interval, state, out)` that writes the
  time derivative into a translation.
- an `Executor` that carries runtime tolerance, safety, and regulator
  policy.

Use this explicit core shape when the interface layer is not enough: custom
state objects, custom translation types, implicit or IMEX method setup,
problem-specific fast paths, or detailed control over schemes, resolvents, and
inverters.

`Executor` also carries the selected `Accelerator`. Built-in accelerators live
under `stark.accelerators` as concrete configured workers, parallel to schemes,
resolvents, and inverters. The default is `AcceleratorAbsent()`, so
acceleration is always opt-in rather than a hidden dependency.

For split problems, STARK also exposes:

```python
from stark import ImExDerivative

imex = ImExDerivative(
    implicit=implicit_derivative,
    explicit=explicit_derivative,
)
```

See the functionality guide for the full contract surface and built-in worker
inventory.

## Implicit shape

Implicit and IMEX schemes add a few more moving parts. Alongside the
workbench and derivative, users may provide:

- a stage `Resolvent`, such as `ResolventNewton` or `ResolventAnderson`;
- for Newton-backed resolvents, a `Linearizer` that supplies the Jacobian
  action of the derivative;
- for Newton-backed resolvents, an `Inverter`, such as `InverterBiCGStab`.

For example:

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
`stark.accelerators`, the contracts live under `stark.contracts`, and custom
accelerators can be checked with `Auditor(..., accelerator=...)` before a run.

For the mathematical view of the contracts, see
[`docs/contracts_math.md`](docs/contracts_math.md). It explains how STARK's
`State`, `Translation`, `Linearizer`, `Residual`, `Resolvent`, and `Inverter`
concepts correspond to affine-space, vector-space, norm, and operator
structures.

## Example

A guided notebook is available at
[`examples/three_body_stark.ipynb`](examples/three_body_stark.ipynb). It starts
from a structured three-body model with an Euler stepper, shows why fixed-step
Euler is fragile for Moore's figure-eight orbit, then adds the small STARK
adapter layer needed for adaptive integration and checkpointed plotting.

For a fuller method-building example, see
[`examples/allen_cahn.ipynb`](examples/allen_cahn.ipynb). It walks through a
structured one-dimensional Allen-Cahn problem, starting with explicit methods,
then moving through implicit resolvents, Newton linearizers, and finally a
custom IMEX spectral resolvent. The notebook uses `Comparator` comparisons
where they help explain which method family improves the problem.

From a source checkout:

```powershell
python -m pip install -e ".[notebooks]"
python -m jupyter lab examples/three_body_stark.ipynb
```

## Comparison Reports

Comparison reports live under [`examples/comparison/`](examples/comparison/). They compare STARK,
SciPy, and Diffrax implementations of the same problems while keeping each
implementation close to its native idiom.

```powershell
python -m pip install -e ".[comparison]"
python -m examples.comparison.brusselator_2d.report
python -m examples.comparison.fput.report
python -m examples.comparison.fitzhugh_nagumo_1d.report
python -m examples.comparison.robertson.report
```

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m pytest -m slow
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
