# stark-ode

**State Translation Adaptive Runge-Kutta** for user-defined Python state types.

STARK is an ODE integration package for problems whose state is not naturally a
single flat vector. It lets users keep rich mutable state objects, define the
linear translation objects used by Runge-Kutta methods, and then integrate those
states with adaptive, fixed-step, implicit, and IMEX schemes.

This is useful when a simulation already has its own domain model: particles,
fields, lattices, structured arrays, nested dataclasses, or other objects where
flattening everything just to call a solver would obscure the code.

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

## Installation

Install directly from GitHub:

```powershell
python -m pip install git+https://github.com/jonmfellows/stark-ode.git
```

For a local editable checkout:

```powershell
git clone https://github.com/jonmfellows/stark-ode.git
cd stark-ode
python -m pip install -e .
```

Optional extras are available by task:

```powershell
python -m pip install -e ".[accelerators]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[notebooks]"
python -m pip install -e ".[benchmarks]"
```

- Core install:
  all built-in schemes, resolvents, inverters, auditing, and comparison tools.
- `.[accelerators]`:
  `AcceleratorNumba` and `AcceleratorJax`.
- `.[examples]`:
  plotting and accelerator dependencies used by the script-style examples.
- `.[notebooks]`:
  Jupyter and plotting dependencies for the public notebooks.
- `.[benchmarks]`:
  SciPy, Diffrax, JAX, and accelerator dependencies used by the benchmark
  reports.

## Basic shape

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

`Executor` also carries the selected `Accelerator`. Accelerators now live under
`stark.accelerators` as concrete configured workers, parallel to schemes,
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

The compact functionality guide is
[`docs/README.md`](docs/README.md). It lists the integration APIs, built-in
schemes, resolvents, inverters, accelerators, execution tools, auditing hooks,
custom scheme contracts, `Algebraist`, and translation fast paths.

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

## Benchmarks

Benchmark reports live under [`benchmarks/`](benchmarks/). They compare STARK,
SciPy, and Diffrax implementations of the same problems while keeping each
implementation close to its native idiom.

```powershell
python -m pip install -e ".[benchmarks]"
python -m benchmarks.brusselator_2d.report
python -m benchmarks.fput.report
python -m benchmarks.fitzhugh_nagumo_1d.report
python -m benchmarks.robertson.report
```

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m pytest -m slow
```

## Citation

If you use `stark-ode` in research or published work, please cite the package
repository. Citation metadata is provided in [`CITATION.cff`](CITATION.cff), and
GitHub should render it as a ready-to-copy citation.

## License

`stark-ode` is released under the MIT license. See [`LICENSE`](LICENSE).
