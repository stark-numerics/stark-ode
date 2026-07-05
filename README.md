# stark-ode

**State Translation Adaptive Runge-Kutta** for Python values, NumPy arrays, and
structured simulation state.

`stark-ode` is an ODE integration package for initial-value problems. Its
default user path is:

```text
System + Frame + Method + Engine
```

A `System` describes the problem, a `Frame` names the state fields, a `Method`
chooses the numerical stack, and an engine chooses where state is stored and
where algebra is performed.

## Why STARK Is Different

Many ODE packages start by flattening state into one vector. STARK instead
keeps a visible split between:

```text
state        the model configuration
translation  the solver increment or tangent object
engine       the storage and algebra policy
method       the numerical time-stepping stack
```

That split is the point of the package. It means STARK can support ordinary
array-valued IVPs while also leaving room for more specialised models:

- **Foreign object models** can keep their own state and increment classes when
  flattening would damage the model design.
- **Arbitrary state representations** can be described either with a high-level
  `Frame` or, for advanced users, with custom state, translation, and allocator
  objects.
- **The translation layer** makes solver increments explicit, so schemes can
  combine, scale, apply, and measure changes without assuming every state is a
  NumPy vector.
- **Implicit and IMEX machinery** is built from composable schemes, resolvents,
  linearizers, and inverters rather than a single opaque stiff-solver switch.
- **Extensible algebra** lets `Frame`-backed systems use generated Algebraist
  kernels, while advanced models can provide their own fast paths when the
  generic runtime route is not enough.

For the high-level path, start with [`docs/getting-started.md`](docs/getting-started.md).
For custom models, see [`docs/foreign-models.md`](docs/foreign-models.md). For
extension points, see [`docs/extending.md`](docs/extending.md).

## What STARK Provides

- Explicit fixed-step schemes such as Euler, midpoint, Heun, Kutta3, RK4,
  RK38, Ralston, and SSP RK33.
- Explicit adaptive schemes such as Bogacki-Shampine, Cash-Karp,
  Dormand-Prince, Fehlberg 4(5), and Tsitouras 5.
- Implicit schemes such as backward Euler, implicit midpoint,
  Crank-Nicolson, Crouzeix DIRK3, Gauss-Legendre 4, Lobatto IIIC 4,
  Radau IIA 5, BDF2, SDIRK21, Kvaerno3, Kvaerno4, and Kvaerno5.
- IMEX schemes including IMEX Euler and Kennedy-Carpenter adaptive ARK pairs.
- Resolvents for implicit correction problems, including Picard, Newton,
  coupled Newton, chord, very-chord, Anderson, and Broyden families.
- Inverters for linear correction problems, including dense, Krylov Arnoldi,
  Richardson, Jacobi, and specialist relaxation paths.
- Engines for native Python values and NumPy arrays, plus optional JAX and CuPy
  engines when those dependencies are installed and usable.
- Diagnostics for monitoring integrations and comparing methods on short runs.
- Runnable examples organised by topic.
- Generated API reference and contributor design notes.

## Installation

Before the beta is published, install directly from GitHub:

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
python -m pip install -e ".[docs]"
python -m pip install -e ".[comparison]"
python -m pip install -e ".[asv]"
```

- `.[accelerators]`: JAX and Numba dependencies used by accelerator paths.
- `.[examples]`: dependencies used by the runnable examples.
- `.[docs]`: Sphinx, MyST, and the documentation theme.
- `.[comparison]`: SciPy, Diffrax, JAX, and other dependencies for comparison
  reports.
- `.[asv]`: contributor benchmarking tools.

`jax`, `cupy`, and `numba` support depends on the local Python and hardware
environment. Optional backend examples skip quietly when their dependency is
not installed.

## Quick Start

This is the smallest high-level solve: one named scalar-like state field,
integrated with Cash-Karp on the NumPy engine.

```python
import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def exponential_decay(t, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


system = System(
    dynamics=exponential_decay,
    frame=Frame.scalar("y", translation="dy"),
)
ivp = system.ivp(
    initial={"y": np.array([2.0])},
    interval=Interval(present=0.0, step=0.1, stop=1.0),
    method=Method(SchemeCashKarp),
    engine=EngineNumpy,
)

for interval, state in ivp.stable_trajectory():
    print(interval.present, state.y[0])
```

The examples in `examples/getting_started/` are the best first stop after this
snippet.

## Problem Shape

Most user code starts by declaring a `System`:

- `dynamics`: the state-change law;
- `frame`: the names and shapes of state fields and translation fields;
- optional `linearizer`: the Jacobian action needed by Newton-style implicit
  methods.

The `Frame` helpers cover common state layouts:

```python
Frame.scalar("y", translation="dy")
Frame.vector("y", translation="dy", length=2)
Frame.array("u", translation="du", shape=(32, 32))
Frame.fields(...)
```

Use `DynamicsStyle` decorators when a dynamics function needs an explicit
signature, for example return-style functions or field kernels that are easier
to accelerate.

## Methods

A `Method` is a stack:

```text
scheme -> optional resolvent -> optional inverter
```

For ordinary non-stiff problems, start with an explicit adaptive scheme:

```python
from stark import Method
from stark.methods import SchemeCashKarp

method = Method(SchemeCashKarp)
```

The method catalogue gives named recipes:

```python
from stark.methods import METHOD_CATALOGUE

method = METHOD_CATALOGUE.method("kvaerno5-newton-dense")
```

Implicit and IMEX methods may need a linearizer, a resolvent, and an inverter.
See `docs/implicit.md` and the examples under `examples/methods/` and
`examples/inverters/` before building those stacks by hand.

## Engines

Engines choose storage and algebra policy:

```python
from stark.engines import EngineNative, EngineNumpy
```

Optional engines are exported when their dependencies import successfully:

```python
from stark.engines import EngineJax, EngineCupy
```

The JAX and CuPy engines are backend storage/arithmetic paths. They do not imply
that the whole solver is automatically `jax.jit`, `jax.grad`, GPU-optimal, or
appropriate for every problem shape.

## Documentation

The manual starts at [`docs/index.md`](docs/index.md). The most useful path for
new users is:

```text
concepts -> getting-started -> problem -> methods
```

Useful pages:

- [`docs/getting-started.md`](docs/getting-started.md)
- [`docs/problem.md`](docs/problem.md)
- [`docs/methods.md`](docs/methods.md)
- [`docs/implicit.md`](docs/implicit.md)
- [`docs/engines.md`](docs/engines.md)
- [`docs/diagnostics.md`](docs/diagnostics.md)
- [`docs/examples.md`](docs/examples.md)

Contributor notes start at [`docs/contributing/README.md`](docs/contributing/README.md).

## Examples

Examples are executable documentation. From a source checkout:

```powershell
python -m pip install -e ".[examples]"
python -m examples.getting_started
python -m examples.problem
python -m examples.methods
python -m examples.diagnostics
python -m examples.engines
python -m examples.inverters
```

Useful starting points include:

```powershell
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
python -m examples.problem.dynamics_styles
python -m examples.methods.choose_scheme
python -m examples.methods.resolvent_linearized
python -m examples.diagnostics.compare_custom_scheme
```

The maintained example inventory lives in [`examples/manifest.py`](examples/manifest.py),
and the examples guide is [`examples/README.md`](examples/README.md).

## Competition Reports

Competition reports live under [`competition/`](competition/). They compare
STARK, SciPy, and Diffrax implementations of the same problems while keeping
each implementation close to its native idiom.

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
python -m pytest -q
.\devtools\check-types.ps1
.\devtools\check-release-surface.ps1
```

Contributor benchmarks use ASV and are described in
[`benchmarks/README.md`](benchmarks/README.md).

## Citation

If you use `stark-ode` in research or published work, please cite the package
repository. Citation metadata is provided in [`CITATION.cff`](CITATION.cff), and
GitHub should render it as a ready-to-copy citation.

## License

`stark-ode` is released under the MIT license. See [`LICENSE`](LICENSE).
