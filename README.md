# stark-ode

**State Translation Adaptive Runge-Kutta** for structured ordinary differential
equations.

`stark-ode` is an initial-value ODE solver that works with ordinary Python
values, NumPy arrays, and structured simulation state.

Unlike most Python ODE libraries, STARK does **not** assume that every model
state is a single flat vector. Instead it separates the mathematical problem,
the state representation, the numerical method, and the execution backend into
independent concepts:

```text
System -> Frame -> Method -> Engine
```

- **System** describes the differential equations.
- **Frame** describes the structure of the state.
- **Method** chooses the numerical integration algorithm.
- **Engine** chooses where state is stored and where algebra is performed.

That separation allows the same numerical methods to work with simple array
problems, structured state, and more specialised simulation models without
changing the solver architecture.

## Why STARK?

Most ODE libraries start by flattening state into one vector.

STARK instead keeps the distinction between:

- the **state** of the model;
- the **translation** or increment applied by the solver;
- the **numerical method**;
- the **execution backend**.

This makes several things possible without changing the solver architecture:

- named and structured state fields;
- existing object-oriented simulation models;
- explicit, implicit, and IMEX methods under one API;
- alternative storage backends including NumPy, JAX, and CuPy;
- extensible algebra for specialised state representations.

If your model is naturally a single dense vector, STARK still works well. If it
is not naturally a vector, STARK is designed specifically for that use case.

## Installation

Until the beta release is published on PyPI:

```bash
python -m pip install git+https://github.com/stark-numerics/stark-ode.git
```

Once a beta is published on PyPI:

```bash
python -m pip install --pre stark-ode
```

For development:

```bash
git clone https://github.com/stark-numerics/stark-ode.git
cd stark-ode
python -m pip install -e .
```

Optional extras are available for documentation, examples, benchmarking,
accelerators, and comparison reports:

```bash
python -m pip install -e ".[accelerators]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[docs]"
python -m pip install -e ".[comparison]"
python -m pip install -e ".[asv]"
```

## Quick Start

```python
import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def exponential_decay(t, state, out):
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

## Documentation

The documentation is organised by learning path.

| If you want to... | Read |
|-------------------|------|
| Learn the design | `docs/concepts.md` |
| Solve your first problem | `docs/getting-started.md` |
| Build and configure systems | `docs/problem.md` |
| Choose numerical methods | `docs/methods.md` |
| Use implicit and IMEX methods | `docs/implicit.md` |
| Understand execution backends | `docs/engines.md` |
| Monitor integrations | `docs/diagnostics.md` |
| Browse runnable examples | `docs/examples.md` |

For contributors, design notes begin in `docs/contributing/README.md`.

## Examples

Examples are executable documentation. Recommended starting points are:

```bash
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
python -m examples.problem.dynamics_styles
python -m examples.methods.choose_scheme
python -m examples.diagnostics.compare_custom_scheme
```

Each example demonstrates one aspect of the library and can be run directly
after installing the `examples` extra.

## Features

STARK includes:

- explicit Runge-Kutta methods;
- adaptive embedded Runge-Kutta methods;
- implicit DIRK, SDIRK, Gauss, Radau, and BDF methods;
- IMEX methods;
- Newton, Picard, Anderson, Broyden, and related nonlinear resolvents;
- dense and iterative linear inversion strategies;
- native Python, NumPy, JAX, and CuPy execution engines;
- diagnostics, comparison tooling, and benchmarking support;
- extension points for custom state representations and numerical algorithms.

See the documentation for the complete method catalogue.

## Who Is STARK For?

STARK is intended for users who:

- want to preserve the structure of simulation state;
- work with dataclasses or object-oriented models;
- need explicit, implicit, and IMEX methods within one framework;
- want to experiment with numerical algorithms or extend solver components.

If every problem you solve is already naturally represented as a single dense
NumPy vector, SciPy's `solve_ivp` may be the simpler choice.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Citation

If you use STARK in published work, please cite the repository. Citation
metadata is provided in `CITATION.cff`.

## License

MIT License.
