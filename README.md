# stark-ode

**State Translation Adaptive Runge-Kutta** for user-defined Python state types.

STARK is an ODE integration package for problems whose state is not naturally a
single flat vector. It lets users keep rich mutable state objects, define the
linear translation objects used by Runge-Kutta methods, and then integrate those
states with adaptive or fixed-step schemes.

This is useful when a simulation already has its own domain model: particles,
fields, lattices, structured arrays, nested dataclasses, or other objects where
flattening everything just to call a solver would obscure the code.

## What STARK Provides

- Adaptive embedded Runge-Kutta schemes, including Cash-Karp,
  Dormand-Prince, Fehlberg 4(5), Bogacki-Shampine, and Tsitouras 5.
- Classic fixed-step schemes, including Euler, Heun, midpoint, Kutta third
  order, RK4, RK38, Ralston, and SSP RK33.
- Snapshot and live integration loops.
- Optional checkpoints for evenly spaced outputs or user-specified output
  times.
- An auditor that checks whether user objects satisfy the STARK contracts.
- Extension points for custom schemes and problem-specific fast translation
  kernels.

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

Optional extras are available for notebooks and benchmarks:

```powershell
python -m pip install -e ".[notebooks]"
python -m pip install -e ".[benchmarks]"
```

## Basic Shape

A STARK integration usually has four user-side objects:

```python
from stark import Marcher, Integrator, Interval, Tolerance
from stark.scheme_library import SchemeDormandPrince

workbench = MyWorkbench()
derivative = MyDerivative()
scheme = SchemeDormandPrince(derivative, workbench)
marcher = Marcher(scheme, tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6))
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
- a derivative callable that writes the time derivative into a translation.

See the functionality guide for the full contract.

## Documentation

The compact functionality guide is
[`docs/README.md`](docs/README.md). It lists the integration APIs, built-in
schemes, checkpoints, auditing tools, custom scheme contract, and translation
fast paths.

## Example

A guided notebook is available at
[`examples/three_body_stark.ipynb`](examples/three_body_stark.ipynb). It starts
from a structured three-body model with an Euler stepper, shows why fixed-step
Euler is fragile for Moore's figure-eight orbit, then adds the small STARK
adapter layer needed for adaptive integration and checkpointed plotting.

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
