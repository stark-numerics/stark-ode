# Benchmarks

STARK includes two public benchmark cases:

- `brusselator_2d`: a periodic two-species reaction-diffusion problem.
- `fput`: a Fermi-Pasta-Ulam-Tsingou beta lattice with fixed endpoints.

Run them with:

```powershell
python -m benchmarks.brusselator_2d.report
python -m benchmarks.fput.report
```

Each report generates a tight SciPy DOP853 reference solution, runs STARK,
SciPy, and Diffrax at a shared tolerance setting, then prints:

- an error table against the reference solution
- a preparation timing table for setup plus one untimed warmup solve
- a run timing table for repeated solves after warmup
- a short summary of the best accuracy and fastest timings

Reference generation is reported separately and is not included in the solver
timing tables.

## Benchmark layout

Each benchmark case directory follows the same structure:

- `common.py` contains plain dictionaries and NumPy arrays for problem
  parameters, tolerances, benchmark settings, and initial conditions.
- `stark.py` contains a STARK-native implementation using structured mutable
  states, translations, a workbench, callable derivative objects, and optional
  Numba-backed fast paths.
- `scipy.py` contains a SciPy-native implementation using `solve_ivp`, flat
  NumPy arrays, and plain RHS callbacks.
- `diffrax.py` contains a Diffrax-native implementation using `ODETerm`, solver
  objects, save policies, controllers, JAX arrays, and small functional vector
  fields.
- `report.py` owns reference generation, timing, table rendering, and the
  human-readable benchmark description.

The benchmark modules intentionally avoid a shared runner abstraction. The
point is to show each library in the coding style it naturally encourages, not
to force all solvers through one generic harness.

## Benchmarking principles

STARK is compared with SciPy and Diffrax because they represent three different
ways to build ODE solves in Python:

### SciPy idiom

- Problem-first, object-light API
- One top-level `solve_ivp` call with keyword configuration
- Plain callback functions for the vector field
- Flat numeric array state
- Optional behavior supplied through extra callbacks or flags

### Diffrax idiom

- Composable object graph rather than one convenience wrapper
- Solver configuration split into terms, solvers, save policies, controllers,
  and arguments
- Functional/JAX style with small pure callables
- PyTree state and transformation-friendly workflows
- Explicit policy objects for adaptation and output
- Swappable components as a first-class design goal

### STARK idiom

- Rich structured mutable states
- Explicit separation between states and translations
- User-defined allocation and execution workbench
- Performance-critical behavior attached to protocol objects
- Repeated setup moved to construction time
- Domain-specific fast paths for translation algebra and derivatives
- Solver core decoupled from domain mechanics through user-defined protocols

There is no single best coding style. These benchmarks are intended to show how
the STARK design behaves on non-stiff ODE problems whose natural state is richer
than one flat vector, while still comparing against idiomatic SciPy and Diffrax
implementations.

## Interpreting results

The reports are CPU wall-clock comparisons for repeated solves of one
medium-sized problem. They are not a claim that one library is universally
faster or more accurate.

Important details:

- Equal `rtol` and `atol` values do not imply identical error norms across
  libraries.
- Diffrax warmup can include JAX tracing and compilation.
- STARK warmup can include Numba compilation when Numba is installed.
- STARK reports whether it is using Numba-jitted kernels or NumPy fallback
  kernels.
- Preparation cost and steady repeated-run cost are shown separately because
  different libraries place work in different phases.
