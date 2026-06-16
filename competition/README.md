# Competition Reports

STARK currently includes five public competition cases:

- `brusselator_2d`: a periodic two-species reaction-diffusion problem.
- `fitzhugh_nagumo_1d`: a periodic stiff excitable-medium reaction-diffusion problem.
- `fput`: a Fermi-Pasta-Ulam-Tsingou beta lattice with fixed endpoints.
- `hires`: an 8-variable stiff chemical kinetics problem.
- `robertson`: a stiff chemical kinetics problem.

Run them with:

```powershell
python -m competition.brusselator_2d.report
python -m competition.fitzhugh_nagumo_1d.report
python -m competition.fput.report
python -m competition.hires.report
python -m competition.robertson.report
```

To run every report as a guard against timeouts and large local regressions:

```powershell
python -m competition.check_reports --timeout 180
```

The guard can also compare against a local machine-specific baseline. The
`competition/local/` directory is git-ignored, so it is a good place to keep
these results:

```powershell
python -m competition.check_reports --write-baseline competition/local/competition-baseline.json
python -m competition.check_reports --baseline competition/local/competition-baseline.json
```

Each report generates a tight SciPy reference solution, runs the methods used
for that competition case, then prints:

- an error table against the reference solution
- a preparation timing table for setup plus one untimed warmup solve
- a run timing table for repeated solves after warmup
- a short summary of the best accuracy and fastest timings

Reference generation is reported separately and is not included in the solver
timing tables.

## Competition Frame

Each competition case directory follows the same structure:

- `common.py` contains plain dictionaries and NumPy arrays for problem
  parameters, tolerances, timing settings, and initial conditions.
- `stark.py` contains a STARK-native implementation using structured mutable
  states, translations, a allocator, callable derivative objects, and optional
  Numba-backed fast paths.
- `scipy.py` contains a SciPy-native implementation using `solve_ivp`, flat
  NumPy arrays, and plain RHS callbacks.
- `diffrax.py`, when present, contains a Diffrax-native implementation using
  `ODETerm`, solver objects, save policies, controllers, JAX arrays, and small
  functional vector fields.
- `report.py` owns reference generation, timing, table rendering, and the
  human-readable benchmark description.
- `runner.py` contains the local competition timing harness shared by the
  reports.

The backend modules still show each library in the coding style it naturally
encourages. The shared runner only owns comparison mechanics: row declarations,
optional dependency failures, setup timing, one untimed warmup solve, repeated
timed solves, and simple text tables.

## Competition Principles

Where they are included, STARK is compared with SciPy and Diffrax because they
represent three different ways to build ODE solves in Python:

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
- User-defined allocation and execution allocator
- Performance-critical behavior attached to protocol objects
- Repeated setup moved to construction time
- Domain-specific fast paths for translation algebra and derivatives
- Solver core decoupled from domain mechanics through user-defined protocols

There is no single best coding style. These reports are intended to show how
the STARK design behaves on both structured non-stiff problems and small stiff
systems, while still comparing against idiomatic SciPy and Diffrax
implementations where those comparisons are informative.

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
- Entry construction in `report.py` is not timed. Timed setup starts when the
  runner calls an entry's `prepare` function, so every solver row gets the same
  setup/warmup/repeat treatment.







