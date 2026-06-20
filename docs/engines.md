# Use engines and backends

This page is for users choosing where STARK stores arrays and performs arithmetic.

An engine chooses:

```text
state/translation carriers
allocator behaviour
backend array type
accelerator support where available
prepared algebra support
```

## Start with NumPy

Use NumPy for the default high-level path.

```python
from stark.engines import EngineNumpy

ivp = system.ivp(..., engine=EngineNumpy)
```

NumPy is usually the best first backend: predictable, debuggable, and fast for small-to-medium problems. If Numba is installed and enabled by the engine, generated STARK algebra kernels may compile on first use.

## Use JAX

Use JAX when you want JAX arrays and JAX-compatible derivative expressions.

Prefer return-style derivatives:

```python
@DerivativeStyle.returning
def rhs(t, state):
    return {"dy": -0.5 * state.y}
```

Then use:

```python
from stark.engines import EngineJax

ivp = system.ivp(..., engine=EngineJax)
```

Important caveat: JAX array support does not automatically mean the whole adaptive solver loop is one JIT-compiled JAX program. STARK's intended accelerated high-level path is through generated Algebraist kernels where the `Frame` is known. Contributor notes for that path live in [Algebraist backend paths](contributing/algebraist_backends.md).

Run:

```powershell
python -m examples.getting_started.interface.jax
```

## Use CuPy

Use CuPy when you want GPU-backed arrays.

```python
from stark.engines import EngineCupy

ivp = system.ivp(..., engine=EngineCupy)
```

For timing, synchronize before stopping the clock. Otherwise CPU timing may not include queued GPU work.

Run:

```powershell
python -m examples.getting_started.interface.cupy
```

## Backend case study

The backend case study is intended to answer two questions:

```text
Can the same problem run on NumPy, JAX, and CuPy?
At what size do alternative backends become useful, if at all?
```

Run:

```powershell
python -m examples.case_studies.backends
```

The lessons should show backend setup directly. They should not hide all problem construction in a shared helper file; users need to see the syntax for each backend.

## Acceleration boundaries

Backend support has layers.

```text
array backend          NumPy / JAX / CuPy field storage
carrier arithmetic     field-level arithmetic on backend arrays
Algebraist generator   prepared code for known Frame-backed state
accelerator            optional compiler/fuser for generated kernels
solver control flow    adaptive stepping, acceptance, monitors, reports
```

A backend may accelerate array operations while the solver control flow remains Python-level. That is not a correctness problem, but it matters for performance.

## Common mistakes

### Comparing total time to warm time

Preparation may include compilation or GPU setup. Warm repeated timings exclude that work. Total timings include setup, warmup, and one measured solve.

### Expecting GPU speedups for tiny arrays

GPU launch and synchronization overhead can dominate small problems.

### Expecting JAX speedups from eager small operations

JAX is strongest when it can compile large pure functions with stable shapes. Many small Python-dispatched operations are a poor shape for JAX.
