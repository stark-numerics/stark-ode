# Changelog

## 0.1.0b2 - 2026-07-19

This beta keeps the public problem-solving surface focused while tightening the
engine internals that downstream packages, including PDE extensions, need to
inspect and reuse.

### Added

- Generator-backed engine internals for compiled linear combinations, fixed
  linear scheme paths, norms, and inner products.
- Allocator decorators for custom state/translation users who want STARK to add
  standard linear-combination support while keeping their own allocation model.
- Release-surface and engine-generation benchmark checks for beta validation.

### Changed

- Engine construction now carries a clearer generator/allocator split and no
  longer exposes the removed algebraist implementation.
- Core contract files are organised by domain while retaining the public
  convenience import path from `stark.core.contracts`.
- Installation wording now points beta users at `python -m pip install --pre
  stark-ode`.

## 0.1.0b1 - 2026-07-05

Initial beta release.

This release establishes the public package shape for early users and for
packages that want to build on STARK.

### Added

- Public problem API centred on `System`, `Frame`, `Dynamics`, `Linearizer`,
  `Method`, and `Engine`.
- Named scalar, vector, array, and multi-field frames for structured ODE state.
- Explicit, implicit, and IMEX scheme families with composable resolvents and
  inverters.
- NumPy and native engines as the primary beta-supported execution paths.
- Optional JAX, CuPy, and Numba-backed paths for users with matching local
  environments.
- Diagnostics examples for monitoring, scheme comparison, timing, checkpoints,
  and error-ratio traces.
- Generated API reference, contributor design notes, and runnable example
  suites.

### Changed

- The user-facing problem language now uses `Dynamics` for the state evolution
  object rather than the older derivative naming.

### Beta Notes

- The core NumPy path is the main compatibility target for this beta.
- Optional accelerator and array-backend support depends on the installed
  versions of JAX, CuPy, Numba, and the local Python environment.
- Benchmarking and comparison tools are included for contributors, but should
  be treated as beta-era tooling rather than stable user workflows.
