# STARK examples

The examples are executable documentation. The maintained inventory lives in
`examples/manifest.py`, and the group runners use that manifest.

## Getting started

Small first-contact examples for the high-level interface:

```powershell
python -m examples.getting_started
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
```

## Problem

Examples about describing a problem: dynamics, frames, linearizers,
structured state, foreign object models, and norms.

```powershell
python -m examples.problem
python -m examples.problem.returning_dynamics
python -m examples.problem.in_place_dynamics
python -m examples.problem.multiple_fields
python -m examples.problem.structured_state_minimal
python -m examples.problem.foreign_model_allocator
python -m examples.problem.foreign_model_plug_in_solver
python -m examples.problem.foreign_model_audit
python -m examples.problem.reaction_diffusion_array
python -m examples.problem.dynamics_styles
python -m examples.problem.linearizer_styles
python -m examples.problem.norm_policy
```

## Methods

Examples about choosing or extending the numerical method: schemes, resolvents,
predictors, IMEX splitting, and matrix-free Jacobian actions.

```powershell
python -m examples.methods
python -m examples.methods.choose_scheme
python -m examples.methods.custom_scheme_fixed_explicit
python -m examples.methods.imex_with_custom_spectral_resolvent
python -m examples.methods.matrix_free_jacobian
python -m examples.methods.resolvent_fixed_point
python -m examples.methods.resolvent_linearized
python -m examples.methods.scheme_family_explicit_fixed
python -m examples.methods.scheme_family_explicit_adaptive
python -m examples.methods.scheme_family_implicit_fixed
python -m examples.methods.scheme_family_implicit_adaptive
python -m examples.methods.scheme_family_imex_fixed
python -m examples.methods.scheme_family_imex_adaptive
python -m examples.methods.scheme_predictor
```

## Diagnostics

Examples about observing, comparing, and explaining integrations.

```powershell
python -m examples.diagnostics
python -m examples.diagnostics.checkpoints
python -m examples.diagnostics.compare_two_schemes
python -m examples.diagnostics.compare_custom_scheme
python -m examples.diagnostics.monitor_scheme_steps
python -m examples.diagnostics.error_ratio_trace
python -m examples.diagnostics.compare_with_monitor_summary
python -m examples.diagnostics.monitor_vs_timing
python -m examples.diagnostics.monitoring_levels
```

## Engines

Examples about backend availability and acceleration. Optional backend examples
are listed here, but the group runner skips them unless the manifest marks them
as default.

```powershell
python -m examples.engines
python -m examples.engines.engine_acceleration
python -m examples.engines.backend_native
python -m examples.engines.backend_numpy
python -m examples.engines.backend_jax
python -m examples.engines.backend_cupy
```

## Inverters

Examples about linear correction solves used by implicit resolvents.

```powershell
python -m examples.inverters
python -m examples.inverters.inverter_request_and_defect
python -m examples.inverters.inverter_dense
python -m examples.inverters.inverter_krylov
python -m examples.inverters.inverter_relaxation_richardson
python -m examples.inverters.inverter_relaxation_jacobi
python -m examples.inverters.inverter_relaxation_linear_fixed
```

## Core

Lower-level examples for users who deliberately want the core integration
objects rather than the high-level `System` path.

```powershell
python -m examples.core
python -m examples.core.manual_stepper_setup
```

## Comparison

Method-comparison reports live under diagnostics examples. Competition reports
are separate showcase scripts with their own timing tables, and benchmark timing
caveats belong in ASV benchmarks or backend-focused examples.
