# STARK examples

The examples are grouped by how much context they assume. They are intended to
be readable scripts first and test fixtures second: each file should teach one
piece of the public API or one extension point.

## Getting started

Small first-contact examples for the high-level `System` / `Frame` / `Method` /
`Engine` interface:

```powershell
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
python -m examples.getting_started.returning_derivative
python -m examples.getting_started.in_place_derivative
python -m examples.getting_started.multiple_fields
python -m examples.getting_started.choose_scheme
python -m examples.getting_started.checkpoints
python -m examples.getting_started.engine_acceleration
python -m examples.getting_started.interface.native
python -m examples.getting_started.interface.numpy
```

Optional backend examples skip cleanly when the backend is not installed:

```powershell
python -m examples.getting_started.interface.jax
python -m examples.getting_started.interface.cupy
```

## Features

Focused examples for specific STARK extension points:

```powershell
python -m examples.features.manual_stepper_setup
python -m examples.features.custom_scheme_fixed_explicit
python -m examples.features.structured_state_minimal
python -m examples.features.derivative_styles
python -m examples.features.linearizer_styles
python -m examples.features.norm_policy
python -m examples.features.compare_two_schemes
python -m examples.features.monitor_scheme_steps
python -m examples.features.compare_with_monitor_summary
python -m examples.features.inverter_request_and_defect
python -m examples.features.inverter_dense
python -m examples.features.inverter_krylov
python -m examples.features.scheme_predictor
python -m examples.features.monitor_vs_timing
python -m examples.features.inverter_relaxation_richardson
python -m examples.features.inverter_relaxation_jacobi
python -m examples.features.inverter_relaxation_specialist
python -m examples.features.monitoring_levels
```

These cover the main extension seams:

- derivative adapters: in-place, returning, and field-level kernels;
- linearizer adapters: matrix-free apply and dense fill;
- frame norm policy;
- scheme predictor workers;
- request-shaped inverters: dense, Krylov, and relaxation;
- diagnostics and the monitor-free timing convention.

## Case studies

Longer teaching examples:

```powershell
python -m examples.case_studies.three_body
python -m examples.case_studies.allen_cahn
```

The Allen-Cahn lessons show why a large structured problem may need a
matrix-free Krylov inverter or a problem-specific IMEX split.

## Comparison

Comparison reports and backend-oriented experiments live under `competition/`.
Those reports deliberately separate preparation time, warm repeated-solve time,
and total time.
