# STARK examples

The examples are executable documentation. The maintained inventory lives in
`examples/manifest.py`, and the group runners use that manifest.

## Getting started

Small first-contact examples for the high-level interface:

```powershell
python -m examples.getting_started
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
python -m examples.getting_started.returning_derivative
python -m examples.getting_started.in_place_derivative
python -m examples.getting_started.multiple_fields
python -m examples.getting_started.choose_scheme
python -m examples.getting_started.checkpoints
python -m examples.getting_started.engine_acceleration
```

## Backends

Backend examples show the same interface with different storage engines:

```powershell
python -m examples.backends
python -m examples.backends.native
python -m examples.backends.numpy
python -m examples.backends.jax
python -m examples.backends.cupy
```

JAX and CuPy examples skip when their optional dependencies are not installed.

## Features

Focused examples for specific STARK extension points:

```powershell
python -m examples.features
python -m examples.features.manual_stepper_setup
python -m examples.features.custom_scheme_fixed_explicit
python -m examples.features.structured_state_minimal
python -m examples.features.foreign_model_allocator
python -m examples.features.derivative_styles
python -m examples.features.linearizer_styles
python -m examples.features.norm_policy
python -m examples.features.compare_two_schemes
python -m examples.features.monitor_scheme_steps
python -m examples.features.compare_with_monitor_summary
python -m examples.features.inverter_dense
python -m examples.features.inverter_krylov
```

## Case studies

Longer teaching examples:

```powershell
python -m examples.case_studies
python -m examples.case_studies.three_body
python -m examples.case_studies.allen_cahn
python -m examples.case_studies.backends
```

## Comparison

Comparison reports and backend-oriented experiments live under
`competition/`.
