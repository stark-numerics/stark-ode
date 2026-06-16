# Examples

The examples are the executable companion to the manual. They are grouped by how much context they assume.

Run the top-level example suites with:

```powershell
python -m examples.getting_started
python -m examples.features
```

## Getting started

Use these examples first:

```powershell
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
python -m examples.getting_started.returning_derivative
python -m examples.getting_started.in_place_derivative
python -m examples.getting_started.multiple_fields
python -m examples.getting_started.choose_scheme
python -m examples.getting_started.checkpoints
```

Backend-specific first-contact examples:

```powershell
python -m examples.getting_started.interface.native
python -m examples.getting_started.interface.numpy
python -m examples.getting_started.interface.jax
python -m examples.getting_started.interface.cupy
```

Optional backend examples should report missing dependencies clearly.

## Features

Feature examples each answer one question:

```powershell
python -m examples.features.derivative_styles
python -m examples.features.linearizer_styles
python -m examples.features.norm_policy
python -m examples.features.inverter_dense
python -m examples.features.inverter_krylov
python -m examples.features.scheme_predictor
python -m examples.features.monitor_vs_timing
python -m examples.features.custom_scheme_fixed_explicit
python -m examples.features.structured_state_minimal
python -m examples.features.compare_two_schemes
```

## Case studies

Case studies are longer narrative examples. They are allowed to introduce several concepts in sequence.

Use them after the focused examples, not before.

## Competition reports

Competition reports live under `competition/`. They are not tutorials; they are comparative runs that show timing and accuracy under named solver configurations.
