# STARK examples

The examples are grouped by how much context they assume.

## Getting started

Small first-contact examples for the high-level interface:

```powershell
python -m examples.getting_started.scalar_decay
python -m examples.getting_started.numpy_oscillator
python -m examples.getting_started.in_place_derivative
python -m examples.getting_started.choose_scheme
python -m examples.getting_started.checkpoints
python -m examples.getting_started.interface.native
python -m examples.getting_started.interface.numpy
```

## Features

Focused examples for specific STARK extension points:

```powershell
python -m examples.features.manual_marcher_setup
python -m examples.features.custom_scheme_fixed_explicit
python -m examples.features.structured_state_minimal
python -m examples.features.compare_two_schemes
python -m examples.features.monitor_scheme_steps
python -m examples.features.compare_with_monitor_summary
```

## Case studies

Longer teaching examples:

```powershell
python -m examples.case_studies.three_body
python -m examples.case_studies.allen_cahn
```

## Comparison

Comparison reports and backend-oriented experiments live under
`examples/comparison/`.
