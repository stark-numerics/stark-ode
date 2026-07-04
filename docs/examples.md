# Examples guide

Examples are executable documentation. Prefer them over long copied code blocks
in the manual. The maintained inventory lives in `examples/manifest.py`; keep
new runnable modules there so the runners, docs, and tests do not drift apart.

Run all example groups:

```powershell
python -m examples
```

Run a specific group:

```powershell
python -m examples.getting_started
python -m examples.problem
python -m examples.methods
python -m examples.diagnostics
python -m examples.engines
python -m examples.inverters
python -m examples.core
```

## Getting started

| Example | Teaches |
|---|---|
| `examples.getting_started.scalar_decay` | smallest high-level solve |
| `examples.getting_started.numpy_oscillator` | vector field with NumPy |

## Problem

| Example | Teaches |
|---|---|
| `examples.problem.returning_dynamics` | return-style dynamics |
| `examples.problem.in_place_dynamics` | in-place dynamics |
| `examples.problem.multiple_fields` | structured state with more than one field |
| `examples.problem.structured_state_minimal` | nested structured state through named Frame paths |
| `examples.problem.foreign_model_allocator` | custom allocator for an existing object model |
| `examples.problem.foreign_model_plug_in_solver` | replace an existing time stepper without flattening the object model |
| `examples.problem.foreign_model_audit` | audit a custom adapter for an existing object model |
| `examples.problem.reaction_diffusion_array` | PDE-like array state through System |
| `examples.problem.dynamics_styles` | dynamics adapters |
| `examples.problem.linearizer_styles` | real linearizer in implicit Newton context |
| `examples.problem.norm_policy` | frame norm subtleties |

## Methods

| Example | Teaches |
|---|---|
| `examples.methods.choose_scheme` | choosing schemes |
| `examples.methods.custom_scheme_fixed_explicit` | minimal custom scheme |
| `examples.methods.imex_with_custom_spectral_resolvent` | custom Fourier-space resolvent for a split problem |
| `examples.methods.matrix_free_jacobian` | Newton with a Jacobian action and Krylov inverter |
| `examples.methods.resolvent_fixed_point` | Picard fixed-point resolvent |
| `examples.methods.resolvent_linearized` | Newton resolvent with a linearizer and inverter |
| `examples.methods.scheme_family_explicit_fixed` | fixed-step explicit scheme |
| `examples.methods.scheme_family_explicit_adaptive` | adaptive explicit scheme |
| `examples.methods.scheme_family_implicit_fixed` | fixed-step implicit scheme |
| `examples.methods.scheme_family_implicit_adaptive` | adaptive implicit scheme |
| `examples.methods.scheme_family_imex_fixed` | fixed-step IMEX scheme |
| `examples.methods.scheme_family_imex_adaptive` | adaptive IMEX scheme |
| `examples.methods.scheme_predictor` | implicit stage predictor policies |

## Diagnostics

| Example | Teaches |
|---|---|
| `examples.diagnostics.checkpoints` | output checkpoints vs internal steps |
| `examples.diagnostics.compare_two_schemes` | comparison report over method choices |
| `examples.diagnostics.monitor_scheme_steps` | accepted and rejected scheme step monitoring |
| `examples.diagnostics.error_ratio_trace` | interpreting adaptive scheme error ratios |
| `examples.diagnostics.compare_with_monitor_summary` | comparison report with monitor summaries |
| `examples.diagnostics.monitor_vs_timing` | diagnostics vs performance measurement |
| `examples.diagnostics.monitoring_levels` | monitor detail levels |

## Engines

| Example | Teaches |
|---|---|
| `examples.engines.engine_acceleration` | inspecting NumPy acceleration fallback |
| `examples.engines.backend_native` | native Python backend |
| `examples.engines.backend_numpy` | NumPy backend |
| `examples.engines.backend_jax` | optional JAX backend syntax |
| `examples.engines.backend_cupy` | optional CuPy backend syntax |

## Inverters

| Example | Teaches |
|---|---|
| `examples.inverters.inverter_request_and_defect` | inverter request and defect records |
| `examples.inverters.inverter_dense` | dense linear correction solve |
| `examples.inverters.inverter_krylov` | matrix-free Krylov solve and preconditioner hook |
| `examples.inverters.inverter_relaxation_richardson` | relaxation inverter baseline |
| `examples.inverters.inverter_relaxation_jacobi` | Jacobi-style relaxation |
| `examples.inverters.inverter_relaxation_specialist` | specialist relaxation inverter |

## Core

| Example | Teaches |
|---|---|
| `examples.core.manual_stepper_setup` | lower-level setup |

## Rule for new examples

An example should answer one question. If a file needs too many concepts, split
it. Repeating a little setup is fine when it helps the reader see the whole
lesson in one script.
