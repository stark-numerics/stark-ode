# Examples guide

Examples are executable documentation. Prefer them over long copied code blocks in the manual.
The maintained inventory lives in `examples/manifest.py`; keep new runnable
modules there so the runners, docs, and tests do not drift apart.

Run all example groups:

```powershell
python -m examples
```

Run a specific group:

```powershell
python -m examples.getting_started
python -m examples.backends
python -m examples.features
python -m examples.case_studies
```

## Getting started

| Example | Teaches |
|---|---|
| `examples.getting_started.scalar_decay` | smallest high-level solve |
| `examples.getting_started.numpy_oscillator` | vector field with NumPy |
| `examples.getting_started.returning_derivative` | return-style derivative |
| `examples.getting_started.in_place_derivative` | in-place derivative |
| `examples.getting_started.multiple_fields` | structured state with more than one field |
| `examples.getting_started.choose_scheme` | choosing schemes |
| `examples.getting_started.checkpoints` | output checkpoints vs internal steps |
| `examples.getting_started.engine_acceleration` | inspecting NumPy acceleration fallback |

## Backends

| Example | Teaches |
|---|---|
| `examples.backends.native` | native Python backend |
| `examples.backends.numpy` | NumPy backend |
| `examples.backends.jax` | optional JAX backend syntax |
| `examples.backends.cupy` | optional CuPy backend syntax |

## Focused features

| Example | Teaches |
|---|---|
| `examples.features.derivative_styles` | derivative adapters |
| `examples.features.linearizer_styles` | real linearizer in implicit Newton context |
| `examples.features.norm_policy` | frame norm subtleties |
| `examples.features.inverter_dense` | dense linear correction solve |
| `examples.features.inverter_krylov` | matrix-free Krylov solve and preconditioner hook |
| `examples.features.compare_two_schemes` | comparison report over method choices |
| `examples.features.monitor_scheme_steps` | accepted and rejected scheme step monitoring |
| `examples.features.compare_with_monitor_summary` | comparison report with monitor summaries |
| `examples.features.inverter_request_and_defect` | inverter request and defect records |
| `examples.features.inverter_relaxation_richardson` | relaxation inverter baseline |
| `examples.features.inverter_relaxation_jacobi` | Jacobi-style relaxation |
| `examples.features.inverter_relaxation_specialist` | specialist relaxation inverter |
| `examples.features.scheme_predictor` | implicit stage predictor policies |
| `examples.features.monitor_vs_timing` | diagnostics vs performance measurement |
| `examples.features.monitoring_levels` | monitor detail levels |
| `examples.features.manual_stepper_setup` | lower-level setup |
| `examples.features.structured_state_minimal` | nested structured state through named Frame paths |
| `examples.features.foreign_model_allocator` | custom allocator for an existing object model |

## Case studies

| Case study | Teaches |
|---|---|
| `examples.case_studies.three_body` | structured multi-field dynamics |
| `examples.case_studies.allen_cahn` | PDE-like stiffness, IMEX, implicit/Krylov methods |
| `examples.case_studies.backends` | NumPy/JAX/CuPy setup and timing caveats |

## Rule for new examples

A feature example should answer one question. A case study can tell a longer story. If a file needs too many concepts, split it.
