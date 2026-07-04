# Extend STARK

This page is for users implementing one method component while keeping the rest of STARK.

Use [foreign models](foreign-models.md) instead when your main problem is custom state and translation objects.

## Extension map

| You want to change... | Implement... | Domain |
|---|---|---|
| right-hand side adaptation | dynamics / `DynamicsStyle` | problem |
| Jacobian action | linearizer / `LinearizerStyle` | problem |
| time stepping | scheme | methods |
| nonlinear implicit solve | resolvent | methods |
| linear correction solve | inverter | methods |
| Krylov acceleration | preconditioner | methods |
| stage initial guess | scheme predictor | methods |
| observation | monitor | diagnostics |
| storage/backend | engine/carrier/accelerator | engines |

## Write a custom scheme

Start from a fixed explicit scheme. A scheme should look like a callable worker and prepare repeated work in `__init__`.

Run:

```powershell
python -m examples.methods.custom_scheme_fixed_explicit
```

Keep the call shape simple:

```python
scheme(interval, state, output)
```

## Write a custom inverter

An inverter solves a linear correction equation represented by an `InverterRequest`.

The current shape is:

```python
inverter(request, output)
```

Use `output_mode` to describe whether the inverter writes the output or improves an existing guess.

Start from:

```powershell
python -m examples.inverters.inverter_request_and_defect
python -m examples.inverters.inverter_dense
python -m examples.inverters.inverter_krylov
```

## Write a preconditioner

A preconditioner belongs with Krylov-style inversion. It should approximate the linear correction solve or apply an easier inverse-like operation.

Keep it explicit. Do not hide problem-specific preconditioning inside a generic Krylov inverter.

## Write a monitor

A monitor observes; it should not change the solve. Keep monitored and unmonitored timing separate.

Start from:

```powershell
python -m examples.diagnostics.monitor_scheme_steps
python -m examples.diagnostics.monitoring_levels
```

## Design rules for extensions

- Put protocols in contracts only when they decouple domains.
- Put concrete workers in the domain that owns them.
- Prepare repeated work in `__init__`.
- Keep `__call__` lean.
- Do not import concrete method workers into `core`.
- Do not add hidden optional dependencies to generic paths.

Read [House style](contributing/house-style.md) before adding a new public family.
