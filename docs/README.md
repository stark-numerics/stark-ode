# STARK documentation map

This directory is being refreshed after the package was reorganized around the
current public domains:

```text
stark.problem      problem declarations: System, Frame, Derivative, Linearizer
stark.methods      numerical methods: schemes, resolvents, inverters
stark.engines      backend execution: Native, NumPy, JAX, CuPy, accelerators
stark.core         integration runtime and extension contracts
stark.diagnostics  monitoring, comparison, and report helpers
```

The executable examples are the most reliable guide to the current API:

- [`examples/README.md`](../examples/README.md) maps the example tree.
- [`examples/getting_started`](../examples/getting_started) shows the high-level
  `System` + `Frame` interface.
- [`examples/features`](../examples/features) shows focused extension points.
- [`examples/case_studies`](../examples/case_studies) contains narrative examples.
- [`competition`](../competition) contains SciPy/Diffrax comparison reports.

## Current docs

- [`interface.md`](interface.md) gives a short current guide to the high-level
  problem layer.
- [`object_map.md`](object_map.md) summarizes the main object families.
- [`contracts_math.md`](contracts_math.md) gives the mathematical view of state,
  translation, derivative, linearizer, resolvent, and inverter contracts.
- [`house_style.md`](house_style.md) records contributor style and naming rules.
- [`benchmarking.md`](benchmarking.md) describes contributor performance tooling.

## Status note

Some advanced material is intentionally brief until the Krylov inverter family
has been refreshed around the newer request-shaped inverter protocol. For now,
prefer executable examples over older prose when they disagree.
