# Contracts Design Notes

`stark.core.contracts` defines stable protocol shapes for users and advanced
contributors who want to provide custom pieces.

Contracts are not concrete implementations. They are the vocabulary that lets
domains cooperate without importing each other's classes.

## Role

Use contracts when a boundary should accept user-defined or domain-defined
objects:

```text
custom derivative       -> DerivativeLike
custom linearizer       -> LinearizerLike
custom scheme           -> SchemeLike
custom inverter         -> Inverter
custom state/translation -> StateLike / TranslationLike
```

Auditors can use these protocols to produce clearer errors than a later hot
path crash. Implementations can also use smaller local protocols when a local
shape is intentionally narrower than the public contract.

## Dependency Rule

Contracts belong in core because every other domain may need to depend on them.
They should avoid importing concrete problem, method, engine, or diagnostic
classes. If a contract needs to mention another shape, prefer another contract
or a minimal protocol.

## Import Map

Prefer focused imports when writing docs or implementation code:

- `stark.core.contracts.accelerator`: accelerator backends
- `stark.core.contracts.block`: grouped translation containers
- `stark.core.contracts.carrier`: carrier bundles
- `stark.core.contracts.derivative`: right-hand-side workers
- `stark.core.contracts.derivative_split`: IMEX derivative split protocol
- `stark.core.contracts.inner_product`: translation-space inner products
- `stark.core.contracts.integrator`: trajectory-building workers
- `stark.core.contracts.interval`: timeline cursors
- `stark.core.contracts.linear_combine`: translation algebra fast paths
- `stark.core.contracts.linearizer`: Jacobian-action workers
- `stark.core.contracts.stepper`: step-accepting workers
- `stark.core.contracts.operator`: matrix-free linear operators
- `stark.core.contracts.residual`: residual workers for nonlinear solves
- `stark.core.contracts.resolvent`: nonlinear stage solvers
- `stark.core.contracts.inverter`: linear solvers and preconditioners
- `stark.core.contracts.state`: unconstrained mutable state objects
- `stark.core.contracts.translation`: linear update objects
- `stark.core.contracts.allocator`: state and translation factories

Use the package-level `stark.core.contracts` imports for convenience in examples and interactive exploration.
