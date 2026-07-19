# Contracts Design Notes

`stark.core.contracts` defines stable protocol shapes for users and advanced
contributors who want to provide custom pieces.

Contracts are not concrete implementations. They are the vocabulary that lets
domains cooperate without importing each other's classes.

## Role

Use contracts when a boundary should accept user-defined or domain-defined
objects:

```text
custom dynamics       -> DynamicsLike
custom linearizer       -> LinearizerLike
custom scheme           -> SchemeLike
custom inverter         -> Inverter
custom state/translation -> State / Translation
```

Auditors can use these protocols to produce clearer errors than a later hot
path crash. Implementations can also use smaller local protocols when a local
shape is intentionally narrower than the public contract.

## Dependency Rule

Contracts belong in core because every other domain may need to depend on them.
They should avoid importing concrete problem, method, engine, or diagnostic
classes. If a contract needs to mention another shape, prefer another contract
or a minimal protocol.

## Type Documentation

Protocol and `TypeVar` docstrings are part of the public documentation layer.
Many users and contributors will meet these names first through IDE hover text,
not by opening the source file, so comments that explain variance, ownership,
and intended use should usually be docstrings attached to the symbol.

This is especially important for generic contract variables such as
`StateType`, `TranslationType`, and their covariant or contravariant variants.
Their names are short because they appear in many signatures, so their
docstrings must explain when to use each one. Good hover text keeps type hints
from becoming decorative noise.

Consistent use of these types is also what lets Pyright and similar tools give
useful warnings. If a contract falls back to `object`, `Any`, or an untyped
protocol too early, the checker stops seeing the state/translation handshake
that STARK relies on.

## Import Map

The package-level `stark.core.contracts` imports remain the convenience surface
for examples, docs, and interactive work. Implementation modules are grouped by
ownership so the package is easier to navigate:

- `stark.core.contracts.engines.accelerator`: accelerator backends
- `stark.core.contracts.methods.block`: grouped translation containers
- `stark.core.contracts.engines.carrier`: carrier bundles
- `stark.core.contracts.problem.dynamics`: right-hand-side workers
- `stark.core.contracts.problem.dynamics_split`: IMEX dynamics split protocol
- `stark.core.contracts.problem.field`: structured fields inside frame-like declarations
- `stark.core.contracts.problem.inner_product`: translation-space inner products
- `stark.core.contracts.methods.integrator`: trajectory-building workers
- `stark.core.contracts.shared.interval`: timeline cursors
- `stark.core.contracts.engines.linear_combine`: translation algebra fast paths
- `stark.core.contracts.problem.linearizer`: Jacobian-action workers
- `stark.core.contracts.problem.norm`: norm policies
- `stark.core.contracts.methods.stepper`: step-accepting workers
- `stark.core.contracts.methods.translation_operator`: matrix-free linear operators
- `stark.core.contracts.methods.residual`: residual workers for nonlinear solves
- `stark.core.contracts.methods.resolvent`: nonlinear stage solvers
- `stark.core.contracts.methods.inverter`: linear solvers and preconditioners
- `stark.core.contracts.problem.state`: unconstrained mutable state objects
- `stark.core.contracts.problem.translation`: linear update objects
- `stark.core.contracts.engines.allocator`: state and translation factories
