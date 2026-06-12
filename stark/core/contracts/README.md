# STARK Contracts

`stark.core.contracts` is the public documentation layer for the shapes STARK
accepts from user code. Runtime modules may use smaller local protocols when
that reduces coupling, but these files should explain the stable concepts a
user needs in order to supply custom objects.

Prefer focused imports when writing docs or implementation code:

- `stark.core.contracts.accelerator`: accelerator backends
- `stark.core.contracts.block`: grouped translation containers
- `stark.core.contracts.carrier`: carrier bundles
- `stark.core.contracts.derivative`: right-hand-side workers
- `stark.core.contracts.derivative_imex`: IMEX derivative split carrier
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

Use the package-level `stark.core.contracts` imports for convenience in examples and
interactive exploration.
